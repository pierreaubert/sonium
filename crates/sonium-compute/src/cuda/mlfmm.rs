//! Device-resident Complex64 MLFMM execution.
//!
//! The mesh tree and its numerical coefficients are deliberately supplied by
//! the owning BEM crate.  That keeps tree construction and quadrature policy
//! backend-independent, while this module owns the execution contract: after
//! construction, an action only reads resident hierarchy data and caller
//! supplied device vectors.

use num_complex::Complex64;

use crate::backend::{Backend, Transpose};
use crate::error::{BackendError, BackendResult};
use crate::pool::{Buffer, MemoryPool};
use crate::scalar::BufferElement;

use super::{CudaBackend, CudaBuffer};
use crate::{DeviceBufferLike, DeviceOperator};

/// A row-major dense transfer with explicit source and destination maps.
///
/// The operation represented by one value is
/// `dst[dst_indices] += matrix * src[src_indices]`.
#[derive(Debug, Clone)]
pub struct CudaMlfmmTransferC64Input {
    /// Row-major transfer coefficients.
    pub matrix: Vec<Complex64>,
    /// Source indices addressed by the matrix columns.
    pub src_indices: Vec<i32>,
    /// Destination indices addressed by the matrix rows.
    pub dst_indices: Vec<i32>,
}

/// One compact inter-level phase transfer.
#[derive(Debug, Clone)]
pub struct CudaMlfmmPhaseTransferC64Input {
    /// Source cluster in the adjacent level.
    pub source_cluster: usize,
    /// Destination cluster in the current level.
    pub destination_cluster: usize,
    /// Pointwise phase coefficients shared by the source and destination rows.
    pub phase: Vec<Complex64>,
}

/// One M2L diagonal translation supplied to the CUDA MLFMM operator.
#[derive(Debug, Clone)]
pub struct CudaMlfmmTranslationC64Input {
    /// Source expansion cluster at this level.
    pub source_cluster: usize,
    /// Field expansion cluster at this level.
    pub field_cluster: usize,
    /// Diagonal phase/translation coefficients, one per sphere point.
    pub diagonal: Vec<Complex64>,
}

/// One near-field block in global DOF coordinates.
#[derive(Debug, Clone)]
pub struct CudaMlfmmNearBlockC64Input {
    /// Row (output/source-side) global DOF indices.
    pub row_indices: Vec<i32>,
    /// Column (input/field-side) global DOF indices.
    pub col_indices: Vec<i32>,
    /// Row-major near-field coefficients.
    pub matrix: Vec<Complex64>,
}

/// All transfer and translation data for one MLFMM level.
#[derive(Debug, Clone)]
pub struct CudaMlfmmLevelC64Input {
    /// Number of expansion points per cluster at this level.
    pub sphere_points: usize,
    /// M2M transfers.  The final level contains leaf element-to-multipole
    /// transfers; preceding levels contain child-to-parent transfers.
    pub t_transfers: Vec<CudaMlfmmTransferC64Input>,
    /// L2L transfers.  The final level contains multipole-to-element output
    /// transfers; preceding levels contain parent-to-child transfers.
    pub s_transfers: Vec<CudaMlfmmTransferC64Input>,
    /// Compact M2M transfers used by the CPU tree when phase plans are valid.
    pub t_phase_transfers: Vec<CudaMlfmmPhaseTransferC64Input>,
    /// Compact L2L transfers used by the CPU tree when phase plans are valid.
    pub s_phase_transfers: Vec<CudaMlfmmPhaseTransferC64Input>,
    /// M2L translations at this level.
    pub translations: Vec<CudaMlfmmTranslationC64Input>,
}

#[derive(Clone, Copy)]
enum Action {
    Forward,
    Transpose,
    Hermitian,
}

struct DeviceDenseC64 {
    rows: usize,
    cols: usize,
    matrix: Buffer<Complex64, CudaBackend>,
    transpose: Buffer<Complex64, CudaBackend>,
    hermitian: Buffer<Complex64, CudaBackend>,
    src_indices: Buffer<i32, CudaBackend>,
    dst_indices: Buffer<i32, CudaBackend>,
}

struct DeviceTranslationC64 {
    diagonal: Buffer<Complex64, CudaBackend>,
    hermitian_diagonal: Buffer<Complex64, CudaBackend>,
    src_indices: Buffer<i32, CudaBackend>,
    dst_indices: Buffer<i32, CudaBackend>,
}

struct DevicePhaseTransferC64 {
    phase: Buffer<Complex64, CudaBackend>,
    hermitian_phase: Buffer<Complex64, CudaBackend>,
    src_indices: Buffer<i32, CudaBackend>,
    dst_indices: Buffer<i32, CudaBackend>,
}

struct DeviceLevelC64 {
    sphere_points: usize,
    t_transfers: Vec<DeviceDenseC64>,
    s_transfers: Vec<DeviceDenseC64>,
    t_phase_transfers: Vec<DevicePhaseTransferC64>,
    s_phase_transfers: Vec<DevicePhaseTransferC64>,
    translations: Vec<DeviceTranslationC64>,
}

struct PhaseWorkspaceC64 {
    len: usize,
    input: Buffer<Complex64, CudaBackend>,
    output: Buffer<Complex64, CudaBackend>,
}

struct DeviceWorkspaceC64 {
    multipoles: Vec<Buffer<Complex64, CudaBackend>>,
    locals: Vec<Buffer<Complex64, CudaBackend>>,
    phase: Vec<Vec<PhaseWorkspaceC64>>,
    result: Buffer<Complex64, CudaBackend>,
}

/// Device-resident MLFMM operator for Complex64.
///
/// Setup is intentionally host-shared: callers construct and validate the
/// tree and upload it through this constructor.  Forward, transpose, and
/// Hermitian actions thereafter use resident matrices, index maps, phases,
/// near blocks, and workspaces; no host staging occurs in `try_apply_device`.
pub struct CudaMlfmmC64 {
    pool: MemoryPool<CudaBackend>,
    num_dofs: usize,
    levels: Vec<DeviceLevelC64>,
    near_blocks: Vec<DeviceDenseC64>,
    workspace: std::sync::Mutex<DeviceWorkspaceC64>,
}

impl DeviceOperator<CudaBackend, Complex64> for CudaMlfmmC64 {
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
            log::error!("CUDA MLFMM action failed: {error}");
        }
    }

    fn try_apply(
        &self,
        backend: &CudaBackend,
        x: &CudaBuffer<Complex64>,
        y: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        if !std::sync::Arc::ptr_eq(backend.stream(), self.pool.backend().stream()) {
            return Err(invalid(
                "cuda_mlfmm_operator",
                "solver and operator stream mismatch",
            ));
        }
        self.try_apply_raw(x, y, Transpose::NoTrans)
    }

    fn try_apply_colmajor(
        &self,
        backend: &CudaBackend,
        rhs: usize,
        x: &CudaBuffer<Complex64>,
        y: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        if rhs == 1 {
            return DeviceOperator::try_apply(self, backend, x, y);
        }
        if rhs == 0 && x.len() == 0 && y.len() == 0 {
            return Ok(());
        }
        // The generic default stages whole vectors on the host. Do not expose
        // that fallback through this native scalar adapter; block residency
        // needs its own checked workspace and production integration.
        Err(BackendError::Unsupported {
            backend: "cuda",
            operation: "cuda_mlfmm_block",
            scalar: "Complex64 multi-RHS MLFMM",
        })
    }

    fn try_residual(
        &self,
        backend: &CudaBackend,
        x: &CudaBuffer<Complex64>,
        b: &CudaBuffer<Complex64>,
        out: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        if b.len() != self.num_dofs
            || (self.num_dofs > 0
                && b.raw().is_none_or(|raw| {
                    !std::sync::Arc::ptr_eq(raw.stream(), self.pool.backend().stream())
                        || b.len().checked_mul(std::mem::size_of::<Complex64>()) != Some(raw.len())
                }))
        {
            return Err(invalid(
                "cuda_mlfmm_residual",
                "RHS dimension, allocation or stream mismatch",
            ));
        }
        DeviceOperator::try_apply(self, backend, x, out)?;
        backend.try_scale(Complex64::new(-1.0, 0.0), out)?;
        backend.try_axpy(Complex64::new(1.0, 0.0), b, out)
    }
}

fn invalid(operation: &'static str, reason: impl Into<String>) -> BackendError {
    BackendError::InvalidArgument {
        operation,
        reason: reason.into(),
    }
}

fn checked_i32_index(index: i32, limit: usize, operation: &'static str) -> BackendResult<usize> {
    if index < 0
        || usize::try_from(index)
            .ok()
            .is_none_or(|value| value >= limit)
    {
        return Err(invalid(
            operation,
            format!("index {index} is outside a buffer of length {limit}"),
        ));
    }
    Ok(index as usize)
}

fn matrix_transforms(
    matrix: &[Complex64],
    rows: usize,
    cols: usize,
) -> (Vec<Complex64>, Vec<Complex64>) {
    let mut transpose = vec![Complex64::new(0.0, 0.0); rows.saturating_mul(cols)];
    let mut hermitian = vec![Complex64::new(0.0, 0.0); rows.saturating_mul(cols)];
    for row in 0..rows {
        for col in 0..cols {
            let value = matrix[row * cols + col];
            transpose[col * rows + row] = value;
            hermitian[col * rows + row] = value.conj();
        }
    }
    (transpose, hermitian)
}

fn upload<T: BufferElement + Copy>(
    pool: &MemoryPool<CudaBackend>,
    values: &[T],
) -> BackendResult<Buffer<T, CudaBackend>> {
    let mut buffer = pool.try_alloc(values.len()).map_err(BackendError::from)?;
    if !values.is_empty() {
        pool.backend().try_copy_h2d(values, buffer.raw_mut())?;
    }
    Ok(buffer)
}

fn validate_indices(indices: &[i32], limit: usize, operation: &'static str) -> BackendResult<()> {
    for &index in indices {
        checked_i32_index(index, limit, operation)?;
    }
    Ok(())
}

fn cluster_index_range(
    cluster: usize,
    stride: usize,
    length: usize,
    operation: &'static str,
) -> BackendResult<Vec<i32>> {
    let start = cluster
        .checked_mul(stride)
        .ok_or_else(|| invalid(operation, "cluster index range overflows usize"))?;
    (0..length)
        .map(|point| {
            let index = start
                .checked_add(point)
                .ok_or_else(|| invalid(operation, "cluster point index overflows usize"))?;
            i32::try_from(index)
                .map_err(|_| invalid(operation, "cluster point index exceeds CUDA index range"))
        })
        .collect()
}

impl CudaMlfmmC64 {
    /// Upload a validated MLFMM hierarchy and allocate all reusable action
    /// workspaces.
    pub fn new(
        pool: MemoryPool<CudaBackend>,
        num_dofs: usize,
        levels: Vec<CudaMlfmmLevelC64Input>,
        near_blocks: Vec<CudaMlfmmNearBlockC64Input>,
    ) -> BackendResult<Self> {
        let operation = "cuda_mlfmm_constructor";
        if levels.is_empty() && num_dofs != 0 {
            return Err(invalid(
                operation,
                "a non-empty operator requires at least one level",
            ));
        }
        if levels.iter().any(|level| level.sphere_points == 0) {
            return Err(invalid(operation, "sphere-point counts must be non-zero"));
        }
        let cluster_counts: Vec<usize> = levels
            .iter()
            .map(|level| {
                if level.t_transfers.len() != level.s_transfers.len() {
                    return Err(invalid(
                        operation,
                        "T and S transfer counts must match at every level",
                    ));
                }
                Ok(level.t_transfers.len())
            })
            .collect::<BackendResult<_>>()?;
        let level_points: Vec<usize> = levels.iter().map(|level| level.sphere_points).collect();
        if num_dofs != 0 && cluster_counts.contains(&0) {
            return Err(invalid(
                operation,
                "a non-empty operator requires at least one cluster at every level",
            ));
        }

        let mut device_levels = Vec::with_capacity(levels.len());
        for (level_index, level) in levels.into_iter().enumerate() {
            let points = level.sphere_points;
            let cluster_count = cluster_counts[level_index];
            let next_points = level_points.get(level_index + 1).copied().unwrap_or(points);
            let next_cluster_count = cluster_counts
                .get(level_index + 1)
                .copied()
                .unwrap_or(cluster_count);
            let current_entries = cluster_count
                .checked_mul(points)
                .ok_or_else(|| invalid(operation, "level index range overflows usize"))?;
            let next_entries = next_cluster_count
                .checked_mul(next_points)
                .ok_or_else(|| invalid(operation, "adjacent level index range overflows usize"))?;
            let mut t_transfers = Vec::with_capacity(cluster_count);
            let mut s_transfers = Vec::with_capacity(cluster_count);
            for (cluster, transfer) in level.t_transfers.into_iter().enumerate() {
                t_transfers.push(Self::upload_dense(
                    &pool,
                    transfer,
                    if level_index + 1 == cluster_counts.len() {
                        num_dofs
                    } else {
                        next_entries
                    },
                    current_entries,
                    operation,
                    format!("T level {level_index} cluster {cluster}"),
                )?);
            }
            for (cluster, transfer) in level.s_transfers.into_iter().enumerate() {
                s_transfers.push(Self::upload_dense(
                    &pool,
                    transfer,
                    current_entries,
                    if level_index + 1 == cluster_counts.len() {
                        num_dofs
                    } else {
                        next_entries
                    },
                    operation,
                    format!("S level {level_index} cluster {cluster}"),
                )?);
            }
            let mut t_phase_transfers = Vec::new();
            for (entry, transfer) in level.t_phase_transfers.into_iter().enumerate() {
                if level_index + 1 >= cluster_counts.len()
                    || transfer.source_cluster >= next_cluster_count
                    || transfer.destination_cluster >= cluster_count
                {
                    return Err(invalid(
                        operation,
                        format!(
                            "M2M phase level {level_index} entry {entry} references an invalid cluster"
                        ),
                    ));
                }
                let expected = points.min(next_points);
                if transfer.phase.len() != expected
                    || transfer
                        .phase
                        .iter()
                        .any(|value| !value.re.is_finite() || !value.im.is_finite())
                {
                    return Err(invalid(
                        operation,
                        format!(
                            "M2M phase level {level_index} entry {entry} has invalid length or coefficients"
                        ),
                    ));
                }
                t_phase_transfers.push(Self::upload_phase(
                    &pool,
                    transfer,
                    next_points,
                    points,
                    operation,
                )?);
            }
            let mut s_phase_transfers = Vec::new();
            for (entry, transfer) in level.s_phase_transfers.into_iter().enumerate() {
                if level_index + 1 >= cluster_counts.len()
                    || transfer.source_cluster >= cluster_count
                    || transfer.destination_cluster >= next_cluster_count
                {
                    return Err(invalid(
                        operation,
                        format!(
                            "L2L phase level {level_index} entry {entry} references an invalid cluster"
                        ),
                    ));
                }
                let expected = points.min(next_points);
                if transfer.phase.len() != expected
                    || transfer
                        .phase
                        .iter()
                        .any(|value| !value.re.is_finite() || !value.im.is_finite())
                {
                    return Err(invalid(
                        operation,
                        format!(
                            "L2L phase level {level_index} entry {entry} has invalid length or coefficients"
                        ),
                    ));
                }
                s_phase_transfers.push(Self::upload_phase(
                    &pool,
                    transfer,
                    points,
                    next_points,
                    operation,
                )?);
            }
            let mut translations = Vec::with_capacity(level.translations.len());
            for (entry, translation) in level.translations.into_iter().enumerate() {
                if translation.source_cluster >= cluster_count
                    || translation.field_cluster >= cluster_count
                {
                    return Err(invalid(
                        operation,
                        format!(
                            "M2L level {level_index} entry {entry} references an invalid cluster"
                        ),
                    ));
                }
                if translation.diagonal.len() != points
                    || translation
                        .diagonal
                        .iter()
                        .any(|value| !value.re.is_finite() || !value.im.is_finite())
                {
                    return Err(invalid(
                        operation,
                        format!("M2L level {level_index} entry {entry} has invalid diagonal data"),
                    ));
                }
                let src_indices =
                    cluster_index_range(translation.source_cluster, points, points, operation)?;
                let dst_indices =
                    cluster_index_range(translation.field_cluster, points, points, operation)?;
                let hermitian_diagonal = translation
                    .diagonal
                    .iter()
                    .map(Complex64::conj)
                    .collect::<Vec<_>>();
                translations.push(DeviceTranslationC64 {
                    diagonal: upload(&pool, &translation.diagonal)?,
                    hermitian_diagonal: upload(&pool, &hermitian_diagonal)?,
                    src_indices: upload(&pool, &src_indices)?,
                    dst_indices: upload(&pool, &dst_indices)?,
                });
            }
            device_levels.push(DeviceLevelC64 {
                sphere_points: points,
                t_transfers,
                s_transfers,
                t_phase_transfers,
                s_phase_transfers,
                translations,
            });
        }

        let mut device_near = Vec::with_capacity(near_blocks.len());
        for (block, near) in near_blocks.into_iter().enumerate() {
            validate_indices(&near.row_indices, num_dofs, operation)?;
            validate_indices(&near.col_indices, num_dofs, operation)?;
            device_near.push(Self::upload_dense(
                &pool,
                CudaMlfmmTransferC64Input {
                    matrix: near.matrix,
                    src_indices: near.col_indices,
                    dst_indices: near.row_indices,
                },
                num_dofs,
                num_dofs,
                operation,
                format!("near block {block}"),
            )?);
        }

        let mut multipoles = Vec::with_capacity(device_levels.len());
        let mut locals = Vec::with_capacity(device_levels.len());
        let mut phase = Vec::with_capacity(device_levels.len());
        for level in &device_levels {
            let count = level.t_transfers.len();
            let entries = count
                .checked_mul(level.sphere_points)
                .ok_or_else(|| invalid(operation, "workspace size overflows usize"))?;
            multipoles.push(pool.try_alloc(entries).map_err(BackendError::from)?);
            locals.push(pool.try_alloc(entries).map_err(BackendError::from)?);
            // Phase kernels require exact-length source, phase, and destination
            // buffers.  Keep one persistent pair for every phase/translation
            // length used at this level; adjacent levels may have different
            // expansion orders.
            let mut lengths = level
                .t_phase_transfers
                .iter()
                .chain(level.s_phase_transfers.iter())
                .map(|transfer| transfer.phase.len())
                .chain(
                    level
                        .translations
                        .iter()
                        .map(|translation| translation.diagonal.len()),
                )
                .collect::<Vec<_>>();
            lengths.sort_unstable();
            lengths.dedup();
            let mut level_phase = Vec::with_capacity(lengths.len());
            for len in lengths {
                level_phase.push(PhaseWorkspaceC64 {
                    len,
                    input: pool.try_alloc(len).map_err(BackendError::from)?,
                    output: pool.try_alloc(len).map_err(BackendError::from)?,
                });
            }
            phase.push(level_phase);
        }
        let result = pool.try_alloc(num_dofs).map_err(BackendError::from)?;
        Ok(Self {
            pool,
            num_dofs,
            levels: device_levels,
            near_blocks: device_near,
            workspace: std::sync::Mutex::new(DeviceWorkspaceC64 {
                multipoles,
                locals,
                phase,
                result,
            }),
        })
    }

    fn upload_dense(
        pool: &MemoryPool<CudaBackend>,
        input: CudaMlfmmTransferC64Input,
        src_limit: usize,
        dst_limit: usize,
        operation: &'static str,
        label: String,
    ) -> BackendResult<DeviceDenseC64> {
        let rows = input.dst_indices.len();
        let cols = input.src_indices.len();
        let expected = rows
            .checked_mul(cols)
            .ok_or_else(|| invalid(operation, format!("{label} shape overflows usize")))?;
        if input.matrix.len() != expected {
            return Err(invalid(
                operation,
                format!(
                    "{label} matrix has {} entries, expected {expected}",
                    input.matrix.len()
                ),
            ));
        }
        validate_indices(&input.src_indices, src_limit, operation)?;
        validate_indices(&input.dst_indices, dst_limit, operation)?;
        if input
            .matrix
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(invalid(
                operation,
                format!("{label} contains a non-finite coefficient"),
            ));
        }
        let (transpose, hermitian) = matrix_transforms(&input.matrix, rows, cols);
        Ok(DeviceDenseC64 {
            rows,
            cols,
            matrix: upload(pool, &input.matrix)?,
            transpose: upload(pool, &transpose)?,
            hermitian: upload(pool, &hermitian)?,
            src_indices: upload(pool, &input.src_indices)?,
            dst_indices: upload(pool, &input.dst_indices)?,
        })
    }

    fn upload_phase(
        pool: &MemoryPool<CudaBackend>,
        input: CudaMlfmmPhaseTransferC64Input,
        source_points: usize,
        destination_points: usize,
        operation: &'static str,
    ) -> BackendResult<DevicePhaseTransferC64> {
        let length = input.phase.len();
        let src_indices =
            cluster_index_range(input.source_cluster, source_points, length, operation)?;
        let dst_indices = cluster_index_range(
            input.destination_cluster,
            destination_points,
            length,
            operation,
        )?;
        let hermitian_phase = input.phase.iter().map(Complex64::conj).collect::<Vec<_>>();
        Ok(DevicePhaseTransferC64 {
            phase: upload(pool, &input.phase)?,
            hermitian_phase: upload(pool, &hermitian_phase)?,
            src_indices: upload(pool, &src_indices)?,
            dst_indices: upload(pool, &dst_indices)?,
        })
    }

    fn phase_workspace_index(
        slots: &[PhaseWorkspaceC64],
        len: usize,
        operation: &'static str,
    ) -> BackendResult<usize> {
        slots
            .iter()
            .position(|slot| slot.len == len)
            .ok_or_else(|| invalid(operation, "missing resident phase workspace"))
    }

    /// Number of global rows/columns handled by this operator.
    pub fn num_dofs(&self) -> usize {
        self.num_dofs
    }

    /// Apply the resident operator to device vectors.
    ///
    /// The output is copied from a private device workspace only after all
    /// launches synchronize successfully, so a failed launch or synchronization
    /// leaves the caller's output buffer untouched.
    pub fn try_apply_device(
        &self,
        input: &Buffer<Complex64, CudaBackend>,
        output: &mut Buffer<Complex64, CudaBackend>,
        transpose: Transpose,
    ) -> BackendResult<()> {
        self.try_apply_raw(input.raw(), output.raw_mut(), transpose)
    }

    /// Apply an oriented MLFMM action to borrowed solver buffers.
    ///
    /// Reuses resident workspaces without allocating or staging vectors on the
    /// host. Output publication follows `try_apply_device` synchronization.
    ///
    /// # Errors
    /// Rejects dimension, allocation and stream mismatches and backend failures.
    pub fn try_apply_raw(
        &self,
        input: &CudaBuffer<Complex64>,
        output: &mut CudaBuffer<Complex64>,
        transpose: Transpose,
    ) -> BackendResult<()> {
        let operation = "cuda_mlfmm_apply";
        if input.len() != self.num_dofs || output.len() != self.num_dofs {
            return Err(invalid(
                operation,
                format!(
                    "input/output lengths must equal operator dimension {} ({} / {})",
                    self.num_dofs,
                    input.len(),
                    output.len()
                ),
            ));
        }
        if self.num_dofs > 0 {
            for buffer in [input, &*output] {
                if buffer.raw().is_none_or(|raw| {
                    !std::sync::Arc::ptr_eq(raw.stream(), self.pool.backend().stream())
                        || buffer.len().checked_mul(std::mem::size_of::<Complex64>())
                            != Some(raw.len())
                }) {
                    return Err(invalid(
                        operation,
                        "buffer allocation or CUDA stream mismatch",
                    ));
                }
            }
        }
        self.pool.backend().try_fmm_action_preflight()?;
        let action = match transpose {
            Transpose::NoTrans => Action::Forward,
            Transpose::Trans => Action::Transpose,
            Transpose::ConjTrans => Action::Hermitian,
        };
        let mut workspace = self
            .workspace
            .lock()
            .map_err(|_| BackendError::OperationFailed {
                operation,
                reason: "MLFMM workspace mutex poisoned".into(),
            })?;
        for buffer in &mut workspace.multipoles {
            self.pool
                .backend()
                .try_scale(Complex64::new(0.0, 0.0), buffer.raw_mut())?;
        }
        for buffer in &mut workspace.locals {
            self.pool
                .backend()
                .try_scale(Complex64::new(0.0, 0.0), buffer.raw_mut())?;
        }
        if action_is_forward(action) {
            self.apply_near(action, input, &mut workspace)?;
            self.apply_forward_far(input, &mut workspace)?;
            self.publish_forward(&mut workspace)?;
        } else {
            self.apply_near(action, input, &mut workspace)?;
            self.apply_adjoint_far(action, input, &mut workspace)?;
            self.publish_adjoint(action, &mut workspace)?;
        }
        self.pool.backend().try_synchronize()?;
        self.pool
            .backend()
            .try_copy_d2d(workspace.result.raw(), output)
    }

    fn apply_near(
        &self,
        action: Action,
        input: &CudaBuffer<Complex64>,
        workspace: &mut DeviceWorkspaceC64,
    ) -> BackendResult<()> {
        self.pool
            .backend()
            .try_scale(Complex64::new(0.0, 0.0), workspace.result.raw_mut())?;
        for block in &self.near_blocks {
            self.apply_dense(block, action, input, &mut workspace.result)?;
        }
        // `workspace` is accepted here to make the near/far staging boundary
        // explicit and to prevent future implementations from introducing a
        // host temporary for near blocks.
        let _ = workspace;
        Ok(())
    }

    fn apply_dense(
        &self,
        dense: &DeviceDenseC64,
        action: Action,
        input: &CudaBuffer<Complex64>,
        output: &mut Buffer<Complex64, CudaBackend>,
    ) -> BackendResult<()> {
        let (matrix, rows, cols, src, dst) = match action {
            Action::Forward => (
                &dense.matrix,
                dense.rows,
                dense.cols,
                &dense.src_indices,
                &dense.dst_indices,
            ),
            Action::Transpose => (
                &dense.transpose,
                dense.cols,
                dense.rows,
                &dense.dst_indices,
                &dense.src_indices,
            ),
            Action::Hermitian => (
                &dense.hermitian,
                dense.cols,
                dense.rows,
                &dense.dst_indices,
                &dense.src_indices,
            ),
        };
        self.pool
            .backend()
            .try_fmm_indexed_dense_matvec_accumulate_c64(
                matrix.raw(),
                rows,
                cols,
                cols,
                input,
                src.raw(),
                output.raw_mut(),
                dst.raw(),
            )
    }

    fn apply_phase(
        &self,
        transfer: &DevicePhaseTransferC64,
        action: Action,
        input: &Buffer<Complex64, CudaBackend>,
        output: &mut Buffer<Complex64, CudaBackend>,
        phase_input: &mut Buffer<Complex64, CudaBackend>,
        phase_output: &mut Buffer<Complex64, CudaBackend>,
    ) -> BackendResult<()> {
        let (phase, src, dst) = match action {
            Action::Forward | Action::Transpose => (
                &transfer.phase,
                if matches!(action, Action::Forward) {
                    &transfer.src_indices
                } else {
                    &transfer.dst_indices
                },
                if matches!(action, Action::Forward) {
                    &transfer.dst_indices
                } else {
                    &transfer.src_indices
                },
            ),
            Action::Hermitian => (
                &transfer.hermitian_phase,
                &transfer.dst_indices,
                &transfer.src_indices,
            ),
        };
        self.pool.backend().try_fmm_compact_indexed_gather_c64(
            input.raw(),
            src.raw(),
            phase_input.raw_mut(),
        )?;
        self.pool
            .backend()
            .try_scale(Complex64::new(0.0, 0.0), phase_output.raw_mut())?;
        self.pool.backend().try_fmm_phase_mul_accumulate_c64(
            phase_output.raw_mut(),
            phase.raw(),
            phase_input.raw(),
        )?;
        self.pool
            .backend()
            .try_fmm_compact_indexed_scatter_accumulate_c64(
                phase_output.raw(),
                dst.raw(),
                output.raw_mut(),
            )
    }

    fn apply_forward_far(
        &self,
        input: &CudaBuffer<Complex64>,
        workspace: &mut DeviceWorkspaceC64,
    ) -> BackendResult<()> {
        let last = self.levels.len().saturating_sub(1);
        if self.levels.len() <= 1 {
            return Ok(());
        }
        for transfer in &self.levels[last].t_transfers {
            self.apply_dense(
                transfer,
                Action::Forward,
                input,
                &mut workspace.multipoles[last],
            )?;
        }
        for level_index in (0..last).rev() {
            let (parents, children) = workspace.multipoles.split_at_mut(level_index + 1);
            let source = &children[0];
            let destination = &mut parents[level_index];
            if self.levels[level_index].t_phase_transfers.is_empty() {
                for transfer in &self.levels[level_index].t_transfers {
                    self.apply_dense(transfer, Action::Forward, source.raw(), destination)?;
                }
            } else {
                for transfer in &self.levels[level_index].t_phase_transfers {
                    let slot_index = Self::phase_workspace_index(
                        &workspace.phase[level_index],
                        transfer.phase.len(),
                        "cuda_mlfmm_apply",
                    )?;
                    let slot = &mut workspace.phase[level_index][slot_index];
                    self.apply_phase(
                        transfer,
                        Action::Forward,
                        source,
                        destination,
                        &mut slot.input,
                        &mut slot.output,
                    )?;
                }
            }
        }
        self.apply_translations(Action::Forward, workspace)?;
        for level_index in 0..last {
            let (parents, children) = workspace.locals.split_at_mut(level_index + 1);
            let source = &parents[level_index];
            let destination = &mut children[0];
            if self.levels[level_index].s_phase_transfers.is_empty() {
                for transfer in &self.levels[level_index].s_transfers {
                    self.apply_dense(transfer, Action::Forward, source.raw(), destination)?;
                }
            } else {
                for transfer in &self.levels[level_index].s_phase_transfers {
                    let slot_index = Self::phase_workspace_index(
                        &workspace.phase[level_index],
                        transfer.phase.len(),
                        "cuda_mlfmm_apply",
                    )?;
                    let slot = &mut workspace.phase[level_index][slot_index];
                    self.apply_phase(
                        transfer,
                        Action::Forward,
                        source,
                        destination,
                        &mut slot.input,
                        &mut slot.output,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn apply_adjoint_far(
        &self,
        action: Action,
        input: &CudaBuffer<Complex64>,
        workspace: &mut DeviceWorkspaceC64,
    ) -> BackendResult<()> {
        let last = self.levels.len().saturating_sub(1);
        if self.levels.len() <= 1 {
            return Ok(());
        }
        for transfer in &self.levels[last].s_transfers {
            self.apply_dense(transfer, action, input, &mut workspace.locals[last])?;
        }
        for level_index in (0..last).rev() {
            let (parents, children) = workspace.locals.split_at_mut(level_index + 1);
            let source = &children[0];
            let destination = &mut parents[level_index];
            if self.levels[level_index].s_phase_transfers.is_empty() {
                for transfer in &self.levels[level_index].s_transfers {
                    self.apply_dense(transfer, action, source.raw(), destination)?;
                }
            } else {
                for transfer in &self.levels[level_index].s_phase_transfers {
                    let slot_index = Self::phase_workspace_index(
                        &workspace.phase[level_index],
                        transfer.phase.len(),
                        "cuda_mlfmm_apply",
                    )?;
                    let slot = &mut workspace.phase[level_index][slot_index];
                    self.apply_phase(
                        transfer,
                        action,
                        source,
                        destination,
                        &mut slot.input,
                        &mut slot.output,
                    )?;
                }
            }
        }
        self.apply_translations(action, workspace)?;
        for level_index in 0..last {
            let (parents, children) = workspace.multipoles.split_at_mut(level_index + 1);
            let source = &parents[level_index];
            let destination = &mut children[0];
            if self.levels[level_index].t_phase_transfers.is_empty() {
                for transfer in &self.levels[level_index].t_transfers {
                    self.apply_dense(transfer, action, source.raw(), destination)?;
                }
            } else {
                for transfer in &self.levels[level_index].t_phase_transfers {
                    let slot_index = Self::phase_workspace_index(
                        &workspace.phase[level_index],
                        transfer.phase.len(),
                        "cuda_mlfmm_apply",
                    )?;
                    let slot = &mut workspace.phase[level_index][slot_index];
                    self.apply_phase(
                        transfer,
                        action,
                        source,
                        destination,
                        &mut slot.input,
                        &mut slot.output,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn apply_translations(
        &self,
        action: Action,
        workspace: &mut DeviceWorkspaceC64,
    ) -> BackendResult<()> {
        for (level_index, level) in self.levels.iter().enumerate() {
            for translation in &level.translations {
                let (phase, src, dst) = match action {
                    Action::Forward => (
                        &translation.diagonal,
                        &translation.src_indices,
                        &translation.dst_indices,
                    ),
                    Action::Transpose => (
                        &translation.diagonal,
                        &translation.dst_indices,
                        &translation.src_indices,
                    ),
                    Action::Hermitian => (
                        &translation.hermitian_diagonal,
                        &translation.dst_indices,
                        &translation.src_indices,
                    ),
                };
                let source = if action_is_forward(action) {
                    &workspace.multipoles[level_index]
                } else {
                    &workspace.locals[level_index]
                };
                let slot_index = Self::phase_workspace_index(
                    &workspace.phase[level_index],
                    phase.len(),
                    "cuda_mlfmm_apply",
                )?;
                let slot = &mut workspace.phase[level_index][slot_index];
                self.pool.backend().try_fmm_compact_indexed_gather_c64(
                    source.raw(),
                    src.raw(),
                    slot.input.raw_mut(),
                )?;
                self.pool
                    .backend()
                    .try_scale(Complex64::new(0.0, 0.0), slot.output.raw_mut())?;
                self.pool.backend().try_fmm_phase_mul_accumulate_c64(
                    slot.output.raw_mut(),
                    phase.raw(),
                    slot.input.raw(),
                )?;
                let destination = if action_is_forward(action) {
                    &mut workspace.locals[level_index]
                } else {
                    &mut workspace.multipoles[level_index]
                };
                self.pool
                    .backend()
                    .try_fmm_compact_indexed_scatter_accumulate_c64(
                        slot.output.raw(),
                        dst.raw(),
                        destination.raw_mut(),
                    )?;
            }
        }
        Ok(())
    }

    fn publish_forward(&self, workspace: &mut DeviceWorkspaceC64) -> BackendResult<()> {
        if self.levels.len() <= 1 {
            return Ok(());
        }
        let last = self.levels.len() - 1;
        for transfer in &self.levels[last].s_transfers {
            self.apply_dense(
                transfer,
                Action::Forward,
                workspace.locals[last].raw(),
                &mut workspace.result,
            )?;
        }
        Ok(())
    }

    fn publish_adjoint(
        &self,
        action: Action,
        workspace: &mut DeviceWorkspaceC64,
    ) -> BackendResult<()> {
        if self.levels.len() <= 1 {
            return Ok(());
        }
        let last = self.levels.len() - 1;
        for transfer in &self.levels[last].t_transfers {
            self.apply_dense(
                transfer,
                action,
                workspace.multipoles[last].raw(),
                &mut workspace.result,
            )?;
        }
        Ok(())
    }

    /// Apply the resident operator and explicitly publish a host result.
    pub fn try_apply(
        &self,
        input: &[Complex64],
        transpose: Transpose,
    ) -> BackendResult<Vec<Complex64>> {
        if input.len() != self.num_dofs {
            return Err(invalid(
                "cuda_mlfmm_apply",
                format!(
                    "input length {} != operator dimension {}",
                    input.len(),
                    self.num_dofs
                ),
            ));
        }
        let device_input = upload(&self.pool, input)?;
        let mut device_output = self
            .pool
            .try_alloc(self.num_dofs)
            .map_err(BackendError::from)?;
        self.try_apply_device(&device_input, &mut device_output, transpose)?;
        let mut output = vec![Complex64::new(0.0, 0.0); self.num_dofs];
        self.pool
            .backend()
            .try_copy_d2h(device_output.raw(), &mut output)?;
        Ok(output)
    }
}

fn action_is_forward(action: Action) -> bool {
    matches!(action, Action::Forward)
}
