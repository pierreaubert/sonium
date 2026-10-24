//! Backend-aware Jacobi/Schur preconditioner for saddle-point systems.
//!
//! This mirrors the CPU upper-triangular saddle-point preconditioner with
//! diagonal block solves:
//!
//! ```text
//! y_p = S_diag^{-1} r_p
//! y_u = A_diag^{-1} (r_u - B y_p)
//! ```
//!
//! The block solves can use polynomial Jacobi/Richardson or Chebyshev
//! approximations.
//! Everything is embedded as full-size CSR matrices so `apply` stays entirely
//! in existing backend primitives: SpMV, scale, and axpy. No device slicing
//! or triangular solves are needed.

use crate::gmres::{BackendCsr, BackendPreconditioner};
use crate::operator::{DeviceCsr, PersistentDeviceCsr, validate_csr_inputs};
use crate::{
    Backend, DeviceBufferLike, DeviceScalar, HostCsrRef, PreconditionerFailureInjectionPoint,
};
use std::sync::Mutex;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum PressureSchurBackendKind {
    #[default]
    Polynomial,
    Chebyshev,
    External,
}

/// Upper-triangular Jacobi/Schur preconditioner over backend buffers.
pub struct SaddlePointJacobiSchurBackend<'a, B: Backend, T: DeviceScalar = f64> {
    a: PersistentDeviceCsr<B, T>,
    velocity_block: PersistentDeviceCsr<B, T>,
    pressure_schur: PersistentDeviceCsr<B, T>,
    pressure_preconditioner: Option<&'a dyn BackendPreconditioner<B, T>>,
    owned_pressure_preconditioner: Option<Box<dyn BackendPreconditioner<B, T>>>,
    pressure_mask: PersistentDeviceCsr<B, T>,
    velocity_diag_inv: PersistentDeviceCsr<B, T>,
    pressure_diag_inv: PersistentDeviceCsr<B, T>,
    n: usize,
    velocity_size: usize,
    host_rows: Vec<usize>,
    host_columns: Vec<usize>,
    schur_rows: Vec<usize>,
    schur_columns: Vec<usize>,
    schur_values: Vec<T>,
    numeric_valid: bool,
    velocity_degree: usize,
    velocity_damping: T,
    pressure_degree: usize,
    pressure_damping: T,
    pressure_kind: PressureSchurBackendKind,
    pressure_chebyshev_degree: usize,
    pressure_chebyshev_eigenvalue_ratio: f64,
    pressure_lambda_max: f64,
    workspace: Mutex<SaddlePointWorkspace<B, T>>,
    runtime_diagnostics: crate::DevicePreconditionerDiagnostics,
}

struct SaddlePointWorkspace<B: Backend, T: DeviceScalar> {
    result: B::Buffer<T>,
    y_pressure: B::Buffer<T>,
    modified_rhs: B::Buffer<T>,
    residual: B::Buffer<T>,
    correction: B::Buffer<T>,
    d_new: B::Buffer<T>,
    external_input: B::Buffer<T>,
    external_output: B::Buffer<T>,
    external_masked: B::Buffer<T>,
}

impl<B: Backend, T: DeviceScalar> SaddlePointWorkspace<B, T> {
    fn new(backend: &B, len: usize) -> crate::BackendResult<Self> {
        Ok(Self {
            result: backend
                .raw_alloc::<T>(len)
                .map_err(crate::BackendError::from)?,
            y_pressure: backend
                .raw_alloc::<T>(len)
                .map_err(crate::BackendError::from)?,
            modified_rhs: backend
                .raw_alloc::<T>(len)
                .map_err(crate::BackendError::from)?,
            residual: backend
                .raw_alloc::<T>(len)
                .map_err(crate::BackendError::from)?,
            correction: backend
                .raw_alloc::<T>(len)
                .map_err(crate::BackendError::from)?,
            d_new: backend
                .raw_alloc::<T>(len)
                .map_err(crate::BackendError::from)?,
            external_input: backend
                .raw_alloc::<T>(len)
                .map_err(crate::BackendError::from)?,
            external_output: backend
                .raw_alloc::<T>(len)
                .map_err(crate::BackendError::from)?,
            external_masked: backend
                .raw_alloc::<T>(len)
                .map_err(crate::BackendError::from)?,
        })
    }

    fn ensure_len(&mut self, backend: &B, len: usize) -> crate::BackendResult<()> {
        for buffer in [
            &mut self.result,
            &mut self.y_pressure,
            &mut self.modified_rhs,
            &mut self.residual,
            &mut self.correction,
            &mut self.d_new,
        ] {
            if buffer.len() != len {
                *buffer = backend
                    .raw_alloc::<T>(len)
                    .map_err(crate::BackendError::from)?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SaddlePointJacobiSchurBackendConfig {
    pub velocity_block_size: usize,
    pub velocity_degree: usize,
    pub velocity_damping: f64,
    pub pressure_degree: usize,
    pub pressure_damping: f64,
    pub pressure_kind: PressureSchurBackendKind,
    pub pressure_chebyshev_degree: usize,
    pub pressure_chebyshev_eigenvalue_ratio: f64,
}

/// Convert an f64 configuration scalar into `T::Real`.
fn real_from_f64<T: DeviceScalar>(value: f64) -> T::Real {
    use num_traits::{FromPrimitive, Zero};
    T::Real::from_f64(value).unwrap_or_else(T::Real::zero)
}

/// Estimate the pressure-Schur spectral bound from f64 host data (the
/// estimate only scales the Chebyshev bracket, so evaluating it in f64
/// is valid for every working precision).
fn schur_lambda_max_f64(
    n: usize,
    velocity_block_size: usize,
    schur: HostCsrRef<'_, f64>,
) -> crate::BackendResult<f64> {
    HostCsrRef::try_new(
        schur.num_rows,
        schur.num_cols,
        schur.row_ptrs,
        schur.col_indices,
        schur.values,
    )
    .map_err(|error| crate::BackendError::InvalidArgument {
        operation: "saddle_preconditioner_constructor",
        reason: error.to_string(),
    })?;
    if velocity_block_size > n {
        return Err(crate::BackendError::InvalidArgument {
            operation: "saddle_preconditioner_constructor",
            reason: "velocity block must fit inside matrix".into(),
        });
    }
    let pressure_size = n - velocity_block_size;
    if schur.num_rows != pressure_size || schur.num_cols != pressure_size {
        return Err(crate::BackendError::InvalidArgument {
            operation: "saddle_preconditioner_constructor",
            reason: format!(
                "Schur dimensions {}/{} do not match pressure block size {pressure_size}",
                schur.num_rows, schur.num_cols
            ),
        });
    }
    let mut pressure_diag = vec![0.0_f64; n];
    for row in 0..schur.num_rows {
        let lo = schur.row_ptrs[row];
        let hi = schur.row_ptrs[row + 1];
        for k in lo..hi {
            if schur.col_indices[k] == row {
                pressure_diag[velocity_block_size + row] = schur.values[k];
                break;
            }
        }
    }
    Ok(estimate_lambda_max_host_f64(
        schur.num_rows,
        schur.row_ptrs,
        schur.col_indices,
        schur.values,
        &pressure_diag[velocity_block_size..],
    )
    .max(f64::EPSILON))
}

impl<'a, B: Backend> SaddlePointJacobiSchurBackend<'a, B, f64> {
    /// Refresh fixed-pattern numeric data while retaining all device storage.
    ///
    /// Validation failures preserve the previous action. A failed upload invalidates
    /// scalar and block applications until a complete refresh succeeds. Diagonal
    /// extraction and pressure spectral estimation remain host operations. An external
    /// pressure action is retained only when the supplied pressure matrix is unchanged.
    ///
    /// # Errors
    /// Rejects changed shapes/patterns, non-finite values, changed external pressure
    /// matrices and backend transfer failures. Changed external pressure setup must rebuild.
    pub fn try_update_numeric_host(
        &mut self,
        backend: &B,
        matrix: HostCsrRef<'_, f64>,
        schur: HostCsrRef<'_, f64>,
    ) -> crate::BackendResult<()> {
        self.update_numeric_host(backend, matrix, schur, false)
    }

    /// Refresh numeric blocks and attach a prepared replacement pressure action.
    ///
    /// The caller supplies a full-system pressure action matching the new Schur
    /// approximation. The replacement is published only after every block upload
    /// succeeds. Shared mutable pressure actions require exclusive producer access
    /// across preparation and this call; discard the session if either step fails.
    ///
    /// # Errors
    /// Rejects changed patterns or invalid data and propagates backend failures.
    /// Interrupted uploads invalidate application until a successful full refresh.
    pub fn try_update_numeric_host_with_pressure(
        &mut self,
        backend: &B,
        matrix: HostCsrRef<'_, f64>,
        schur: HostCsrRef<'_, f64>,
        pressure: Box<dyn BackendPreconditioner<B, f64>>,
    ) -> crate::BackendResult<()> {
        self.update_numeric_host(backend, matrix, schur, true)?;
        self.pressure_preconditioner = None;
        self.owned_pressure_preconditioner = Some(pressure);
        self.pressure_kind = PressureSchurBackendKind::External;
        Ok(())
    }

    fn update_numeric_host(
        &mut self,
        backend: &B,
        matrix: HostCsrRef<'_, f64>,
        schur: HostCsrRef<'_, f64>,
        replace_pressure: bool,
    ) -> crate::BackendResult<()> {
        let invalid = |reason: &str| crate::BackendError::InvalidArgument {
            operation: "saddle_preconditioner_update",
            reason: reason.into(),
        };
        if !replace_pressure
            && self.pressure_kind == PressureSchurBackendKind::External
            && schur.values != self.schur_values
        {
            return Err(invalid(
                "changed external pressure matrix requires rebuilding",
            ));
        }
        if matrix.num_rows != self.n
            || matrix.num_cols != self.n
            || matrix.row_ptrs != self.host_rows
            || matrix.col_indices != self.host_columns
            || matrix.values.len() != self.host_columns.len()
            || schur.num_rows != self.n - self.velocity_size
            || schur.num_cols != self.n - self.velocity_size
            || schur.row_ptrs != self.schur_rows
            || schur.col_indices != self.schur_columns
            || schur.values.len() != self.schur_columns.len()
        {
            return Err(invalid(
                "numeric refresh requires the retained full and pressure CSR patterns",
            ));
        }
        if matrix
            .values
            .iter()
            .chain(schur.values)
            .any(|v| !v.is_finite())
        {
            return Err(invalid("matrix values must be finite"));
        }
        // Recovery must restore every block: a failed prior refresh may have
        // uploaded pressure values without publishing the new host snapshot.
        let refresh_pressure = !self.numeric_valid || schur.values != self.schur_values;
        let bound = if refresh_pressure {
            schur_lambda_max_f64(self.n, self.velocity_size, schur)?
        } else {
            self.pressure_lambda_max
        };
        if !bound.is_finite() || bound <= 0.0 {
            return Err(invalid(
                "pressure spectral bound must be finite and positive",
            ));
        }
        let mut velocity_values = Vec::new();
        let mut velocity_inverse = vec![0.0; self.n];
        let mut pressure_inverse = if refresh_pressure {
            vec![0.0; self.n]
        } else {
            Vec::new()
        };
        // Match constructor semantics: first diagonal entry, zero for missing or tiny diagonals.
        let inverse = |v: f64| if v.abs() > f64::EPSILON { 1.0 / v } else { 0.0 };
        for (row, diagonal) in velocity_inverse
            .iter_mut()
            .enumerate()
            .take(self.velocity_size)
        {
            let range = matrix.row_ptrs[row]..matrix.row_ptrs[row + 1];
            if let Some(j) = range.clone().find(|&j| matrix.col_indices[j] == row) {
                *diagonal = inverse(matrix.values[j]);
            }
            velocity_values.extend(
                range
                    .filter(|&j| matrix.col_indices[j] < self.velocity_size)
                    .map(|j| matrix.values[j]),
            );
        }
        if refresh_pressure {
            for row in 0..schur.num_rows {
                if let Some(j) = (schur.row_ptrs[row]..schur.row_ptrs[row + 1])
                    .find(|&j| schur.col_indices[j] == row)
                {
                    pressure_inverse[self.velocity_size + row] = inverse(schur.values[j]);
                }
            }
        }
        self.numeric_valid = false;
        self.a.try_copy_values_from_host(backend, matrix.values)?;
        self.velocity_block
            .try_copy_values_from_host(backend, &velocity_values)?;
        if refresh_pressure {
            self.pressure_schur
                .try_copy_values_from_host(backend, schur.values)?;
        }
        backend.try_inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::DiagonalUpload,
        )?;
        self.velocity_diag_inv
            .try_copy_values_from_host(backend, &velocity_inverse)?;
        if refresh_pressure {
            self.pressure_diag_inv
                .try_copy_values_from_host(backend, &pressure_inverse)?;
        }
        self.pressure_lambda_max = bound;
        self.schur_values.copy_from_slice(schur.values);
        self.numeric_valid = true;
        Ok(())
    }

    /// Build an owned Jacobi/Schur action directly from host CSR matrices.
    ///
    /// The full matrix and all block operators upload once and remain owned by
    /// the preconditioner. No borrowed device matrix or intermediate upload is
    /// required. Setup and spectral estimation are host work. Polynomial and
    /// Chebyshev pressure solves are supported. External pressure solvers can
    /// be attached with [`Self::with_owned_pressure_preconditioner`] or use the
    /// separate borrowed-preconditioner constructor.
    ///
    /// # Errors
    /// Rejects malformed or non-finite CSR, incompatible block dimensions,
    /// non-finite relaxation controls, unsupported external pressure solvers,
    /// index overflow and native setup/allocation failures.
    pub fn try_from_host_csr(
        backend: &B,
        matrix: HostCsrRef<'_, f64>,
        pressure_schur: HostCsrRef<'_, f64>,
        config: SaddlePointJacobiSchurBackendConfig,
    ) -> crate::BackendResult<Self> {
        let invalid = |reason: String| crate::BackendError::InvalidArgument {
            operation: "saddle_preconditioner_constructor",
            reason,
        };
        HostCsrRef::try_new(
            matrix.num_rows,
            matrix.num_cols,
            matrix.row_ptrs,
            matrix.col_indices,
            matrix.values,
        )
        .map_err(|error| invalid(error.to_string()))?;
        if config.pressure_kind == PressureSchurBackendKind::External {
            return Err(invalid(
                "external pressure solver requires the borrowed-preconditioner constructor".into(),
            ));
        }
        let convert = |indices: &[usize]| -> crate::BackendResult<Vec<i32>> {
            indices
                .iter()
                .map(|&index| {
                    i32::try_from(index).map_err(|_| invalid("CSR index exceeds i32 range".into()))
                })
                .collect()
        };
        let rows = convert(matrix.row_ptrs)?;
        let columns = convert(matrix.col_indices)?;
        let lambda =
            schur_lambda_max_f64(matrix.num_rows, config.velocity_block_size, pressure_schur)?;
        Self::try_build(
            backend,
            (matrix.num_rows, matrix.num_cols),
            || PersistentDeviceCsr::try_from_host_csr_ref(backend, matrix),
            &rows,
            &columns,
            matrix.values,
            pressure_schur,
            config,
            None,
            lambda,
        )
    }

    /// Build from raw host CSR parts.
    ///
    /// Panics on construction failure; production paths must use
    /// [`Self::try_from_csr_host_parts`] instead.
    #[deprecated(note = "panics on construction failure; use try_from_csr_host_parts")]
    pub fn from_csr_host_parts(
        backend: &B,
        a: &BackendCsr<B, f64>,
        row_ptr_host: &[i32],
        col_idx_host: &[i32],
        values_host: &[f64],
        pressure_schur: HostCsrRef<'_, f64>,
        config: SaddlePointJacobiSchurBackendConfig,
    ) -> Self {
        Self::try_from_csr_host_parts_with_pressure_preconditioner(
            backend,
            a,
            row_ptr_host,
            col_idx_host,
            values_host,
            pressure_schur,
            config,
            None,
        )
        .expect("saddle-point backend preconditioner construction failed")
    }

    /// Checked variant of [`Self::from_csr_host_parts`].
    pub fn try_from_csr_host_parts(
        backend: &B,
        a: &BackendCsr<B, f64>,
        row_ptr_host: &[i32],
        col_idx_host: &[i32],
        values_host: &[f64],
        pressure_schur: HostCsrRef<'_, f64>,
        config: SaddlePointJacobiSchurBackendConfig,
    ) -> crate::BackendResult<Self> {
        Self::try_from_csr_host_parts_with_pressure_preconditioner(
            backend,
            a,
            row_ptr_host,
            col_idx_host,
            values_host,
            pressure_schur,
            config,
            None,
        )
    }

    /// Build from raw host CSR parts with an optional full-size pressure
    /// preconditioner. The external preconditioner must operate on a matrix
    /// where velocity rows are identity and pressure rows contain the embedded
    /// Schur approximation.
    ///
    /// Panics on construction failure; production paths must use
    /// [`Self::try_from_csr_host_parts_with_pressure_preconditioner`]
    /// instead.
    #[allow(clippy::too_many_arguments)]
    #[deprecated(
        note = "panics on construction failure; use try_from_csr_host_parts_with_pressure_preconditioner"
    )]
    pub fn from_csr_host_parts_with_pressure_preconditioner(
        backend: &B,
        a: &BackendCsr<B, f64>,
        row_ptr_host: &[i32],
        col_idx_host: &[i32],
        values_host: &[f64],
        pressure_schur: HostCsrRef<'_, f64>,
        config: SaddlePointJacobiSchurBackendConfig,
        pressure_preconditioner: Option<&'a dyn BackendPreconditioner<B, f64>>,
    ) -> Self {
        Self::try_from_csr_host_parts_with_pressure_preconditioner(
            backend,
            a,
            row_ptr_host,
            col_idx_host,
            values_host,
            pressure_schur,
            config,
            pressure_preconditioner,
        )
        .expect("saddle-point backend preconditioner construction failed")
    }

    /// Checked variant of
    /// [`Self::from_csr_host_parts_with_pressure_preconditioner`].
    #[allow(clippy::too_many_arguments)]
    pub fn try_from_csr_host_parts_with_pressure_preconditioner(
        backend: &B,
        a: &BackendCsr<B, f64>,
        row_ptr_host: &[i32],
        col_idx_host: &[i32],
        values_host: &[f64],
        pressure_schur: HostCsrRef<'_, f64>,
        config: SaddlePointJacobiSchurBackendConfig,
        pressure_preconditioner: Option<&'a dyn BackendPreconditioner<B, f64>>,
    ) -> crate::BackendResult<Self> {
        let pressure_lambda_max =
            schur_lambda_max_f64(a.m, config.velocity_block_size, pressure_schur)?;
        Self::try_build(
            backend,
            (a.m, a.n),
            || PersistentDeviceCsr::try_clone_from_device_csr(backend, a),
            row_ptr_host,
            col_idx_host,
            values_host,
            pressure_schur,
            config,
            pressure_preconditioner,
            pressure_lambda_max,
        )
    }
}

impl<'a, B: Backend> SaddlePointJacobiSchurBackend<'a, B, f32> {
    /// Checked f32 construction from raw host CSR parts (the
    /// mixed-precision inner preconditioner). The pressure spectral
    /// estimate is evaluated in f64 from widened host values.
    #[allow(clippy::too_many_arguments)]
    pub fn try_from_csr_host_parts_f32(
        backend: &B,
        a: &BackendCsr<B, f32>,
        row_ptr_host: &[i32],
        col_idx_host: &[i32],
        values_host: &[f32],
        pressure_schur: HostCsrRef<'_, f32>,
        config: SaddlePointJacobiSchurBackendConfig,
    ) -> crate::BackendResult<Self> {
        let schur_values_f64: Vec<f64> = pressure_schur
            .values
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        let schur_f64 = HostCsrRef::new(
            pressure_schur.num_rows,
            pressure_schur.num_cols,
            pressure_schur.row_ptrs,
            pressure_schur.col_indices,
            &schur_values_f64,
        );
        let pressure_lambda_max = schur_lambda_max_f64(a.m, config.velocity_block_size, schur_f64)?;
        Self::try_build(
            backend,
            (a.m, a.n),
            || PersistentDeviceCsr::try_clone_from_device_csr(backend, a),
            row_ptr_host,
            col_idx_host,
            values_host,
            pressure_schur,
            config,
            None,
            pressure_lambda_max,
        )
    }
}

impl<'a, B: Backend, T: DeviceScalar> SaddlePointJacobiSchurBackend<'a, B, T> {
    /// Replace the pressure solve with an owned full-system device action.
    ///
    /// The action must use the same backend/stream and full-system vector layout,
    /// with pressure unknowns after the velocity block. Its output is masked to
    /// the pressure block. Ownership allows an auxiliary AMG hierarchy to survive
    /// repeated solves without borrowing a temporary setup object.
    pub fn with_owned_pressure_preconditioner(
        mut self,
        preconditioner: Box<dyn BackendPreconditioner<B, T>>,
    ) -> Self {
        self.pressure_preconditioner = None;
        self.owned_pressure_preconditioner = Some(preconditioner);
        self.pressure_kind = PressureSchurBackendKind::External;
        self
    }

    /// Shared checked construction for every working precision.
    #[allow(clippy::too_many_arguments)]
    fn try_build(
        backend: &B,
        shape: (usize, usize),
        make_matrix: impl FnOnce() -> crate::BackendResult<PersistentDeviceCsr<B, T>>,
        row_ptr_host: &[i32],
        col_idx_host: &[i32],
        values_host: &[T],
        pressure_schur: HostCsrRef<'_, T>,
        config: SaddlePointJacobiSchurBackendConfig,
        pressure_preconditioner: Option<&'a dyn BackendPreconditioner<B, T>>,
        pressure_lambda_max: f64,
    ) -> crate::BackendResult<Self> {
        let (matrix_m, matrix_n) = shape;
        let velocity_block_size = config.velocity_block_size;
        if !config.velocity_damping.is_finite()
            || !config.pressure_damping.is_finite()
            || !config.pressure_chebyshev_eigenvalue_ratio.is_finite()
            || values_host.iter().any(|value| !value.is_finite())
            || pressure_schur.values.iter().any(|value| !value.is_finite())
        {
            return Err(crate::BackendError::InvalidArgument {
                operation: "saddle_preconditioner_constructor",
                reason: "matrix values and relaxation controls must be finite".into(),
            });
        }
        if matrix_m != matrix_n {
            return Err(crate::BackendError::InvalidArgument {
                operation: "saddle_preconditioner_constructor",
                reason: "saddle-point backend preconditioner needs square A".into(),
            });
        }
        if matrix_m > i32::MAX as usize {
            return Err(crate::BackendError::InvalidArgument {
                operation: "saddle_preconditioner_constructor",
                reason: format!("matrix dimension {} exceeds i32 CSR index range", matrix_m),
            });
        }
        validate_csr_inputs(matrix_m, matrix_n, row_ptr_host, col_idx_host, values_host)?;
        HostCsrRef::try_new(
            pressure_schur.num_rows,
            pressure_schur.num_cols,
            pressure_schur.row_ptrs,
            pressure_schur.col_indices,
            pressure_schur.values,
        )
        .map_err(|error| crate::BackendError::InvalidArgument {
            operation: "saddle_preconditioner_constructor",
            reason: error.to_string(),
        })?;
        if velocity_block_size > matrix_m {
            return Err(crate::BackendError::InvalidArgument {
                operation: "saddle_preconditioner_constructor",
                reason: "velocity block must fit inside matrix".into(),
            });
        }
        let pressure_size = matrix_m - velocity_block_size;
        if pressure_schur.num_rows != pressure_size || pressure_schur.num_cols != pressure_size {
            return Err(crate::BackendError::InvalidArgument {
                operation: "saddle_preconditioner_constructor",
                reason: format!(
                    "Schur dimensions {}/{} do not match pressure block size {pressure_size}",
                    pressure_schur.num_rows, pressure_schur.num_cols
                ),
            });
        }

        let mut velocity_diag = vec![T::zero(); matrix_m];
        let (velocity_block_row_ptr, velocity_block_col_idx, velocity_block_values) =
            extract_embedded_velocity_block(
                matrix_m,
                row_ptr_host,
                col_idx_host,
                values_host,
                velocity_block_size,
            );
        let (pressure_schur_row_ptr, pressure_schur_col_idx, pressure_schur_values) =
            embed_pressure_schur(matrix_m, velocity_block_size, pressure_schur);
        for row in 0..velocity_block_size {
            let lo = row_ptr_host[row] as usize;
            let hi = row_ptr_host[row + 1] as usize;
            for k in lo..hi {
                if col_idx_host[k] as usize == row {
                    velocity_diag[row] = values_host[k];
                    break;
                }
            }
        }

        let mut pressure_diag = vec![T::zero(); matrix_m];
        for row in 0..pressure_schur.num_rows {
            let lo = pressure_schur.row_ptrs[row];
            let hi = pressure_schur.row_ptrs[row + 1];
            for k in lo..hi {
                if pressure_schur.col_indices[k] == row {
                    pressure_diag[velocity_block_size + row] = pressure_schur.values[k];
                    break;
                }
            }
        }

        backend.try_inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::HierarchyUpload,
        )?;
        let a = make_matrix()?;
        backend.try_inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::DiagonalUpload,
        )?;
        let velocity_diag_inv = PersistentDeviceCsr::try_from_device_csr(
            backend,
            DeviceCsr::try_from_host_inverse_diagonal(backend, &velocity_diag)?,
        )?;
        let pressure_diag_inv = PersistentDeviceCsr::try_from_device_csr(
            backend,
            DeviceCsr::try_from_host_inverse_diagonal(backend, &pressure_diag)?,
        )?;
        let pressure_mask = upload_pressure_mask(backend, matrix_m, velocity_block_size)?;
        let velocity_block = PersistentDeviceCsr::try_from_device_csr(
            backend,
            BackendCsr::try_from_host(
                backend,
                matrix_m,
                matrix_n,
                &velocity_block_row_ptr,
                &velocity_block_col_idx,
                &velocity_block_values,
            )?,
        )?;
        let schur_rows = pressure_schur.row_ptrs.to_vec();
        let schur_columns = pressure_schur.col_indices.to_vec();
        let schur_values = pressure_schur.values.to_vec();
        let pressure_schur = PersistentDeviceCsr::try_from_device_csr(
            backend,
            BackendCsr::try_from_host(
                backend,
                matrix_m,
                matrix_n,
                &pressure_schur_row_ptr,
                &pressure_schur_col_idx,
                &pressure_schur_values,
            )?,
        )?;
        backend.try_inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::WorkspaceAllocation,
        )?;
        let workspace = SaddlePointWorkspace::new(backend, matrix_m)?;
        let runtime_diagnostics = crate::DevicePreconditionerDiagnostics::default();
        runtime_diagnostics.record_setup();

        Ok(Self {
            a,
            velocity_block,
            pressure_schur,
            pressure_preconditioner,
            owned_pressure_preconditioner: None,
            pressure_mask,
            velocity_diag_inv,
            pressure_diag_inv,
            n: matrix_m,
            velocity_size: velocity_block_size,
            host_rows: row_ptr_host.iter().map(|&i| i as usize).collect(),
            host_columns: col_idx_host.iter().map(|&i| i as usize).collect(),
            schur_rows,
            schur_columns,
            schur_values,
            numeric_valid: true,
            velocity_degree: config.velocity_degree.max(1),
            velocity_damping: T::from_real(real_from_f64::<T>(config.velocity_damping)),
            pressure_degree: config.pressure_degree.max(1),
            pressure_damping: T::from_real(real_from_f64::<T>(config.pressure_damping)),
            pressure_kind: config.pressure_kind,
            pressure_chebyshev_degree: config.pressure_chebyshev_degree.max(1),
            pressure_chebyshev_eigenvalue_ratio: config
                .pressure_chebyshev_eigenvalue_ratio
                .max(1.0),
            pressure_lambda_max,
            workspace: Mutex::new(workspace),
            runtime_diagnostics,
        })
    }

    fn try_apply_pressure_schur(
        &self,
        backend: &B,
        r: &B::Buffer<T>,
        y_pressure: &mut B::Buffer<T>,
        residual: &mut B::Buffer<T>,
        correction: &mut B::Buffer<T>,
        d_new: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        match self.pressure_kind {
            PressureSchurBackendKind::Polynomial => {
                self.try_apply_pressure_polynomial(backend, r, y_pressure, residual, correction)?;
            }
            PressureSchurBackendKind::Chebyshev => {
                self.try_apply_pressure_chebyshev(
                    backend, r, y_pressure, residual, correction, d_new,
                )?;
            }
            PressureSchurBackendKind::External => {
                if let Some(preconditioner) = self
                    .owned_pressure_preconditioner
                    .as_deref()
                    .or(self.pressure_preconditioner)
                {
                    preconditioner.try_apply(backend, r, y_pressure)?;
                    self.pressure_mask
                        .try_apply(backend, y_pressure, correction)?;
                    backend.try_copy_d2d(correction, y_pressure)?;
                } else {
                    self.try_apply_pressure_polynomial(
                        backend, r, y_pressure, residual, correction,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn try_apply_pressure_polynomial(
        &self,
        backend: &B,
        r: &B::Buffer<T>,
        y_pressure: &mut B::Buffer<T>,
        residual: &mut B::Buffer<T>,
        correction: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        self.pressure_diag_inv.try_apply(backend, r, y_pressure)?;
        for _ in 1..self.pressure_degree {
            self.pressure_schur
                .try_apply(backend, y_pressure, residual)?;
            backend.try_scale::<T>(-T::one(), residual)?;
            backend.try_axpy::<T>(T::one(), r, residual)?;
            self.pressure_diag_inv
                .try_apply(backend, residual, correction)?;
            backend.try_axpy::<T>(self.pressure_damping, correction, y_pressure)?;
        }
        Ok(())
    }

    fn try_apply_pressure_chebyshev(
        &self,
        backend: &B,
        r: &B::Buffer<T>,
        y_pressure: &mut B::Buffer<T>,
        residual: &mut B::Buffer<T>,
        correction: &mut B::Buffer<T>,
        d_new: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        if self.pressure_lambda_max <= 0.0 || self.pressure_chebyshev_eigenvalue_ratio <= 1.0 {
            self.try_apply_pressure_polynomial(backend, r, y_pressure, residual, correction)?;
            return Ok(());
        }

        let lambda_min = self.pressure_lambda_max / self.pressure_chebyshev_eigenvalue_ratio;
        let theta = 0.5 * (self.pressure_lambda_max + lambda_min);
        let delta = 0.5 * (self.pressure_lambda_max - lambda_min);
        if theta <= 0.0 || delta <= 0.0 {
            self.try_apply_pressure_polynomial(backend, r, y_pressure, residual, correction)?;
            return Ok(());
        }

        let sigma = theta / delta;
        let mut rho = 1.0 / sigma;

        self.pressure_diag_inv.try_apply(backend, r, correction)?;
        backend.try_scale::<T>(T::from_real(real_from_f64::<T>(1.0 / theta)), correction)?;
        backend.try_copy_d2d(correction, y_pressure)?;

        for _ in 1..self.pressure_chebyshev_degree {
            let rho_new = 1.0 / (2.0 * sigma - rho);
            let coeff_old = rho_new * rho;
            let coeff_res = 2.0 * rho_new / delta;

            self.pressure_schur
                .try_apply(backend, y_pressure, residual)?;
            backend.try_scale::<T>(-T::one(), residual)?;
            backend.try_axpy::<T>(T::one(), r, residual)?;
            self.pressure_diag_inv.try_apply(backend, residual, d_new)?;

            backend.try_scale::<T>(T::from_real(real_from_f64::<T>(coeff_old)), correction)?;
            backend.try_axpy::<T>(
                T::from_real(real_from_f64::<T>(coeff_res)),
                d_new,
                correction,
            )?;
            backend.try_axpy::<T>(T::one(), correction, y_pressure)?;
            rho = rho_new;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn try_apply_pressure_schur_colmajor(
        &self,
        backend: &B,
        rhs: usize,
        r: &B::Buffer<T>,
        y_pressure: &mut B::Buffer<T>,
        residual: &mut B::Buffer<T>,
        correction: &mut B::Buffer<T>,
        d_new: &mut B::Buffer<T>,
        external_input: &mut B::Buffer<T>,
        external_output: &mut B::Buffer<T>,
        external_masked: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        match self.pressure_kind {
            PressureSchurBackendKind::Polynomial => {
                self.try_apply_pressure_polynomial_colmajor(
                    backend, rhs, r, y_pressure, residual, correction,
                )?;
            }
            PressureSchurBackendKind::Chebyshev => {
                self.try_apply_pressure_chebyshev_colmajor(
                    backend, rhs, r, y_pressure, residual, correction, d_new,
                )?;
            }
            PressureSchurBackendKind::External => {
                if let Some(preconditioner) = self
                    .owned_pressure_preconditioner
                    .as_deref()
                    .or(self.pressure_preconditioner)
                {
                    // The default col-major adapter is allowed to stage through the host.
                    // Apply one column at a time through persistent device buffers so a
                    // composed pressure preconditioner remains device-resident as well.
                    for column in 0..rhs {
                        let view = crate::BufferView::contiguous(column * self.n, self.n);
                        backend.try_copy_views_d2d(
                            r,
                            view,
                            external_input,
                            crate::BufferView::contiguous(0, self.n),
                        )?;
                        preconditioner.try_apply(backend, external_input, external_output)?;
                        self.pressure_mask
                            .try_apply(backend, external_output, external_masked)?;
                        backend.try_copy_views_d2d(
                            external_masked,
                            crate::BufferView::contiguous(0, self.n),
                            y_pressure,
                            view,
                        )?;
                    }
                } else {
                    self.try_apply_pressure_polynomial_colmajor(
                        backend, rhs, r, y_pressure, residual, correction,
                    )?;
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn try_apply_pressure_polynomial_colmajor(
        &self,
        backend: &B,
        rhs: usize,
        r: &B::Buffer<T>,
        y_pressure: &mut B::Buffer<T>,
        residual: &mut B::Buffer<T>,
        correction: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        self.pressure_diag_inv
            .try_apply_colmajor(backend, rhs, r, y_pressure)?;
        for _ in 1..self.pressure_degree {
            self.pressure_schur
                .try_apply_colmajor(backend, rhs, y_pressure, residual)?;
            backend.try_scale::<T>(-T::one(), residual)?;
            backend.try_axpy::<T>(T::one(), r, residual)?;
            self.pressure_diag_inv
                .try_apply_colmajor(backend, rhs, residual, correction)?;
            backend.try_axpy::<T>(self.pressure_damping, correction, y_pressure)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn try_apply_pressure_chebyshev_colmajor(
        &self,
        backend: &B,
        rhs: usize,
        r: &B::Buffer<T>,
        y_pressure: &mut B::Buffer<T>,
        residual: &mut B::Buffer<T>,
        correction: &mut B::Buffer<T>,
        d_new: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        if self.pressure_lambda_max <= 0.0 || self.pressure_chebyshev_eigenvalue_ratio <= 1.0 {
            self.try_apply_pressure_polynomial_colmajor(
                backend, rhs, r, y_pressure, residual, correction,
            )?;
            return Ok(());
        }

        let lambda_min = self.pressure_lambda_max / self.pressure_chebyshev_eigenvalue_ratio;
        let theta = 0.5 * (self.pressure_lambda_max + lambda_min);
        let delta = 0.5 * (self.pressure_lambda_max - lambda_min);
        if theta <= 0.0 || delta <= 0.0 {
            self.try_apply_pressure_polynomial_colmajor(
                backend, rhs, r, y_pressure, residual, correction,
            )?;
            return Ok(());
        }

        let sigma = theta / delta;
        let mut rho = 1.0 / sigma;

        self.pressure_diag_inv
            .try_apply_colmajor(backend, rhs, r, correction)?;
        backend.try_scale::<T>(T::from_real(real_from_f64::<T>(1.0 / theta)), correction)?;
        backend.try_copy_d2d(correction, y_pressure)?;

        for _ in 1..self.pressure_chebyshev_degree {
            let rho_new = 1.0 / (2.0 * sigma - rho);
            let coeff_old = rho_new * rho;
            let coeff_res = 2.0 * rho_new / delta;

            self.pressure_schur
                .try_apply_colmajor(backend, rhs, y_pressure, residual)?;
            backend.try_scale::<T>(-T::one(), residual)?;
            backend.try_axpy::<T>(T::one(), r, residual)?;
            self.pressure_diag_inv
                .try_apply_colmajor(backend, rhs, residual, d_new)?;

            backend.try_scale::<T>(T::from_real(real_from_f64::<T>(coeff_old)), correction)?;
            backend.try_axpy::<T>(
                T::from_real(real_from_f64::<T>(coeff_res)),
                d_new,
                correction,
            )?;
            backend.try_axpy::<T>(T::one(), correction, y_pressure)?;
            rho = rho_new;
        }
        Ok(())
    }
}

impl<B: Backend, T: DeviceScalar> BackendPreconditioner<B, T>
    for SaddlePointJacobiSchurBackend<'_, B, T>
{
    fn apply(&self, backend: &B, r: &B::Buffer<T>, z: &mut B::Buffer<T>) {
        if let Err(error) = self.try_apply(backend, r, z) {
            log::error!(
                target: "sonium_compute::preconditioner",
                "saddle-point preconditioner application failed: {error}"
            );
        }
    }

    fn try_apply(
        &self,
        backend: &B,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        let result = self.try_apply_untracked(backend, r, z);
        match &result {
            Ok(()) => self.runtime_diagnostics.record_apply(),
            Err(error) => self.runtime_diagnostics.record_failure(error),
        }
        result
    }

    fn try_apply_untracked(
        &self,
        backend: &B,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        if !self.numeric_valid {
            return Err(crate::BackendError::InvalidArgument {
                operation: "saddle_preconditioner_apply",
                reason: "numeric refresh failed; complete a successful refresh before applying"
                    .into(),
            });
        }
        backend.try_inject_preconditioner_failure(PreconditionerFailureInjectionPoint::Launch)?;
        if r.len() != self.n || z.len() != self.n {
            return Err(crate::BackendError::InvalidArgument {
                operation: "saddle_preconditioner_apply",
                reason: format!(
                    "expected input/output length {}, got {}/{}",
                    self.n,
                    r.len(),
                    z.len()
                ),
            });
        }

        let mut workspace =
            self.workspace
                .lock()
                .map_err(|_| crate::BackendError::OperationFailed {
                    operation: "saddle_preconditioner_apply",
                    reason: "workspace mutex poisoned".into(),
                })?;
        backend.try_inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::WorkspaceAllocation,
        )?;
        workspace.ensure_len(backend, self.n)?;
        let SaddlePointWorkspace {
            result,
            y_pressure,
            modified_rhs,
            residual,
            correction,
            d_new,
            external_input: _,
            external_output: _,
            external_masked: _,
        } = &mut *workspace;

        self.try_apply_pressure_schur(backend, r, y_pressure, residual, correction, d_new)?;

        self.a.try_apply(backend, y_pressure, modified_rhs)?;
        backend.try_scale::<T>(-T::one(), modified_rhs)?;
        backend.try_axpy::<T>(T::one(), r, modified_rhs)?;
        self.velocity_diag_inv
            .try_apply(backend, modified_rhs, result)?;
        backend.try_scale::<T>(self.velocity_damping, result)?;

        for _ in 1..self.velocity_degree {
            self.velocity_block.try_apply(backend, result, residual)?;
            backend.try_scale::<T>(-T::one(), residual)?;
            backend.try_axpy::<T>(T::one(), modified_rhs, residual)?;
            self.velocity_diag_inv
                .try_apply(backend, residual, correction)?;
            backend.try_axpy::<T>(self.velocity_damping, correction, result)?;
        }

        backend.try_axpy::<T>(T::one(), y_pressure, result)?;
        backend.try_copy_d2d(result, z)
    }

    fn apply_colmajor(&self, backend: &B, rhs: usize, r: &B::Buffer<T>, z: &mut B::Buffer<T>) {
        if let Err(error) = self.try_apply_colmajor(backend, rhs, r, z) {
            log::error!(
                target: "sonium_compute::preconditioner",
                "saddle-point multi-RHS preconditioner application failed: {error}"
            );
        }
    }

    fn try_apply_colmajor(
        &self,
        backend: &B,
        rhs: usize,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        let result = self.try_apply_colmajor_untracked(backend, rhs, r, z);
        match &result {
            Ok(()) => self.runtime_diagnostics.record_apply(),
            Err(error) => self.runtime_diagnostics.record_failure(error),
        }
        result
    }

    fn try_apply_colmajor_untracked(
        &self,
        backend: &B,
        rhs: usize,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        if !self.numeric_valid {
            return Err(crate::BackendError::InvalidArgument {
                operation: "saddle_preconditioner_apply",
                reason: "numeric refresh failed; complete a successful refresh before applying"
                    .into(),
            });
        }
        backend.try_inject_preconditioner_failure(PreconditionerFailureInjectionPoint::Launch)?;
        let len = self
            .n
            .checked_mul(rhs)
            .ok_or(crate::BackendError::InvalidArgument {
                operation: "saddle_preconditioner_apply_colmajor",
                reason: format!("workspace length overflow: n={}, rhs={rhs}", self.n),
            })?;
        if r.len() != len || z.len() != len {
            return Err(crate::BackendError::InvalidArgument {
                operation: "saddle_preconditioner_apply_colmajor",
                reason: format!(
                    "expected input/output length {len}, got {}/{}",
                    r.len(),
                    z.len()
                ),
            });
        }
        if len == 0 {
            return Ok(());
        }

        let mut workspace =
            self.workspace
                .lock()
                .map_err(|_| crate::BackendError::OperationFailed {
                    operation: "saddle_preconditioner_apply_colmajor",
                    reason: "workspace mutex poisoned".into(),
                })?;
        backend.try_inject_preconditioner_failure(
            PreconditionerFailureInjectionPoint::WorkspaceAllocation,
        )?;
        workspace.ensure_len(backend, len)?;
        let SaddlePointWorkspace {
            result,
            y_pressure,
            modified_rhs,
            residual,
            correction,
            d_new,
            external_input,
            external_output,
            external_masked,
        } = &mut *workspace;

        self.try_apply_pressure_schur_colmajor(
            backend,
            rhs,
            r,
            y_pressure,
            residual,
            correction,
            d_new,
            external_input,
            external_output,
            external_masked,
        )?;

        self.a
            .try_apply_colmajor(backend, rhs, y_pressure, modified_rhs)?;
        backend.try_scale::<T>(-T::one(), modified_rhs)?;
        backend.try_axpy::<T>(T::one(), r, modified_rhs)?;
        self.velocity_diag_inv
            .try_apply_colmajor(backend, rhs, modified_rhs, result)?;
        backend.try_scale::<T>(self.velocity_damping, result)?;

        for _ in 1..self.velocity_degree {
            self.velocity_block
                .try_apply_colmajor(backend, rhs, result, residual)?;
            backend.try_scale::<T>(-T::one(), residual)?;
            backend.try_axpy::<T>(T::one(), modified_rhs, residual)?;
            self.velocity_diag_inv
                .try_apply_colmajor(backend, rhs, residual, correction)?;
            backend.try_axpy::<T>(self.velocity_damping, correction, result)?;
        }

        backend.try_axpy::<T>(T::one(), y_pressure, result)?;
        backend.try_copy_d2d(result, z)
    }

    fn diagnostics(&self) -> crate::DevicePreconditionerDiagnosticsSnapshot {
        self.runtime_diagnostics.snapshot()
    }

    fn try_apply_transpose(
        &self,
        _backend: &B,
        _r: &B::Buffer<T>,
        _z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        let error = crate::BackendError::Unsupported {
            backend: "device",
            operation: "saddle_preconditioner_transpose",
            scalar: "f64",
        };
        self.runtime_diagnostics.record_transpose_failure(&error);
        Err(error)
    }

    fn try_apply_hermitian(
        &self,
        _backend: &B,
        _r: &B::Buffer<T>,
        _z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        let error = crate::BackendError::Unsupported {
            backend: "device",
            operation: "saddle_preconditioner_hermitian",
            scalar: "f64",
        };
        self.runtime_diagnostics.record_hermitian_failure(&error);
        Err(error)
    }

    fn try_apply_transpose_colmajor(
        &self,
        _backend: &B,
        _rhs: usize,
        _r: &B::Buffer<T>,
        _z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        let error = crate::BackendError::Unsupported {
            backend: "device",
            operation: "saddle_preconditioner_transpose_colmajor",
            scalar: "f64",
        };
        self.runtime_diagnostics.record_transpose_failure(&error);
        Err(error)
    }

    fn try_apply_hermitian_colmajor(
        &self,
        _backend: &B,
        _rhs: usize,
        _r: &B::Buffer<T>,
        _z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        let error = crate::BackendError::Unsupported {
            backend: "device",
            operation: "saddle_preconditioner_hermitian_colmajor",
            scalar: "f64",
        };
        self.runtime_diagnostics.record_hermitian_failure(&error);
        Err(error)
    }
}

fn extract_embedded_velocity_block<T: Copy>(
    n: usize,
    row_ptr_host: &[i32],
    col_idx_host: &[i32],
    values_host: &[T],
    velocity_block_size: usize,
) -> (Vec<i32>, Vec<i32>, Vec<T>) {
    let mut row_ptr = vec![0_i32; n + 1];
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    for row in 0..n {
        if row < velocity_block_size {
            let lo = row_ptr_host[row] as usize;
            let hi = row_ptr_host[row + 1] as usize;
            for k in lo..hi {
                let col = col_idx_host[k] as usize;
                if col < velocity_block_size {
                    col_idx.push(col_idx_host[k]);
                    values.push(values_host[k]);
                }
            }
        }
        row_ptr[row + 1] = col_idx.len() as i32;
    }
    (row_ptr, col_idx, values)
}

fn embed_pressure_schur<T: Copy>(
    n: usize,
    velocity_block_size: usize,
    schur: HostCsrRef<'_, T>,
) -> (Vec<i32>, Vec<i32>, Vec<T>) {
    let mut row_ptr = vec![0_i32; n + 1];
    let mut col_idx = Vec::with_capacity(schur.nnz());
    let mut values = Vec::with_capacity(schur.nnz());
    for row in 0..n {
        if row >= velocity_block_size {
            let pressure_row = row - velocity_block_size;
            let lo = schur.row_ptrs[pressure_row];
            let hi = schur.row_ptrs[pressure_row + 1];
            for k in lo..hi {
                col_idx.push((velocity_block_size + schur.col_indices[k]) as i32);
                values.push(schur.values[k]);
            }
        }
        row_ptr[row + 1] = col_idx.len() as i32;
    }
    (row_ptr, col_idx, values)
}

pub fn embed_pressure_schur_with_velocity_identity(
    n: usize,
    velocity_block_size: usize,
    schur: HostCsrRef<'_, f64>,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut row_ptr = vec![0_usize; n + 1];
    let mut col_idx = Vec::with_capacity(schur.nnz() + velocity_block_size);
    let mut values = Vec::with_capacity(schur.nnz() + velocity_block_size);
    for row in 0..n {
        if row < velocity_block_size {
            col_idx.push(row);
            values.push(1.0);
        } else {
            let pressure_row = row - velocity_block_size;
            let lo = schur.row_ptrs[pressure_row];
            let hi = schur.row_ptrs[pressure_row + 1];
            for k in lo..hi {
                col_idx.push(velocity_block_size + schur.col_indices[k]);
                values.push(schur.values[k]);
            }
        }
        row_ptr[row + 1] = col_idx.len();
    }
    (row_ptr, col_idx, values)
}

fn upload_pressure_mask<B: Backend, T: DeviceScalar>(
    backend: &B,
    n: usize,
    velocity_block_size: usize,
) -> crate::BackendResult<PersistentDeviceCsr<B, T>> {
    let _n_i32 = i32::try_from(n).map_err(|_| crate::BackendError::InvalidArgument {
        operation: "pressure_mask_upload",
        reason: format!("matrix dimension {n} exceeds i32 CSR index range"),
    })?;
    let values_host: Vec<T> = (0..n)
        .map(|row| {
            if row < velocity_block_size {
                T::zero()
            } else {
                T::one()
            }
        })
        .collect();
    PersistentDeviceCsr::try_from_host_diagonal(backend, &values_host)
}

fn estimate_lambda_max_host_f64(
    n: usize,
    row_ptr: &[usize],
    col_idx: &[usize],
    values: &[f64],
    diag: &[f64],
) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let mut v: Vec<f64> = (0..n)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let mut w = vec![0.0; n];
    let mut lambda = 1.0;
    for _ in 0..20 {
        for i in 0..n {
            let mut acc = 0.0;
            for k in row_ptr[i]..row_ptr[i + 1] {
                acc += values[k] * v[col_idx[k]];
            }
            w[i] = if diag[i].abs() > f64::EPSILON {
                acc / diag[i]
            } else {
                acc
            };
        }

        let mut num = 0.0;
        let mut den = 0.0;
        for (&vi, &wi) in v.iter().zip(w.iter()) {
            num += vi * wi;
            den += vi * vi;
        }
        if den > 0.0 {
            lambda = (num / den).abs();
        }

        let nrm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nrm > 0.0 {
            for i in 0..n {
                v[i] = w[i] / nrm;
            }
        }
    }
    (lambda * 1.05).max(1.0)
}

#[cfg(test)]
// Fixtures intentionally use the panicking convenience constructors on
// tiny fixed matrices; production paths must use the `try_*` variants.
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::gmres::BackendCsr;
    use crate::{Backend, CpuBackend, PointJacobiBackend};

    #[test]
    fn backend_apply_matches_upper_triangular_jacobi_schur() {
        let backend = CpuBackend::new();
        let row_ptr = vec![0, 2, 4, 5];
        let col_idx = vec![0, 2, 1, 2, 2];
        let values = vec![4.0_f64, 1.0, 4.0, 1.0, 0.1];
        let a = BackendCsr::from_host(&backend, 3, 3, &row_ptr, &col_idx, &values);
        let schur_row_ptrs = vec![0_usize, 1];
        let schur_col_indices = vec![0_usize];
        let schur_values = vec![1.0_f64];
        let precond = SaddlePointJacobiSchurBackend::from_csr_host_parts(
            &backend,
            &a,
            &row_ptr,
            &col_idx,
            &values,
            HostCsrRef::new(1, 1, &schur_row_ptrs, &schur_col_indices, &schur_values),
            SaddlePointJacobiSchurBackendConfig {
                velocity_block_size: 2,
                velocity_degree: 1,
                velocity_damping: 1.0,
                pressure_degree: 1,
                pressure_damping: 1.0,
                pressure_kind: PressureSchurBackendKind::Polynomial,
                pressure_chebyshev_degree: 5,
                pressure_chebyshev_eigenvalue_ratio: 30.0,
            },
        );

        let mut r = backend.raw_alloc::<f64>(3).unwrap();
        backend.copy_h2d(&[1.0_f64, 1.0, 1.0], &mut r);
        let mut z = backend.raw_alloc::<f64>(3).unwrap();
        precond.apply(&backend, &r, &mut z);

        let mut host = vec![0.0_f64; 3];
        backend.copy_d2h(&z, &mut host);
        assert!((host[0] - 0.0).abs() < 1.0e-12);
        assert!((host[1] - 0.0).abs() < 1.0e-12);
        assert!((host[2] - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn f32_backend_apply_matches_f64_within_precision() {
        // The mixed-precision (f32) saddle preconditioner must reproduce
        // the f64 action to f32 accuracy on every backend (the code is
        // backend-generic, so the CPU backend proves the arithmetic for
        // CUDA and Metal as well).
        let backend = CpuBackend::new();
        let row_ptr = vec![0, 2, 4, 5];
        let col_idx = vec![0, 2, 1, 2, 2];
        let values = vec![4.0_f64, 1.0, 4.0, 1.0, 0.1];
        let schur_row_ptrs = vec![0_usize, 1];
        let schur_col_indices = vec![0_usize];
        let schur_values = vec![1.0_f64];
        for pressure_kind in [
            PressureSchurBackendKind::Polynomial,
            PressureSchurBackendKind::Chebyshev,
        ] {
            let config = SaddlePointJacobiSchurBackendConfig {
                velocity_block_size: 2,
                velocity_degree: 2,
                velocity_damping: 0.9,
                pressure_degree: 2,
                pressure_damping: 0.9,
                pressure_kind,
                pressure_chebyshev_degree: 5,
                pressure_chebyshev_eigenvalue_ratio: 30.0,
            };
            let a64 = BackendCsr::from_host(&backend, 3, 3, &row_ptr, &col_idx, &values);
            let precond64 = SaddlePointJacobiSchurBackend::try_from_csr_host_parts(
                &backend,
                &a64,
                &row_ptr,
                &col_idx,
                &values,
                HostCsrRef::new(1, 1, &schur_row_ptrs, &schur_col_indices, &schur_values),
                config,
            )
            .expect("f64 saddle construction");
            let mut r64 = backend.raw_alloc::<f64>(3).unwrap();
            backend.copy_h2d(&[1.0_f64, -2.0, 0.5], &mut r64);
            let mut z64 = backend.raw_alloc::<f64>(3).unwrap();
            precond64.apply(&backend, &r64, &mut z64);
            let mut host64 = vec![0.0_f64; 3];
            backend.copy_d2h(&z64, &mut host64);

            let values32: Vec<f32> = values.iter().map(|&v| v as f32).collect();
            let schur_values32: Vec<f32> = schur_values.iter().map(|&v| v as f32).collect();
            let a32 = BackendCsr::from_host(&backend, 3, 3, &row_ptr, &col_idx, &values32);
            let precond32 = SaddlePointJacobiSchurBackend::try_from_csr_host_parts_f32(
                &backend,
                &a32,
                &row_ptr,
                &col_idx,
                &values32,
                HostCsrRef::new(1, 1, &schur_row_ptrs, &schur_col_indices, &schur_values32),
                config,
            )
            .expect("f32 saddle construction");
            let mut r32 = backend.raw_alloc::<f32>(3).unwrap();
            backend.copy_h2d(&[1.0_f32, -2.0, 0.5], &mut r32);
            let mut z32 = backend.raw_alloc::<f32>(3).unwrap();
            precond32.apply(&backend, &r32, &mut z32);
            let mut host32 = vec![0.0_f32; 3];
            backend.copy_d2h(&z32, &mut host32);

            for (index, (&got, &want)) in host32.iter().zip(&host64).enumerate() {
                assert!(
                    (f64::from(got) - want).abs() <= 1.0e-5 * (1.0 + want.abs()),
                    "f32 saddle apply differs from f64 at {index} ({pressure_kind:?}): {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn backend_from_csr_host_parts_matches_matrix_adapter() {
        let backend = CpuBackend::new();
        let row_ptr = vec![0, 2, 4, 5];
        let col_idx = vec![0, 2, 1, 2, 2];
        let values = vec![4.0_f64, 1.0, 4.0, 1.0, 0.1];
        let a = BackendCsr::from_host(&backend, 3, 3, &row_ptr, &col_idx, &values);
        let schur_row_ptrs = vec![0_usize, 1];
        let schur_col_indices = vec![0_usize];
        let schur_values = vec![1.0_f64];
        let config = SaddlePointJacobiSchurBackendConfig {
            velocity_block_size: 2,
            velocity_degree: 1,
            velocity_damping: 1.0,
            pressure_degree: 1,
            pressure_damping: 1.0,
            pressure_kind: PressureSchurBackendKind::Polynomial,
            pressure_chebyshev_degree: 5,
            pressure_chebyshev_eigenvalue_ratio: 30.0,
        };
        let precond = SaddlePointJacobiSchurBackend::from_csr_host_parts(
            &backend,
            &a,
            &row_ptr,
            &col_idx,
            &values,
            HostCsrRef::new(1, 1, &schur_row_ptrs, &schur_col_indices, &schur_values),
            config,
        );

        let mut r = backend.raw_alloc::<f64>(3).unwrap();
        backend.copy_h2d(&[1.0_f64, 1.0, 1.0], &mut r);
        let mut z = backend.raw_alloc::<f64>(3).unwrap();
        precond.apply(&backend, &r, &mut z);

        let mut host = vec![0.0_f64; 3];
        backend.copy_d2h(&z, &mut host);
        assert!((host[0] - 0.0).abs() < 1.0e-12);
        assert!((host[1] - 0.0).abs() < 1.0e-12);
        assert!((host[2] - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn stronger_pressure_schur_polynomial_reduces_pressure_coupling_error() {
        let backend = CpuBackend::new();
        let row_ptr = vec![0, 1, 2, 3, 4];
        let col_idx = vec![0, 1, 2, 3];
        let values = vec![4.0_f64, 4.0, 0.1, 0.1];
        let a = BackendCsr::from_host(&backend, 4, 4, &row_ptr, &col_idx, &values);
        let schur_row_ptrs = vec![0_usize, 2, 4];
        let schur_col_indices = vec![0_usize, 1, 0, 1];
        let schur_values = vec![2.0_f64, 0.5, 0.5, 2.0];
        let precond = SaddlePointJacobiSchurBackend::from_csr_host_parts(
            &backend,
            &a,
            &row_ptr,
            &col_idx,
            &values,
            HostCsrRef::new(2, 2, &schur_row_ptrs, &schur_col_indices, &schur_values),
            SaddlePointJacobiSchurBackendConfig {
                velocity_block_size: 2,
                velocity_degree: 1,
                velocity_damping: 1.0,
                pressure_degree: 2,
                pressure_damping: 1.0,
                pressure_kind: PressureSchurBackendKind::Polynomial,
                pressure_chebyshev_degree: 5,
                pressure_chebyshev_eigenvalue_ratio: 30.0,
            },
        );

        let mut r = backend.raw_alloc::<f64>(4).unwrap();
        backend.copy_h2d(&[0.0_f64, 0.0, 1.0, 0.0], &mut r);
        let mut z = backend.raw_alloc::<f64>(4).unwrap();
        precond.apply(&backend, &r, &mut z);

        let mut host = vec![0.0_f64; 4];
        backend.copy_d2h(&z, &mut host);
        assert!((host[2] - 0.5).abs() < 1.0e-12);
        assert!((host[3] + 0.125).abs() < 1.0e-12);
    }

    #[test]
    fn backend_apply_colmajor_matches_column_apply() {
        let backend = CpuBackend::new();
        let row_ptr = vec![0, 3, 6, 7, 8];
        let col_idx = vec![0, 1, 2, 0, 1, 3, 2, 3];
        let values = vec![4.0_f64, 0.5, 1.0, 0.25, 3.0, -0.5, 0.1, 0.1];
        let a = BackendCsr::from_host(&backend, 4, 4, &row_ptr, &col_idx, &values);
        let schur_row_ptrs = vec![0_usize, 2, 4];
        let schur_col_indices = vec![0_usize, 1, 0, 1];
        let schur_values = vec![2.0_f64, 0.5, 0.5, 2.0];
        let precond = SaddlePointJacobiSchurBackend::from_csr_host_parts(
            &backend,
            &a,
            &row_ptr,
            &col_idx,
            &values,
            HostCsrRef::new(2, 2, &schur_row_ptrs, &schur_col_indices, &schur_values),
            SaddlePointJacobiSchurBackendConfig {
                velocity_block_size: 2,
                velocity_degree: 2,
                velocity_damping: 0.9,
                pressure_degree: 2,
                pressure_damping: 1.0,
                pressure_kind: PressureSchurBackendKind::Polynomial,
                pressure_chebyshev_degree: 5,
                pressure_chebyshev_eigenvalue_ratio: 30.0,
            },
        );
        let r_host = vec![1.0_f64, -2.0, 0.5, 3.0, -1.5, 0.25, 2.0, -0.75];
        let mut r = backend.raw_alloc::<f64>(r_host.len()).unwrap();
        backend.copy_h2d(&r_host, &mut r);
        let mut z_multi = backend.raw_alloc::<f64>(r_host.len()).unwrap();

        precond.apply_colmajor(&backend, 2, &r, &mut z_multi);

        let mut expected = vec![0.0_f64; r_host.len()];
        for col in 0..2 {
            let range = col * 4..(col + 1) * 4;
            let mut r_col = backend.raw_alloc::<f64>(4).unwrap();
            backend.copy_h2d(&r_host[range.clone()], &mut r_col);
            let mut z_col = backend.raw_alloc::<f64>(4).unwrap();
            precond.try_apply(&backend, &r_col, &mut z_col).unwrap();
            backend.copy_d2h(&z_col, &mut expected[range]);
        }

        let mut actual = vec![0.0_f64; r_host.len()];
        backend.copy_d2h(&z_multi, &mut actual);
        for (index, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1.0e-12,
                "index {index}: got {got} want {want}; actual={actual:?} expected={expected:?}"
            );
        }

        let diagnostics = precond.diagnostics();
        assert_eq!(diagnostics.apply_count, 3);
        assert_eq!(diagnostics.apply_failures, 0);

        let mut r_failure = backend.raw_alloc::<f64>(4).unwrap();
        backend.copy_h2d(&r_host[..4], &mut r_failure);
        let sentinel = vec![7.0_f64; 4];
        let mut z_failure = backend.raw_alloc::<f64>(4).unwrap();
        backend.copy_h2d(&sentinel, &mut z_failure);
        CpuBackend::inject_pending_error();
        let error = precond
            .try_apply(&backend, &r_failure, &mut z_failure)
            .expect_err("injected backend failure must reach the checked boundary");
        assert!(matches!(error, crate::BackendError::OperationFailed { .. }));
        let mut after_failure = vec![0.0_f64; 4];
        backend.copy_d2h(&z_failure, &mut after_failure);
        assert_eq!(after_failure, sentinel);

        let diagnostics = precond.diagnostics();
        assert_eq!(diagnostics.apply_count, 3);
        assert_eq!(diagnostics.apply_failures, 1);
        assert_eq!(
            diagnostics.first_failure_stage,
            Some(crate::DevicePreconditionerFailureStage::Launch)
        );
    }

    #[test]
    fn external_pressure_block_apply_does_not_stage_through_host() {
        let backend = CpuBackend::new();
        let row_ptr = vec![0, 3, 6, 7, 8];
        let col_idx = vec![0, 1, 2, 0, 1, 3, 2, 3];
        let values = vec![4.0_f64, 0.5, 1.0, 0.25, 3.0, -0.5, 0.1, 0.1];
        let a = BackendCsr::from_host(&backend, 4, 4, &row_ptr, &col_idx, &values);
        let schur_row_ptrs = vec![0_usize, 2, 4];
        let schur_col_indices = vec![0_usize, 1, 0, 1];
        let schur_values = vec![2.0_f64, 0.5, 0.5, 2.0];
        let external = PointJacobiBackend::from_diagonal(&backend, &[1.0, 1.0, 2.0, 2.0]);
        let config = SaddlePointJacobiSchurBackendConfig {
            velocity_block_size: 2,
            velocity_degree: 1,
            velocity_damping: 1.0,
            pressure_degree: 1,
            pressure_damping: 1.0,
            pressure_kind: PressureSchurBackendKind::External,
            pressure_chebyshev_degree: 5,
            pressure_chebyshev_eigenvalue_ratio: 30.0,
        };
        let precond =
            SaddlePointJacobiSchurBackend::from_csr_host_parts_with_pressure_preconditioner(
                &backend,
                &a,
                &row_ptr,
                &col_idx,
                &values,
                HostCsrRef::new(2, 2, &schur_row_ptrs, &schur_col_indices, &schur_values),
                config,
                Some(&external),
            );
        let r_host = vec![1.0_f64, -2.0, 0.5, 3.0, -1.5, 0.25, 2.0, -0.75];
        let mut r = backend.raw_alloc::<f64>(r_host.len()).unwrap();
        backend.copy_h2d(&r_host, &mut r);
        let mut expected = vec![0.0_f64; r_host.len()];
        for column in 0..2 {
            let range = column * 4..(column + 1) * 4;
            let mut r_col = backend.raw_alloc::<f64>(4).unwrap();
            backend.copy_h2d(&r_host[range.clone()], &mut r_col);
            let mut z_col = backend.raw_alloc::<f64>(4).unwrap();
            precond.try_apply(&backend, &r_col, &mut z_col).unwrap();
            backend.copy_d2h(&z_col, &mut expected[range]);
        }

        let mut z = backend.raw_alloc::<f64>(r_host.len()).unwrap();
        backend.reset_transfer_stats();
        precond.try_apply_colmajor(&backend, 2, &r, &mut z).unwrap();
        let device_only = backend.transfer_stats();
        assert_eq!(device_only.h2d_operations, 0);
        assert_eq!(device_only.d2h_operations, 0);

        let mut actual = vec![0.0_f64; r_host.len()];
        backend.copy_d2h(&z, &mut actual);
        for (index, (got, want)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1.0e-12,
                "index {index}: got {got} want {want}"
            );
        }
    }

    #[test]
    fn checked_constructor_rejects_malformed_velocity_csr() {
        let backend = CpuBackend::new();
        let valid_row_ptr = vec![0, 1, 2, 3];
        let valid_col_idx = vec![0, 1, 2];
        let valid_values = vec![1.0_f64; 3];
        let a = BackendCsr::from_host(
            &backend,
            3,
            3,
            &valid_row_ptr,
            &valid_col_idx,
            &valid_values,
        );
        let malformed_row_ptr = vec![0, 2, 1, 3];
        let schur_row_ptrs = vec![0_usize, 1];
        let schur_col_indices = vec![0_usize];
        let schur_values = vec![1.0_f64];

        let result = SaddlePointJacobiSchurBackend::try_from_csr_host_parts(
            &backend,
            &a,
            &malformed_row_ptr,
            &valid_col_idx,
            &valid_values,
            HostCsrRef::new(1, 1, &schur_row_ptrs, &schur_col_indices, &schur_values),
            SaddlePointJacobiSchurBackendConfig {
                velocity_block_size: 2,
                velocity_degree: 1,
                velocity_damping: 1.0,
                pressure_degree: 1,
                pressure_damping: 1.0,
                pressure_kind: PressureSchurBackendKind::Polynomial,
                pressure_chebyshev_degree: 2,
                pressure_chebyshev_eigenvalue_ratio: 2.0,
            },
        );

        assert!(matches!(
            result,
            Err(crate::BackendError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn checked_apply_rejects_wrong_buffer_lengths() {
        let backend = CpuBackend::new();
        let row_ptr = vec![0, 1, 2, 3];
        let col_idx = vec![0, 1, 2];
        let values = vec![1.0_f64; 3];
        let a = BackendCsr::from_host(&backend, 3, 3, &row_ptr, &col_idx, &values);
        let schur_row_ptrs = vec![0_usize, 1];
        let schur_col_indices = vec![0_usize];
        let schur_values = vec![1.0_f64];
        let precond = SaddlePointJacobiSchurBackend::from_csr_host_parts(
            &backend,
            &a,
            &row_ptr,
            &col_idx,
            &values,
            HostCsrRef::new(1, 1, &schur_row_ptrs, &schur_col_indices, &schur_values),
            SaddlePointJacobiSchurBackendConfig {
                velocity_block_size: 2,
                velocity_degree: 1,
                velocity_damping: 1.0,
                pressure_degree: 1,
                pressure_damping: 1.0,
                pressure_kind: PressureSchurBackendKind::Polynomial,
                pressure_chebyshev_degree: 2,
                pressure_chebyshev_eigenvalue_ratio: 2.0,
            },
        );
        let r = backend.raw_alloc::<f64>(2).unwrap();
        let mut z = backend.raw_alloc::<f64>(3).unwrap();

        let result = precond.try_apply(&backend, &r, &mut z);
        assert!(matches!(
            result,
            Err(crate::BackendError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn checked_colmajor_rejects_workspace_overflow() {
        let backend = CpuBackend::new();
        let row_ptr = vec![0, 1, 2, 3];
        let col_idx = vec![0, 1, 2];
        let values = vec![1.0_f64; 3];
        let a = BackendCsr::from_host(&backend, 3, 3, &row_ptr, &col_idx, &values);
        let schur_row_ptrs = vec![0_usize, 1];
        let schur_col_indices = vec![0_usize];
        let schur_values = vec![1.0_f64];
        let precond = SaddlePointJacobiSchurBackend::from_csr_host_parts(
            &backend,
            &a,
            &row_ptr,
            &col_idx,
            &values,
            HostCsrRef::new(1, 1, &schur_row_ptrs, &schur_col_indices, &schur_values),
            SaddlePointJacobiSchurBackendConfig {
                velocity_block_size: 2,
                velocity_degree: 1,
                velocity_damping: 1.0,
                pressure_degree: 1,
                pressure_damping: 1.0,
                pressure_kind: PressureSchurBackendKind::Polynomial,
                pressure_chebyshev_degree: 2,
                pressure_chebyshev_eigenvalue_ratio: 2.0,
            },
        );
        let r = backend.raw_alloc::<f64>(0).unwrap();
        let mut z = backend.raw_alloc::<f64>(0).unwrap();

        let result = precond.try_apply_colmajor(&backend, usize::MAX, &r, &mut z);
        assert!(matches!(
            result,
            Err(crate::BackendError::InvalidArgument { .. })
        ));
    }
}
