//! Device-resident operators and preconditioners.
//!
//! These types sit at the compute boundary: they own backend buffers and expose
//! the minimal `apply` contracts that portable Krylov solvers need. Backend-
//! specific storage details stay here instead of leaking into solver code.

use crate::host_csr::HostCsrRef;
use crate::pool::Buffer;
use crate::{Backend, DeviceBufferLike, DeviceScalar, LowerPrecision, MemoryPool};
use num_traits::Float;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

/// CSR sparse matrix stored on a [`Backend`] device.
///
/// Three buffers live on-device: row pointers (`i32`), column indices (`i32`),
/// and values (`T`). Applications can upload the structure once and reuse the
/// values buffer across solver calls when the backend supports in-place updates.
pub struct DeviceCsr<B: Backend, T: DeviceScalar = f64> {
    pub m: usize,
    pub n: usize,
    pub nnz: usize,
    pub row_ptr: B::Buffer<i32>,
    pub col_idx: B::Buffer<i32>,
    pub values: B::Buffer<T>,
}

impl<B: Backend, T: DeviceScalar> DeviceCsr<B, T> {
    /// Build a backend-prebaked SpMV handle for this matrix.
    pub fn prebake<'a>(&'a self, backend: &'a B) -> B::CsrPrebaked<'a, T> {
        backend.prebake_csr::<T>(self.m, self.n, &self.row_ptr, &self.col_idx, &self.values)
    }

    /// Checked variant of [`Self::prebake`].
    pub fn try_prebake<'a>(
        &'a self,
        backend: &'a B,
    ) -> crate::BackendResult<B::CsrPrebaked<'a, T>> {
        backend.try_prebake_csr::<T>(self.m, self.n, &self.row_ptr, &self.col_idx, &self.values)
    }

    /// Upload a host CSR matrix to the device.
    ///
    /// Panics on allocation or shape failure; production paths must use
    /// [`Self::try_from_host`] and propagate the typed error instead.
    #[deprecated(note = "panics on allocation/shape failure; use try_from_host")]
    pub fn from_host(
        backend: &B,
        m: usize,
        n: usize,
        row_ptr_host: &[i32],
        col_idx_host: &[i32],
        values_host: &[T],
    ) -> Self {
        Self::try_from_host(backend, m, n, row_ptr_host, col_idx_host, values_host)
            .expect("CSR device allocation failed")
    }

    /// Fallible variant of [`Self::from_host`] for production paths that
    /// already return a backend error.  Allocation failures are propagated
    /// instead of becoming process-aborting panics.
    pub fn try_from_host(
        backend: &B,
        m: usize,
        n: usize,
        row_ptr_host: &[i32],
        col_idx_host: &[i32],
        values_host: &[T],
    ) -> crate::BackendResult<Self> {
        let nnz = validate_csr_inputs(m, n, row_ptr_host, col_idx_host, values_host)?;
        let mut row_ptr = backend
            .raw_alloc::<i32>(m + 1)
            .map_err(crate::BackendError::from)?;
        let mut col_idx = backend
            .raw_alloc::<i32>(nnz)
            .map_err(crate::BackendError::from)?;
        let mut values = backend
            .raw_alloc::<T>(nnz)
            .map_err(crate::BackendError::from)?;

        backend.try_copy_h2d(row_ptr_host, &mut row_ptr)?;
        backend.try_copy_h2d(col_idx_host, &mut col_idx)?;
        backend.try_copy_h2d(values_host, &mut values)?;

        Ok(Self {
            m,
            n,
            nnz,
            row_ptr,
            col_idx,
            values,
        })
    }

    /// Replaces only the values buffer from the host.
    ///
    /// The sparsity structure (row pointers and column indices) stays
    /// resident, so repeated solves with changing values avoid the
    /// structure re-upload that `try_from_host` performs.
    ///
    /// # Errors
    /// Returns [`crate::BackendError`] when `values_host` does not hold
    /// exactly `nnz` entries or the copy fails.
    pub fn update_values(&mut self, backend: &B, values_host: &[T]) -> crate::BackendResult<()> {
        if values_host.len() != self.nnz {
            return Err(crate::BackendError::InvalidArgument {
                operation: "device_csr_update_values",
                reason: format!(
                    "expected {} values for the resident sparsity, found {}",
                    self.nnz,
                    values_host.len()
                ),
            });
        }
        backend.try_copy_h2d(values_host, &mut self.values)?;
        Ok(())
    }

    /// Upload a borrowed host CSR matrix to the device.
    ///
    /// The host interchange type uses `usize` indices; backend CSR storage
    /// uses `i32` indices to match BLAS/cuSPARSE-style kernels.
    ///
    /// Panics on allocation or shape failure; production paths must use
    /// [`Self::try_from_host_csr_ref`] instead.
    #[deprecated(note = "panics on allocation/shape failure; use try_from_host_csr_ref")]
    pub fn from_host_csr_ref(backend: &B, matrix: HostCsrRef<'_, T>) -> Self {
        Self::try_from_host_csr_ref(backend, matrix).expect("CSR host-reference upload failed")
    }

    /// Checked variant of [`Self::from_host_csr_ref`].
    pub fn try_from_host_csr_ref(
        backend: &B,
        matrix: HostCsrRef<'_, T>,
    ) -> crate::BackendResult<Self> {
        let row_ptr: Vec<i32> = matrix
            .row_ptrs
            .iter()
            .map(|&idx| {
                i32::try_from(idx).map_err(|_| crate::BackendError::InvalidArgument {
                    operation: "csr_upload",
                    reason: format!("row pointer {idx} exceeds i32 CSR storage"),
                })
            })
            .collect::<Result<_, _>>()?;
        let col_idx: Vec<i32> = matrix
            .col_indices
            .iter()
            .map(|&idx| {
                i32::try_from(idx).map_err(|_| crate::BackendError::InvalidArgument {
                    operation: "csr_upload",
                    reason: format!("column index {idx} exceeds i32 CSR storage"),
                })
            })
            .collect::<Result<_, _>>()?;
        Self::try_from_host(
            backend,
            matrix.num_rows,
            matrix.num_cols,
            &row_ptr,
            &col_idx,
            matrix.values,
        )
    }

    /// Upload a host diagonal as a square CSR matrix.
    ///
    /// Panics on allocation failure; production paths must use
    /// [`Self::try_from_host_diagonal`] instead.
    #[deprecated(note = "panics on allocation failure; use try_from_host_diagonal")]
    pub fn from_host_diagonal(backend: &B, diagonal: &[T]) -> Self {
        Self::try_from_host_diagonal(backend, diagonal).expect("CSR diagonal allocation failed")
    }

    /// Checked variant of [`Self::from_host_diagonal`].
    pub fn try_from_host_diagonal(backend: &B, diagonal: &[T]) -> crate::BackendResult<Self> {
        let n = diagonal.len();
        let row_ptr: Vec<i32> = (0..=n)
            .map(|idx| {
                i32::try_from(idx).map_err(|_| crate::BackendError::InvalidArgument {
                    operation: "csr_diagonal",
                    reason: format!("diagonal length {n} exceeds i32 CSR index range"),
                })
            })
            .collect::<Result<_, _>>()?;
        let col_idx: Vec<i32> = (0..n)
            .map(|idx| {
                i32::try_from(idx).map_err(|_| crate::BackendError::InvalidArgument {
                    operation: "csr_diagonal",
                    reason: format!("diagonal length {n} exceeds i32 CSR index range"),
                })
            })
            .collect::<Result<_, _>>()?;
        Self::try_from_host(backend, n, n, &row_ptr, &col_idx, diagonal)
    }

    /// Upload the elementwise inverse of a host diagonal as a square CSR matrix.
    ///
    /// Entries with magnitude at machine epsilon or below are mapped to zero.
    ///
    /// Panics on allocation failure; production paths must use
    /// [`Self::try_from_host_inverse_diagonal`] instead.
    #[deprecated(note = "panics on allocation failure; use try_from_host_inverse_diagonal")]
    pub fn from_host_inverse_diagonal(backend: &B, diagonal: &[T]) -> Self {
        Self::try_from_host_inverse_diagonal(backend, diagonal)
            .expect("CSR inverse diagonal allocation failed")
    }

    /// Checked variant of [`Self::from_host_inverse_diagonal`].
    pub fn try_from_host_inverse_diagonal(
        backend: &B,
        diagonal: &[T],
    ) -> crate::BackendResult<Self> {
        if diagonal.iter().any(|value| !value.is_finite()) {
            return Err(crate::BackendError::InvalidArgument {
                operation: "csr_inverse_diagonal",
                reason: "diagonal values must be finite".into(),
            });
        }
        let eps = T::Real::epsilon();
        let inverse: Vec<T> = diagonal
            .iter()
            .map(|&d| {
                if d.norm_sqr().sqrt() > eps {
                    T::one() / d
                } else {
                    T::zero()
                }
            })
            .collect();
        let n = inverse.len();
        let row_ptr: Vec<i32> = (0..=n)
            .map(|idx| {
                i32::try_from(idx).map_err(|_| crate::BackendError::InvalidArgument {
                    operation: "csr_inverse_diagonal",
                    reason: format!("diagonal length {n} exceeds i32 CSR index range"),
                })
            })
            .collect::<Result<_, _>>()?;
        let col_idx: Vec<i32> = (0..n)
            .map(|idx| {
                i32::try_from(idx).map_err(|_| crate::BackendError::InvalidArgument {
                    operation: "csr_inverse_diagonal",
                    reason: format!("diagonal length {n} exceeds i32 CSR index range"),
                })
            })
            .collect::<Result<_, _>>()?;
        Self::try_from_host(backend, n, n, &row_ptr, &col_idx, &inverse)
    }

    /// Upload a host CSR matrix through a [`MemoryPool`] so matrix storage is
    /// counted against the configured device-memory quota.
    pub fn from_pool(
        pool: &MemoryPool<B>,
        m: usize,
        n: usize,
        row_ptr_host: &[i32],
        col_idx_host: &[i32],
        values_host: &[T],
    ) -> Result<PooledDeviceCsr<B, T>, crate::error::OomError> {
        let nnz = validate_csr_inputs(m, n, row_ptr_host, col_idx_host, values_host)
            .map_err(|error| crate::error::OomError::Invalid(error.to_string()))?;

        let mut row_ptr = pool.try_alloc::<i32>(m + 1)?;
        let mut col_idx = pool.try_alloc::<i32>(nnz)?;
        let mut values = pool.try_alloc::<T>(nnz)?;

        let backend = pool.backend();
        backend
            .try_copy_h2d(row_ptr_host, row_ptr.raw_mut())
            .map_err(|error| crate::error::OomError::Invalid(error.to_string()))?;
        backend
            .try_copy_h2d(col_idx_host, col_idx.raw_mut())
            .map_err(|error| crate::error::OomError::Invalid(error.to_string()))?;
        backend
            .try_copy_h2d(values_host, values.raw_mut())
            .map_err(|error| crate::error::OomError::Invalid(error.to_string()))?;

        Ok(PooledDeviceCsr {
            m,
            n,
            nnz,
            row_ptr,
            col_idx,
            values,
        })
    }

    /// Checked pool-backed CSR upload. Quota, driver-allocation, and transfer
    /// failures are preserved as [`crate::BackendError`] values.
    pub fn try_from_pool(
        pool: &MemoryPool<B>,
        m: usize,
        n: usize,
        row_ptr_host: &[i32],
        col_idx_host: &[i32],
        values_host: &[T],
    ) -> crate::BackendResult<PooledDeviceCsr<B, T>> {
        let nnz = validate_csr_inputs(m, n, row_ptr_host, col_idx_host, values_host)?;
        let mut row_ptr = pool
            .try_alloc::<i32>(m + 1)
            .map_err(crate::BackendError::from)?;
        let mut col_idx = pool
            .try_alloc::<i32>(nnz)
            .map_err(crate::BackendError::from)?;
        let mut values = pool
            .try_alloc::<T>(nnz)
            .map_err(crate::BackendError::from)?;

        let backend = pool.backend();
        backend.try_copy_h2d(row_ptr_host, row_ptr.raw_mut())?;
        backend.try_copy_h2d(col_idx_host, col_idx.raw_mut())?;
        backend.try_copy_h2d(values_host, values.raw_mut())?;

        Ok(PooledDeviceCsr {
            m,
            n,
            nnz,
            row_ptr,
            col_idx,
            values,
        })
    }

    /// Upload only the CSR sparsity structure and allocate an uninitialised
    /// values buffer for later updates.
    ///
    /// Use [`Self::copy_values_from_host`] before applying the operator.
    ///
    /// Panics on allocation or shape failure; production paths must use
    /// [`Self::try_from_host_structure`] instead.
    #[deprecated(note = "panics on allocation/shape failure; use try_from_host_structure")]
    pub fn from_host_structure(
        backend: &B,
        m: usize,
        n: usize,
        row_ptr_host: &[i32],
        col_idx_host: &[i32],
    ) -> Self {
        Self::try_from_host_structure(backend, m, n, row_ptr_host, col_idx_host)
            .expect("CSR structure upload failed")
    }

    /// Checked variant of [`Self::from_host_structure`].
    pub fn try_from_host_structure(
        backend: &B,
        m: usize,
        n: usize,
        row_ptr_host: &[i32],
        col_idx_host: &[i32],
    ) -> crate::BackendResult<Self> {
        let nnz = validate_csr_structure(m, n, row_ptr_host, col_idx_host)?;
        let mut row_ptr = backend
            .raw_alloc::<i32>(m + 1)
            .map_err(crate::BackendError::from)?;
        let mut col_idx = backend
            .raw_alloc::<i32>(nnz)
            .map_err(crate::BackendError::from)?;
        let values = backend
            .raw_alloc::<T>(nnz)
            .map_err(crate::BackendError::from)?;

        backend.try_copy_h2d(row_ptr_host, &mut row_ptr)?;
        backend.try_copy_h2d(col_idx_host, &mut col_idx)?;

        Ok(Self {
            m,
            n,
            nnz,
            row_ptr,
            col_idx,
            values,
        })
    }

    /// Upload CSR sparsity through a [`MemoryPool`] and allocate an
    /// uninitialised values buffer for later updates.
    pub fn from_pool_structure(
        pool: &MemoryPool<B>,
        m: usize,
        n: usize,
        row_ptr_host: &[i32],
        col_idx_host: &[i32],
    ) -> Result<PooledDeviceCsr<B, T>, crate::error::OomError> {
        let nnz = validate_csr_structure(m, n, row_ptr_host, col_idx_host)
            .map_err(|error| crate::error::OomError::Invalid(error.to_string()))?;

        let mut row_ptr = pool.try_alloc::<i32>(m + 1)?;
        let mut col_idx = pool.try_alloc::<i32>(nnz)?;
        // SAFETY: callers must fill values via copy_values_from_host before apply.
        let values = unsafe { pool.try_alloc_uninitialized::<T>(nnz)? };

        let backend = pool.backend();
        backend
            .try_copy_h2d(row_ptr_host, row_ptr.raw_mut())
            .map_err(|error| crate::error::OomError::Invalid(error.to_string()))?;
        backend
            .try_copy_h2d(col_idx_host, col_idx.raw_mut())
            .map_err(|error| crate::error::OomError::Invalid(error.to_string()))?;

        Ok(PooledDeviceCsr {
            m,
            n,
            nnz,
            row_ptr,
            col_idx,
            values,
        })
    }

    /// Checked pool-backed structure upload. The returned values buffer must
    /// be initialized through [`PooledDeviceCsr::try_copy_values_from_host`]
    /// before applying the operator.
    pub fn try_from_pool_structure(
        pool: &MemoryPool<B>,
        m: usize,
        n: usize,
        row_ptr_host: &[i32],
        col_idx_host: &[i32],
    ) -> crate::BackendResult<PooledDeviceCsr<B, T>> {
        let nnz = validate_csr_structure(m, n, row_ptr_host, col_idx_host)?;
        let mut row_ptr = pool
            .try_alloc::<i32>(m + 1)
            .map_err(crate::BackendError::from)?;
        let mut col_idx = pool
            .try_alloc::<i32>(nnz)
            .map_err(crate::BackendError::from)?;
        // SAFETY: this constructor documents that the values buffer must be
        // initialized before any operator application.
        let values = unsafe {
            pool.try_alloc_uninitialized::<T>(nnz)
                .map_err(crate::BackendError::from)?
        };

        let backend = pool.backend();
        backend.try_copy_h2d(row_ptr_host, row_ptr.raw_mut())?;
        backend.try_copy_h2d(col_idx_host, col_idx.raw_mut())?;

        Ok(PooledDeviceCsr {
            m,
            n,
            nnz,
            row_ptr,
            col_idx,
            values,
        })
    }

    /// Replace the values buffer while keeping the CSR sparsity structure
    /// resident on the backend.
    ///
    /// Panics on shape or upload failure; production paths must use
    /// [`Self::try_copy_values_from_host`] instead.
    #[deprecated(note = "panics on shape/upload failure; use try_copy_values_from_host")]
    pub fn copy_values_from_host(&mut self, backend: &B, values_host: &[T]) {
        self.try_copy_values_from_host(backend, values_host)
            .expect("CSR value update failed");
    }

    /// Checked value-buffer update for a CSR matrix whose structure is reused.
    pub fn try_copy_values_from_host(
        &mut self,
        backend: &B,
        values_host: &[T],
    ) -> crate::BackendResult<()> {
        if values_host.len() != self.nnz {
            return Err(crate::BackendError::InvalidArgument {
                operation: "csr_value_update",
                reason: format!(
                    "expected {} values for the existing CSR structure, got {}",
                    self.nnz,
                    values_host.len()
                ),
            });
        }
        if values_host.iter().any(|value| !value.is_finite()) {
            return Err(crate::BackendError::InvalidArgument {
                operation: "csr_value_update",
                reason: "CSR values must be finite".into(),
            });
        }
        backend.try_copy_h2d(values_host, &mut self.values)
    }

    /// Upload a host CSR matrix, casting values from the paired upper
    /// precision to `T` before upload.
    ///
    /// Panics on allocation or shape failure; production paths must use
    /// [`Self::try_from_host_cast`] instead.
    #[deprecated(note = "panics on allocation/shape failure; use try_from_host_cast")]
    pub fn from_host_cast<UpperT>(
        backend: &B,
        m: usize,
        n: usize,
        row_ptr_host: &[i32],
        col_idx_host: &[i32],
        values_upper: &[UpperT],
    ) -> Self
    where
        UpperT: LowerPrecision<Lower = T>,
    {
        Self::try_from_host_cast(backend, m, n, row_ptr_host, col_idx_host, values_upper)
            .expect("CSR cast upload failed")
    }

    /// Checked variant of [`Self::from_host_cast`].
    pub fn try_from_host_cast<UpperT>(
        backend: &B,
        m: usize,
        n: usize,
        row_ptr_host: &[i32],
        col_idx_host: &[i32],
        values_upper: &[UpperT],
    ) -> crate::BackendResult<Self>
    where
        UpperT: LowerPrecision<Lower = T>,
    {
        let nnz = validate_csr_structure(m, n, row_ptr_host, col_idx_host)?;
        if col_idx_host.len() != values_upper.len() {
            return Err(crate::BackendError::InvalidArgument {
                operation: "csr_cast_upload",
                reason: format!(
                    "column-index length {} differs from value length {}",
                    col_idx_host.len(),
                    values_upper.len()
                ),
            });
        }
        if values_upper.iter().any(|value| !value.is_finite()) {
            return Err(crate::BackendError::InvalidArgument {
                operation: "csr_cast_upload",
                reason: "CSR values must be finite before precision cast".into(),
            });
        }

        let mut row_ptr = backend
            .raw_alloc::<i32>(m + 1)
            .map_err(crate::BackendError::from)?;
        let mut col_idx = backend
            .raw_alloc::<i32>(nnz)
            .map_err(crate::BackendError::from)?;
        let mut values = backend
            .raw_alloc::<T>(nnz)
            .map_err(crate::BackendError::from)?;

        backend.try_copy_h2d(row_ptr_host, &mut row_ptr)?;
        backend.try_copy_h2d(col_idx_host, &mut col_idx)?;
        backend.try_copy_h2d_cast(values_upper, &mut values)?;

        Ok(Self {
            m,
            n,
            nnz,
            row_ptr,
            col_idx,
            values,
        })
    }

    /// Upload a host CSR matrix through a [`MemoryPool`], casting values from
    /// the paired upper precision into the device buffer.
    pub fn from_pool_cast<UpperT>(
        pool: &MemoryPool<B>,
        m: usize,
        n: usize,
        row_ptr_host: &[i32],
        col_idx_host: &[i32],
        values_upper: &[UpperT],
    ) -> Result<PooledDeviceCsr<B, T>, crate::error::OomError>
    where
        UpperT: LowerPrecision<Lower = T>,
    {
        if row_ptr_host.len() != m.saturating_add(1) {
            return Err(crate::error::OomError::Invalid(format!(
                "CSR row-pointer length {} differs expected {}",
                row_ptr_host.len(),
                m.saturating_add(1)
            )));
        }
        if col_idx_host.len() != values_upper.len() {
            return Err(crate::error::OomError::Invalid(format!(
                "CSR column/value lengths differ: {} versus {}",
                col_idx_host.len(),
                values_upper.len()
            )));
        }
        if let Err(error) = validate_csr_structure(m, n, row_ptr_host, col_idx_host) {
            return Err(crate::error::OomError::Invalid(error.to_string()));
        }
        if values_upper.iter().any(|value| !value.is_finite()) {
            return Err(crate::error::OomError::Invalid(
                "CSR values must be finite before precision cast".into(),
            ));
        }

        let nnz = values_upper.len();
        let mut row_ptr = pool.try_alloc::<i32>(m + 1)?;
        let mut col_idx = pool.try_alloc::<i32>(nnz)?;
        let mut values = pool.try_alloc::<T>(nnz)?;

        let backend = pool.backend();
        backend
            .try_copy_h2d(row_ptr_host, row_ptr.raw_mut())
            .map_err(|error| crate::error::OomError::Invalid(error.to_string()))?;
        backend
            .try_copy_h2d(col_idx_host, col_idx.raw_mut())
            .map_err(|error| crate::error::OomError::Invalid(error.to_string()))?;
        backend
            .try_copy_h2d_cast(values_upper, values.raw_mut())
            .map_err(|error| crate::error::OomError::Invalid(error.to_string()))?;

        Ok(PooledDeviceCsr {
            m,
            n,
            nnz,
            row_ptr,
            col_idx,
            values,
        })
    }

    /// Checked pool-backed precision-cast upload.
    pub fn try_from_pool_cast<UpperT>(
        pool: &MemoryPool<B>,
        m: usize,
        n: usize,
        row_ptr_host: &[i32],
        col_idx_host: &[i32],
        values_upper: &[UpperT],
    ) -> crate::BackendResult<PooledDeviceCsr<B, T>>
    where
        UpperT: LowerPrecision<Lower = T>,
    {
        let nnz = validate_csr_structure(m, n, row_ptr_host, col_idx_host)?;
        if col_idx_host.len() != values_upper.len() {
            return Err(crate::BackendError::InvalidArgument {
                operation: "pooled_csr_cast_upload",
                reason: format!(
                    "column-index length {} differs from value length {}",
                    col_idx_host.len(),
                    values_upper.len()
                ),
            });
        }
        if values_upper.iter().any(|value| !value.is_finite()) {
            return Err(crate::BackendError::InvalidArgument {
                operation: "pooled_csr_cast_upload",
                reason: "CSR values must be finite before precision cast".into(),
            });
        }

        let mut row_ptr = pool
            .try_alloc::<i32>(m + 1)
            .map_err(crate::BackendError::from)?;
        let mut col_idx = pool
            .try_alloc::<i32>(nnz)
            .map_err(crate::BackendError::from)?;
        let mut values = pool
            .try_alloc::<T>(nnz)
            .map_err(crate::BackendError::from)?;

        let backend = pool.backend();
        backend.try_copy_h2d(row_ptr_host, row_ptr.raw_mut())?;
        backend.try_copy_h2d(col_idx_host, col_idx.raw_mut())?;
        backend.try_copy_h2d_cast(values_upper, values.raw_mut())?;

        Ok(PooledDeviceCsr {
            m,
            n,
            nnz,
            row_ptr,
            col_idx,
            values,
        })
    }

    pub fn n(&self) -> usize {
        self.n
    }
}

/// An owned CSR matrix with a prebaked backend descriptor.
///
/// `DeviceCsr` is intentionally a simple storage type and its `apply` method
/// is consequently a setup-on-every-call compatibility path.  Preconditioners
/// and other steady-state operators should use this type: the CSR buffers and
/// the backend descriptor/workspace live together and remain resident for the
/// lifetime of the object.
///
/// The backend trait uses one lifetime for both the backend and the source
/// buffers.  The concrete prebake implementations only retain source-buffer
/// references (and CUDA/Metal opaque descriptors), so the handle can safely be
/// tied to the owned `Arc` below.  The handle is declared before `matrix` so it
/// is dropped first; this ordering is part of the safety invariant.
pub struct PersistentDeviceCsr<B: Backend, T: DeviceScalar = f64> {
    prebaked: B::CsrPrebaked<'static, T>,
    matrix: Arc<DeviceCsr<B, T>>,
}

impl<B: Backend, T: DeviceScalar> PersistentDeviceCsr<B, T> {
    /// Turn an already-uploaded CSR matrix into a persistent sparse action.
    pub fn try_from_device_csr(backend: &B, matrix: DeviceCsr<B, T>) -> crate::BackendResult<Self> {
        let matrix = Arc::new(matrix);
        let prebaked = matrix.try_prebake(backend)?;
        // SAFETY: every current `CsrPrebaked` implementation stores only
        // pointers/references to the CSR buffers and backend-owned state. The
        // `matrix` Arc is pinned for this value's lifetime and is dropped
        // after `prebaked`, so extending the source-buffer lifetime cannot
        // create a dangling descriptor. The Backend contract documents this
        // ownership requirement for CsrPrebaked.
        let prebaked = unsafe {
            std::mem::transmute::<B::CsrPrebaked<'_, T>, B::CsrPrebaked<'static, T>>(prebaked)
        };
        Ok(Self { prebaked, matrix })
    }

    /// Clone an already-uploaded CSR matrix into independent persistent
    /// storage.  This is useful when a caller lends a matrix to a
    /// preconditioner but the backend descriptor must outlive the borrow.
    pub fn try_clone_from_device_csr(
        backend: &B,
        source: &DeviceCsr<B, T>,
    ) -> crate::BackendResult<Self> {
        let mut row_ptr = backend
            .raw_alloc::<i32>(source.row_ptr.len())
            .map_err(crate::BackendError::from)?;
        let mut col_idx = backend
            .raw_alloc::<i32>(source.col_idx.len())
            .map_err(crate::BackendError::from)?;
        let mut values = backend
            .raw_alloc::<T>(source.values.len())
            .map_err(crate::BackendError::from)?;
        backend.try_copy_d2d(&source.row_ptr, &mut row_ptr)?;
        backend.try_copy_d2d(&source.col_idx, &mut col_idx)?;
        backend.try_copy_d2d(&source.values, &mut values)?;
        Self::try_from_device_csr(
            backend,
            DeviceCsr {
                m: source.m,
                n: source.n,
                nnz: source.nnz,
                row_ptr,
                col_idx,
                values,
            },
        )
    }

    /// Upload and prebake a host CSR matrix in one setup transaction.
    pub fn try_from_host_csr_ref(
        backend: &B,
        matrix: HostCsrRef<'_, T>,
    ) -> crate::BackendResult<Self> {
        Self::try_from_device_csr(backend, DeviceCsr::try_from_host_csr_ref(backend, matrix)?)
    }

    /// Upload and prebake a host diagonal matrix in one setup transaction.
    pub fn try_from_host_diagonal(backend: &B, diagonal: &[T]) -> crate::BackendResult<Self> {
        Self::try_from_device_csr(
            backend,
            DeviceCsr::try_from_host_diagonal(backend, diagonal)?,
        )
    }

    #[must_use]
    pub fn m(&self) -> usize {
        self.matrix.m
    }

    #[must_use]
    pub fn n(&self) -> usize {
        self.matrix.n
    }

    #[must_use]
    pub fn nnz(&self) -> usize {
        self.matrix.nnz
    }

    /// Access the resident CSR storage for diagnostics or value inspection.
    #[must_use]
    pub fn matrix(&self) -> &DeviceCsr<B, T> {
        &self.matrix
    }

    /// Update only the resident CSR values while preserving row/column
    /// storage and the backend's prebaked descriptor/workspace.
    ///
    /// This is the steady-state frequency-sweep path: changing values does
    /// not invalidate the sparsity pattern or require a new cuSPARSE/MPS
    /// handle.  The `Arc` must remain uniquely owned by this operator; any
    /// leaked storage alias is rejected rather than silently rebuilding the
    /// descriptor.
    pub fn try_copy_values_from_host(
        &mut self,
        backend: &B,
        values_host: &[T],
    ) -> crate::BackendResult<()> {
        if values_host.len() != self.nnz() {
            return Err(crate::BackendError::InvalidArgument {
                operation: "persistent_csr_update_values",
                reason: format!("expected {} values, got {}", self.nnz(), values_host.len()),
            });
        }
        if values_host.iter().any(|value| !value.is_finite()) {
            return Err(crate::BackendError::InvalidArgument {
                operation: "persistent_csr_update_values",
                reason: "CSR values must be finite".into(),
            });
        }
        let matrix =
            Arc::get_mut(&mut self.matrix).ok_or_else(|| crate::BackendError::OperationFailed {
                operation: "persistent_csr_update_values",
                reason: "resident CSR storage is aliased; descriptor cannot be updated safely"
                    .into(),
            })?;
        backend.try_copy_h2d(values_host, &mut matrix.values)
    }

    /// Copy resident numeric values without rebuilding sparse descriptors.
    ///
    /// The source must belong to the same backend context. Numeric validity is
    /// the producer's responsibility; no device values are downloaded for checks.
    ///
    /// # Errors
    /// Rejects length or storage-aliasing mismatches and propagates checked
    /// backend copy failures. Backend errors may leave values unusable.
    pub fn try_copy_values_from_device(
        &mut self,
        backend: &B,
        values: &B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        if values.len() != self.matrix.nnz {
            return Err(crate::BackendError::InvalidArgument {
                operation: "persistent_csr_device_update",
                reason: "device value count must match CSR nnz".into(),
            });
        }
        let matrix =
            Arc::get_mut(&mut self.matrix).ok_or_else(|| crate::BackendError::OperationFailed {
                operation: "persistent_csr_device_update",
                reason: "resident CSR storage is aliased".into(),
            })?;
        backend.try_copy_d2d(values, &mut matrix.values)
    }

    /// Combine resident numeric bases without rebuilding the sparse descriptor.
    ///
    /// Sets values to `base + sum(alpha * term)`, in slice order. All buffers
    /// must belong to this operator's backend context. Sparsity, storage addresses
    /// and descriptors remain unchanged; no host staging or scratch is introduced.
    ///
    /// # Errors
    /// Rejects mismatched lengths, non-finite coefficients or aliased storage
    /// before mutation. Backend errors may leave a partial update; complete a
    /// successful replacement before applying the operator again.
    pub fn try_combine_values(
        &mut self,
        backend: &B,
        base: &B::Buffer<T>,
        terms: &[(T, &B::Buffer<T>)],
    ) -> crate::BackendResult<()> {
        self.combine_values(backend, None, base, terms)
    }

    /// Combine a scaled resident base and terms while preserving sparse storage.
    ///
    /// Sets values to `base_scale * base + sum(alpha * term)`. Buffers must
    /// belong to this backend context. No host staging or scratch is introduced.
    ///
    /// # Errors
    /// Rejects invalid lengths, non-finite coefficients or aliased matrix storage.
    /// Backend failures can leave partial values; complete a replacement before use.
    pub fn try_combine_scaled_values(
        &mut self,
        backend: &B,
        base_scale: T,
        base: &B::Buffer<T>,
        terms: &[(T, &B::Buffer<T>)],
    ) -> crate::BackendResult<()> {
        self.combine_values(backend, Some(base_scale), base, terms)
    }

    fn combine_values(
        &mut self,
        backend: &B,
        base_scale: Option<T>,
        base: &B::Buffer<T>,
        terms: &[(T, &B::Buffer<T>)],
    ) -> crate::BackendResult<()> {
        if base_scale.is_some_and(|scale| !scale.is_finite())
            || base.len() != self.nnz()
            || terms
                .iter()
                .any(|(alpha, values)| !alpha.is_finite() || values.len() != self.nnz())
        {
            return Err(crate::BackendError::InvalidArgument {
                operation: "persistent_csr_combine_values",
                reason: "all bases must match nnz and coefficients must be finite".into(),
            });
        }
        let matrix =
            Arc::get_mut(&mut self.matrix).ok_or_else(|| crate::BackendError::OperationFailed {
                operation: "persistent_csr_combine_values",
                reason: "resident CSR storage is aliased".into(),
            })?;
        if let [(beta, y)] = terms
            && backend.try_linear_combination2_native(
                base_scale.unwrap_or_else(T::one),
                base,
                *beta,
                y,
                &mut matrix.values,
            )?
        {
            return Ok(());
        }
        backend.try_copy_d2d(base, &mut matrix.values)?;
        if let Some(scale) = base_scale {
            backend.try_scale(scale, &mut matrix.values)?;
        }
        for &(alpha, values) in terms {
            backend.try_axpy(alpha, values, &mut matrix.values)?;
        }
        Ok(())
    }

    /// Promote resident numeric values without rebuilding the sparse descriptor.
    ///
    /// The source must belong to the same backend context. Numeric validity is
    /// the producer's responsibility; values are not downloaded for validation.
    ///
    /// # Errors
    /// Rejects a mismatched length or aliased storage. Propagates conversion
    /// failures; complete a successful replacement before reusing failed output.
    pub fn try_cast_values_from_lower(
        &mut self,
        backend: &B,
        values: &B::Buffer<T::Lower>,
    ) -> crate::BackendResult<()>
    where
        T: LowerPrecision,
    {
        if values.len() != self.nnz() {
            return Err(crate::BackendError::InvalidArgument {
                operation: "persistent_csr_cast_values",
                reason: "source value count must match CSR nnz".into(),
            });
        }
        let matrix =
            Arc::get_mut(&mut self.matrix).ok_or_else(|| crate::BackendError::OperationFailed {
                operation: "persistent_csr_cast_values",
                reason: "resident CSR storage is aliased".into(),
            })?;
        backend.try_cast_from_lower::<T>(values, &mut matrix.values)
    }

    /// Update the resident CSR values from a higher-precision host slice,
    /// casting on the way in (device-side cast where the backend supports
    /// it, see [`Backend::try_copy_h2d_cast`]).
    ///
    /// This is the mixed-precision frequency-sweep path: it deletes the
    /// serial host-side `f64 -> f32` cast loop plus its temporary `N^2`
    /// buffer. Length, finiteness, and aliasing checks mirror
    /// [`Self::try_copy_values_from_host`].
    pub fn try_copy_values_from_host_cast<UpperT>(
        &mut self,
        backend: &B,
        values_host: &[UpperT],
    ) -> crate::BackendResult<()>
    where
        UpperT: LowerPrecision<Lower = T>,
    {
        if values_host.len() != self.nnz() {
            return Err(crate::BackendError::InvalidArgument {
                operation: "persistent_csr_update_values_cast",
                reason: format!("expected {} values, got {}", self.nnz(), values_host.len()),
            });
        }
        if values_host.iter().any(|value| !value.is_finite()) {
            return Err(crate::BackendError::InvalidArgument {
                operation: "persistent_csr_update_values_cast",
                reason: "CSR values must be finite".into(),
            });
        }
        let matrix =
            Arc::get_mut(&mut self.matrix).ok_or_else(|| crate::BackendError::OperationFailed {
                operation: "persistent_csr_update_values_cast",
                reason: "resident CSR storage is aliased; descriptor cannot be updated safely"
                    .into(),
            })?;
        backend.try_copy_h2d_cast(values_host, &mut matrix.values)
    }

    pub fn try_apply(
        &self,
        backend: &B,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        if x.len() != self.n() || y.len() != self.m() {
            return Err(crate::BackendError::InvalidArgument {
                operation: "persistent_csr_apply",
                reason: format!(
                    "expected input/output lengths {}/{}, got {}/{}",
                    self.n(),
                    self.m(),
                    x.len(),
                    y.len()
                ),
            });
        }
        backend.try_spmv_prebaked::<T>(&self.prebaked, x, y)
    }

    /// Form `b - A*x` with retained sparse storage and native vector fusion.
    ///
    /// Supports rectangular matrices with `x` of length `n` and `b`/`out` of
    /// length `m`. Reuses the existing descriptor and caller-owned output.
    /// Backends without native residual updates use checked scale and AXPY.
    ///
    /// # Errors
    /// Rejects incompatible lengths before mutation and propagates backend
    /// failures. Execution failures may leave partial output; do not publish it.
    pub fn try_residual(
        &self,
        backend: &B,
        x: &B::Buffer<T>,
        b: &B::Buffer<T>,
        out: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        if x.len() != self.n() || b.len() != self.m() || out.len() != self.m() {
            return Err(crate::BackendError::InvalidArgument {
                operation: "persistent_csr_residual",
                reason: format!(
                    "expected x={}, b/out={}, got {}/{}/{}",
                    self.n(),
                    self.m(),
                    x.len(),
                    b.len(),
                    out.len()
                ),
            });
        }
        self.try_apply(backend, x, out)?;
        if !backend.try_residual_update_native(b, out)? {
            backend.try_scale::<T>(-T::one(), out)?;
            backend.try_axpy::<T>(T::one(), b, out)?;
        }
        Ok(())
    }

    pub fn try_apply_colmajor(
        &self,
        backend: &B,
        rhs: usize,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        validate_colmajor_lengths(
            "persistent_csr_apply_colmajor",
            self.m(),
            self.n(),
            rhs,
            x.len(),
            y.len(),
        )?;
        if rhs == 0 {
            return Ok(());
        }
        backend.try_spmm_prebaked_colmajor::<T>(&self.prebaked, rhs, x, y)
    }
}

/// Device-resident sparse operator paired with a precomputed Hermitian CSR.
///
/// Building the adjoint once avoids host transfers and sparse transposition in
/// forward/adjoint optimization loops. Both applications use the backend's
/// ordinary CSR SpMV/SpMM path, so capability and failure semantics remain
/// identical to forward sparse execution.
pub struct AdjointDeviceCsr<B: Backend, T: DeviceScalar = f64> {
    forward: DeviceCsr<B, T>,
    hermitian: DeviceCsr<B, T>,
}

impl<B: Backend, T: DeviceScalar> AdjointDeviceCsr<B, T> {
    /// Upload a square host CSR matrix and its deterministic Hermitian adjoint.
    pub fn try_from_host_csr_ref(
        backend: &B,
        matrix: HostCsrRef<'_, T>,
    ) -> crate::BackendResult<Self> {
        if !matrix.is_square() {
            return Err(crate::BackendError::InvalidArgument {
                operation: "adjoint_csr_upload",
                reason: format!(
                    "device operator contract requires a square matrix, got {}x{}",
                    matrix.num_rows, matrix.num_cols
                ),
            });
        }
        let hermitian_host =
            matrix
                .try_hermitian()
                .map_err(|error| crate::BackendError::InvalidArgument {
                    operation: "adjoint_csr_upload",
                    reason: error.to_string(),
                })?;
        let forward = DeviceCsr::try_from_host_csr_ref(backend, matrix)?;
        let hermitian = DeviceCsr::try_from_host_csr_ref(backend, hermitian_host.as_ref())?;
        Ok(Self { forward, hermitian })
    }

    /// Forward sparse matrix stored on the backend.
    #[must_use]
    pub const fn forward(&self) -> &DeviceCsr<B, T> {
        &self.forward
    }

    /// Precomputed Hermitian sparse matrix stored on the backend.
    #[must_use]
    pub const fn hermitian(&self) -> &DeviceCsr<B, T> {
        &self.hermitian
    }

    /// Create a persistent forward/Hermitian SpMV pair for a steady-state
    /// solve loop.
    ///
    /// The returned operator owns both backend-prebaked descriptors. Keeping
    /// it alive across applications avoids rebuilding sparse descriptors (and
    /// any associated backend workspace) for every forward or adjoint step.
    pub fn prebake<'a>(&'a self, backend: &'a B) -> PrebakedAdjointSpmv<'a, B, T> {
        PrebakedAdjointSpmv::new(
            self.forward.n(),
            self.forward.prebake(backend),
            self.hermitian.prebake(backend),
        )
    }

    /// Checked variant of [`Self::prebake`].
    pub fn try_prebake<'a>(
        &'a self,
        backend: &'a B,
    ) -> crate::BackendResult<PrebakedAdjointSpmv<'a, B, T>> {
        Ok(PrebakedAdjointSpmv::new(
            self.forward.n(),
            self.forward.try_prebake(backend)?,
            self.hermitian.try_prebake(backend)?,
        ))
    }
}

/// Persistent forward/Hermitian sparse operator backed by prebaked descriptors.
///
/// This is the allocation-free steady-state counterpart to
/// [`AdjointDeviceCsr`]. Descriptor construction belongs at setup time; all
/// application methods below dispatch directly to the backend handles.
pub struct PrebakedAdjointSpmv<'a, B: Backend, T: DeviceScalar> {
    n: usize,
    forward: B::CsrPrebaked<'a, T>,
    hermitian: B::CsrPrebaked<'a, T>,
}

impl<'a, B: Backend, T: DeviceScalar> PrebakedAdjointSpmv<'a, B, T> {
    fn new(n: usize, forward: B::CsrPrebaked<'a, T>, hermitian: B::CsrPrebaked<'a, T>) -> Self {
        Self {
            n,
            forward,
            hermitian,
        }
    }

    fn validate_colmajor(
        &self,
        rhs: usize,
        x: &B::Buffer<T>,
        y: &B::Buffer<T>,
        operation: &'static str,
    ) -> crate::BackendResult<()> {
        let total = self
            .n
            .checked_mul(rhs)
            .ok_or(crate::BackendError::InvalidArgument {
                operation,
                reason: format!("size overflow: n={}, rhs={rhs}", self.n),
            })?;
        if x.len() != total || y.len() != total {
            return Err(crate::BackendError::InvalidArgument {
                operation,
                reason: format!(
                    "expected input/output length {total}, got {}/{}",
                    x.len(),
                    y.len()
                ),
            });
        }
        Ok(())
    }
}

impl<'a, B: Backend, T: DeviceScalar> DeviceOperator<B, T> for PrebakedAdjointSpmv<'a, B, T> {
    fn n(&self) -> usize {
        self.n
    }

    fn apply(&self, backend: &B, x: &B::Buffer<T>, y: &mut B::Buffer<T>) {
        backend.spmv_prebaked::<T>(&self.forward, x, y);
    }

    fn try_apply(
        &self,
        backend: &B,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        if x.len() != self.n || y.len() != self.n {
            return Err(crate::BackendError::InvalidArgument {
                operation: "prebaked_adjoint_apply",
                reason: format!(
                    "expected input/output length {}, got {}/{}",
                    self.n,
                    x.len(),
                    y.len()
                ),
            });
        }
        backend.try_spmv_prebaked::<T>(&self.forward, x, y)
    }

    fn apply_colmajor(&self, backend: &B, rhs: usize, x: &B::Buffer<T>, y: &mut B::Buffer<T>) {
        backend.spmm_prebaked_colmajor::<T>(&self.forward, rhs, x, y);
    }

    fn try_residual(
        &self,
        backend: &B,
        x: &B::Buffer<T>,
        b: &B::Buffer<T>,
        out: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        if x.len() != self.n || b.len() != self.n || out.len() != self.n {
            return Err(crate::BackendError::InvalidArgument {
                operation: "prebaked_adjoint_residual",
                reason: format!(
                    "expected vector length {}, got x={}, b={}, out={}",
                    self.n,
                    x.len(),
                    b.len(),
                    out.len()
                ),
            });
        }
        backend.try_spmv_prebaked::<T>(&self.forward, x, out)?;
        if !backend.try_residual_update_native(b, out)? {
            backend.try_scale::<T>(-T::one(), out)?;
            backend.try_axpy::<T>(T::one(), b, out)?;
        }
        Ok(())
    }

    fn try_apply_colmajor(
        &self,
        backend: &B,
        rhs: usize,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        self.validate_colmajor(rhs, x, y, "prebaked_adjoint_apply_colmajor")?;
        backend.try_spmm_prebaked_colmajor::<T>(&self.forward, rhs, x, y)
    }
}

impl<'a, B: Backend, T: DeviceScalar> DeviceAdjointOperator<B, T>
    for PrebakedAdjointSpmv<'a, B, T>
{
    fn try_apply_hermitian(
        &self,
        backend: &B,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        if x.len() != self.n || y.len() != self.n {
            return Err(crate::BackendError::InvalidArgument {
                operation: "prebaked_adjoint_apply_hermitian",
                reason: format!(
                    "expected input/output length {}, got {}/{}",
                    self.n,
                    x.len(),
                    y.len()
                ),
            });
        }
        backend.try_spmv_prebaked::<T>(&self.hermitian, x, y)
    }

    fn try_apply_hermitian_colmajor(
        &self,
        backend: &B,
        rhs: usize,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        self.validate_colmajor(rhs, x, y, "prebaked_adjoint_apply_hermitian_colmajor")?;
        backend.try_spmm_prebaked_colmajor::<T>(&self.hermitian, rhs, x, y)
    }
}

/// CSR sparse matrix whose buffers are owned by a [`MemoryPool`].
pub struct PooledDeviceCsr<B: Backend, T: DeviceScalar = f64> {
    pub m: usize,
    pub n: usize,
    pub nnz: usize,
    pub row_ptr: Buffer<i32, B>,
    pub col_idx: Buffer<i32, B>,
    pub values: Buffer<T, B>,
}

fn validate_csr_structure(
    m: usize,
    n: usize,
    row_ptr: &[i32],
    col_idx: &[i32],
) -> crate::BackendResult<usize> {
    let expected_row_ptr_len = m
        .checked_add(1)
        .ok_or(crate::BackendError::InvalidArgument {
            operation: "csr_upload",
            reason: format!("row count {m} overflows row-pointer length"),
        })?;
    if row_ptr.len() != expected_row_ptr_len {
        return Err(crate::BackendError::InvalidArgument {
            operation: "csr_upload",
            reason: format!(
                "expected {expected_row_ptr_len} row pointers, got {}",
                row_ptr.len()
            ),
        });
    }

    let nnz = col_idx.len();
    let nnz_i32 = i32::try_from(nnz).map_err(|_| crate::BackendError::InvalidArgument {
        operation: "csr_upload",
        reason: format!("nnz {nnz} exceeds i32 CSR storage"),
    })?;

    if row_ptr.first() != Some(&0) {
        return Err(crate::BackendError::InvalidArgument {
            operation: "csr_upload",
            reason: "row pointer must start at zero".to_owned(),
        });
    }
    if row_ptr.last() != Some(&nnz_i32) {
        return Err(crate::BackendError::InvalidArgument {
            operation: "csr_upload",
            reason: format!(
                "terminal row pointer must equal nnz {nnz}, got {:?}",
                row_ptr.last()
            ),
        });
    }
    for (row, pair) in row_ptr.windows(2).enumerate() {
        if pair[0] < 0 || pair[1] < pair[0] {
            return Err(crate::BackendError::InvalidArgument {
                operation: "csr_upload",
                reason: format!(
                    "row pointers are not monotone at row {row}: {} -> {}",
                    pair[0], pair[1]
                ),
            });
        }
    }
    for (entry, &column) in col_idx.iter().enumerate() {
        if column < 0 || usize::try_from(column).map_or(true, |column| column >= n) {
            return Err(crate::BackendError::InvalidArgument {
                operation: "csr_upload",
                reason: format!("column index {column} at entry {entry} is outside 0..{n}"),
            });
        }
    }

    Ok(nnz)
}

pub(crate) fn validate_csr_inputs<T: DeviceScalar>(
    m: usize,
    n: usize,
    row_ptr: &[i32],
    col_idx: &[i32],
    values: &[T],
) -> crate::BackendResult<usize> {
    if col_idx.len() != values.len() {
        return Err(crate::BackendError::InvalidArgument {
            operation: "csr_upload",
            reason: format!(
                "column-index length {} differs from value length {}",
                col_idx.len(),
                values.len()
            ),
        });
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(crate::BackendError::InvalidArgument {
            operation: "csr_upload",
            reason: "CSR values must be finite".into(),
        });
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(crate::BackendError::InvalidArgument {
            operation: "csr_upload",
            reason: "CSR values must be finite".into(),
        });
    }
    validate_csr_structure(m, n, row_ptr, col_idx)
}

impl<B: Backend, T: DeviceScalar> PooledDeviceCsr<B, T> {
    pub fn prebake<'a>(&'a self, backend: &'a B) -> B::CsrPrebaked<'a, T> {
        backend.prebake_csr::<T>(
            self.m,
            self.n,
            self.row_ptr.raw(),
            self.col_idx.raw(),
            self.values.raw(),
        )
    }

    /// Checked variant of [`Self::prebake`].
    pub fn try_prebake<'a>(
        &'a self,
        backend: &'a B,
    ) -> crate::BackendResult<B::CsrPrebaked<'a, T>> {
        backend.try_prebake_csr::<T>(
            self.m,
            self.n,
            self.row_ptr.raw(),
            self.col_idx.raw(),
            self.values.raw(),
        )
    }

    /// Panics on shape or upload failure; production paths must use
    /// [`Self::try_copy_values_from_host`] instead.
    #[deprecated(note = "panics on shape/upload failure; use try_copy_values_from_host")]
    pub fn copy_values_from_host(&mut self, backend: &B, values_host: &[T]) {
        self.try_copy_values_from_host(backend, values_host)
            .expect("pooled CSR value update failed");
    }

    /// Checked value-buffer update for a pooled CSR matrix.
    pub fn try_copy_values_from_host(
        &mut self,
        backend: &B,
        values_host: &[T],
    ) -> crate::BackendResult<()> {
        if values_host.len() != self.nnz {
            return Err(crate::BackendError::InvalidArgument {
                operation: "pooled_csr_value_update",
                reason: format!(
                    "expected {} values for the existing CSR structure, got {}",
                    self.nnz,
                    values_host.len()
                ),
            });
        }
        if values_host.iter().any(|value| !value.is_finite()) {
            return Err(crate::BackendError::InvalidArgument {
                operation: "pooled_csr_value_update",
                reason: "CSR values must be finite".into(),
            });
        }
        backend.try_copy_h2d(values_host, self.values.raw_mut())
    }

    pub fn n(&self) -> usize {
        self.n
    }
}

impl<B: Backend> DeviceCsr<B, f32> {
    /// Backward-compatible helper for f64 host values uploaded as f32.
    ///
    /// Panics on allocation or shape failure; production paths must use
    /// [`Self::try_from_host_cast`] instead.
    #[deprecated(note = "panics on allocation/shape failure; use try_from_host_cast")]
    pub fn from_host_f64_values(
        backend: &B,
        m: usize,
        n: usize,
        row_ptr_host: &[i32],
        col_idx_host: &[i32],
        values_f64: &[f64],
    ) -> Self {
        Self::try_from_host_cast::<f64>(backend, m, n, row_ptr_host, col_idx_host, values_f64)
            .expect("CSR f64-value upload failed")
    }
}

impl<B: Backend> PooledDeviceCsr<B, f32> {
    /// Backward-compatible helper for f64 host values uploaded as f32.
    pub fn from_pool_f64_values(
        pool: &MemoryPool<B>,
        m: usize,
        n: usize,
        row_ptr_host: &[i32],
        col_idx_host: &[i32],
        values_f64: &[f64],
    ) -> Result<Self, crate::error::OomError> {
        DeviceCsr::<B, f32>::from_pool_cast::<f64>(
            pool,
            m,
            n,
            row_ptr_host,
            col_idx_host,
            values_f64,
        )
    }
}

/// Backend-aware linear operator over device buffers.
fn validate_colmajor_lengths(
    operation: &'static str,
    rows: usize,
    cols: usize,
    rhs: usize,
    x_len: usize,
    y_len: usize,
) -> crate::BackendResult<()> {
    let expected_x = cols
        .checked_mul(rhs)
        .ok_or_else(|| crate::BackendError::InvalidArgument {
            operation,
            reason: format!("input shape overflows: cols={cols}, rhs={rhs}"),
        })?;
    let expected_y = rows
        .checked_mul(rhs)
        .ok_or_else(|| crate::BackendError::InvalidArgument {
            operation,
            reason: format!("output shape overflows: rows={rows}, rhs={rhs}"),
        })?;
    if x_len != expected_x || y_len != expected_y {
        return Err(crate::BackendError::InvalidArgument {
            operation,
            reason: format!(
                "expected input/output lengths {expected_x}/{expected_y}, got {x_len}/{y_len}"
            ),
        });
    }
    Ok(())
}

pub trait DeviceOperator<B: Backend, T: DeviceScalar>: Send + Sync {
    fn n(&self) -> usize;

    fn apply(&self, backend: &B, x: &B::Buffer<T>, y: &mut B::Buffer<T>);

    /// Fallible single-right-hand-side application with shape errors
    /// propagated to the caller. The legacy [`Self::apply`] method remains
    /// available for already-validated hot paths.
    fn try_apply(
        &self,
        backend: &B,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        let n = self.n();
        if x.len() != n || y.len() != n {
            return Err(crate::BackendError::InvalidArgument {
                operation: "operator_apply",
                reason: format!(
                    "expected input/output length {n}, got {}/{}",
                    x.len(),
                    y.len()
                ),
            });
        }
        crate::backend::catch_backend_panic("operator_apply", || self.apply(backend, x, y))?;
        backend.check_pending_error()
    }

    /// Compatibility hook for implementations that retain a historical
    /// staging path while exposing a newer checked implementation.
    #[doc(hidden)]
    fn try_apply_legacy(
        &self,
        backend: &B,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        self.try_apply(backend, x, y)
    }

    /// Compatibility hook for implementations that retain a historical
    /// multi-RHS staging path while exposing a newer checked implementation.
    #[doc(hidden)]
    fn try_apply_colmajor_legacy(
        &self,
        backend: &B,
        rhs: usize,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        self.try_apply_colmajor(backend, rhs, x, y)
    }

    fn residual(&self, backend: &B, x: &B::Buffer<T>, b: &B::Buffer<T>, out: &mut B::Buffer<T>) {
        self.apply(backend, x, out);
        backend.scale::<T>(-T::one(), out);
        backend.axpy::<T>(T::one(), b, out);
    }

    /// Checked residual evaluation. The legacy [`Self::residual`] method is
    /// retained for already-validated hot paths.
    fn try_residual(
        &self,
        backend: &B,
        x: &B::Buffer<T>,
        b: &B::Buffer<T>,
        out: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        if x.len() != self.n() || b.len() != self.n() || out.len() != self.n() {
            return Err(crate::BackendError::InvalidArgument {
                operation: "operator_residual",
                reason: format!(
                    "expected all buffers to have length {}, got x={}, b={}, out={}",
                    self.n(),
                    x.len(),
                    b.len(),
                    out.len()
                ),
            });
        }
        crate::backend::catch_backend_panic("operator_residual", || {
            self.residual(backend, x, b, out)
        })?;
        backend.check_pending_error()
    }

    /// Apply this square operator to multiple dense right-hand sides:
    /// `Y <- A * X`, where `X` and `Y` are `n × rhs` column-major buffers.
    ///
    /// The default path preserves correctness for arbitrary operators by
    /// applying each column separately. Operators with backend packing or
    /// prebaked sparse descriptors should override this with a real SpMM path.
    fn apply_colmajor(&self, backend: &B, rhs: usize, x: &B::Buffer<T>, y: &mut B::Buffer<T>) {
        if let Err(error) = self.try_apply_colmajor(backend, rhs, x, y) {
            log::error!(
                target: "sonium_compute::operator",
                "device operator multi-RHS application failed: {error}"
            );
        }
    }

    /// Fallible multi-RHS application with allocation and shape errors
    /// propagated to the caller. The legacy [`Self::apply_colmajor`] method
    /// logs these errors for backwards compatibility.
    fn try_apply_colmajor(
        &self,
        backend: &B,
        rhs: usize,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        let n = self.n();
        let total = n
            .checked_mul(rhs)
            .ok_or(crate::BackendError::InvalidArgument {
                operation: "operator_apply_colmajor",
                reason: format!("size overflow: n={n}, rhs={rhs}"),
            })?;
        if x.len() != total || y.len() != total {
            return Err(crate::BackendError::InvalidArgument {
                operation: "operator_apply_colmajor",
                reason: format!(
                    "expected input/output length {total}, got {}/{}",
                    x.len(),
                    y.len()
                ),
            });
        }
        if rhs == 0 || n == 0 {
            return Ok(());
        }
        if rhs == 1 {
            return self.try_apply(backend, x, y);
        }

        let mut x_host = crate::backend::try_zeroed_host(total, "operator_apply_colmajor")?;
        let mut y_host = crate::backend::try_zeroed_host(total, "operator_apply_colmajor")?;
        backend.try_copy_d2h(x, &mut x_host)?;

        let mut x_col = backend
            .raw_alloc::<T>(n)
            .map_err(crate::BackendError::from)?;
        let mut y_col = backend
            .raw_alloc::<T>(n)
            .map_err(crate::BackendError::from)?;

        for col in 0..rhs {
            let range = col * n..(col + 1) * n;
            backend.try_copy_h2d(&x_host[range.clone()], &mut x_col)?;
            self.try_apply(backend, &x_col, &mut y_col)?;
            backend.try_copy_d2h(&y_col, &mut y_host[range])?;
        }

        backend.try_copy_h2d(&y_host, y)?;
        Ok(())
    }
}

/// Device operator with an exact Hermitian-adjoint application contract.
///
/// Implementors must keep the adjoint device-resident after construction;
/// callers can therefore reuse the operation in optimization loops without
/// hidden host staging or per-application sparse transposition.
pub trait DeviceAdjointOperator<B: Backend, T: DeviceScalar>: DeviceOperator<B, T> {
    /// Apply `Aᴴ` to one device-resident right-hand side.
    fn try_apply_hermitian(
        &self,
        backend: &B,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()>;

    /// Apply `Aᴴ` to `rhs` column-major device-resident right-hand sides.
    fn try_apply_hermitian_colmajor(
        &self,
        backend: &B,
        rhs: usize,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()>;
}

impl<B: Backend, T: DeviceScalar> DeviceOperator<B, T> for DeviceCsr<B, T> {
    fn n(&self) -> usize {
        DeviceCsr::n(self)
    }

    /// Slow path: rebuilds the prebaked handle on every call. For CUDA this
    /// re-creates the `cusparseSpMatDescr_t` and re-allocates the SpMV
    /// workspace per SpMV — measurable Krylov-inner-loop cost.
    ///
    /// Hot loops should wrap the matrix in [`PrebakedSpmv`] once outside the
    /// loop and call its `apply` instead.
    fn apply(&self, backend: &B, x: &B::Buffer<T>, y: &mut B::Buffer<T>) {
        let prebaked = self.prebake(backend);
        backend.spmv_prebaked::<T>(&prebaked, x, y);
    }

    fn try_apply(
        &self,
        backend: &B,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        if x.len() != self.n || y.len() != self.m {
            return Err(crate::BackendError::InvalidArgument {
                operation: "csr_operator_apply",
                reason: format!(
                    "expected input length {} and output length {}, got {}/{}",
                    self.n,
                    self.m,
                    x.len(),
                    y.len()
                ),
            });
        }
        let prebaked = backend.try_prebake_csr::<T>(
            self.m,
            self.n,
            &self.row_ptr,
            &self.col_idx,
            &self.values,
        )?;
        backend.try_spmv_prebaked::<T>(&prebaked, x, y)
    }

    fn residual(&self, backend: &B, x: &B::Buffer<T>, b: &B::Buffer<T>, out: &mut B::Buffer<T>) {
        let prebaked = self.prebake(backend);
        backend.spmv_prebaked::<T>(&prebaked, x, out);
        backend.scale::<T>(-T::one(), out);
        backend.axpy::<T>(T::one(), b, out);
    }

    fn try_residual(
        &self,
        backend: &B,
        x: &B::Buffer<T>,
        b: &B::Buffer<T>,
        out: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        if x.len() != self.n || b.len() != self.m || out.len() != self.m {
            return Err(crate::BackendError::InvalidArgument {
                operation: "csr_operator_residual",
                reason: format!(
                    "expected x={}, b={}, out={} for {}x{} CSR, got {}/{}/{}",
                    self.n,
                    self.m,
                    self.m,
                    self.m,
                    self.n,
                    x.len(),
                    b.len(),
                    out.len()
                ),
            });
        }
        let prebaked = backend.try_prebake_csr::<T>(
            self.m,
            self.n,
            &self.row_ptr,
            &self.col_idx,
            &self.values,
        )?;
        backend.try_spmv_prebaked::<T>(&prebaked, x, out)?;
        if !backend.try_residual_update_native(b, out)? {
            backend.try_scale::<T>(-T::one(), out)?;
            backend.try_axpy::<T>(T::one(), b, out)?;
        }
        Ok(())
    }

    fn apply_colmajor(&self, backend: &B, rhs: usize, x: &B::Buffer<T>, y: &mut B::Buffer<T>) {
        let prebaked = self.prebake(backend);
        backend.spmm_prebaked_colmajor::<T>(&prebaked, rhs, x, y);
    }

    fn try_apply_colmajor(
        &self,
        backend: &B,
        rhs: usize,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        validate_colmajor_lengths(
            "csr_operator_apply_colmajor",
            self.m,
            self.n,
            rhs,
            x.len(),
            y.len(),
        )?;
        if rhs == 0 {
            return Ok(());
        }
        let prebaked = backend.try_prebake_csr::<T>(
            self.m,
            self.n,
            &self.row_ptr,
            &self.col_idx,
            &self.values,
        )?;
        backend.try_spmm_prebaked_colmajor::<T>(&prebaked, rhs, x, y)
    }
}

impl<B: Backend, T: DeviceScalar> DeviceOperator<B, T> for PersistentDeviceCsr<B, T> {
    fn n(&self) -> usize {
        self.n()
    }

    fn apply(&self, backend: &B, x: &B::Buffer<T>, y: &mut B::Buffer<T>) {
        if let Err(error) = self.try_apply(backend, x, y) {
            log::error!(
                target: "sonium_compute::operator",
                "persistent CSR application failed: {error}"
            );
        }
    }

    fn try_apply(
        &self,
        backend: &B,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        PersistentDeviceCsr::try_apply(self, backend, x, y)
    }

    fn residual(&self, backend: &B, x: &B::Buffer<T>, b: &B::Buffer<T>, out: &mut B::Buffer<T>) {
        if let Err(error) = PersistentDeviceCsr::try_residual(self, backend, x, b, out) {
            log::error!("persistent CSR residual failed: {error}");
        }
    }

    fn try_residual(
        &self,
        backend: &B,
        x: &B::Buffer<T>,
        b: &B::Buffer<T>,
        out: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        PersistentDeviceCsr::try_residual(self, backend, x, b, out)
    }

    fn apply_colmajor(&self, backend: &B, rhs: usize, x: &B::Buffer<T>, y: &mut B::Buffer<T>) {
        if let Err(error) = self.try_apply_colmajor(backend, rhs, x, y) {
            log::error!(
                target: "sonium_compute::operator",
                "persistent CSR multi-RHS application failed: {error}"
            );
        }
    }

    fn try_apply_colmajor(
        &self,
        backend: &B,
        rhs: usize,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        PersistentDeviceCsr::try_apply_colmajor(self, backend, rhs, x, y)
    }
}

impl<B: Backend, T: DeviceScalar> DeviceOperator<B, T> for AdjointDeviceCsr<B, T> {
    fn n(&self) -> usize {
        self.forward.n()
    }

    fn apply(&self, backend: &B, x: &B::Buffer<T>, y: &mut B::Buffer<T>) {
        self.forward.apply(backend, x, y);
    }

    fn try_apply(
        &self,
        backend: &B,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        self.forward.try_apply(backend, x, y)
    }

    fn try_residual(
        &self,
        backend: &B,
        x: &B::Buffer<T>,
        b: &B::Buffer<T>,
        out: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        self.forward.try_residual(backend, x, b, out)
    }

    fn apply_colmajor(&self, backend: &B, rhs: usize, x: &B::Buffer<T>, y: &mut B::Buffer<T>) {
        self.forward.apply_colmajor(backend, rhs, x, y);
    }

    fn try_apply_colmajor(
        &self,
        backend: &B,
        rhs: usize,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        self.forward.try_apply_colmajor(backend, rhs, x, y)
    }
}

impl<B: Backend, T: DeviceScalar> DeviceAdjointOperator<B, T> for AdjointDeviceCsr<B, T> {
    fn try_apply_hermitian(
        &self,
        backend: &B,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        self.hermitian.try_apply(backend, x, y)
    }

    fn try_apply_hermitian_colmajor(
        &self,
        backend: &B,
        rhs: usize,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        self.hermitian.try_apply_colmajor(backend, rhs, x, y)
    }
}

impl<B: Backend, T: DeviceScalar> DeviceOperator<B, T> for PooledDeviceCsr<B, T> {
    fn n(&self) -> usize {
        PooledDeviceCsr::n(self)
    }

    /// Slow path. See [`DeviceCsr::apply`] for the rationale and the
    /// [`PrebakedSpmv`] fast-path alternative.
    fn apply(&self, backend: &B, x: &B::Buffer<T>, y: &mut B::Buffer<T>) {
        let prebaked = self.prebake(backend);
        backend.spmv_prebaked::<T>(&prebaked, x, y);
    }

    fn try_apply(
        &self,
        backend: &B,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        if x.len() != self.n || y.len() != self.m {
            return Err(crate::BackendError::InvalidArgument {
                operation: "pooled_csr_operator_apply",
                reason: format!(
                    "expected input length {} and output length {}, got {}/{}",
                    self.n,
                    self.m,
                    x.len(),
                    y.len()
                ),
            });
        }
        let prebaked = backend.try_prebake_csr::<T>(
            self.m,
            self.n,
            self.row_ptr.raw(),
            self.col_idx.raw(),
            self.values.raw(),
        )?;
        backend.try_spmv_prebaked::<T>(&prebaked, x, y)
    }

    fn residual(&self, backend: &B, x: &B::Buffer<T>, b: &B::Buffer<T>, out: &mut B::Buffer<T>) {
        let prebaked = self.prebake(backend);
        backend.spmv_prebaked::<T>(&prebaked, x, out);
        backend.scale::<T>(-T::one(), out);
        backend.axpy::<T>(T::one(), b, out);
    }

    fn try_residual(
        &self,
        backend: &B,
        x: &B::Buffer<T>,
        b: &B::Buffer<T>,
        out: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        if x.len() != self.n || b.len() != self.m || out.len() != self.m {
            return Err(crate::BackendError::InvalidArgument {
                operation: "pooled_csr_operator_residual",
                reason: format!(
                    "expected x={}, b={}, out={} for {}x{} CSR, got {}/{}/{}",
                    self.n,
                    self.m,
                    self.m,
                    self.m,
                    self.n,
                    x.len(),
                    b.len(),
                    out.len()
                ),
            });
        }
        let prebaked = backend.try_prebake_csr::<T>(
            self.m,
            self.n,
            self.row_ptr.raw(),
            self.col_idx.raw(),
            self.values.raw(),
        )?;
        backend.try_spmv_prebaked::<T>(&prebaked, x, out)?;
        if !backend.try_residual_update_native(b, out)? {
            backend.try_scale::<T>(-T::one(), out)?;
            backend.try_axpy::<T>(T::one(), b, out)?;
        }
        Ok(())
    }

    fn apply_colmajor(&self, backend: &B, rhs: usize, x: &B::Buffer<T>, y: &mut B::Buffer<T>) {
        let prebaked = self.prebake(backend);
        backend.spmm_prebaked_colmajor::<T>(&prebaked, rhs, x, y);
    }

    fn try_apply_colmajor(
        &self,
        backend: &B,
        rhs: usize,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        validate_colmajor_lengths(
            "pooled_csr_operator_apply_colmajor",
            self.m,
            self.n,
            rhs,
            x.len(),
            y.len(),
        )?;
        if rhs == 0 {
            return Ok(());
        }
        let prebaked = self.try_prebake(backend)?;
        backend.try_spmm_prebaked_colmajor::<T>(&prebaked, rhs, x, y)
    }
}

/// Fast-path SpMV operator: owns the backend's prebaked CSR handle so
/// repeated `apply` calls skip the descriptor / workspace rebuild that
/// [`DeviceCsr::apply`] / [`PooledDeviceCsr::apply`] pay every call.
///
/// Construct once outside the Krylov inner loop:
///
/// ```ignore
/// let prebaked = matrix.prebake(backend);
/// let op = PrebakedSpmv::new(matrix.n(), prebaked);
/// for _ in 0..max_iter {
///     op.apply(backend, &x, &mut y); // no rebuild
/// }
/// ```
///
/// The lifetime `'a` ties the operator to the source CSR buffers — the
/// matrix must outlive the operator (enforced by the borrow checker).
pub struct PrebakedSpmv<'a, B: Backend, T: DeviceScalar> {
    m: usize,
    n: usize,
    prebaked: B::CsrPrebaked<'a, T>,
}

impl<'a, B: Backend, T: DeviceScalar> PrebakedSpmv<'a, B, T> {
    pub fn new(n: usize, prebaked: B::CsrPrebaked<'a, T>) -> Self {
        Self { m: n, n, prebaked }
    }

    /// Convenience: prebake a `DeviceCsr` and wrap the result.
    pub fn from_csr(backend: &'a B, matrix: &'a DeviceCsr<B, T>) -> Self {
        Self {
            m: matrix.m,
            n: matrix.n(),
            prebaked: matrix.prebake(backend),
        }
    }

    /// Checked variant of [`Self::from_csr`].
    pub fn try_from_csr(backend: &'a B, matrix: &'a DeviceCsr<B, T>) -> crate::BackendResult<Self> {
        Ok(Self {
            m: matrix.m,
            n: matrix.n(),
            prebaked: matrix.try_prebake(backend)?,
        })
    }

    /// Convenience: prebake a `PooledDeviceCsr` and wrap the result.
    pub fn from_pooled(backend: &'a B, matrix: &'a PooledDeviceCsr<B, T>) -> Self {
        Self {
            m: matrix.m,
            n: matrix.n(),
            prebaked: matrix.prebake(backend),
        }
    }

    /// Checked variant of [`Self::from_pooled`].
    pub fn try_from_pooled(
        backend: &'a B,
        matrix: &'a PooledDeviceCsr<B, T>,
    ) -> crate::BackendResult<Self> {
        Ok(Self {
            m: matrix.m,
            n: matrix.n(),
            prebaked: matrix.try_prebake(backend)?,
        })
    }

    /// Apply this prebaked CSR matrix to multiple dense right-hand sides:
    /// `Y <- A * X`, where `X` is `n × rhs` column-major and `Y` is
    /// `m × rhs` column-major.
    pub fn apply_colmajor(&self, backend: &B, rhs: usize, x: &B::Buffer<T>, y: &mut B::Buffer<T>) {
        backend.spmm_prebaked_colmajor::<T>(&self.prebaked, rhs, x, y);
    }
}

impl<'a, B: Backend, T: DeviceScalar> DeviceOperator<B, T> for PrebakedSpmv<'a, B, T> {
    fn n(&self) -> usize {
        self.n
    }

    fn apply(&self, backend: &B, x: &B::Buffer<T>, y: &mut B::Buffer<T>) {
        backend.spmv_prebaked::<T>(&self.prebaked, x, y);
    }

    fn try_apply(
        &self,
        backend: &B,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        if x.len() != self.n || y.len() != self.m {
            return Err(crate::BackendError::InvalidArgument {
                operation: "prebaked_spmv_apply",
                reason: format!(
                    "expected input/output lengths {}/{}, got {}/{}",
                    self.n,
                    self.m,
                    x.len(),
                    y.len()
                ),
            });
        }
        backend.try_spmv_prebaked::<T>(&self.prebaked, x, y)
    }

    fn residual(&self, backend: &B, x: &B::Buffer<T>, b: &B::Buffer<T>, out: &mut B::Buffer<T>) {
        backend.spmv_prebaked::<T>(&self.prebaked, x, out);
        backend.scale::<T>(-T::one(), out);
        backend.axpy::<T>(T::one(), b, out);
    }

    fn try_residual(
        &self,
        backend: &B,
        x: &B::Buffer<T>,
        b: &B::Buffer<T>,
        out: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        if x.len() != self.n || b.len() != self.m || out.len() != self.m {
            return Err(crate::BackendError::InvalidArgument {
                operation: "prebaked_spmv_residual",
                reason: format!(
                    "expected input/output lengths {}/{}, got x={}, b={}, out={}",
                    self.n,
                    self.m,
                    x.len(),
                    b.len(),
                    out.len()
                ),
            });
        }
        backend.try_spmv_prebaked::<T>(&self.prebaked, x, out)?;
        if !backend.try_residual_update_native(b, out)? {
            backend.try_scale::<T>(-T::one(), out)?;
            backend.try_axpy::<T>(T::one(), b, out)?;
        }
        Ok(())
    }

    fn apply_colmajor(&self, backend: &B, rhs: usize, x: &B::Buffer<T>, y: &mut B::Buffer<T>) {
        backend.spmm_prebaked_colmajor::<T>(&self.prebaked, rhs, x, y);
    }

    fn try_apply_colmajor(
        &self,
        backend: &B,
        rhs: usize,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        validate_colmajor_lengths(
            "prebaked_spmv_apply_colmajor",
            self.m,
            self.n,
            rhs,
            x.len(),
            y.len(),
        )?;
        if rhs == 0 {
            return Ok(());
        }
        backend.try_spmm_prebaked_colmajor::<T>(&self.prebaked, rhs, x, y)
    }
}

/// Capability of a device preconditioner adjoint action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceAdjointCapability {
    /// The implementation has an exact, device-resident action.
    Exact,
    /// The action is deliberately unavailable for this preconditioner.
    Unsupported,
}

/// Stable stage vocabulary shared by device-preconditioner diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DevicePreconditionerFailureStage {
    Unknown = 1,
    Setup = 2,
    WorkspaceAllocation = 3,
    Transfer = 4,
    Launch = 5,
    Synchronization = 6,
    DeviceLoss = 7,
    Fallback = 8,
    NonFiniteGeneration = 9,
}

/// Allocation-free counters for a device-preconditioner action boundary.
#[derive(Debug, Default)]
pub struct DevicePreconditionerDiagnostics {
    setup_count: AtomicU64,
    setup_failures: AtomicU64,
    apply_count: AtomicU64,
    apply_failures: AtomicU64,
    transpose_count: AtomicU64,
    transpose_failures: AtomicU64,
    hermitian_count: AtomicU64,
    hermitian_failures: AtomicU64,
    fallback_count: AtomicU64,
    first_failure_stage: AtomicU8,
}

/// Snapshot of [`DevicePreconditionerDiagnostics`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DevicePreconditionerDiagnosticsSnapshot {
    pub setup_count: u64,
    pub setup_failures: u64,
    pub apply_count: u64,
    pub apply_failures: u64,
    pub transpose_count: u64,
    pub transpose_failures: u64,
    pub hermitian_count: u64,
    pub hermitian_failures: u64,
    pub fallback_count: u64,
    pub first_failure_stage: Option<DevicePreconditionerFailureStage>,
}

impl DevicePreconditionerDiagnostics {
    pub fn record_setup(&self) {
        self.setup_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_setup_failure(&self, error: &crate::BackendError) {
        self.setup_failures.fetch_add(1, Ordering::Relaxed);
        self.record_failure_stage(error);
    }

    pub fn record_apply(&self) {
        self.apply_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_transpose(&self) {
        self.transpose_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_hermitian(&self) {
        self.hermitian_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_transpose_failure(&self, error: &crate::BackendError) {
        self.transpose_failures.fetch_add(1, Ordering::Relaxed);
        self.record_failure_stage(error);
    }

    pub fn record_hermitian_failure(&self, error: &crate::BackendError) {
        self.hermitian_failures.fetch_add(1, Ordering::Relaxed);
        self.record_failure_stage(error);
    }

    pub fn record_fallback(&self) {
        self.fallback_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_failure(&self, error: &crate::BackendError) {
        self.apply_failures.fetch_add(1, Ordering::Relaxed);
        self.record_failure_stage(error);
    }

    fn record_failure_stage(&self, error: &crate::BackendError) {
        let stage = match error {
            crate::BackendError::Allocation { .. } => {
                DevicePreconditionerFailureStage::WorkspaceAllocation
            }
            crate::BackendError::Transfer { .. } => DevicePreconditionerFailureStage::Transfer,
            crate::BackendError::Launch { .. } => DevicePreconditionerFailureStage::Launch,
            crate::BackendError::InvalidArgument { operation, .. }
                if operation.contains("nonfinite") || operation.contains("non_finite") =>
            {
                DevicePreconditionerFailureStage::NonFiniteGeneration
            }
            crate::BackendError::OperationFailed { operation, .. }
                if operation.contains("nonfinite") || operation.contains("non_finite") =>
            {
                DevicePreconditionerFailureStage::NonFiniteGeneration
            }
            crate::BackendError::OperationFailed { .. } => DevicePreconditionerFailureStage::Launch,
            crate::BackendError::Synchronization { .. } => {
                DevicePreconditionerFailureStage::Synchronization
            }
            crate::BackendError::DeviceLost { .. } | crate::BackendError::CommandBuffer { .. } => {
                DevicePreconditionerFailureStage::DeviceLoss
            }
            crate::BackendError::HostFallback { .. } => DevicePreconditionerFailureStage::Fallback,
            _ => DevicePreconditionerFailureStage::Unknown,
        };
        let _ = self.first_failure_stage.compare_exchange(
            0,
            stage as u8,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    #[must_use]
    pub fn snapshot(&self) -> DevicePreconditionerDiagnosticsSnapshot {
        let stage = match self.first_failure_stage.load(Ordering::Acquire) {
            0 => None,
            1 => Some(DevicePreconditionerFailureStage::Unknown),
            2 => Some(DevicePreconditionerFailureStage::Setup),
            3 => Some(DevicePreconditionerFailureStage::WorkspaceAllocation),
            4 => Some(DevicePreconditionerFailureStage::Transfer),
            5 => Some(DevicePreconditionerFailureStage::Launch),
            6 => Some(DevicePreconditionerFailureStage::Synchronization),
            7 => Some(DevicePreconditionerFailureStage::DeviceLoss),
            8 => Some(DevicePreconditionerFailureStage::Fallback),
            9 => Some(DevicePreconditionerFailureStage::NonFiniteGeneration),
            _ => Some(DevicePreconditionerFailureStage::Unknown),
        };
        DevicePreconditionerDiagnosticsSnapshot {
            setup_count: self.setup_count.load(Ordering::Acquire),
            setup_failures: self.setup_failures.load(Ordering::Acquire),
            apply_count: self.apply_count.load(Ordering::Acquire),
            apply_failures: self.apply_failures.load(Ordering::Acquire),
            transpose_count: self.transpose_count.load(Ordering::Acquire),
            transpose_failures: self.transpose_failures.load(Ordering::Acquire),
            hermitian_count: self.hermitian_count.load(Ordering::Acquire),
            hermitian_failures: self.hermitian_failures.load(Ordering::Acquire),
            fallback_count: self.fallback_count.load(Ordering::Acquire),
            first_failure_stage: stage,
        }
    }
}

/// Backend-aware preconditioner over device buffers.
pub trait DevicePreconditioner<B: Backend, T: DeviceScalar = f64>: Send + Sync {
    fn apply(&self, backend: &B, r: &B::Buffer<T>, z: &mut B::Buffer<T>);

    /// Capability of the forward action.  Concrete preconditioners may
    /// override this when setup has intentionally selected a fallback.
    fn forward_capability(&self) -> DeviceAdjointCapability {
        DeviceAdjointCapability::Exact
    }

    /// Capability of the transpose action.  Unsupported actions are rejected
    /// before touching any device buffer or backend dispatch path.
    fn transpose_capability(&self) -> DeviceAdjointCapability {
        DeviceAdjointCapability::Unsupported
    }

    /// Capability of the Hermitian action.  Implementations must opt in
    /// explicitly; a transpose is not a Hermitian action for complex scalars.
    fn hermitian_capability(&self) -> DeviceAdjointCapability {
        DeviceAdjointCapability::Unsupported
    }

    /// Snapshot normalized action diagnostics for this preconditioner.
    /// Implementations with a runtime boundary should override this method;
    /// the default keeps lightweight stateless implementations source
    /// compatible.
    fn diagnostics(&self) -> DevicePreconditionerDiagnosticsSnapshot {
        DevicePreconditionerDiagnosticsSnapshot::default()
    }

    fn try_apply_transpose(
        &self,
        _backend: &B,
        _r: &B::Buffer<T>,
        _z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        Err(crate::BackendError::Unsupported {
            backend: "device",
            operation: "preconditioner_transpose",
            scalar: "device scalar",
        })
    }

    fn try_apply_hermitian(
        &self,
        _backend: &B,
        _r: &B::Buffer<T>,
        _z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        Err(crate::BackendError::Unsupported {
            backend: "device",
            operation: "preconditioner_hermitian",
            scalar: "device scalar",
        })
    }

    fn try_apply_transpose_colmajor(
        &self,
        _backend: &B,
        _rhs: usize,
        _r: &B::Buffer<T>,
        _z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        Err(crate::BackendError::Unsupported {
            backend: "device",
            operation: "preconditioner_transpose_colmajor",
            scalar: "device scalar",
        })
    }

    fn try_apply_hermitian_colmajor(
        &self,
        _backend: &B,
        _rhs: usize,
        _r: &B::Buffer<T>,
        _z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        Err(crate::BackendError::Unsupported {
            backend: "device",
            operation: "preconditioner_hermitian_colmajor",
            scalar: "device scalar",
        })
    }

    /// Hidden compatibility hook retained for implementations that still
    /// carry an unchecked device path internally. New callers must use
    /// [`Self::try_apply`].
    #[doc(hidden)]
    fn apply_legacy_unchecked(&self, backend: &B, r: &B::Buffer<T>, z: &mut B::Buffer<T>) {
        self.apply(backend, r, z);
    }

    /// Fallible single-right-hand-side application. Implementations with
    /// device workspaces or native dispatch should override this method so
    /// allocation, shape, and backend failures reach checked solvers.
    fn try_apply(
        &self,
        backend: &B,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        if r.len() != z.len() {
            return Err(crate::BackendError::InvalidArgument {
                operation: "preconditioner_apply",
                reason: format!("input/output lengths {}/{} differ", r.len(), z.len()),
            });
        }
        crate::backend::catch_backend_panic("preconditioner_apply", || self.apply(backend, r, z))
    }

    /// Internal hook for implementations that need to wrap an action with
    /// normalized diagnostics without recursively calling their public checked
    /// entry point.
    #[doc(hidden)]
    fn try_apply_untracked(
        &self,
        backend: &B,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        self.try_apply(backend, r, z)
    }

    /// Apply this preconditioner to multiple dense right-hand sides:
    /// `Z <- M^-1 * R`, where `R` and `Z` are `n × rhs` column-major buffers.
    fn apply_colmajor(&self, backend: &B, rhs: usize, r: &B::Buffer<T>, z: &mut B::Buffer<T>) {
        if let Err(error) = self.try_apply_colmajor(backend, rhs, r, z) {
            log::error!(
                target: "sonium_compute::operator",
                "device preconditioner multi-RHS application failed: {error}"
            );
        }
    }

    /// Hidden compatibility hook for a legacy multi-RHS implementation.
    #[doc(hidden)]
    fn apply_colmajor_legacy_unchecked(
        &self,
        backend: &B,
        rhs: usize,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) {
        self.apply_colmajor(backend, rhs, r, z);
    }

    /// Fallible multi-RHS application with shape and workspace failures
    /// propagated to the caller.
    fn try_apply_colmajor(
        &self,
        backend: &B,
        rhs: usize,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        if r.len() != z.len() {
            return Err(crate::BackendError::InvalidArgument {
                operation: "preconditioner_apply_colmajor",
                reason: format!("input/output lengths {}/{} differ", r.len(), z.len()),
            });
        }
        if rhs == 0 || r.len() == 0 {
            return Ok(());
        }
        if !r.len().is_multiple_of(rhs) {
            return Err(crate::BackendError::InvalidArgument {
                operation: "preconditioner_apply_colmajor",
                reason: format!("input length {} is not divisible by rhs {rhs}", r.len()),
            });
        }
        let n = r.len() / rhs;
        if rhs == 1 {
            return self.try_apply(backend, r, z);
        }

        let mut r_host = crate::backend::try_zeroed_host(r.len(), "preconditioner_apply_colmajor")?;
        let mut z_host = crate::backend::try_zeroed_host(z.len(), "preconditioner_apply_colmajor")?;
        backend.try_copy_d2h(r, &mut r_host)?;

        let mut r_col = backend
            .raw_alloc::<T>(n)
            .map_err(crate::BackendError::from)?;
        let mut z_col = backend
            .raw_alloc::<T>(n)
            .map_err(crate::BackendError::from)?;

        for col in 0..rhs {
            let range = col * n..(col + 1) * n;
            backend.try_copy_h2d(&r_host[range.clone()], &mut r_col)?;
            self.try_apply(backend, &r_col, &mut z_col)?;
            backend.try_copy_d2h(&z_col, &mut z_host[range])?;
        }

        backend.try_copy_h2d(&z_host, z)
    }

    /// Internal counterpart to [`Self::try_apply_untracked`].
    #[doc(hidden)]
    fn try_apply_colmajor_untracked(
        &self,
        backend: &B,
        rhs: usize,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        self.try_apply_colmajor(backend, rhs, r, z)
    }
}

/// Identity preconditioner: `z <- r`.
pub struct IdentityDevicePreconditioner;

impl<B: Backend, T: DeviceScalar> DevicePreconditioner<B, T> for IdentityDevicePreconditioner {
    fn transpose_capability(&self) -> DeviceAdjointCapability {
        DeviceAdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> DeviceAdjointCapability {
        DeviceAdjointCapability::Exact
    }

    fn apply(&self, backend: &B, r: &B::Buffer<T>, z: &mut B::Buffer<T>) {
        backend.copy_d2d(r, z);
    }

    fn try_apply(
        &self,
        backend: &B,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        backend.try_copy_d2d(r, z)
    }

    fn try_apply_transpose(
        &self,
        backend: &B,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        backend.try_copy_d2d(r, z)
    }

    fn try_apply_hermitian(
        &self,
        backend: &B,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        backend.try_copy_d2d(r, z)
    }

    fn apply_colmajor(&self, backend: &B, _rhs: usize, r: &B::Buffer<T>, z: &mut B::Buffer<T>) {
        backend.copy_d2d(r, z);
    }

    fn try_apply_colmajor(
        &self,
        backend: &B,
        _rhs: usize,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        backend.try_copy_d2d(r, z)
    }

    fn try_apply_transpose_colmajor(
        &self,
        backend: &B,
        _rhs: usize,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        backend.try_copy_d2d(r, z)
    }

    fn try_apply_hermitian_colmajor(
        &self,
        backend: &B,
        _rhs: usize,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> crate::BackendResult<()> {
        backend.try_copy_d2d(r, z)
    }
}

#[cfg(test)]
// Fixtures intentionally use the panicking convenience constructors on
// tiny fixed matrices; production paths must use the `try_*` variants.
#[allow(deprecated)]
mod tests {
    use super::*;
    use crate::{CpuBackend, MemoryPoolConfig};
    use num_complex::Complex64;

    struct PanickingOperator;

    impl DeviceOperator<CpuBackend, f64> for PanickingOperator {
        fn n(&self) -> usize {
            2
        }

        fn apply(
            &self,
            _backend: &CpuBackend,
            _x: &<CpuBackend as Backend>::Buffer<f64>,
            _y: &mut <CpuBackend as Backend>::Buffer<f64>,
        ) {
            panic!("simulated legacy operator failure");
        }
    }

    #[test]
    fn device_csr_update_values_replaces_values_without_reupload() {
        let backend = CpuBackend::new();
        let mut csr =
            DeviceCsr::try_from_host(&backend, 2, 2, &[0, 1, 2], &[0, 1], &[2.0_f64, 3.0])
                .expect("initial upload");
        csr.update_values(&backend, &[4.0_f64, 5.0])
            .expect("values-only update");

        let mut x = backend.raw_alloc::<f64>(2).expect("x alloc");
        backend.copy_h2d(&[1.0_f64, 1.0], &mut x);
        let mut y = backend.raw_alloc::<f64>(2).expect("y alloc");
        let prebaked = csr.prebake(&backend);
        backend.spmv_prebaked::<f64>(&prebaked, &x, &mut y);
        let mut host = vec![0.0; 2];
        backend.copy_d2h(&y, &mut host);
        assert_eq!(host, vec![4.0, 5.0]);

        assert!(csr.update_values(&backend, &[1.0]).is_err());
    }

    #[test]
    fn checked_operator_application_contains_legacy_panics() {
        let backend = CpuBackend::new();
        let x = backend.raw_alloc::<f64>(2).expect("input allocation");
        let mut y = backend.raw_alloc::<f64>(2).expect("output allocation");

        let error = PanickingOperator
            .try_apply(&backend, &x, &mut y)
            .expect_err("checked operator boundary must contain legacy panic");

        assert!(matches!(
            error,
            crate::BackendError::OperationFailed {
                operation: "operator_apply",
                ..
            }
        ));
    }

    #[test]
    fn checked_operator_boundaries_propagate_pending_backend_errors() {
        struct CopyOperator;

        impl DeviceOperator<CpuBackend, f64> for CopyOperator {
            fn n(&self) -> usize {
                2
            }

            fn apply(
                &self,
                backend: &CpuBackend,
                x: &<CpuBackend as Backend>::Buffer<f64>,
                y: &mut <CpuBackend as Backend>::Buffer<f64>,
            ) {
                backend.copy_d2d(x, y);
            }
        }

        let backend = CpuBackend::new();
        let operator = CopyOperator;
        let x = backend.raw_alloc::<f64>(2).expect("input allocation");
        let b = backend.raw_alloc::<f64>(2).expect("rhs allocation");
        let mut y = backend.raw_alloc::<f64>(2).expect("output allocation");

        CpuBackend::inject_pending_error();
        let apply_error = operator
            .try_apply(&backend, &x, &mut y)
            .expect_err("checked apply must surface a pending backend error");
        assert!(matches!(
            apply_error,
            crate::BackendError::OperationFailed {
                operation: "cpu_pending_error",
                ..
            }
        ));

        CpuBackend::inject_pending_error();
        let residual_error = operator
            .try_residual(&backend, &x, &b, &mut y)
            .expect_err("checked residual must surface a pending backend error");
        assert!(matches!(
            residual_error,
            crate::BackendError::OperationFailed {
                operation: "cpu_pending_error",
                ..
            }
        ));
    }

    #[test]
    fn paired_device_csr_satisfies_complex_hermitian_identity() {
        let backend = CpuBackend::new();
        let row_ptrs = [0, 2, 4, 6];
        let col_indices = [0, 1, 1, 2, 0, 2];
        let values = [
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, -1.0),
            Complex64::new(0.0, 4.0),
            Complex64::new(5.0, 0.0),
            Complex64::new(-2.0, 0.5),
        ];
        let host = HostCsrRef::try_new(3, 3, &row_ptrs, &col_indices, &values).unwrap();
        let operator = AdjointDeviceCsr::try_from_host_csr_ref(&backend, host).unwrap();
        let prebaked = operator.prebake(&backend);

        let x_host = [
            Complex64::new(0.5, -1.0),
            Complex64::new(2.0, 0.25),
            Complex64::new(-0.75, 0.5),
        ];
        let y_host = [
            Complex64::new(-1.0, 0.5),
            Complex64::new(0.25, 1.5),
            Complex64::new(2.0, -0.25),
        ];
        let mut x = backend.raw_alloc::<Complex64>(3).unwrap();
        let mut y = backend.raw_alloc::<Complex64>(3).unwrap();
        backend.copy_h2d(&x_host, &mut x);
        backend.copy_h2d(&y_host, &mut y);
        let mut ax = backend.raw_alloc::<Complex64>(3).unwrap();
        let mut ah_y = backend.raw_alloc::<Complex64>(3).unwrap();
        prebaked.try_apply(&backend, &x, &mut ax).unwrap();
        prebaked
            .try_apply_hermitian(&backend, &y, &mut ah_y)
            .unwrap();

        let mut ax_host = [Complex64::new(0.0, 0.0); 3];
        let mut ah_y_host = [Complex64::new(0.0, 0.0); 3];
        backend.copy_d2h(&ax, &mut ax_host);
        backend.copy_d2h(&ah_y, &mut ah_y_host);
        let lhs = ax_host
            .iter()
            .zip(&y_host)
            .map(|(&left, &right)| left.conj() * right)
            .sum::<Complex64>();
        let rhs = x_host
            .iter()
            .zip(&ah_y_host)
            .map(|(&left, &right)| left.conj() * right)
            .sum::<Complex64>();
        assert!((lhs - rhs).norm() < 1.0e-12, "lhs={lhs:?}, rhs={rhs:?}");

        let mut short = backend.raw_alloc::<Complex64>(2).unwrap();
        assert!(matches!(
            prebaked.try_apply_hermitian(&backend, &y, &mut short),
            Err(crate::BackendError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn paired_device_csr_block_adjoint_matches_column_applications() {
        let backend = CpuBackend::new();
        let row_ptrs = [0, 2, 4];
        let col_indices = [0, 1, 0, 1];
        let values = [
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(-0.5, 0.25),
            Complex64::new(3.0, 2.0),
        ];
        let host = HostCsrRef::try_new(2, 2, &row_ptrs, &col_indices, &values).unwrap();
        let operator = AdjointDeviceCsr::try_from_host_csr_ref(&backend, host).unwrap();
        let prebaked = operator.try_prebake(&backend).unwrap();
        let columns = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(-2.0, 0.5),
            Complex64::new(1.5, -0.25),
        ];
        let mut x = backend.raw_alloc::<Complex64>(4).unwrap();
        backend.copy_h2d(&columns, &mut x);
        let mut block = backend.raw_alloc::<Complex64>(4).unwrap();
        prebaked
            .try_apply_hermitian_colmajor(&backend, 2, &x, &mut block)
            .unwrap();

        let mut expected = [Complex64::new(0.0, 0.0); 4];
        for column in 0..2 {
            let range = column * 2..column * 2 + 2;
            let mut input = backend.raw_alloc::<Complex64>(2).unwrap();
            backend.copy_h2d(&columns[range.clone()], &mut input);
            let mut output = backend.raw_alloc::<Complex64>(2).unwrap();
            prebaked
                .try_apply_hermitian(&backend, &input, &mut output)
                .unwrap();
            backend.copy_d2h(&output, &mut expected[range]);
        }
        let mut got = [Complex64::new(0.0, 0.0); 4];
        backend.copy_d2h(&block, &mut got);
        assert_eq!(got, expected);
    }

    #[test]
    fn paired_device_csr_rejects_rectangular_and_malformed_inputs() {
        let backend = CpuBackend::new();
        let rectangular = HostCsrRef::try_new(1, 2, &[0, 1], &[1], &[1.0_f64]).unwrap();
        assert!(matches!(
            AdjointDeviceCsr::try_from_host_csr_ref(&backend, rectangular),
            Err(crate::BackendError::InvalidArgument { .. })
        ));

        let malformed = HostCsrRef {
            num_rows: 2,
            num_cols: 2,
            row_ptrs: &[0, 2],
            col_indices: &[0, 1],
            values: &[1.0_f64, 2.0],
        };
        assert!(matches!(
            AdjointDeviceCsr::try_from_host_csr_ref(&backend, malformed),
            Err(crate::BackendError::InvalidArgument { .. })
        ));
    }

    #[test]
    fn device_csr_cpu_apply_matches_expected() {
        let backend = CpuBackend::new();
        let row_ptr = [0, 2, 4];
        let col_idx = [0, 1, 0, 1];
        let values = [2.0_f64, 1.0, -1.0, 3.0];
        let a = DeviceCsr::from_host(&backend, 2, 2, &row_ptr, &col_idx, &values);

        let mut x = backend.raw_alloc::<f64>(2).unwrap();
        backend.copy_h2d(&[4.0, 5.0], &mut x);
        let mut y = backend.raw_alloc::<f64>(2).unwrap();

        a.apply(&backend, &x, &mut y);

        let mut out = [0.0; 2];
        backend.copy_d2h(&y, &mut out);
        assert_eq!(out, [13.0, 11.0]);
    }

    #[test]
    fn prebaked_spmv_apply_colmajor_uses_backend_spmm() {
        let backend = CpuBackend::new();
        let row_ptr = [0, 2, 4];
        let col_idx = [0, 1, 0, 1];
        let values = [2.0_f64, 1.0, -1.0, 3.0];
        let a = DeviceCsr::from_host(&backend, 2, 2, &row_ptr, &col_idx, &values);
        let op = PrebakedSpmv::try_from_csr(&backend, &a).unwrap();

        let mut x = backend.raw_alloc::<f64>(4).unwrap();
        backend.copy_h2d(&[4.0, 5.0, -1.0, 2.0], &mut x);
        let mut y = backend.raw_alloc::<f64>(4).unwrap();

        op.apply_colmajor(&backend, 2, &x, &mut y);

        let mut out = [0.0; 4];
        backend.copy_d2h(&y, &mut out);
        assert_eq!(out, [13.0, 11.0, 0.0, 7.0]);
    }

    #[test]
    fn device_csr_checked_apply_colmajor_matches_spmm() {
        let backend = CpuBackend::new();
        let a = DeviceCsr::from_host(
            &backend,
            2,
            2,
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[2.0_f64, 1.0, -1.0, 3.0],
        );
        let mut x = backend.raw_alloc::<f64>(4).unwrap();
        backend.copy_h2d(&[4.0, 5.0, -1.0, 2.0], &mut x);
        let mut y = backend.raw_alloc::<f64>(4).unwrap();

        a.try_apply_colmajor(&backend, 2, &x, &mut y).unwrap();

        let mut out = [0.0; 4];
        backend.copy_d2h(&y, &mut out);
        assert_eq!(out, [13.0, 11.0, 0.0, 7.0]);
    }

    #[test]
    fn pooled_device_csr_checked_apply_colmajor_matches_spmm() {
        let pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(4096)).unwrap();
        let backend = pool.backend();
        let a = DeviceCsr::from_pool(
            &pool,
            2,
            2,
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[2.0_f64, 1.0, -1.0, 3.0],
        )
        .unwrap();
        let op = PrebakedSpmv::try_from_pooled(backend, &a).unwrap();
        let mut x = backend.raw_alloc::<f64>(4).unwrap();
        backend.copy_h2d(&[4.0, 5.0, -1.0, 2.0], &mut x);
        let mut y = backend.raw_alloc::<f64>(4).unwrap();

        op.try_apply_colmajor(backend, 2, &x, &mut y).unwrap();

        let mut out = [0.0; 4];
        backend.copy_d2h(&y, &mut out);
        assert_eq!(out, [13.0, 11.0, 0.0, 7.0]);
    }

    #[test]
    fn csr_colmajor_rejects_rectangular_shape_before_dispatch() {
        let backend = CpuBackend::new();
        let a = DeviceCsr::from_host(&backend, 1, 2, &[0, 2], &[0, 1], &[2.0_f64, 3.0]);
        let op = PrebakedSpmv::try_from_csr(&backend, &a).unwrap();

        let mut valid_x = backend.raw_alloc::<f64>(4).unwrap();
        let mut valid_y = backend.raw_alloc::<f64>(2).unwrap();
        backend.copy_h2d(&[1.0, 2.0, 3.0, 4.0], &mut valid_x);
        a.try_apply_colmajor(&backend, 2, &valid_x, &mut valid_y)
            .expect("rectangular DeviceCsr SpMM");
        let mut valid_out = [0.0; 2];
        backend.copy_d2h(&valid_y, &mut valid_out);
        assert_eq!(valid_out, [8.0, 18.0]);
        backend.copy_h2d(&[0.0, 0.0], &mut valid_y);
        op.try_apply_colmajor(&backend, 2, &valid_x, &mut valid_y)
            .expect("rectangular PrebakedSpmv SpMM");
        backend.copy_d2h(&valid_y, &mut valid_out);
        assert_eq!(valid_out, [8.0, 18.0]);

        let x = backend.raw_alloc::<f64>(2).unwrap();
        let mut y = backend.raw_alloc::<f64>(1).unwrap();

        let error = a
            .try_apply_colmajor(&backend, 2, &x, &mut y)
            .expect_err("DeviceCsr must reject short column-major buffers");
        assert!(matches!(error, crate::BackendError::InvalidArgument { .. }));
        let error = op
            .try_apply_colmajor(&backend, 2, &x, &mut y)
            .expect_err("PrebakedSpmv must retain rectangular shape checks");
        assert!(matches!(error, crate::BackendError::InvalidArgument { .. }));
    }

    #[test]
    fn checked_device_csr_colmajor_contains_malformed_prebake() {
        let backend = CpuBackend::new();
        let mut row_ptr = backend.raw_alloc::<i32>(2).unwrap();
        let mut col_idx = backend.raw_alloc::<i32>(2).unwrap();
        let mut values = backend.raw_alloc::<f64>(2).unwrap();
        backend.copy_h2d(&[0, 2], &mut row_ptr);
        backend.copy_h2d(&[0, 1], &mut col_idx);
        backend.copy_h2d(&[1.0, 2.0], &mut values);
        let a = DeviceCsr {
            m: 2,
            n: 2,
            nnz: 2,
            row_ptr,
            col_idx,
            values,
        };
        let x = backend.raw_alloc::<f64>(4).unwrap();
        let mut y = backend.raw_alloc::<f64>(4).unwrap();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            a.try_apply_colmajor(&backend, 2, &x, &mut y)
        }));
        assert!(result.is_ok(), "checked column-major CSR path panicked");
        assert!(result.unwrap().is_err());
    }

    #[test]
    fn device_operator_residual_computes_b_minus_ax() {
        let backend = CpuBackend::new();
        let row_ptr = [0, 2, 4];
        let col_idx = [0, 1, 0, 1];
        let values = [2.0_f64, 1.0, -1.0, 3.0];
        let a = DeviceCsr::from_host(&backend, 2, 2, &row_ptr, &col_idx, &values);

        let mut x = backend.raw_alloc::<f64>(2).unwrap();
        backend.copy_h2d(&[4.0, 5.0], &mut x);
        let mut b = backend.raw_alloc::<f64>(2).unwrap();
        backend.copy_h2d(&[20.0, 7.0], &mut b);
        let mut r = backend.raw_alloc::<f64>(2).unwrap();

        a.residual(&backend, &x, &b, &mut r);

        let mut out = [0.0; 2];
        backend.copy_d2h(&r, &mut out);
        assert_eq!(out, [7.0, -4.0]);
    }

    #[test]
    fn prebaked_spmv_residual_computes_b_minus_ax() {
        let backend = CpuBackend::new();
        let row_ptr = [0, 2, 4];
        let col_idx = [0, 1, 0, 1];
        let values = [2.0_f64, 1.0, -1.0, 3.0];
        let a = DeviceCsr::from_host(&backend, 2, 2, &row_ptr, &col_idx, &values);
        let op = PrebakedSpmv::from_csr(&backend, &a);

        let mut x = backend.raw_alloc::<f64>(2).unwrap();
        backend.copy_h2d(&[4.0, 5.0], &mut x);
        let mut b = backend.raw_alloc::<f64>(2).unwrap();
        backend.copy_h2d(&[20.0, 7.0], &mut b);
        let mut r = backend.raw_alloc::<f64>(2).unwrap();

        op.residual(&backend, &x, &b, &mut r);

        let mut out = [0.0; 2];
        backend.copy_d2h(&r, &mut out);
        assert_eq!(out, [7.0, -4.0]);

        op.try_residual(&backend, &x, &b, &mut r).unwrap();
        backend.copy_d2h(&r, &mut out);
        assert_eq!(out, [7.0, -4.0]);
    }

    #[test]
    fn device_operator_apply_colmajor_fallback_matches_column_apply() {
        struct ScaleByTwo;

        impl DeviceOperator<CpuBackend, f64> for ScaleByTwo {
            fn n(&self) -> usize {
                2
            }

            fn apply(
                &self,
                backend: &CpuBackend,
                x: &<CpuBackend as Backend>::Buffer<f64>,
                y: &mut <CpuBackend as Backend>::Buffer<f64>,
            ) {
                backend.copy_d2d(x, y);
                backend.scale::<f64>(2.0, y);
            }
        }

        let backend = CpuBackend::new();
        let mut x = backend.raw_alloc::<f64>(4).unwrap();
        backend.copy_h2d(&[4.0, 5.0, -1.0, 2.0], &mut x);
        let mut y = backend.raw_alloc::<f64>(4).unwrap();

        ScaleByTwo.apply_colmajor(&backend, 2, &x, &mut y);

        let mut out = [0.0; 4];
        backend.copy_d2h(&y, &mut out);
        assert_eq!(out, [8.0, 10.0, -2.0, 4.0]);
    }

    #[test]
    fn device_csr_apply_colmajor_is_available_through_operator_trait() {
        let backend = CpuBackend::new();
        let row_ptr = [0, 2, 4];
        let col_idx = [0, 1, 0, 1];
        let values = [2.0_f64, 1.0, -1.0, 3.0];
        let a = DeviceCsr::from_host(&backend, 2, 2, &row_ptr, &col_idx, &values);
        let op: &dyn DeviceOperator<CpuBackend, f64> = &a;

        let mut x = backend.raw_alloc::<f64>(4).unwrap();
        backend.copy_h2d(&[4.0, 5.0, -1.0, 2.0], &mut x);
        let mut y = backend.raw_alloc::<f64>(4).unwrap();

        op.apply_colmajor(&backend, 2, &x, &mut y);

        let mut out = [0.0; 4];
        backend.copy_d2h(&y, &mut out);
        assert_eq!(out, [13.0, 11.0, 0.0, 7.0]);
    }

    #[test]
    fn device_csr_uploads_host_csr_ref() {
        let backend = CpuBackend::new();
        let row_ptrs = [0, 2, 4];
        let col_indices = [0, 1, 0, 1];
        let values = [2.0_f64, 1.0, -1.0, 3.0];
        let matrix = HostCsrRef::new(2, 2, &row_ptrs, &col_indices, &values);
        let a = DeviceCsr::from_host_csr_ref(&backend, matrix);

        let mut x = backend.raw_alloc::<f64>(2).unwrap();
        backend.copy_h2d(&[4.0, 5.0], &mut x);
        let mut y = backend.raw_alloc::<f64>(2).unwrap();

        a.apply(&backend, &x, &mut y);

        let mut out = [0.0; 2];
        backend.copy_d2h(&y, &mut out);
        assert_eq!(out, [13.0, 11.0]);
    }

    #[test]
    fn device_csr_uploads_host_diagonal() {
        let backend = CpuBackend::new();
        let a = DeviceCsr::from_host_diagonal(&backend, &[2.0_f64, 3.0, 4.0]);

        let mut x = backend.raw_alloc::<f64>(3).unwrap();
        backend.copy_h2d(&[5.0, 6.0, 7.0], &mut x);
        let mut y = backend.raw_alloc::<f64>(3).unwrap();

        a.apply(&backend, &x, &mut y);

        let mut out = [0.0; 3];
        backend.copy_d2h(&y, &mut out);
        assert_eq!(out, [10.0, 18.0, 28.0]);
    }

    #[test]
    fn device_csr_uploads_inverse_host_diagonal() {
        let backend = CpuBackend::new();
        let a = DeviceCsr::from_host_inverse_diagonal(&backend, &[2.0_f64, 0.0, 4.0]);

        let mut x = backend.raw_alloc::<f64>(3).unwrap();
        backend.copy_h2d(&[6.0, 6.0, 8.0], &mut x);
        let mut y = backend.raw_alloc::<f64>(3).unwrap();

        a.apply(&backend, &x, &mut y);

        let mut out = [0.0; 3];
        backend.copy_d2h(&y, &mut out);
        assert_eq!(out, [3.0, 0.0, 2.0]);
    }

    #[test]
    fn identity_preconditioner_copies_device_buffer() {
        let backend = CpuBackend::new();
        let mut src = backend.raw_alloc::<f32>(3).unwrap();
        backend.copy_h2d(&[1.0, 2.0, 3.0], &mut src);
        let mut dst = backend.raw_alloc::<f32>(3).unwrap();

        IdentityDevicePreconditioner.apply(&backend, &src, &mut dst);

        let mut out = [0.0; 3];
        backend.copy_d2h(&dst, &mut out);
        assert_eq!(out, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn identity_preconditioner_apply_colmajor_copies_all_columns() {
        let backend = CpuBackend::new();
        let mut src = backend.raw_alloc::<f32>(6).unwrap();
        backend.copy_h2d(&[1.0, 2.0, 3.0, -1.0, -2.0, -3.0], &mut src);
        let mut dst = backend.raw_alloc::<f32>(6).unwrap();

        IdentityDevicePreconditioner.apply_colmajor(&backend, 2, &src, &mut dst);

        let mut out = [0.0; 6];
        backend.copy_d2h(&dst, &mut out);
        assert_eq!(out, [1.0, 2.0, 3.0, -1.0, -2.0, -3.0]);
    }

    #[test]
    fn device_csr_structure_can_reuse_values_buffer() {
        let backend = CpuBackend::new();
        let row_ptr = [0, 2, 4];
        let col_idx = [0, 1, 0, 1];
        let mut a =
            DeviceCsr::<CpuBackend, f64>::from_host_structure(&backend, 2, 2, &row_ptr, &col_idx);

        let mut x = backend.raw_alloc::<f64>(2).unwrap();
        backend.copy_h2d(&[4.0, 5.0], &mut x);
        let mut y = backend.raw_alloc::<f64>(2).unwrap();

        a.copy_values_from_host(&backend, &[2.0, 1.0, -1.0, 3.0]);
        a.apply(&backend, &x, &mut y);
        let mut out = [0.0; 2];
        backend.copy_d2h(&y, &mut out);
        assert_eq!(out, [13.0, 11.0]);

        a.copy_values_from_host(&backend, &[1.0, 0.0, 0.0, 1.0]);
        a.apply(&backend, &x, &mut y);
        backend.copy_d2h(&y, &mut out);
        assert_eq!(out, [4.0, 5.0]);
    }

    #[test]
    fn pooled_device_csr_counts_storage_against_quota() {
        let pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(60)).unwrap();
        let row_ptr = [0, 2, 4];
        let col_idx = [0, 1, 0, 1];
        let values = [2.0_f64, 1.0, -1.0, 3.0];

        let a = DeviceCsr::from_pool(&pool, 2, 2, &row_ptr, &col_idx, &values).unwrap();

        assert_eq!(a.nnz, 4);
        assert_eq!(pool.bytes_in_use(), 60);
    }

    #[test]
    fn pooled_device_csr_rejects_over_quota_before_upload() {
        let pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(59)).unwrap();
        let row_ptr = [0, 2, 4];
        let col_idx = [0, 1, 0, 1];
        let values = [2.0_f64, 1.0, -1.0, 3.0];

        let err = match DeviceCsr::from_pool(&pool, 2, 2, &row_ptr, &col_idx, &values) {
            Ok(_) => panic!("pooled CSR allocation should exceed quota"),
            Err(err) => err,
        };

        assert!(matches!(err, crate::OomError::QuotaExceeded { .. }));
        assert_eq!(pool.bytes_in_use(), 0);
    }

    #[test]
    fn checked_pooled_csr_helpers_propagate_validation_and_quota_errors() {
        let row_ptr = [0, 2, 4];
        let col_idx = [0, 1, 0, 1];
        let values = [2.0_f64, 1.0, -1.0, 3.0];

        let pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(128)).unwrap();
        let malformed = DeviceCsr::try_from_pool(&pool, 2, 2, &[0, 2, 1], &col_idx, &values);
        assert!(matches!(
            malformed,
            Err(crate::BackendError::InvalidArgument { .. })
        ));
        assert_eq!(pool.bytes_in_use(), 0);

        let malformed_structure =
            DeviceCsr::<CpuBackend, f64>::try_from_pool_structure(&pool, 1, 1, &[0, 1], &[1]);
        assert!(matches!(
            malformed_structure,
            Err(crate::BackendError::InvalidArgument { .. })
        ));
        assert_eq!(pool.bytes_in_use(), 0);

        let malformed_cast = DeviceCsr::<CpuBackend, f32>::try_from_pool_cast::<f64>(
            &pool,
            2,
            2,
            &row_ptr,
            &col_idx,
            &[1.0, 2.0],
        );
        assert!(matches!(
            malformed_cast,
            Err(crate::BackendError::InvalidArgument { .. })
        ));
        assert_eq!(pool.bytes_in_use(), 0);

        let small_pool =
            MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(59)).unwrap();
        let quota_error =
            match DeviceCsr::try_from_pool(&small_pool, 2, 2, &row_ptr, &col_idx, &values) {
                Ok(_) => panic!("checked pooled allocation should preserve quota failure"),
                Err(error) => error,
            };
        assert!(matches!(
            quota_error,
            crate::BackendError::Allocation { .. }
        ));
        assert_eq!(small_pool.bytes_in_use(), 0);

        let pooled = DeviceCsr::try_from_pool(&pool, 2, 2, &row_ptr, &col_idx, &values)
            .expect("valid checked pooled upload");
        assert_eq!(pooled.nnz, 4);
        assert_eq!(pool.bytes_in_use(), 60);
        drop(pooled);
        assert_eq!(pool.bytes_in_use(), 0);
    }

    #[test]
    fn device_csr_checked_upload_rejects_malformed_structure_before_allocation() {
        let backend = CpuBackend::new();

        let cases = [
            (&[0_i32][..], &[0_i32][..], &[1.0_f64][..]),
            (&[0_i32, 2, 1][..], &[0_i32, 1][..], &[1.0_f64, 2.0][..]),
            (&[0_i32, 1, 3][..], &[0_i32, 1][..], &[1.0_f64, 2.0][..]),
            (&[0_i32, 1, 2][..], &[-1_i32, 1][..], &[1.0_f64, 2.0][..]),
            (&[0_i32, 1, 2][..], &[0_i32, 2][..], &[1.0_f64, 2.0][..]),
            (&[0_i32, 1, 2][..], &[0_i32][..], &[1.0_f64][..]),
        ];

        for (row_ptr, col_idx, values) in cases {
            let error = match DeviceCsr::<CpuBackend, f64>::try_from_host(
                &backend, 2, 2, row_ptr, col_idx, values,
            ) {
                Ok(_) => panic!("malformed CSR must be rejected"),
                Err(error) => error,
            };
            assert!(matches!(error, crate::BackendError::InvalidArgument { .. }));
        }
    }

    #[test]
    fn device_csr_checked_structure_upload_rejects_out_of_range_columns() {
        let backend = CpuBackend::new();
        let error = match DeviceCsr::<CpuBackend, f64>::try_from_host_structure(
            &backend,
            1,
            1,
            &[0, 1],
            &[1],
        ) {
            Ok(_) => panic!("out-of-range CSR column must be rejected"),
            Err(error) => error,
        };

        assert!(matches!(error, crate::BackendError::InvalidArgument { .. }));
    }

    #[test]
    fn checked_csr_helpers_reject_invalid_indices_values_and_updates() {
        let backend = CpuBackend::new();
        let matrix = HostCsrRef::new(1, 1, &[0, 1], &[0], &[f64::NAN]);
        let error = match DeviceCsr::<CpuBackend, f64>::try_from_host_csr_ref(&backend, matrix) {
            Ok(_) => panic!("non-finite host-reference values must be rejected"),
            Err(error) => error,
        };
        assert!(matches!(error, crate::BackendError::InvalidArgument { .. }));

        let error = match DeviceCsr::<CpuBackend, f32>::try_from_host_cast::<f64>(
            &backend,
            1,
            1,
            &[0, 1],
            &[0],
            &[],
        ) {
            Ok(_) => panic!("cast value/structure mismatch must be rejected"),
            Err(error) => error,
        };
        assert!(matches!(error, crate::BackendError::InvalidArgument { .. }));

        let mut a =
            DeviceCsr::<CpuBackend, f64>::from_host_structure(&backend, 1, 1, &[0, 1], &[0]);
        let error = a
            .try_copy_values_from_host(&backend, &[])
            .expect_err("wrong value-update length must be rejected");
        assert!(matches!(error, crate::BackendError::InvalidArgument { .. }));
        let error = a
            .try_copy_values_from_host(&backend, &[f64::INFINITY])
            .expect_err("non-finite value updates must be rejected");
        assert!(matches!(error, crate::BackendError::InvalidArgument { .. }));
    }

    #[test]
    fn pooled_csr_checked_upload_rejects_invalid_structure_without_quota_charge() {
        let pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(128)).unwrap();
        let error = match DeviceCsr::from_pool(&pool, 1, 1, &[0, 2], &[0], &[1.0_f64]) {
            Ok(_) => panic!("pooled CSR must reject a terminal pointer mismatch"),
            Err(error) => error,
        };

        assert!(matches!(error, crate::error::OomError::Invalid(_)));
        assert_eq!(pool.bytes_in_use(), 0);
    }

    #[test]
    fn from_host_cast_does_not_collect_full_values_vec() {
        let source = include_str!("operator.rs");
        let body = source
            .split("pub fn from_host_cast<UpperT>")
            .nth(1)
            .expect("from_host_cast should exist")
            .split("pub fn n(&self)")
            .next()
            .expect("from_host_cast body should be followed by n()");
        assert!(
            !body.contains(".collect()"),
            "from_host_cast should copy-cast directly into the backend buffer"
        );
    }

    #[test]
    fn pooled_csr_cast_rejects_invalid_input_without_panicking() {
        let pool = MemoryPool::new(CpuBackend::new(), MemoryPoolConfig::with_cap(128)).unwrap();

        let error = match DeviceCsr::<CpuBackend, f32>::from_pool_cast::<f64>(
            &pool,
            2,
            2,
            &[0, 1],
            &[0],
            &[1.0],
        ) {
            Ok(_) => panic!("malformed row pointer must return a typed error"),
            Err(error) => error,
        };
        assert!(matches!(error, crate::error::OomError::Invalid(_)));
        assert_eq!(pool.bytes_in_use(), 0);

        let error = match DeviceCsr::<CpuBackend, f32>::from_pool_cast::<f64>(
            &pool,
            1,
            1,
            &[0, 1],
            &[0],
            &[f64::NAN],
        ) {
            Ok(_) => panic!("non-finite cast values must return a typed error"),
            Err(error) => error,
        };
        assert!(matches!(error, crate::error::OomError::Invalid(_)));
        assert_eq!(pool.bytes_in_use(), 0);
    }

    #[test]
    fn persistent_csr_accepts_resident_value_updates() {
        let backend = CpuBackend::new();
        let matrix =
            DeviceCsr::try_from_host(&backend, 2, 2, &[0, 1, 2], &[0, 1], &[2.0_f64, 3.0]).unwrap();
        let mut persistent = PersistentDeviceCsr::try_from_device_csr(&backend, matrix).unwrap();
        let pool = MemoryPool::new(backend, crate::MemoryPoolConfig::with_cap(4096)).unwrap();
        let mut values = pool.try_alloc::<f64>(2).unwrap();
        let mut x = pool.try_alloc::<f64>(2).unwrap();
        let mut y = pool.try_alloc::<f64>(2).unwrap();
        pool.backend()
            .try_copy_h2d(&[4., 5.], values.raw_mut())
            .unwrap();
        pool.backend().try_copy_h2d(&[1., 2.], x.raw_mut()).unwrap();
        persistent
            .try_copy_values_from_device(pool.backend(), values.raw())
            .unwrap();
        persistent
            .try_apply(pool.backend(), x.raw(), y.raw_mut())
            .unwrap();
        let mut actual = [0.; 2];
        pool.backend().try_copy_d2h(y.raw(), &mut actual).unwrap();
        assert_eq!(actual, [4., 10.]);
        let short = pool.try_alloc::<f64>(1).unwrap();
        assert!(
            persistent
                .try_copy_values_from_device(pool.backend(), short.raw())
                .is_err()
        );
        persistent
            .try_apply(pool.backend(), x.raw(), y.raw_mut())
            .unwrap();
        pool.backend().try_copy_d2h(y.raw(), &mut actual).unwrap();
        assert_eq!(actual, [4., 10.]);
    }

    #[test]
    fn persistent_csr_updates_values_without_rebuilding_operator_storage() {
        let backend = CpuBackend::new();
        let matrix =
            DeviceCsr::try_from_host(&backend, 2, 2, &[0, 1, 2], &[0, 1], &[2.0_f64, 3.0]).unwrap();
        let mut persistent = PersistentDeviceCsr::try_from_device_csr(&backend, matrix).unwrap();
        let row_ptr = persistent.matrix().row_ptr.len();
        persistent
            .try_copy_values_from_host(&backend, &[4.0, 5.0])
            .unwrap();

        let mut x = backend.raw_alloc::<f64>(2).unwrap();
        let mut y = backend.raw_alloc::<f64>(2).unwrap();
        backend.copy_h2d(&[1.0, 1.0], &mut x);
        persistent.try_apply(&backend, &x, &mut y).unwrap();
        let mut result = [0.0_f64; 2];
        backend.copy_d2h(&y, &mut result);

        assert_eq!(row_ptr, persistent.matrix().row_ptr.len());
        assert_eq!(result, [4.0, 5.0]);
    }

    #[test]
    fn persistent_csr_cast_update_matches_host_side_cast() {
        use num_complex::Complex32;
        let backend = CpuBackend::new();
        // 2x2 complex system, resident in f32.
        let init32 = vec![
            Complex32::new(4.0, 0.0),
            Complex32::new(0.0, 1.0),
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
        ];
        let matrix =
            DeviceCsr::try_from_host(&backend, 2, 2, &[0, 2, 4], &[0, 1, 0, 1], &init32).unwrap();
        let mut persistent = PersistentDeviceCsr::try_from_device_csr(&backend, matrix).unwrap();
        // New f64 values; the production path used to cast these on the
        // host into a temporary N^2 buffer before upload.
        let update64 = vec![
            Complex64::new(8.0, 0.5),
            Complex64::new(0.0, -2.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(1.0, 1.0),
        ];
        persistent
            .try_copy_values_from_host_cast(&backend, &update64)
            .unwrap();
        // Length and finiteness gates mirror the same-precision update.
        assert!(
            persistent
                .try_copy_values_from_host_cast(&backend, &update64[..3])
                .is_err()
        );
        assert!(
            persistent
                .try_copy_values_from_host_cast(
                    &backend,
                    &[
                        Complex64::new(1.0, 0.0),
                        Complex64::new(0.0, 1.0),
                        Complex64::new(0.0, 0.0),
                        Complex64::new(f64::NAN, 0.0),
                    ]
                )
                .is_err()
        );

        let mut x = backend.raw_alloc::<Complex32>(2).unwrap();
        let mut y = backend.raw_alloc::<Complex32>(2).unwrap();
        backend.copy_h2d(
            &[Complex32::new(1.0, 0.0), Complex32::new(0.0, 1.0)],
            &mut x,
        );
        persistent.try_apply(&backend, &x, &mut y).unwrap();
        let mut result = [Complex32::new(0.0, 0.0); 2];
        backend.copy_d2h(&y, &mut result);
        // (8+0.5i)*1 + (0-2i)*i = 8+0.5i+2 = 10+0.5i;
        // 3*1 + (1+i)*i = 3-1+i = 2+i.
        assert!((result[0] - Complex32::new(10.0, 0.5)).norm() < 1e-5);
        assert!((result[1] - Complex32::new(2.0, 1.0)).norm() < 1e-5);
    }
}

#[cfg(test)]
mod scaled_combination_tests {
    use super::*;
    use crate::{CpuBackend, MemoryPoolConfig};

    fn check<B: Backend, T: DeviceScalar + PartialEq + std::fmt::Debug>(backend: B) {
        let pool = MemoryPool::new(backend, MemoryPoolConfig::with_cap(1 << 20)).unwrap();
        let backend = pool.backend();
        let one = T::one();
        let two = one + one;
        let three = two + one;
        let mut matrix = PersistentDeviceCsr::try_from_host_csr_ref(
            backend,
            HostCsrRef::new(2, 2, &[0, 1, 2], &[0, 1], &[one, one]),
        )
        .unwrap();
        let mut base = pool.try_alloc(2).unwrap();
        let mut term = pool.try_alloc(2).unwrap();
        backend.try_copy_h2d(&[one, two], base.raw_mut()).unwrap();
        backend.try_copy_h2d(&[three, one], term.raw_mut()).unwrap();
        matrix
            .try_combine_scaled_values(backend, two, base.raw(), &[(three, term.raw())])
            .unwrap();
        let expected = [two * one + three * three, two * two + three * one];
        let mut actual = [T::zero(); 2];
        backend
            .try_copy_d2h(&matrix.matrix().values, &mut actual)
            .unwrap();
        for (a, b) in actual.into_iter().zip(expected) {
            assert_eq!(a, b);
        }
        let short = pool.try_alloc::<T>(1).unwrap();
        assert!(
            matrix
                .try_combine_scaled_values(backend, two, short.raw(), &[])
                .is_err()
        );
        backend
            .try_copy_d2h(&matrix.matrix().values, &mut actual)
            .unwrap();
        for (a, b) in actual.into_iter().zip(expected) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn scaled_sparse_combination_all_scalars() {
        check::<_, f32>(CpuBackend::new());
        check::<_, f64>(CpuBackend::new());
        check::<_, num_complex::Complex32>(CpuBackend::new());
        check::<_, num_complex::Complex64>(CpuBackend::new());
    }
}
