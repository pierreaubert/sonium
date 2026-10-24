//! Resident ILUT/ILUTP factors with native permutation and triangular actions.

// Rust guideline compliant 2026-02-21
use super::{IlutConfig, IlutPreconditioner};
use crate::cuda_ilu::{CudaIluPreconditioner, TriangularFactors};
use sonium_compute::{
    Backend, BackendError, BackendResult, CudaBackend, CudaBuffer, DeviceBufferLike,
    DevicePreconditioner, DeviceScalar, PersistentDeviceCsr,
};
use sonium_solvers_core::traits::ComplexField;
use sonium_solvers_sparse::CsrMatrix;
use std::sync::Mutex;

fn invalid(reason: impl Into<String>) -> BackendError {
    BackendError::InvalidArgument {
        operation: "cuda_ilut",
        reason: reason.into(),
    }
}

struct Workspace<T: DeviceScalar> {
    gathered: CudaBuffer<T>,
    solved: CudaBuffer<T>,
    result: CudaBuffer<T>,
}

/// Retains threshold ILU factors and both permutations for native CUDA application.
///
/// Host factorization preserves the CPU drop, fill and pivot policy. Importing
/// an existing host factor also preserves its symmetric fill-reducing ordering.
/// Native sparse maps gather the RHS and scatter the pivoted solution; the
/// shared dependency-level triangular solver applies the factors. All buffers
/// are allocated during setup. Long triangular dependency chains still require
/// many launches. This adapter implements the forward preconditioner action;
/// it does not claim transpose/Hermitian or block preconditioning support.
pub struct CudaIlutPreconditioner<T: DeviceScalar> {
    factors: CudaIluPreconditioner<T>,
    gather: PersistentDeviceCsr<CudaBackend, T>,
    scatter: PersistentDeviceCsr<CudaBackend, T>,
    gather_columns: Vec<usize>,
    scatter_rows: Vec<usize>,
    workspace: Mutex<Workspace<T>>,
}

impl<T: DeviceScalar> std::fmt::Debug for CudaIlutPreconditioner<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaIlutPreconditioner")
            .field("dimension", &self.gather_columns.len())
            .finish_non_exhaustive()
    }
}

impl<T> CudaIlutPreconditioner<T>
where
    T: ComplexField + DeviceScalar<Real = <T as ComplexField>::Real>,
{
    /// Prepare checked default ILUT factors and resident native application state.
    ///
    /// # Errors
    /// Rejects invalid factorization, regularized pivots or missing native support.
    /// Propagates device allocation and upload failures.
    pub fn try_from_csr(backend: &CudaBackend, matrix: &CsrMatrix<T>) -> BackendResult<Self> {
        Self::try_from_csr_with_config(backend, matrix, &IlutConfig::default())
    }

    /// Preserve explicit drop, fill and pivot controls during host factorization.
    ///
    /// # Errors
    /// Rejects malformed input, invalid controls, regularized/non-finite factors
    /// and unavailable native primitives. Propagates allocation/upload failures.
    pub fn try_from_csr_with_config(
        backend: &CudaBackend,
        matrix: &CsrMatrix<T>,
        config: &IlutConfig,
    ) -> BackendResult<Self> {
        backend.require_native_gs_color_sweep::<T>()?;
        let host = IlutPreconditioner::try_from_csr_with_config(matrix, config)
            .map_err(|error| invalid(error.to_string()))?;
        Self::try_from_host(backend, &host)
    }

    /// Upload existing host factors, preserving symmetric ordering and column pivoting.
    ///
    /// No refactorization occurs. Host factor storage may be dropped afterward.
    /// Device allocations follow the backend's existing quota policy.
    ///
    /// # Errors
    /// Rejects empty, regularized or non-finite factors, invalid permutations and
    /// unavailable native primitives. Propagates device setup failures.
    pub fn try_from_host(
        backend: &CudaBackend,
        host: &IlutPreconditioner<T>,
    ) -> BackendResult<Self> {
        backend.require_native_gs_color_sweep::<T>()?;
        let (gather_columns, scatter_rows) = mappings(host)?;
        let gather_matrix = CsrMatrix::from_triplets(
            host.n,
            host.n,
            gather_columns
                .iter()
                .enumerate()
                .map(|(row, &column)| (row, column, <T as DeviceScalar>::one()))
                .collect(),
        );
        let scatter_matrix = CsrMatrix::from_triplets(
            host.n,
            host.n,
            scatter_rows
                .iter()
                .enumerate()
                .map(|(column, &row)| (row, column, <T as DeviceScalar>::one()))
                .collect(),
        );
        Ok(Self {
            factors: CudaIluPreconditioner::from_factors(backend, &factor_view(host))?,
            gather: PersistentDeviceCsr::try_from_host_csr_ref(
                backend,
                gather_matrix.as_host_csr_ref(),
            )?,
            scatter: PersistentDeviceCsr::try_from_host_csr_ref(
                backend,
                scatter_matrix.as_host_csr_ref(),
            )?,
            gather_columns,
            scatter_rows,
            workspace: Mutex::new(Workspace {
                gathered: backend.raw_alloc(host.n)?,
                solved: backend.raw_alloc(host.n)?,
                result: backend.raw_alloc(host.n)?,
            }),
        })
    }

    /// Refresh compatible numeric factors while retaining maps, levels and scratch buffers.
    ///
    /// Returns false before mutation if either permutation or factor sparsity
    /// changes; prepare a replacement adapter in that case. Host factorization
    /// remains the caller's responsibility. Successful refresh allocates no
    /// backend buffers. Failed numeric upload invalidates application until a
    /// full successful refresh; validation failures retain the old action.
    ///
    /// # Errors
    /// Rejects invalid host factors and propagates native upload failures.
    pub fn try_refresh_from_host(
        &mut self,
        backend: &CudaBackend,
        host: &IlutPreconditioner<T>,
    ) -> BackendResult<bool> {
        let (gather, scatter) = mappings(host)?;
        if gather != self.gather_columns || scatter != self.scatter_rows {
            return Ok(false);
        }
        self.factors
            .try_refresh_factors(backend, &factor_view(host))
    }
}

fn factor_view<T: ComplexField>(host: &IlutPreconditioner<T>) -> TriangularFactors<'_, T> {
    TriangularFactors {
        n: host.n,
        l_values: &host.l_values,
        l_col_indices: &host.l_col_indices,
        l_row_ptrs: &host.l_row_ptrs,
        u_values: &host.u_values,
        u_col_indices: &host.u_col_indices,
        u_row_ptrs: &host.u_row_ptrs,
        u_diag_inv: &host.u_diag_inv,
    }
}

fn mappings<T>(host: &IlutPreconditioner<T>) -> BackendResult<(Vec<usize>, Vec<usize>)>
where
    T: ComplexField + DeviceScalar<Real = <T as ComplexField>::Real>,
{
    if host.n == 0
        || !host.regularized_pivots.is_empty()
        || host.non_finite_entries != 0
        || host
            .l_values
            .iter()
            .chain(&host.u_values)
            .chain(&host.u_diag_inv)
            .any(|v| !DeviceScalar::is_finite(v))
        || host
            .u_diag_inv
            .iter()
            .any(|v| *v == <T as DeviceScalar>::zero())
    {
        return Err(invalid(
            "requires nonempty finite ILUT factors without regularized pivots",
        ));
    }
    let gather: Vec<_> = host.perm.clone().unwrap_or_else(|| (0..host.n).collect());
    let columns: Vec<_> = host
        .col_perm
        .clone()
        .unwrap_or_else(|| (0..host.n).collect());
    for permutation in [&gather, &columns] {
        let mut seen = vec![false; host.n];
        if permutation.len() != host.n
            || permutation
                .iter()
                .any(|&i| i >= host.n || std::mem::replace(&mut seen[i], true))
        {
            return Err(invalid("invalid ILUT permutation"));
        }
    }
    // CPU application first undoes column pivoting in the reordered system,
    // then scatters that result through the symmetric ordering.
    let scatter = columns.iter().map(|&column| gather[column]).collect();
    Ok((gather, scatter))
}

impl<T> DevicePreconditioner<CudaBackend, T> for CudaIlutPreconditioner<T>
where
    T: ComplexField + DeviceScalar<Real = <T as ComplexField>::Real>,
{
    fn apply(&self, backend: &CudaBackend, rhs: &CudaBuffer<T>, output: &mut CudaBuffer<T>) {
        self.try_apply(backend, rhs, output)
            .expect("CUDA ILUT application failed");
    }

    fn try_apply(
        &self,
        backend: &CudaBackend,
        rhs: &CudaBuffer<T>,
        output: &mut CudaBuffer<T>,
    ) -> BackendResult<()> {
        if rhs.len() != self.gather_columns.len() || output.len() != rhs.len() {
            return Err(invalid("input/output dimension mismatch"));
        }
        let mut workspace = self
            .workspace
            .lock()
            .map_err(|_| invalid("workspace mutex poisoned"))?;
        let Workspace {
            gathered,
            solved,
            result,
        } = &mut *workspace;
        self.gather.try_apply(backend, rhs, gathered)?;
        self.factors.try_apply(backend, gathered, solved)?;
        self.scatter.try_apply(backend, solved, result)?;
        backend.try_copy_d2d(result, output)
    }
}

#[cfg(test)]
mod tests;
