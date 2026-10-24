//! Resident ILU(0) application through dependency-ordered native row sweeps.

// Rust guideline compliant 2026-02-21
use crate::ilu::{Ilu0Factors, IluPreconditioner};
use num_traits::One;
use sonium_compute::{
    Backend, BackendError, BackendResult, CudaBackend, CudaBuffer, DeviceBufferLike, DeviceCsr,
    DevicePreconditioner, DeviceScalar, PersistentDeviceCsr,
};
use sonium_solvers_core::traits::ComplexField;
use sonium_solvers_sparse::CsrMatrix;
use std::sync::Mutex;

fn invalid(reason: impl Into<String>) -> BackendError {
    BackendError::InvalidArgument {
        operation: "cuda_ilu",
        reason: reason.into(),
    }
}

/// Borrowed triangular storage shared by ILU(0) and threshold ILU.
pub(crate) struct TriangularFactors<'a, T> {
    pub(crate) n: usize,
    pub(crate) l_values: &'a [T],
    pub(crate) l_col_indices: &'a [u32],
    pub(crate) l_row_ptrs: &'a [usize],
    pub(crate) u_values: &'a [T],
    pub(crate) u_col_indices: &'a [u32],
    pub(crate) u_row_ptrs: &'a [usize],
    pub(crate) u_diag_inv: &'a [T],
}

impl<'a, T: ComplexField> From<&'a Ilu0Factors<T>> for TriangularFactors<'a, T> {
    fn from(f: &'a Ilu0Factors<T>) -> Self {
        Self {
            n: f.n,
            l_values: &f.l_values,
            l_col_indices: &f.l_col_indices,
            l_row_ptrs: &f.l_row_ptrs,
            u_values: &f.u_values,
            u_col_indices: &f.u_col_indices,
            u_row_ptrs: &f.u_row_ptrs,
            u_diag_inv: &f.u_diag_inv,
        }
    }
}

struct Triangular<T: DeviceScalar> {
    matrix: DeviceCsr<CudaBackend, T>,
    levels: Vec<CudaBuffer<i32>>,
    rows: Vec<usize>,
    columns: Vec<usize>,
}

struct Workspace<T: DeviceScalar> {
    lower: CudaBuffer<T>,
    scaled: CudaBuffer<T>,
    result: CudaBuffer<T>,
}

/// Retains checked host ILU(0) factors for native CUDA application.
///
/// Factorization uses the default CPU policy without reordering. Each triangular
/// solve executes dependency levels in order; rows within a level are independent.
/// Upper rows are normalized to unit diagonal to avoid row-sweep pivot cutoffs.
/// Setup remains host work. Long dependency chains can require many launches.
pub struct CudaIluPreconditioner<T: DeviceScalar> {
    valid: bool,
    lower: Triangular<T>,
    upper: Triangular<T>,
    inverse: PersistentDeviceCsr<CudaBackend, T>,
    workspace: Mutex<Workspace<T>>,
}

impl<T> CudaIluPreconditioner<T>
where
    T: ComplexField + DeviceScalar<Real = <T as ComplexField>::Real>,
{
    /// Factor on the host and upload reusable triangular solve state.
    ///
    /// # Errors
    /// Rejects unavailable native kernels, invalid factors and nonfinite normalized
    /// coefficients. Propagates host factorization, allocation and upload failures.
    pub fn try_from_csr(backend: &CudaBackend, matrix: &CsrMatrix<T>) -> BackendResult<Self> {
        backend.require_native_gs_color_sweep::<T>()?;
        let host = IluPreconditioner::try_from_csr(matrix).map_err(|e| invalid(e.to_string()))?;
        let (factors, permutation) = host.into_factors();
        if permutation.is_some() {
            return Err(invalid("unexpected default ILU permutation"));
        }
        Self::from_factors(backend, &(&factors).into())
    }

    /// Upload existing host ILU(0) factors without refactorization.
    ///
    /// Unlike [`Self::try_from_csr`], which re-runs the checked host
    /// policy, this accepts factors built by the compatibility host
    /// constructors (`IluPreconditioner::from_csr*`), including any
    /// pivot regularization they retained; inspect
    /// `host.diagnostics()` before opting in. The device action is
    /// then bit-identical in policy to the CPU preconditioner, which
    /// CPU/device parity harnesses rely on. Host factor storage may
    /// be dropped afterward.
    ///
    /// # Errors
    /// Rejects unavailable native kernels, permuted factors (the
    /// resident apply has no permutation stage) and non-finite
    /// normalized coefficients. Propagates allocation and upload
    /// failures.
    pub fn try_from_host(
        backend: &CudaBackend,
        host: &IluPreconditioner<T>,
    ) -> BackendResult<Self> {
        backend.require_native_gs_color_sweep::<T>()?;
        if host.permutation().is_some() {
            return Err(invalid(
                "resident ILU(0) apply does not support permuted factors",
            ));
        }
        Self::from_factors(backend, &host.factors().into())
    }

    pub(crate) fn from_factors(
        backend: &CudaBackend,
        f: &TriangularFactors<'_, T>,
    ) -> BackendResult<Self> {
        let lower = triangular(backend, f, false)?;
        let upper = triangular(backend, f, true)?;
        let inverse = PersistentDeviceCsr::try_from_host_diagonal(backend, f.u_diag_inv)?;
        Ok(Self {
            valid: true,
            lower,
            upper,
            inverse,
            workspace: Mutex::new(Workspace {
                lower: backend.raw_alloc(f.n)?,
                scaled: backend.raw_alloc(f.n)?,
                result: backend.raw_alloc(f.n)?,
            }),
        })
    }

    /// Refresh compatible factors without replacing device storage or dependency levels.
    ///
    /// Returns false before mutation if factor sparsity changes. Host factorization
    /// remains setup work. Upload failures invalidate application until a complete
    /// refresh succeeds; validation failures preserve the previous valid factors.
    ///
    /// # Errors
    /// Rejects invalid factorization and propagates native admission/upload failures.
    pub fn try_refresh_from_csr(
        &mut self,
        backend: &CudaBackend,
        matrix: &CsrMatrix<T>,
    ) -> BackendResult<bool> {
        backend.require_native_gs_color_sweep::<T>()?;
        let (f, permutation) = IluPreconditioner::try_from_csr(matrix)
            .map_err(|e| invalid(e.to_string()))?
            .into_factors();
        if permutation.is_some() {
            return Err(invalid("unexpected default ILU permutation"));
        }
        self.try_refresh_factors(backend, &(&f).into())
    }

    pub(crate) fn try_refresh_factors(
        &mut self,
        backend: &CudaBackend,
        f: &TriangularFactors<'_, T>,
    ) -> BackendResult<bool> {
        backend.require_native_gs_color_sweep::<T>()?;
        let (lower, _) = triangular_host(f, false)?;
        let (upper, _) = triangular_host(f, true)?;
        if lower.row_ptrs != self.lower.rows
            || lower.col_indices != self.lower.columns
            || upper.row_ptrs != self.upper.rows
            || upper.col_indices != self.upper.columns
        {
            return Ok(false);
        }
        self.valid = false;
        self.lower
            .matrix
            .try_copy_values_from_host(backend, &lower.values)?;
        self.upper
            .matrix
            .try_copy_values_from_host(backend, &upper.values)?;
        backend.try_inject_preconditioner_failure(
            sonium_compute::PreconditionerFailureInjectionPoint::DiagonalUpload,
        )?;
        self.inverse
            .try_copy_values_from_host(backend, f.u_diag_inv)?;
        self.valid = true;
        Ok(true)
    }
}

fn triangular_host<T>(
    f: &TriangularFactors<'_, T>,
    upper: bool,
) -> BackendResult<(CsrMatrix<T>, Vec<Vec<i32>>)>
where
    T: ComplexField + DeviceScalar<Real = <T as ComplexField>::Real>,
{
    let (rows, columns, values) = if upper {
        (&f.u_row_ptrs, &f.u_col_indices, &f.u_values)
    } else {
        (&f.l_row_ptrs, &f.l_col_indices, &f.l_values)
    };
    let mut depths = vec![0usize; f.n];
    let order: Box<dyn Iterator<Item = usize>> = if upper {
        Box::new((0..f.n).rev())
    } else {
        Box::new(0..f.n)
    };
    for row in order {
        for &col in &columns[rows[row]..rows[row + 1]] {
            let col = col as usize;
            if col >= f.n || (upper && col <= row) || (!upper && col >= row) {
                return Err(invalid("invalid triangular dependency"));
            }
            depths[row] = depths[row].max(depths[col] + 1);
        }
    }
    let mut groups = vec![Vec::<i32>::new(); depths.iter().max().map_or(0, |d| d + 1)];
    let mut entries = Vec::with_capacity(values.len() + f.n);
    for row in 0..f.n {
        groups[depths[row]].push(i32::try_from(row).map_err(|_| invalid("row index exceeds i32"))?);
        entries.push((row, row, <T as DeviceScalar>::one()));
        for j in rows[row]..rows[row + 1] {
            let value = if upper {
                values[j] * f.u_diag_inv[row]
            } else {
                values[j]
            };
            if !DeviceScalar::is_finite(&value) {
                return Err(invalid("nonfinite normalized ILU factor"));
            }
            entries.push((row, columns[j] as usize, value));
        }
    }
    let host = CsrMatrix::from_triplets(f.n, f.n, entries);
    Ok((host, groups))
}

fn triangular<T>(
    backend: &CudaBackend,
    f: &TriangularFactors<'_, T>,
    upper: bool,
) -> BackendResult<Triangular<T>>
where
    T: ComplexField + DeviceScalar<Real = <T as ComplexField>::Real>,
{
    let (host, groups) = triangular_host(f, upper)?;
    let matrix = DeviceCsr::try_from_host_csr_ref(backend, host.as_host_csr_ref())?;
    let levels = groups
        .into_iter()
        .map(|rows| {
            let mut buffer = backend.raw_alloc::<i32>(rows.len())?;
            backend.try_copy_h2d(&rows, &mut buffer)?;
            Ok(buffer)
        })
        .collect::<BackendResult<Vec<_>>>()?;
    Ok(Triangular {
        matrix,
        levels,
        rows: host.row_ptrs,
        columns: host.col_indices,
    })
}

impl<T> DevicePreconditioner<CudaBackend, T> for CudaIluPreconditioner<T>
where
    T: ComplexField + DeviceScalar<Real = <T as ComplexField>::Real>,
{
    fn apply(&self, backend: &CudaBackend, rhs: &CudaBuffer<T>, output: &mut CudaBuffer<T>) {
        self.try_apply(backend, rhs, output)
            .expect("CUDA ILU application failed");
    }

    fn try_apply(
        &self,
        backend: &CudaBackend,
        rhs: &CudaBuffer<T>,
        output: &mut CudaBuffer<T>,
    ) -> BackendResult<()> {
        backend.require_native_gs_color_sweep::<T>()?;
        if !self.valid {
            return Err(invalid("numeric factors require a successful refresh"));
        }
        if rhs.len() != self.lower.matrix.m || output.len() != rhs.len() {
            return Err(invalid("input/output dimension mismatch"));
        }
        let mut workspace = self
            .workspace
            .lock()
            .map_err(|_| invalid("workspace mutex poisoned"))?;
        let Workspace {
            lower,
            scaled,
            result,
        } = &mut *workspace;
        // Initialize every lane: the row primitive evaluates (1-omega)*x even
        // for omega=1, so uninitialized values could otherwise propagate NaNs.
        backend.try_copy_d2d(rhs, lower)?;
        for level in &self.lower.levels {
            let a = &self.lower.matrix;
            backend.try_gs_color_sweep(
                &a.row_ptr,
                &a.col_idx,
                &a.values,
                level,
                <T as DeviceScalar>::Real::one(),
                rhs,
                lower,
            )?;
        }
        self.inverse.try_apply(backend, lower, scaled)?;
        backend.try_copy_d2d(scaled, result)?;
        for level in &self.upper.levels {
            let a = &self.upper.matrix;
            backend.try_gs_color_sweep(
                &a.row_ptr,
                &a.col_idx,
                &a.values,
                level,
                <T as DeviceScalar>::Real::one(),
                scaled,
                result,
            )?;
        }
        // Preserve caller output through every intermediate failure.
        backend.try_copy_d2d(result, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::{FromPrimitive, ToPrimitive};
    use sonium_solvers_core::traits::Preconditioner;

    fn check<T>(backend: &CudaBackend, scale: f64, tolerance: f64)
    where
        T: ComplexField + DeviceScalar<Real = <T as ComplexField>::Real>,
    {
        let scalar = |re, im| {
            <T as ComplexField>::from_re_im(
                <T as ComplexField>::Real::from_f64(re).unwrap(),
                <T as ComplexField>::Real::from_f64(im).unwrap(),
            )
        };
        let n = 19;
        let mut entries = Vec::new();
        for row in 0..n {
            entries.push((row, row, scalar(5.0 * scale, 0.1 * scale)));
            for offset in [1, 4] {
                // Disconnected six-row components exercise multiple independent
                // rows per dependency level and an isolated final row.
                if row + offset < n && row / 6 == (row + offset) / 6 {
                    entries.push((row, row + offset, scalar(-0.6 * scale, 0.2 * scale)));
                    entries.push((row + offset, row, scalar(-0.3 * scale, -0.1 * scale)));
                }
            }
        }
        let matrix = CsrMatrix::from_triplets(n, n, entries);
        let host = IluPreconditioner::try_from_csr(&matrix).unwrap();
        let mut device = CudaIluPreconditioner::try_from_csr(backend, &matrix).unwrap();
        let mut rhs = backend.raw_alloc::<T>(n).unwrap();
        let mut output = backend.raw_alloc::<T>(n).unwrap();
        for step in 0..3 {
            let input = (0..n)
                .map(|i| scalar(scale * (i + step + 1) as f64, scale * 0.2))
                .collect::<Vec<_>>();
            let expected = host.apply(&input);
            backend.try_copy_h2d(&input, &mut rhs).unwrap();
            let before = backend.begin_runtime_stats();
            device.try_apply(backend, &rhs, &mut output).unwrap();
            backend.try_synchronize().unwrap();
            let stats = backend.runtime_stats_since(before);
            assert_eq!(stats.allocation_operations, 0);
            assert_eq!(stats.transfer.h2d_bytes, 0);
            assert_eq!(stats.transfer.d2h_bytes, 0);
            let mut actual = vec![<T as DeviceScalar>::zero(); n];
            backend.try_copy_d2h(&output, &mut actual).unwrap();
            for (a, b) in actual.iter().zip(&expected) {
                assert!(
                    (*a - *b).norm().to_f64().unwrap()
                        < tolerance * (1.0 + b.norm().to_f64().unwrap())
                );
            }
        }
        backend.inject_failure(sonium_compute::CudaFailureInjectionPoint::Launch);
        let mut before_output = vec![<T as DeviceScalar>::zero(); n];
        backend.try_copy_d2h(&output, &mut before_output).unwrap();
        assert!(device.try_apply(backend, &rhs, &mut output).is_err());
        backend.recover_after_error().unwrap();
        let mut after_output = before_output.clone();
        backend.try_copy_d2h(&output, &mut after_output).unwrap();
        assert_eq!(before_output, after_output);
        let before = backend.begin_runtime_stats();
        assert!(device.try_refresh_from_csr(backend, &matrix).unwrap());
        assert_eq!(backend.runtime_stats_since(before).allocation_operations, 0);
        device.try_apply(backend, &rhs, &mut output).unwrap();
        backend.inject_preconditioner_failure(
            sonium_compute::PreconditionerFailureInjectionPoint::DiagonalUpload,
        );
        assert!(device.try_refresh_from_csr(backend, &matrix).is_err());
        backend.recover_after_error().unwrap();
        assert!(device.try_apply(backend, &rhs, &mut output).is_err());
        assert!(device.try_refresh_from_csr(backend, &matrix).unwrap());
        device.try_apply(backend, &rhs, &mut output).unwrap();
    }

    #[test]
    fn native_ilu_matches_cpu_factors_all_scalars_and_small_pivots() {
        let backend = match CudaBackend::new(0) {
            Ok(backend) => backend,
            Err(error) => {
                assert!(
                    std::env::var_os("SONIUM_REQUIRE_CUDA").is_none(),
                    "CUDA required: {error}"
                );
                return;
            }
        };
        for scale in [1.0, 1e-20] {
            check::<f32>(&backend, scale, 2e-5);
            check::<f64>(&backend, scale, 1e-11);
            check::<num_complex::Complex32>(&backend, scale, 2e-5);
            check::<num_complex::Complex64>(&backend, scale, 1e-11);
        }
    }

    #[test]
    fn try_from_host_accepts_compat_regularized_factors() {
        let backend = match CudaBackend::new(0) {
            Ok(backend) => backend,
            Err(error) => {
                assert!(
                    std::env::var_os("SONIUM_REQUIRE_CUDA").is_none(),
                    "CUDA required: {error}"
                );
                return;
            }
        };
        // |a_00| sits below the 1e-2 relative pivot threshold:
        // checked construction reports the skipped pivot and
        // rejects, while the compatibility constructor retains
        // fallback factors that try_from_host must upload as-is.
        let matrix = CsrMatrix::from_triplets(
            2,
            2,
            vec![(0, 0, 1.0_f64), (0, 1, 200.0), (1, 0, 200.0), (1, 1, 1.0)],
        );
        assert!(IluPreconditioner::<f64>::try_from_csr(&matrix).is_err());
        assert!(CudaIluPreconditioner::<f64>::try_from_csr(&backend, &matrix).is_err());
        let host = IluPreconditioner::from_csr(&matrix);
        let device = CudaIluPreconditioner::try_from_host(&backend, &host).unwrap();
        let input = vec![3.0, -1.0];
        let expected = host.apply(&input);
        let mut rhs = backend.raw_alloc::<f64>(2).unwrap();
        let mut output = backend.raw_alloc::<f64>(2).unwrap();
        backend.try_copy_h2d(&input, &mut rhs).unwrap();
        device.try_apply(&backend, &rhs, &mut output).unwrap();
        backend.try_synchronize().unwrap();
        let mut actual = vec![0.0; 2];
        backend.try_copy_d2h(&output, &mut actual).unwrap();
        for (a, b) in actual.iter().zip(&expected) {
            assert!(
                (a - b).abs() < 1e-12 * (1.0 + b.abs()),
                "{actual:?} vs {expected:?}"
            );
        }
    }
}
