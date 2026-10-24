//! Packed resident one-level additive Schwarz with the CPU overlap and weights.

// Rust guideline compliant 2026-02-21
use super::AdditiveSchwarzPreconditioner;
use crate::CudaIluPreconditioner;
use sonium_compute::{
    Backend, BackendError, BackendResult, CudaBackend, CudaBuffer, DeviceBufferLike,
    DevicePreconditioner, DeviceScalar, PersistentDeviceCsr,
};
use sonium_solvers_core::traits::ComplexField;
use sonium_solvers_sparse::CsrMatrix;
use std::sync::Mutex;

fn invalid(reason: impl Into<String>) -> BackendError {
    BackendError::InvalidArgument {
        operation: "cuda_schwarz",
        reason: reason.into(),
    }
}

struct Workspace<T: DeviceScalar> {
    rhs: CudaBuffer<T>,
    local: CudaBuffer<T>,
    result: CudaBuffer<T>,
}

/// Retains one-level overlapping Schwarz maps, local factors and device scratch.
///
/// Uses the checked CPU partition, overlap extension and partition-of-unity
/// weights. All subdomain matrices are packed into a block-diagonal ILU(0)
/// action; sparse restriction and weighted assembly keep vectors on the device.
/// Partitioning and local factorization remain host setup/update work.
pub struct CudaSchwarzPreconditioner<T: DeviceScalar> {
    layout: Vec<Vec<usize>>,
    weights: Vec<T>,
    restriction: PersistentDeviceCsr<CudaBackend, T>,
    assembly: PersistentDeviceCsr<CudaBackend, T>,
    local: CudaIluPreconditioner<T>,
    workspace: Mutex<Workspace<T>>,
}

fn packed_matrix<T>(host: &AdditiveSchwarzPreconditioner<T>) -> BackendResult<CsrMatrix<T>>
where
    T: ComplexField,
{
    if host.coarse.is_some() {
        return Err(invalid("two-level coarse correction is not implemented"));
    }
    let mut entries = Vec::new();
    let mut offset = 0usize;
    for domain in &host.subdomains {
        let n = domain.global_indices.len();
        let next = offset
            .checked_add(n)
            .ok_or_else(|| invalid("packed dimension overflow"))?;
        if next > i32::MAX as usize {
            return Err(invalid("packed dimension exceeds i32"));
        }
        for row in 0..n {
            for j in domain.local_row_ptrs[row]..domain.local_row_ptrs[row + 1] {
                entries.push((
                    offset + row,
                    offset + domain.local_col_indices[j],
                    domain.local_values[j],
                ));
            }
        }
        offset = next;
    }
    Ok(CsrMatrix::from_triplets(offset, offset, entries))
}

impl<T> CudaSchwarzPreconditioner<T>
where
    T: ComplexField + DeviceScalar<Real = <T as ComplexField>::Real>,
{
    /// Prepare the facade's automatic partition policy with explicit overlap depth.
    ///
    /// # Errors
    /// Rejects unavailable kernels or invalid CPU Schwarz setup and propagates uploads.
    pub fn try_from_csr_auto(
        backend: &CudaBackend,
        matrix: &CsrMatrix<T>,
        overlap: usize,
    ) -> BackendResult<Self> {
        backend.require_native_gs_color_sweep::<T>()?;
        let host = AdditiveSchwarzPreconditioner::try_from_csr_auto(matrix, overlap)
            .map_err(|e| invalid(e.to_string()))?;
        Self::try_from_host(backend, &host)
    }

    /// Upload a one-level CPU Schwarz configuration with identical local matrices.
    ///
    /// # Errors
    /// Rejects attached coarse corrections, invalid factors and backend failures.
    pub fn try_from_host(
        backend: &CudaBackend,
        host: &AdditiveSchwarzPreconditioner<T>,
    ) -> BackendResult<Self> {
        backend.require_native_gs_color_sweep::<T>()?;
        let packed = packed_matrix(host)?;
        let local = CudaIluPreconditioner::try_from_csr(backend, &packed)?;
        let mut gather = Vec::new();
        let mut scatter = Vec::new();
        let mut offset = 0;
        for domain in &host.subdomains {
            for (row, &global) in domain.global_indices.iter().enumerate() {
                if global >= host.n {
                    return Err(invalid("invalid global subdomain index"));
                }
                gather.push((offset + row, global, <T as DeviceScalar>::one()));
                scatter.push((global, offset + row, host.weights[global]));
            }
            offset += domain.global_indices.len();
        }
        let restriction = CsrMatrix::from_triplets(offset, host.n, gather);
        let assembly = CsrMatrix::from_triplets(host.n, offset, scatter);
        Ok(Self {
            layout: host
                .subdomains
                .iter()
                .map(|d| d.global_indices.clone())
                .collect(),
            weights: host.weights.clone(),
            restriction: PersistentDeviceCsr::try_from_host_csr_ref(
                backend,
                restriction.as_host_csr_ref(),
            )?,
            assembly: PersistentDeviceCsr::try_from_host_csr_ref(
                backend,
                assembly.as_host_csr_ref(),
            )?,
            local,
            workspace: Mutex::new(Workspace {
                rhs: backend.raw_alloc(offset)?,
                local: backend.raw_alloc(offset)?,
                result: backend.raw_alloc(host.n)?,
            }),
        })
    }

    /// Refresh compatible local factors while retaining overlap maps and scratch.
    ///
    /// Returns false before mutation if membership, weights or factor sparsity
    /// changes. A partial upload invalidates the local action until repaired.
    ///
    /// # Errors
    /// Rejects invalid host setup and propagates numeric upload failures.
    pub fn try_refresh_from_host(
        &mut self,
        backend: &CudaBackend,
        host: &AdditiveSchwarzPreconditioner<T>,
    ) -> BackendResult<bool> {
        if host.n != self.weights.len()
            || host.weights != self.weights
            || host.subdomains.len() != self.layout.len()
            || host
                .subdomains
                .iter()
                .zip(&self.layout)
                .any(|(d, old)| d.global_indices != *old)
        {
            return Ok(false);
        }
        self.local
            .try_refresh_from_csr(backend, &packed_matrix(host)?)
    }
}

impl<T> DevicePreconditioner<CudaBackend, T> for CudaSchwarzPreconditioner<T>
where
    T: ComplexField + DeviceScalar<Real = <T as ComplexField>::Real>,
{
    fn apply(&self, backend: &CudaBackend, rhs: &CudaBuffer<T>, output: &mut CudaBuffer<T>) {
        self.try_apply(backend, rhs, output)
            .expect("CUDA Schwarz application failed");
    }

    fn try_apply(
        &self,
        backend: &CudaBackend,
        rhs: &CudaBuffer<T>,
        output: &mut CudaBuffer<T>,
    ) -> BackendResult<()> {
        backend.require_native_gs_color_sweep::<T>()?;
        if rhs.len() != self.weights.len() || output.len() != rhs.len() {
            return Err(invalid("input/output dimension mismatch"));
        }
        let mut workspace = self
            .workspace
            .lock()
            .map_err(|_| invalid("workspace mutex poisoned"))?;
        let Workspace {
            rhs: local_rhs,
            local,
            result,
        } = &mut *workspace;
        self.restriction.try_apply(backend, rhs, local_rhs)?;
        self.local.try_apply(backend, local_rhs, local)?;
        self.assembly.try_apply(backend, local, result)?;
        backend.try_copy_d2d(result, output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::{FromPrimitive, ToPrimitive};
    use sonium_solvers_core::traits::Preconditioner;

    fn check<T>(backend: &CudaBackend, tolerance: f64)
    where
        T: ComplexField + DeviceScalar<Real = <T as ComplexField>::Real>,
    {
        let scalar = |re, im| {
            <T as ComplexField>::from_re_im(
                <T as ComplexField>::Real::from_f64(re).unwrap(),
                <T as ComplexField>::Real::from_f64(im).unwrap(),
            )
        };
        let n = 31;
        let mut entries = Vec::new();
        for row in 0..n {
            entries.push((row, row, scalar(5.0, 0.2)));
            for offset in [1, 5] {
                if row + offset < n {
                    entries.push((row, row + offset, scalar(-0.6, 0.1)));
                    entries.push((row + offset, row, scalar(-0.3, -0.2)));
                }
            }
        }
        let matrix = CsrMatrix::from_triplets(n, n, entries);
        let mut rhs = backend.raw_alloc::<T>(n).unwrap();
        let mut output = backend.raw_alloc::<T>(n).unwrap();
        for overlap in [0, 1, 2] {
            let host = AdditiveSchwarzPreconditioner::try_from_csr(&matrix, 4, overlap).unwrap();
            assert!(host.subdomains.len() > 1);
            if overlap > 0 {
                assert!(
                    host.subdomains
                        .iter()
                        .map(|d| d.global_indices.len())
                        .sum::<usize>()
                        > n
                );
            }
            let mut device = CudaSchwarzPreconditioner::try_from_host(backend, &host).unwrap();
            for scale in [1.0, 1.25] {
                let mut changed = matrix.clone();
                changed
                    .values
                    .iter_mut()
                    .for_each(|v| *v *= scalar(scale, 0.0));
                let updated =
                    AdditiveSchwarzPreconditioner::try_from_csr(&changed, 4, overlap).unwrap();
                let before = backend.begin_runtime_stats();
                assert!(device.try_refresh_from_host(backend, &updated).unwrap());
                let stats = backend.runtime_stats_since(before);
                assert_eq!(stats.allocation_operations, 0);
                assert_eq!(stats.transfer.d2h_bytes, 0);
                for step in 0..2 {
                    let input = (0..n)
                        .map(|i| scalar((i + step + 1) as f64, 0.3))
                        .collect::<Vec<_>>();
                    let expected = updated.apply(&input);
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
                    for (a, b) in actual.iter().zip(expected) {
                        assert!(
                            (*a - b).norm().to_f64().unwrap()
                                < tolerance * (1.0 + b.norm().to_f64().unwrap())
                        );
                    }
                }
            }
            let repartitioned =
                AdditiveSchwarzPreconditioner::try_from_csr(&matrix, 2, overlap).unwrap();
            let before = backend.begin_runtime_stats();
            assert!(
                !device
                    .try_refresh_from_host(backend, &repartitioned)
                    .unwrap()
            );
            let stats = backend.runtime_stats_since(before);
            assert_eq!(stats.allocation_operations, 0);
            assert_eq!(stats.transfer.h2d_bytes, 0);
            backend.inject_preconditioner_failure(
                sonium_compute::PreconditionerFailureInjectionPoint::DiagonalUpload,
            );
            assert!(device.try_refresh_from_host(backend, &host).is_err());
            backend.recover_after_error().unwrap();
            assert!(device.try_apply(backend, &rhs, &mut output).is_err());
            assert!(device.try_refresh_from_host(backend, &host).unwrap());
            device.try_apply(backend, &rhs, &mut output).unwrap();
        }
    }

    #[test]
    fn native_schwarz_matches_cpu_overlap_weights_and_refresh() {
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
        check::<f32>(&backend, 3e-5);
        check::<f64>(&backend, 1e-11);
        check::<num_complex::Complex32>(&backend, 3e-5);
        check::<num_complex::Complex64>(&backend, 1e-11);
    }
}
