//! Persistent device-side helpers used by the room FEM GPU runners.
//!
//! Source directivity and complex amplitudes are still evaluated by the
//! application because they are model-specific.  Once those coefficients are
//! available, the sparse source basis and the listener interpolation remain
//! resident and all accumulation/extraction happens through backend kernels.

use anyhow::{Context, Result, anyhow};
use sonium_compute::{
    Backend, BufferElement, DeviceBufferLike, DeviceCsr, DeviceScalar, PersistentDeviceCsr,
};

/// Reduce block residuals without downloading full vectors on CUDA.
#[cfg(any(feature = "cuda", test))]
pub(crate) fn block_relative_residuals<B: Backend>(
    backend: &B,
    rhs: &B::Buffer<num_complex::Complex64>,
    residual: &B::Buffer<num_complex::Complex64>,
    column: &mut B::Buffer<num_complex::Complex64>,
) -> Result<Vec<f64>> {
    use sonium_compute::{BufferView, ReductionMode};

    let n = column.len();
    anyhow::ensure!(n > 0, "block residual column must be nonempty");
    anyhow::ensure!(
        rhs.len() == residual.len() && rhs.len().is_multiple_of(n),
        "block residual dimensions do not match"
    );
    let destination = BufferView::new(n, 0, n, 1)?;
    (0..rhs.len() / n)
        .map(|index| {
            let source = BufferView::new(rhs.len(), index * n, n, 1)?;
            backend.try_copy_views_d2d(rhs, source, column, destination)?;
            let rhs_norm = backend.try_nrm2_with_mode(ReductionMode::Fast, column)?;
            backend.try_copy_views_d2d(residual, source, column, destination)?;
            let residual_norm = backend.try_nrm2_with_mode(ReductionMode::Fast, column)?;
            Ok(if !rhs_norm.is_finite() || !residual_norm.is_finite() {
                f64::INFINITY
            } else if rhs_norm > 0.0 {
                residual_norm / rhs_norm
            } else if residual_norm == 0.0 {
                0.0
            } else {
                f64::INFINITY
            })
        })
        .collect()
}

/// Device-resident sparse source scatter.
///
/// The matrix has one column per source-basis entry and one row per mesh DOF.
/// Its values are the real scatter weights promoted to `T`. Production room
/// coefficients already include their source-basis weights, so that path
/// supplies unit-valued scatter entries to avoid applying a weight twice.
pub(crate) struct DeviceRhsAssembler<B: Backend, T: DeviceScalar> {
    basis: PersistentDeviceCsr<B, T>,
    coefficients: B::Buffer<T>,
}

/// Device-resident sparse source scatter for a batch of right-hand sides.
///
/// Rows are laid out as `rhs * n_dofs`, with one contiguous block of rows per
/// source.  A single SpMV therefore assembles all source RHS vectors into the
/// same column-major layout consumed by block Krylov solvers. The production
/// room path supplies unit-valued scatter entries because its coefficients
/// already include source-basis weights.
#[cfg(any(feature = "cuda", test))]
pub(crate) struct DeviceRhsBlockAssembler<B: Backend, T: DeviceScalar> {
    basis: PersistentDeviceCsr<B, T>,
    coefficients: B::Buffer<T>,
    n_dofs: usize,
    rhs: usize,
}

#[cfg(any(feature = "cuda", test))]
struct ListenerBlockCache<B: Backend, T: DeviceScalar>
where
    T::Real: BufferElement,
{
    n_dofs: usize,
    rhs: usize,
    indices: B::Buffer<i32>,
    weights: B::Buffer<T::Real>,
    output: B::Buffer<T>,
}

#[cfg(any(feature = "cuda", test))]
impl<B: Backend, T: DeviceScalar> DeviceRhsBlockAssembler<B, T> {
    pub(crate) fn new(
        backend: &B,
        n_dofs: usize,
        entries: &[Vec<(usize, T::Real)>],
    ) -> Result<Self> {
        anyhow::ensure!(
            !entries.is_empty(),
            "block source basis needs at least one RHS"
        );
        let rhs = entries.len();
        let total_entries = entries.iter().map(Vec::len).sum::<usize>();
        let mut row_entries: Vec<Vec<(i32, T)>> = vec![Vec::new(); n_dofs * rhs];
        let mut column_offset = 0usize;
        for (source_idx, source_entries) in entries.iter().enumerate() {
            for &(node, weight) in source_entries {
                anyhow::ensure!(
                    node < n_dofs,
                    "source basis node {node} exceeds {n_dofs} DOFs"
                );
                let column = i32::try_from(column_offset)
                    .context("source basis exceeds i32 column range")?;
                row_entries[source_idx * n_dofs + node].push((column, T::from_real(weight)));
                column_offset += 1;
            }
        }

        debug_assert_eq!(column_offset, total_entries);
        let mut row_ptr = vec![0_i32; n_dofs * rhs + 1];
        for (row, values) in row_entries.iter().enumerate() {
            row_ptr[row + 1] = row_ptr[row]
                .checked_add(i32::try_from(values.len()).context("source basis row too large")?)
                .context("source basis exceeds i32 nnz range")?;
        }
        let nnz = row_ptr[n_dofs * rhs] as usize;
        let mut col_idx = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);
        for row in row_entries {
            for (column, value) in row {
                col_idx.push(column);
                values.push(value);
            }
        }

        let basis = DeviceCsr::try_from_host(
            backend,
            n_dofs * rhs,
            total_entries,
            &row_ptr,
            &col_idx,
            &values,
        )
        .map_err(|error| anyhow!(error))?;
        let coefficients = backend
            .raw_alloc::<T>(total_entries)
            .map_err(|error| anyhow!(error))?;

        Ok(Self {
            basis: PersistentDeviceCsr::try_from_device_csr(backend, basis)?,
            coefficients,
            n_dofs,
            rhs,
        })
    }

    /// Compute all source RHS vectors into one column-major device buffer.
    pub(crate) fn assemble(
        &mut self,
        backend: &B,
        coefficients: &[T],
        rhs: &mut B::Buffer<T>,
    ) -> Result<()> {
        anyhow::ensure!(
            coefficients.len() == self.coefficients.len(),
            "RHS coefficient count mismatch"
        );
        anyhow::ensure!(
            rhs.len() == self.n_dofs * self.rhs,
            "RHS block length mismatch"
        );
        backend.try_copy_h2d(coefficients, &mut self.coefficients)?;
        self.basis.try_apply(backend, &self.coefficients, rhs)?;
        Ok(())
    }
}

impl<B: Backend, T: DeviceScalar> DeviceRhsAssembler<B, T> {
    pub(crate) fn new(backend: &B, n_dofs: usize, entries: &[(usize, T::Real)]) -> Result<Self> {
        let mut row_ptr = vec![0_i32; n_dofs + 1];
        for &(node, _) in entries {
            anyhow::ensure!(
                node < n_dofs,
                "source basis node {node} exceeds {n_dofs} DOFs"
            );
        }
        row_ptr.fill(0);
        for &(node, _) in entries {
            row_ptr[node + 1] += 1;
        }
        for row in 1..=n_dofs {
            row_ptr[row] += row_ptr[row - 1];
        }
        let mut ordered_cols = vec![0_i32; entries.len()];
        let mut ordered_values = vec![T::zero(); entries.len()];
        let mut cursor = row_ptr[..n_dofs].to_vec();
        for (column, &(node, weight)) in entries.iter().enumerate() {
            let slot = cursor[node] as usize;
            ordered_cols[slot] =
                i32::try_from(column).context("source basis exceeds i32 column range")?;
            ordered_values[slot] = T::from_real(weight);
            cursor[node] += 1;
        }
        let mut row_ptr_dev = backend
            .raw_alloc::<i32>(row_ptr.len())
            .map_err(|error| anyhow!(error))?;
        let mut col_idx_dev = backend
            .raw_alloc::<i32>(ordered_cols.len())
            .map_err(|error| anyhow!(error))?;
        let mut values_dev = backend
            .raw_alloc::<T>(ordered_values.len())
            .map_err(|error| anyhow!(error))?;
        let coefficients = backend
            .raw_alloc::<T>(entries.len())
            .map_err(|error| anyhow!(error))?;
        backend.try_copy_h2d(&row_ptr, &mut row_ptr_dev)?;
        backend.try_copy_h2d(&ordered_cols, &mut col_idx_dev)?;
        backend.try_copy_h2d(&ordered_values, &mut values_dev)?;

        Ok(Self {
            basis: PersistentDeviceCsr::try_from_device_csr(
                backend,
                DeviceCsr {
                    m: n_dofs,
                    n: entries.len(),
                    nnz: ordered_values.len(),
                    row_ptr: row_ptr_dev,
                    col_idx: col_idx_dev,
                    values: values_dev,
                },
            )?,
            coefficients,
        })
    }

    /// Compute `rhs <- basis * coefficients` without downloading a nodal RHS.
    pub(crate) fn assemble(
        &mut self,
        backend: &B,
        coefficients: &[T],
        rhs: &mut B::Buffer<T>,
    ) -> Result<()> {
        anyhow::ensure!(
            coefficients.len() == self.coefficients.len(),
            "RHS coefficient count mismatch"
        );
        anyhow::ensure!(rhs.len() == self.basis.m(), "RHS length mismatch");
        backend.try_copy_h2d(coefficients, &mut self.coefficients)?;
        self.basis.try_apply(backend, &self.coefficients, rhs)?;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn entry_count(&self) -> usize {
        self.coefficients.len()
    }
}

/// Persistent four-point listener extraction buffers.
pub(crate) struct DeviceListenerExtraction<B: Backend, T: DeviceScalar>
where
    T::Real: BufferElement,
{
    indices: B::Buffer<i32>,
    weights: B::Buffer<T::Real>,
    output: B::Buffer<T>,
    #[cfg(any(feature = "cuda", test))]
    indices_host: Vec<i32>,
    #[cfg(any(feature = "cuda", test))]
    weights_host: Vec<T::Real>,
    #[cfg(any(feature = "cuda", test))]
    block_cache: Option<ListenerBlockCache<B, T>>,
}

impl<B: Backend, T: DeviceScalar> DeviceListenerExtraction<B, T>
where
    T::Real: BufferElement,
{
    pub(crate) fn new(backend: &B, nodes: &[[usize; 4]], weights: &[[T::Real; 4]]) -> Result<Self> {
        anyhow::ensure!(
            nodes.len() == weights.len(),
            "listener node/weight count mismatch"
        );
        let flat_indices: Vec<i32> = nodes
            .iter()
            .flat_map(|quad| quad.iter().copied())
            .map(|node| i32::try_from(node).map_err(|_| anyhow!("listener node exceeds i32 range")))
            .collect::<Result<_>>()?;
        let flat_weights: Vec<T::Real> = weights
            .iter()
            .flat_map(|quad| quad.iter().copied())
            .collect();
        let mut indices = backend
            .raw_alloc::<i32>(flat_indices.len())
            .map_err(|error| anyhow!(error))?;
        let mut weights = backend
            .raw_alloc::<T::Real>(flat_weights.len())
            .map_err(|error| anyhow!(error))?;
        let output = backend
            .raw_alloc::<T>(nodes.len())
            .map_err(|error| anyhow!(error))?;
        backend.copy_h2d(&flat_indices, &mut indices);
        backend.copy_h2d(&flat_weights, &mut weights);
        Ok(Self {
            indices,
            weights,
            output,
            #[cfg(any(feature = "cuda", test))]
            indices_host: flat_indices,
            #[cfg(any(feature = "cuda", test))]
            weights_host: flat_weights,
            #[cfg(any(feature = "cuda", test))]
            block_cache: None,
        })
    }

    /// Evaluate all listeners into the persistent device output buffer.
    pub(crate) fn evaluate_device<'a>(
        &'a mut self,
        backend: &B,
        values: &B::Buffer<T>,
    ) -> &'a B::Buffer<T> {
        backend.gather4_weighted(values, &self.indices, &self.weights, &mut self.output);
        &self.output
    }

    /// Evaluate and copy only the listener values to the host.
    pub(crate) fn evaluate_to_host(&mut self, backend: &B, values: &B::Buffer<T>) -> Vec<T> {
        let output_len = self.output.len();
        let output = self.evaluate_device(backend, values);
        let mut host = vec![T::zero(); output_len];
        backend.copy_d2h(output, &mut host);
        host
    }

    /// Evaluate a column-major block of nodal vectors with one persistent
    /// gather launch and copy only the listener values back to the host.
    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn evaluate_colmajor_to_host(
        &mut self,
        backend: &B,
        values: &B::Buffer<T>,
        n_dofs: usize,
        rhs: usize,
    ) -> Vec<Vec<T>> {
        assert_eq!(values.len(), n_dofs * rhs);
        if rhs == 0 {
            return Vec::new();
        }
        let cache = if let Some(cache) = &mut self.block_cache {
            cache
        } else {
            let mut indices = Vec::with_capacity(self.indices_host.len() * rhs);
            let mut weights = Vec::with_capacity(self.weights_host.len() * rhs);
            for column in 0..rhs {
                let offset =
                    i32::try_from(column * n_dofs).expect("block listener offset exceeds i32");
                indices.extend(self.indices_host.iter().map(|&index| index + offset));
                weights.extend_from_slice(&self.weights_host);
            }
            let mut indices_dev = backend
                .raw_alloc::<i32>(indices.len())
                .expect("block listener index allocation failed");
            let mut weights_dev = backend
                .raw_alloc::<T::Real>(weights.len())
                .expect("block listener weight allocation failed");
            let output_dev = backend
                .raw_alloc::<T>(self.output.len() * rhs)
                .expect("block listener output allocation failed");
            backend.copy_h2d(&indices, &mut indices_dev);
            backend.copy_h2d(&weights, &mut weights_dev);
            self.block_cache = Some(ListenerBlockCache {
                n_dofs,
                rhs,
                indices: indices_dev,
                weights: weights_dev,
                output: output_dev,
            });
            self.block_cache
                .as_mut()
                .expect("block listener cache was just initialized")
        };
        assert_eq!(cache.n_dofs, n_dofs);
        assert_eq!(cache.rhs, rhs);
        backend.gather4_weighted(values, &cache.indices, &cache.weights, &mut cache.output);
        let mut output = vec![T::zero(); self.output.len() * rhs];
        backend.copy_d2h(&cache.output, &mut output);
        output
            .chunks_exact(self.output.len())
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.output.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use sonium_compute::CpuBackend;

    fn check_block_relative_residuals<B: Backend>(
        backend: &B,
        counters: impl Fn() -> [u64; 3],
    ) {
        let n = 257;
        let mut rhs = vec![Complex64::new(0.0, 0.0); n * 3];
        let mut residual = rhs.clone();
        rhs[0] = Complex64::new(3.0, 4.0);
        residual[0] = Complex64::new(0.3, 0.4);
        residual[2 * n] = Complex64::new(1.0, 0.0);
        let mut b = backend.raw_alloc::<Complex64>(rhs.len()).unwrap();
        let mut r = backend.raw_alloc::<Complex64>(rhs.len()).unwrap();
        let mut scratch = backend.raw_alloc::<Complex64>(n).unwrap();
        backend.try_copy_h2d(&rhs, &mut b).unwrap();
        backend.try_copy_h2d(&residual, &mut r).unwrap();
        for _ in 0..2 {
            let before = counters();
            let values = block_relative_residuals(backend, &b, &r, &mut scratch).unwrap();
            let after = counters();
            assert!((values[0] - 0.1).abs() < 1.0e-14);
            assert_eq!(values[1], 0.0);
            assert_eq!(values[2], f64::INFINITY);
            assert_eq!(after[0], before[0]);
            assert_eq!(after[1], before[1]);
            // At most two scalar norms per column, never N-sized downloads.
            assert!(after[2] - before[2] <= 6 * 16);
        }
        rhs[0] = Complex64::new(f64::NAN, 0.0);
        backend.try_copy_h2d(&rhs, &mut b).unwrap();
        // Checked backends may reject a non-finite norm before returning it;
        // either outcome must prevent acceptance of the block.
        if let Ok(values) = block_relative_residuals(backend, &b, &r, &mut scratch) {
            assert_eq!(values[0], f64::INFINITY);
        }
        let mut wrong_shape = backend.raw_alloc::<Complex64>(n - 1).unwrap();
        assert!(block_relative_residuals(backend, &b, &r, &mut wrong_shape).is_err());
    }

    #[test]
    fn block_residual_norms_handle_zero_and_nonfinite_columns() {
        check_block_relative_residuals(&CpuBackend::new(), || [0; 3]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_block_residual_norms_do_not_download_vectors() {
        let backend = match sonium_compute::CudaBackend::new(0) {
            Ok(backend) => backend,
            Err(error) => {
                assert!(std::env::var_os("SONIUM_SIM_ROOM_REQUIRE_CUDA").is_none(),
                    "CUDA required for block norms: {error}");
                return;
            }
        };
        check_block_relative_residuals(&backend, || {
            let stats = backend.runtime_stats();
            [stats.allocation_operations as u64, stats.transfer.h2d_bytes as u64,
                stats.transfer.d2h_bytes as u64]
        });
    }

    #[test]
    fn rhs_basis_accumulates_on_device_for_arbitrary_entry_order() {
        let backend = CpuBackend::new();
        let mut assembler = DeviceRhsAssembler::<CpuBackend, Complex64>::new(
            &backend,
            3,
            &[(2, 0.25), (0, 0.5), (2, 0.75)],
        )
        .unwrap();
        assert_eq!(assembler.entry_count(), 3);
        let mut rhs = backend.raw_alloc::<Complex64>(3).unwrap();
        assembler
            .assemble(
                &backend,
                &[
                    Complex64::new(2.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(4.0, 0.0),
                ],
                &mut rhs,
            )
            .unwrap();
        let mut host = [Complex64::new(0.0, 0.0); 3];
        backend.copy_d2h(&rhs, &mut host);
        assert_eq!(host[0], Complex64::new(0.5, 0.0));
        assert_eq!(host[1], Complex64::new(0.0, 0.0));
        assert_eq!(host[2], Complex64::new(3.5, 0.0));
        assert!(assembler.assemble(&backend, &[], &mut rhs).is_err());
        let mut unchanged = [Complex64::new(0.0, 0.0); 3];
        backend.copy_d2h(&rhs, &mut unchanged);
        assert_eq!(unchanged, host);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_preweighted_rhs_scatter_matches_cpu_reference() {
        use sonium_compute::{Backend, CudaBackend};

        let backend = match CudaBackend::new(0) {
            Ok(backend) => backend,
            Err(error) => {
                if std::env::var_os("SONIUM_SIM_ROOM_REQUIRE_CUDA").is_some() {
                    panic!("CUDA is required for the RHS scatter regression: {error}");
                }
                eprintln!("no CUDA device — skipping RHS scatter regression: {error}");
                return;
            }
        };

        // The room runner applies each source-basis weight while building its
        // coefficient vector. The device matrix must therefore contain unit
        // scatter values, including when several entries target one node.
        let nodes = [(2, 1.0_f64), (0, 1.0_f64), (2, 1.0_f64)];
        let coefficients = [
            Complex64::new(0.5, -0.25),
            Complex64::new(1.0, 0.75),
            Complex64::new(-1.5, 0.125),
        ];
        let mut assembler =
            DeviceRhsAssembler::<CudaBackend, Complex64>::new(&backend, 3, &nodes).unwrap();
        let mut rhs = backend.raw_alloc::<Complex64>(3).unwrap();
        assembler
            .assemble(&backend, &coefficients, &mut rhs)
            .unwrap();
        backend.synchronize().unwrap();

        let before = backend.runtime_stats();
        assembler
            .assemble(&backend, &coefficients, &mut rhs)
            .unwrap();
        let after = backend.runtime_stats();
        assert_eq!(after.allocation_operations, before.allocation_operations);
        assert_eq!(after.transfer.d2h_bytes, before.transfer.d2h_bytes);

        let mut uploaded = [Complex64::new(0.0, 0.0); 3];
        backend.copy_d2h(&assembler.coefficients, &mut uploaded);
        backend.synchronize().unwrap();
        assert_eq!(uploaded, coefficients);

        let mut actual = [Complex64::new(0.0, 0.0); 3];
        backend.copy_d2h(&rhs, &mut actual);
        backend.synchronize().unwrap();
        let mut expected = [Complex64::new(0.0, 0.0); 3];
        expected[2] = coefficients[0] + coefficients[2];
        expected[0] = coefficients[1];
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (*actual - expected).norm() <= 1.0e-12,
                "CUDA RHS scatter mismatch: actual={actual:?}, expected={expected:?}"
            );
        }
    }

    #[test]
    fn listeners_are_uploaded_once_and_extracted_without_full_solution_copy() {
        let backend = CpuBackend::new();
        let mut listeners = DeviceListenerExtraction::<CpuBackend, Complex64>::new(
            &backend,
            &[[0, 1, 2, 3], [3, 2, 1, 0]],
            &[[0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 0.0]],
        )
        .unwrap();
        assert_eq!(listeners.len(), 2);
        let mut values = backend.raw_alloc::<Complex64>(4).unwrap();
        backend.copy_h2d(
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
            &mut values,
        );
        assert_eq!(
            listeners.evaluate_to_host(&backend, &values),
            vec![Complex64::new(2.5, 0.0), Complex64::new(4.0, 0.0)]
        );
    }

    #[test]
    fn block_rhs_assembler_preserves_source_column_layout() {
        let backend = CpuBackend::new();
        let mut assembler = DeviceRhsBlockAssembler::<CpuBackend, Complex64>::new(
            &backend,
            3,
            &[vec![(2, 0.25), (0, 0.5), (2, 0.75)], vec![(1, 1.0)]],
        )
        .unwrap();
        let mut rhs = backend.raw_alloc::<Complex64>(6).unwrap();
        assembler
            .assemble(
                &backend,
                &[
                    Complex64::new(2.0, 0.0),
                    Complex64::new(1.0, 0.0),
                    Complex64::new(4.0, 0.0),
                    Complex64::new(3.0, 0.0),
                ],
                &mut rhs,
            )
            .unwrap();
        let mut host = [Complex64::new(0.0, 0.0); 6];
        backend.copy_d2h(&rhs, &mut host);
        assert_eq!(
            host,
            [
                Complex64::new(0.5, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(3.5, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(0.0, 0.0),
            ]
        );
    }

    #[test]
    fn block_listener_extraction_matches_per_column_results() {
        let backend = CpuBackend::new();
        let mut listeners = DeviceListenerExtraction::<CpuBackend, Complex64>::new(
            &backend,
            &[[0, 1, 2, 3]],
            &[[0.25, 0.25, 0.25, 0.25]],
        )
        .unwrap();
        let mut values = backend.raw_alloc::<Complex64>(8).unwrap();
        backend.copy_h2d(
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
            &mut values,
        );
        assert_eq!(
            listeners.evaluate_colmajor_to_host(&backend, &values, 4, 2),
            vec![
                vec![Complex64::new(2.5, 0.0)],
                vec![Complex64::new(2.5, 0.0)],
            ]
        );
    }
}
