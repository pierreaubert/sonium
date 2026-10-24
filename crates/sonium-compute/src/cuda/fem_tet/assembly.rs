//! Resident numeric CSR assembly using a fixed tetrahedral contribution map.

// Rust guideline compliant 2026-02-21
use super::*;
use crate::fem_tet::TetCsrPattern;

/// Retains topology, geometry and mapping for repeated GPU numeric assembly.
///
/// Setup builds the CSR pattern on the host. Each assembly writes numeric values
/// in one launch. Feed the output to `PersistentDeviceCsr::try_copy_values_from_device`
/// to update a retained sparse operator without a host round trip.
pub struct CudaTetCsrAssembly<const N: usize> {
    operator: CudaTetC64<N>,
    pattern: TetCsrPattern,
    rows: Buffer<u32, CudaBackend>,
    columns: Buffer<i32, CudaBackend>,
    offsets: Buffer<u32, CudaBackend>,
    contributions: Buffer<u32, CudaBackend>,
}
impl<const N: usize> std::fmt::Debug for CudaTetCsrAssembly<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaTetCsrAssembly")
            .field("nnz", &self.pattern.nnz())
            .finish_non_exhaustive()
    }
}
impl<const N: usize> CudaTetCsrAssembly<N> {
    /// Build and upload the fixed sparsity mapping and validated geometry.
    ///
    /// # Errors
    /// Rejects CSR index overflow or graph capture and propagates pool, upload,
    /// JIT and backend errors. Partial construction releases pooled arrays.
    pub fn try_new(pool: MemoryPool<CudaBackend>, host: &TetAcousticC64<N>) -> BackendResult<Self> {
        pool.backend().try_prepare_tet_kernels::<N>()?;
        let pattern = host.csr_pattern()?;
        let operator = CudaTetC64::try_new(pool.clone(), host)?;
        Ok(Self {
            rows: upload(&pool, &pattern.rows)?,
            columns: upload(&pool, &pattern.columns)?,
            offsets: upload(&pool, &pattern.offsets)?,
            contributions: upload(&pool, &pattern.contributions)?,
            pattern,
            operator,
        })
    }
    /// Borrow the element operator sharing this assembly's geometry.
    pub fn operator(&self) -> &CudaTetC64<N> {
        &self.operator
    }

    /// Borrow the element operator to update its action scales.
    ///
    /// Numeric assembly continues to use the explicit scales passed to
    /// `try_assemble_values`; callers keep the two representations consistent.
    pub fn operator_mut(&mut self) -> &mut CudaTetC64<N> {
        &mut self.operator
    }

    /// Borrow the fixed host CSR structure for sparse-operator setup.
    pub fn pattern(&self) -> &TetCsrPattern {
        &self.pattern
    }

    /// Assemble scaled diffusion/mass values without reallocating or uploading arrays.
    ///
    /// Scales multiply the constructor's per-element coefficients. Constraint
    /// diagonals remain one; constrained off-diagonals remain zero. The output
    /// must have exactly `pattern().nnz()` complex entries on the owning stream.
    ///
    /// # Errors
    /// Rejects invalid scales or output shape/storage/stream before launch.
    /// Backend launch failures follow the typed output/context recovery contract.
    pub fn try_assemble_values(
        &self,
        diffusion_scale: Complex64,
        mass_scale: Complex64,
        out: &mut CudaBuffer<Complex64>,
    ) -> BackendResult<()> {
        let backend = self.operator.pool.backend();
        let nnz = self.pattern.nnz();
        if [diffusion_scale, mass_scale]
            .iter()
            .any(|z| !z.re.is_finite() || !z.im.is_finite())
            || out.len() != nnz
        {
            return Err(invalid("CSR scale or output length mismatch"));
        }
        let bytes = nnz
            .checked_mul(size_of::<Complex64>())
            .ok_or_else(|| invalid("CSR byte count overflow"))?;
        let storage = out
            .raw()
            .ok_or_else(|| invalid("missing CSR output storage"))?;
        if storage.len() != bytes || !Arc::ptr_eq(storage.stream(), backend.stream()) {
            return Err(invalid("CSR output storage size or stream mismatch"));
        }
        backend.try_custom_kernel_preflight()?;
        let count = nnz as u32; // Host pattern checks signed 32-bit CSR limits.
        let output = out
            .raw_mut()
            .ok_or_else(|| invalid("missing CSR output storage"))?;
        backend.record_custom_kernel_launch();
        // SAFETY: private mapping arrays derive from this operator's validated
        // host mesh. Encoded element/local indices and CSR rows/columns are in
        // range. Output byte count and stream were checked. Kernel ABI arguments
        // match sonium_tet_csr_c64; cudarc retains stream-use dependencies.
        unsafe {
            backend
                .stream()
                .launch_builder(&self.operator.assembly_kernel)
                .arg(raw(&self.operator.geometry)?)
                .arg(raw(&self.operator.coefficients)?)
                .arg(raw(&self.operator.dirichlet)?)
                .arg(raw(&self.rows)?)
                .arg(raw(&self.columns)?)
                .arg(raw(&self.offsets)?)
                .arg(raw(&self.contributions)?)
                .arg(output)
                .arg(&count)
                .arg(&diffusion_scale.re)
                .arg(&diffusion_scale.im)
                .arg(&mass_scale.re)
                .arg(&mass_scale.im)
                .launch(launch_config(count, self.operator.assembly_block_size))
                .map_err(|error| backend.custom_kernel_error(OPERATION, format!("{error:?}")))?;
        }
        Ok(())
    }
}
