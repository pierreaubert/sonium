//! Physical volume storage selected independently of the auxiliary AMG hierarchy.

// Rust guideline compliant 2026-02-21
use super::*;
use sonium_compute::cuda::fem_tet::{CudaTetC64, CudaTetCsrAssembly};
use sonium_compute::{Backend, DeviceCsr, DeviceOperator, PersistentDeviceCsr};

pub(super) struct CsrVolume<const N: usize> {
    assembly: CudaTetCsrAssembly<N>,
    values: Buffer<Complex64, CudaBackend>,
    matrix: PersistentDeviceCsr<CudaBackend, Complex64>,
}

pub(super) enum TetVolume<const N: usize> {
    Csr(Box<CsrVolume<N>>),
    Element(Box<CudaTetC64<N>>),
}

impl<const N: usize> TetVolume<N> {
    pub(super) fn prepare(
        pool: &MemoryPool<CudaBackend>,
        host: &TetAcousticC64<N>,
        mode: TetPressureOperator,
    ) -> BackendResult<Self> {
        if mode == TetPressureOperator::Element {
            return Ok(Self::Element(Box::new(CudaTetC64::try_new(
                pool.clone(),
                host,
            )?)));
        }
        let assembly = CudaTetCsrAssembly::try_new(pool.clone(), host)?;
        let pattern = assembly.pattern();
        let n = host.dimension();
        let matrix = PersistentDeviceCsr::try_from_device_csr(
            pool.backend(),
            DeviceCsr::try_from_host(
                pool.backend(),
                n,
                n,
                pattern.row_ptrs(),
                pattern.columns(),
                &vec![Complex64::default(); pattern.nnz()],
            )?,
        )?;
        let values = pool.try_alloc(pattern.nnz()).map_err(BackendError::from)?;
        Ok(Self::Csr(Box::new(CsrVolume {
            assembly,
            values,
            matrix,
        })))
    }

    pub(super) fn element(&self) -> &CudaTetC64<N> {
        match self {
            Self::Csr(c) => c.assembly.operator(),
            Self::Element(e) => e,
        }
    }

    pub(super) fn action(&self) -> &dyn DeviceOperator<CudaBackend, Complex64> {
        match self {
            Self::Csr(c) => &c.matrix,
            Self::Element(e) => e.as_ref(),
        }
    }

    pub(super) fn update(&mut self, backend: &CudaBackend, mass: Complex64) -> BackendResult<()> {
        backend.check_pending_error()?;
        let diffusion = Complex64::new(1.0, 0.0);
        match self {
            Self::Element(e) => e.try_set_scales(diffusion, mass),
            Self::Csr(c) => {
                c.assembly
                    .try_assemble_values(diffusion, mass, c.values.raw_mut())?;
                c.matrix
                    .try_copy_values_from_device(backend, c.values.raw())?;
                c.assembly.operator_mut().try_set_scales(diffusion, mass)
            }
        }
    }
}
