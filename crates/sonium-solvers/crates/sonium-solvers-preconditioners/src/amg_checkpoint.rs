//! Strict portable checkpoints for the host AMG preconditioner.
//!
//! The hierarchy is deliberately rebuilt on restore from the canonical finest
//! CSR operator and the complete public [`AmgConfig`].  This keeps the format
//! independent of machine index widths, sparse-LU internals, mutex workspaces,
//! and setup timing while preserving the mathematical preconditioner policy.

use super::{
    AmgCoarsening, AmgConfig, AmgCycle, AmgInterpolation, AmgPreconditioner, AmgRestriction,
    AmgSmoother, MaterialSocConfig,
};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use sonium_solvers_core::traits::{ComplexField, SolverError};
use sonium_solvers_sparse::CsrMatrix;

/// Version of the portable host-AMG checkpoint envelope.
pub const AMG_CHECKPOINT_FORMAT_VERSION: u16 = 2;
const MAGIC: &[u8; 8] = b"SAMGCP1\0";

fn error(message: impl Into<String>) -> SolverError {
    SolverError::OperationFailed {
        operation: "AMG checkpoint",
        reason: message.into(),
    }
}

struct Writer {
    bytes: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn u64(&mut self, value: usize) -> Result<(), SolverError> {
        self.bytes.extend_from_slice(
            &u64::try_from(value)
                .map_err(|_| error("length does not fit u64"))?
                .to_le_bytes(),
        );
        Ok(())
    }

    fn f64(&mut self, value: f64) -> Result<(), SolverError> {
        if !value.is_finite() {
            return Err(error("checkpoint contains non-finite f64"));
        }
        self.bytes.extend_from_slice(&value.to_le_bytes());
        Ok(())
    }

    fn bool(&mut self, value: bool) {
        self.u8(u8::from(value));
    }

    fn option_f64(&mut self, value: Option<f64>) -> Result<(), SolverError> {
        match value {
            Some(value) => {
                self.bool(true);
                self.f64(value)?;
            }
            None => self.bool(false),
        }
        Ok(())
    }

    fn usize_vec(&mut self, values: &[usize]) -> Result<(), SolverError> {
        self.u64(values.len())?;
        for &value in values {
            self.u64(value)?;
        }
        Ok(())
    }

    fn scalars<T: ComplexField>(&mut self, values: &[T]) -> Result<(), SolverError> {
        self.u64(values.len())?;
        for value in values {
            let re = value
                .re()
                .to_f64()
                .ok_or_else(|| error("real scalar cannot convert to f64"))?;
            let im = value
                .im()
                .to_f64()
                .ok_or_else(|| error("imaginary scalar cannot convert to f64"))?;
            if !re.is_finite() || !im.is_finite() {
                return Err(error("checkpoint contains non-finite scalar"));
            }
            self.f64(re)?;
            self.f64(im)?;
        }
        Ok(())
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], SolverError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| error("checkpoint offset overflow"))?;
        let result = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| error("truncated AMG checkpoint"))?;
        self.offset = end;
        Ok(result)
    }

    fn u8(&mut self) -> Result<u8, SolverError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, SolverError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<usize, SolverError> {
        usize::try_from(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
            .map_err(|_| error("checkpoint length does not fit usize"))
    }

    fn f64(&mut self) -> Result<f64, SolverError> {
        let value = f64::from_le_bytes(self.take(8)?.try_into().unwrap());
        if !value.is_finite() {
            return Err(error("checkpoint contains non-finite f64"));
        }
        Ok(value)
    }

    fn bool(&mut self) -> Result<bool, SolverError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(error("boolean flag is not 0 or 1")),
        }
    }

    fn option_f64(&mut self) -> Result<Option<f64>, SolverError> {
        if self.bool()? {
            Ok(Some(self.f64()?))
        } else {
            Ok(None)
        }
    }

    fn usize_vec(&mut self, label: &str) -> Result<Vec<usize>, SolverError> {
        let length = self.u64()?;
        if length > self.bytes.len().saturating_sub(self.offset) / 8 {
            return Err(error(format!(
                "{label} exceeds remaining checkpoint storage"
            )));
        }
        (0..length)
            .map(|_| self.u64())
            .collect::<Result<Vec<_>, _>>()
    }

    fn scalars<T: ComplexField>(&mut self, label: &str) -> Result<Vec<T>, SolverError> {
        let length = self.u64()?;
        if length > self.bytes.len().saturating_sub(self.offset) / 16 {
            return Err(error(format!(
                "{label} exceeds remaining checkpoint storage"
            )));
        }
        (0..length)
            .map(|_| {
                let re = T::Real::from_f64(self.f64()?)
                    .ok_or_else(|| error("real scalar cannot be decoded"))?;
                let im = T::Real::from_f64(self.f64()?)
                    .ok_or_else(|| error("imaginary scalar cannot be decoded"))?;
                let value = T::from_re_im(re, im);
                if !value.norm().is_finite() {
                    return Err(error(format!("{label} contains non-finite value")));
                }
                Ok(value)
            })
            .collect()
    }

    fn at_end(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

fn write_config(writer: &mut Writer, config: &AmgConfig) -> Result<(), SolverError> {
    writer.u8(match config.coarsening {
        AmgCoarsening::RugeStuben => 0,
        AmgCoarsening::Pmis => 1,
        AmgCoarsening::Hmis => 2,
        AmgCoarsening::SmoothedAggregation => 3,
        AmgCoarsening::Cljp => 4,
        AmgCoarsening::Falgout => 5,
        AmgCoarsening::EnergyMinimizing => 6,
        #[cfg(feature = "experimental-gnn")]
        AmgCoarsening::GnnCoarsening => {
            return Err(error(
                "experimental GNN coarsening is not supported by this checkpoint format",
            ));
        }
    });
    writer.u8(match config.interpolation {
        AmgInterpolation::Standard => 0,
        AmgInterpolation::Extended => 1,
        AmgInterpolation::ExtendedI => 2,
        AmgInterpolation::Direct => 3,
    });
    match config.restriction {
        AmgRestriction::Transpose => writer.u8(0),
        AmgRestriction::ConjugateTranspose => writer.u8(1),
        AmgRestriction::LAir { distance, filter } => {
            writer.u8(2);
            writer.u64(distance)?;
            writer.option_f64(filter)?;
        }
    }
    writer.bool(config.symmetrize_strength);
    writer.u8(match config.smoother {
        AmgSmoother::Jacobi => 0,
        AmgSmoother::L1Jacobi => 1,
        AmgSmoother::SymmetricGaussSeidel => 2,
        AmgSmoother::Chebyshev => 3,
        AmgSmoother::MulticolorGaussSeidel => 4,
        AmgSmoother::CfSymmetricGaussSeidel => 5,
    });
    writer.bool(config.f_point_smoothing);
    writer.u8(match config.cycle {
        AmgCycle::VCycle => 0,
        AmgCycle::WCycle => 1,
        AmgCycle::FCycle => 2,
    });
    for value in [
        config.k_cycle_iterations,
        config.max_levels,
        config.coarse_size,
        config.num_pre_smooth,
        config.num_post_smooth,
        config.max_interp_elements,
        config.aggressive_coarsening_levels,
        config.mis_distance,
        config.energy_min_iterations,
        config.energy_min_pattern_distance,
        config.near_nullspace_dim,
        config.block_size,
    ] {
        writer.u64(value)?;
    }
    for value in [
        config.strong_threshold,
        config.jacobi_weight,
        config.trunc_factor,
    ] {
        writer.f64(value)?;
    }
    writer.bool(config.sa_strength_with_ata);
    match &config.material_soc {
        None => writer.bool(false),
        Some(material) => {
            writer.bool(true);
            writer.u64(material.dim)?;
            writer.u64(material.tensors.len())?;
            for tensor in &material.tensors {
                writer.u64(tensor.len())?;
                for &value in tensor {
                    writer.f64(value)?;
                }
            }
        }
    }
    #[cfg(feature = "experimental-gnn")]
    if config.gnn_weights.is_some() {
        return Err(error(
            "experimental GNN weights are not supported by this checkpoint format",
        ));
    }
    writer.option_f64(config.max_grid_complexity)?;
    writer.option_f64(config.max_operator_complexity)?;
    writer.f64(config.galerkin_drop_tol)?;
    Ok(())
}

fn read_config(reader: &mut Reader<'_>, version: u16) -> Result<AmgConfig, SolverError> {
    let coarsening = match reader.u8()? {
        0 => AmgCoarsening::RugeStuben,
        1 => AmgCoarsening::Pmis,
        2 => AmgCoarsening::Hmis,
        3 => AmgCoarsening::SmoothedAggregation,
        4 => AmgCoarsening::Cljp,
        5 => AmgCoarsening::Falgout,
        6 => AmgCoarsening::EnergyMinimizing,
        _ => return Err(error("unknown AMG coarsening tag")),
    };
    let interpolation = match reader.u8()? {
        0 => AmgInterpolation::Standard,
        1 => AmgInterpolation::Extended,
        2 => AmgInterpolation::ExtendedI,
        3 => AmgInterpolation::Direct,
        _ => return Err(error("unknown AMG interpolation tag")),
    };
    let restriction = match reader.u8()? {
        0 => AmgRestriction::Transpose,
        1 => AmgRestriction::ConjugateTranspose,
        2 => AmgRestriction::LAir {
            distance: reader.u64()?,
            filter: reader.option_f64()?,
        },
        _ => return Err(error("unknown AMG restriction tag")),
    };
    let symmetrize_strength = reader.bool()?;
    let smoother = match reader.u8()? {
        0 => AmgSmoother::Jacobi,
        1 => AmgSmoother::L1Jacobi,
        2 => AmgSmoother::SymmetricGaussSeidel,
        3 => AmgSmoother::Chebyshev,
        4 => AmgSmoother::MulticolorGaussSeidel,
        5 => AmgSmoother::CfSymmetricGaussSeidel,
        _ => return Err(error("unknown AMG smoother tag")),
    };
    let f_point_smoothing = reader.bool()?;
    let cycle = match reader.u8()? {
        0 => AmgCycle::VCycle,
        1 => AmgCycle::WCycle,
        2 => AmgCycle::FCycle,
        _ => return Err(error("unknown AMG cycle tag")),
    };
    let mut values = [0usize; 12];
    for value in &mut values {
        *value = reader.u64()?;
    }
    let strong_threshold = reader.f64()?;
    let jacobi_weight = reader.f64()?;
    let trunc_factor = reader.f64()?;
    let sa_strength_with_ata = reader.bool()?;
    let material_soc = if reader.bool()? {
        let dim = reader.u64()?;
        let tensor_count = reader.u64()?;
        if tensor_count > reader.bytes.len().saturating_sub(reader.offset) / 8 {
            return Err(error("material tensor count exceeds checkpoint storage"));
        }
        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let length = reader.u64()?;
            if length > reader.bytes.len().saturating_sub(reader.offset) / 8 {
                return Err(error("material tensor exceeds checkpoint storage"));
            }
            let mut tensor = Vec::with_capacity(length);
            for _ in 0..length {
                tensor.push(reader.f64()?);
            }
            tensors.push(tensor);
        }
        Some(MaterialSocConfig { tensors, dim })
    } else {
        None
    };
    let mut config = AmgConfig {
        coarsening,
        interpolation,
        restriction,
        symmetrize_strength,
        smoother,
        f_point_smoothing,
        cycle,
        k_cycle_iterations: values[0],
        strong_threshold,
        max_levels: values[1],
        coarse_size: values[2],
        num_pre_smooth: values[3],
        num_post_smooth: values[4],
        jacobi_weight,
        trunc_factor,
        // v1 checkpoints predate galerkin_drop_tol: restore the default.
        galerkin_drop_tol: 0.001,
        max_interp_elements: values[5],
        aggressive_coarsening_levels: values[6],
        mis_distance: values[7],
        sa_strength_with_ata,
        material_soc,
        energy_min_iterations: values[8],
        energy_min_pattern_distance: values[9],
        near_nullspace_dim: values[10],
        block_size: values[11],
        #[cfg(feature = "experimental-gnn")]
        gnn_weights: None,
        max_grid_complexity: reader.option_f64()?,
        max_operator_complexity: reader.option_f64()?,
    };
    if version >= 2 {
        config.galerkin_drop_tol = reader.f64()?;
    }
    config.validate()?;
    Ok(config)
}

impl<T: ComplexField> AmgPreconditioner<T> {
    /// Serialize a host AMG hierarchy as a strict portable checkpoint.
    ///
    /// Setup-derived coarse matrices and sparse-LU factors are rebuilt on
    /// restore, so the envelope is independent of machine index widths and
    /// factor-layout implementation details.
    pub fn to_checkpoint_bytes(&self) -> Result<Vec<u8>, SolverError> {
        let finest = self
            .levels
            .first()
            .ok_or_else(|| error("AMG hierarchy has no finest level"))?;
        finest.matrix.validate_canonical_structure()?;
        self.config.validate()?;
        let mut writer = Writer::new();
        writer.bytes.extend_from_slice(MAGIC);
        writer.u16(AMG_CHECKPOINT_FORMAT_VERSION);
        writer.u8(std::mem::size_of::<T>() as u8);
        writer.u8(std::mem::size_of::<T::Real>() as u8);
        writer.u8(u8::from(
            std::mem::size_of::<T>() == 2 * std::mem::size_of::<T::Real>(),
        ));
        writer.u8(0);
        writer.u64(finest.matrix.num_rows)?;
        writer.u64(finest.matrix.num_cols)?;
        write_config(&mut writer, &self.config)?;
        writer.usize_vec(&finest.matrix.row_ptrs)?;
        writer.usize_vec(&finest.matrix.col_indices)?;
        writer.scalars(&finest.matrix.values)?;
        Ok(writer.bytes)
    }

    /// Restore a host AMG preconditioner from [`Self::to_checkpoint_bytes`].
    ///
    /// Version 1 omitted the Galerkin drop tolerance and restores its historical
    /// value of 0.001. Version 2 preserves the serialized tolerance, including zero.
    pub fn from_checkpoint_bytes(bytes: &[u8]) -> Result<Self, SolverError> {
        let mut reader = Reader::new(bytes);
        if reader.take(MAGIC.len())? != MAGIC {
            return Err(error("invalid AMG checkpoint magic"));
        }
        let version = reader.u16()?;
        if !matches!(version, 1 | AMG_CHECKPOINT_FORMAT_VERSION) {
            return Err(error("unsupported AMG checkpoint version"));
        }
        let scalar_size = reader.u8()? as usize;
        let real_size = reader.u8()? as usize;
        let expected_complex =
            u8::from(std::mem::size_of::<T>() == 2 * std::mem::size_of::<T::Real>());
        if scalar_size != std::mem::size_of::<T>()
            || real_size != std::mem::size_of::<T::Real>()
            || reader.u8()? != expected_complex
        {
            return Err(error(
                "AMG checkpoint scalar metadata does not match requested type",
            ));
        }
        if reader.u8()? != 0 {
            return Err(error("AMG checkpoint reserved flags are non-zero"));
        }
        let rows = reader.u64()?;
        let cols = reader.u64()?;
        let config = read_config(&mut reader, version)?;
        let row_ptrs = reader.usize_vec("row pointers")?;
        let col_indices = reader.usize_vec("column indices")?;
        let values = reader.scalars::<T>("matrix values")?;
        if !reader.at_end() {
            return Err(error("trailing bytes in AMG checkpoint"));
        }
        let matrix = CsrMatrix::try_from_raw_parts(rows, cols, row_ptrs, col_indices, values)?;
        Self::try_from_csr(&matrix, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v2_preserves_drop_policy_and_reads_legacy_v1() {
        for tolerance in [0.0, 0.02] {
            let host = AmgPreconditioner::try_from_csr(
                &CsrMatrix::<f64>::identity(3),
                AmgConfig {
                    galerkin_drop_tol: tolerance,
                    ..AmgConfig::for_backend_pcg()
                },
            )
            .unwrap();
            let bytes = host.to_checkpoint_bytes().unwrap();
            let restored = AmgPreconditioner::<f64>::from_checkpoint_bytes(&bytes).unwrap();
            assert_eq!(restored.config.galerkin_drop_tol, tolerance);
            let mut reader = Reader::new(&bytes);
            // Magic, version, four scalar/flag bytes, then two u64 dimensions.
            reader.take(MAGIC.len() + 2 + 4 + 8 + 8).unwrap();
            read_config(&mut reader, 2).unwrap();
            let drop_offset = reader.offset - 8;
            let mut legacy = bytes.clone();
            legacy.drain(drop_offset..drop_offset + 8);
            legacy[8..10].copy_from_slice(&1_u16.to_le_bytes());
            let restored = AmgPreconditioner::<f64>::from_checkpoint_bytes(&legacy).unwrap();
            assert_eq!(restored.config.galerkin_drop_tol, 0.001);
            for bad in [f64::NAN, -0.1, 0.2] {
                let mut corrupt = bytes.clone();
                corrupt[drop_offset..drop_offset + 8].copy_from_slice(&bad.to_le_bytes());
                assert!(AmgPreconditioner::<f64>::from_checkpoint_bytes(&corrupt).is_err());
            }
        }
    }
}
