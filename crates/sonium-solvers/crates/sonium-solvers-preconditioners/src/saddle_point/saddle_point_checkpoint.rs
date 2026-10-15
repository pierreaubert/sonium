//! Strict portable checkpoints for the bounded saddle-point preconditioner.

use super::{
    BlockPreconditionerConfig, SaddlePointConfig, SaddlePointPreconditioner, SchurStrategy,
};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use sonium_solvers_core::traits::{ComplexField, SolverError};
use sonium_solvers_sparse::CsrMatrix;

pub const SADDLE_POINT_CHECKPOINT_FORMAT_VERSION: u16 = 1;
const MAGIC: &[u8; 8] = b"SSPCP001";

fn error(message: impl Into<String>) -> SolverError {
    SolverError::OperationFailed {
        operation: "saddle-point checkpoint",
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
                .map_err(|_| error("length does not fit checkpoint u64"))?
                .to_le_bytes(),
        );
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
                return Err(error("checkpoint contains a non-finite scalar"));
            }
            self.bytes.extend_from_slice(&re.to_le_bytes());
            self.bytes.extend_from_slice(&im.to_le_bytes());
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
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| error("truncated saddle-point checkpoint"))?;
        self.offset = end;
        Ok(value)
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

    fn read_len(&mut self, label: &str, element_size: usize) -> Result<usize, SolverError> {
        let length = self.u64()?;
        if element_size == 0 || length > self.bytes.len().saturating_sub(self.offset) / element_size
        {
            return Err(error(format!(
                "{label} exceeds remaining checkpoint storage"
            )));
        }
        Ok(length)
    }

    fn usize_vec(&mut self, label: &str) -> Result<Vec<usize>, SolverError> {
        let length = self.read_len(label, 8)?;
        (0..length).map(|_| self.u64()).collect()
    }

    fn scalars<T: ComplexField>(&mut self, label: &str) -> Result<Vec<T>, SolverError> {
        let length = self.read_len(label, 16)?;
        let mut values = Vec::with_capacity(length);
        for _ in 0..length {
            let re = f64::from_le_bytes(self.take(8)?.try_into().unwrap());
            let im = f64::from_le_bytes(self.take(8)?.try_into().unwrap());
            if !re.is_finite() || !im.is_finite() {
                return Err(error(format!("{label} contains a non-finite scalar")));
            }
            let re = T::Real::from_f64(re).ok_or_else(|| error("real scalar cannot be decoded"))?;
            let im =
                T::Real::from_f64(im).ok_or_else(|| error("imaginary scalar cannot be decoded"))?;
            let value = T::from_re_im(re, im);
            if !value.norm().is_finite() {
                return Err(error(format!("{label} contains a non-finite value")));
            }
            values.push(value);
        }
        Ok(values)
    }

    fn at_end(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

fn config_tag(config: &BlockPreconditionerConfig) -> Result<u8, SolverError> {
    match config {
        BlockPreconditionerConfig::Diagonal => Ok(0),
        BlockPreconditionerConfig::None => Ok(1),
        BlockPreconditionerConfig::Amg(_)
        | BlockPreconditionerConfig::Ilut(_)
        | BlockPreconditionerConfig::Ilu => Err(error(
            "checkpoint format v1 does not support AMG, ILUT, or ILU block factors",
        )),
    }
}

fn config_from_tag(tag: u8) -> Result<BlockPreconditionerConfig, SolverError> {
    match tag {
        0 => Ok(BlockPreconditionerConfig::Diagonal),
        1 => Ok(BlockPreconditionerConfig::None),
        _ => Err(error("unsupported block-preconditioner tag")),
    }
}

fn validate_supported<T: ComplexField>(
    preconditioner: &SaddlePointPreconditioner<T>,
) -> Result<(u8, u8), SolverError> {
    if preconditioner.config.schur_strategy != SchurStrategy::Diagonal {
        return Err(error(
            "checkpoint format v1 supports only the diagonal Schur strategy",
        ));
    }
    if preconditioner.config.block1_size == 0
        || preconditioner.config.block1_size >= preconditioner.source_matrix.num_rows
    {
        return Err(error("block1_size does not split the source matrix"));
    }
    preconditioner
        .source_matrix
        .validate_structure()
        .map_err(|e| error(e.to_string()))?;
    if preconditioner.source_matrix.num_rows != preconditioner.source_matrix.num_cols {
        return Err(error("source matrix must be square"));
    }
    Ok((
        config_tag(&preconditioner.config.block1_config)?,
        config_tag(&preconditioner.config.block2_config)?,
    ))
}

impl<T: ComplexField> SaddlePointPreconditioner<T> {
    pub fn to_checkpoint_bytes(&self) -> Result<Vec<u8>, SolverError> {
        let (block1_tag, block2_tag) = validate_supported(self)?;
        let mut writer = Writer::new();
        writer.bytes.extend_from_slice(MAGIC);
        writer.u16(SADDLE_POINT_CHECKPOINT_FORMAT_VERSION);
        writer.u8(std::mem::size_of::<T>() as u8);
        writer.u8(std::mem::size_of::<T::Real>() as u8);
        writer.u8(u8::from(
            std::mem::size_of::<T>() == 2 * std::mem::size_of::<T::Real>(),
        ));
        writer.u8(0);
        writer.u64(self.source_matrix.num_rows)?;
        writer.u64(self.source_matrix.num_cols)?;
        writer.u64(self.config.block1_size)?;
        writer.u8(0);
        writer.u8(block1_tag);
        writer.u8(block2_tag);
        writer.u8(u8::from(self.config.upper_triangular));
        writer.u8(0);
        writer.u64(self.source_matrix.row_ptrs.len())?;
        for &value in &self.source_matrix.row_ptrs {
            writer.u64(value)?;
        }
        writer.u64(self.source_matrix.col_indices.len())?;
        for &value in &self.source_matrix.col_indices {
            writer.u64(value)?;
        }
        writer.scalars(&self.source_matrix.values)?;
        Ok(writer.bytes)
    }

    pub fn from_checkpoint_bytes(bytes: &[u8]) -> Result<Self, SolverError> {
        let mut reader = Reader::new(bytes);
        if reader.take(MAGIC.len())? != MAGIC {
            return Err(error("invalid checkpoint magic"));
        }
        if reader.u16()? != SADDLE_POINT_CHECKPOINT_FORMAT_VERSION {
            return Err(error("unsupported checkpoint version"));
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
                "checkpoint scalar metadata does not match requested type",
            ));
        }
        if reader.u8()? != 0 {
            return Err(error("checkpoint reserved flags are non-zero"));
        }
        let rows = reader.u64()?;
        let cols = reader.u64()?;
        let block1_size = reader.u64()?;
        if rows == 0 || rows != cols || block1_size == 0 || block1_size >= rows {
            return Err(error(
                "checkpoint dimensions do not define two non-empty blocks",
            ));
        }
        if reader.u8()? != 0 {
            return Err(error("unsupported Schur strategy tag"));
        }
        let block1_config = config_from_tag(reader.u8()?)?;
        let block2_config = config_from_tag(reader.u8()?)?;
        let upper_triangular = match reader.u8()? {
            0 => false,
            1 => true,
            _ => return Err(error("upper-triangular flag is not boolean")),
        };
        if reader.u8()? != 0 {
            return Err(error("checkpoint configuration flags are non-zero"));
        }
        let row_ptrs = reader.usize_vec("row pointers")?;
        let col_indices = reader.usize_vec("column indices")?;
        let values = reader.scalars::<T>("matrix values")?;
        if row_ptrs.len() != rows.saturating_add(1) || col_indices.len() != values.len() {
            return Err(error("checkpoint CSR dimensions are inconsistent"));
        }
        let matrix = CsrMatrix::try_from_raw_parts(rows, cols, row_ptrs, col_indices, values)
            .map_err(|e| error(e.to_string()))?;
        matrix
            .validate_structure()
            .map_err(|e| error(e.to_string()))?;
        if !reader.at_end() {
            return Err(error("checkpoint contains trailing bytes"));
        }
        SaddlePointPreconditioner::try_from_matrix(
            &matrix,
            SaddlePointConfig {
                block1_size,
                block1_config,
                block2_config,
                schur_strategy: SchurStrategy::Diagonal,
                upper_triangular,
            },
        )
    }
}
