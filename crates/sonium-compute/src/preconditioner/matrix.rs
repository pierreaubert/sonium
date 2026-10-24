//! Matrix ownership and checked refresh for polynomial preconditioners.

// Rust guideline compliant 2026-02-21
use crate::operator::{DeviceOperator, PersistentDeviceCsr, PrebakedSpmv};
use crate::{Backend, BackendError, BackendResult, DeviceBufferLike, DeviceCsr, DeviceScalar};
use num_traits::Float;

pub(super) enum PreconditionerMatrix<'a, B: Backend, T: DeviceScalar> {
    Borrowed {
        matrix: &'a DeviceCsr<B, T>,
        action: PrebakedSpmv<'a, B, T>,
    },
    Owned {
        matrix: PersistentDeviceCsr<B, T>,
        valid: bool,
    },
}

impl<'a, B: Backend, T: DeviceScalar> PreconditionerMatrix<'a, B, T> {
    pub(super) fn borrowed(backend: &'a B, matrix: &'a DeviceCsr<B, T>) -> BackendResult<Self> {
        Ok(Self::Borrowed {
            matrix,
            action: PrebakedSpmv::try_from_csr(backend, matrix)?,
        })
    }

    pub(super) fn matrix(&self) -> &DeviceCsr<B, T> {
        match self {
            Self::Borrowed { matrix, .. } => matrix,
            Self::Owned { matrix, .. } => matrix.matrix(),
        }
    }

    pub(super) fn ensure_valid(&self) -> BackendResult<()> {
        if matches!(self, Self::Owned { valid: false, .. }) {
            return Err(BackendError::OperationFailed {
                operation: "polynomial_matrix_apply",
                reason: "numeric refresh failed; complete a successful refresh before applying"
                    .into(),
            });
        }
        Ok(())
    }

    pub(super) fn try_apply(
        &self,
        backend: &B,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> BackendResult<()> {
        self.ensure_valid()?;
        match self {
            Self::Borrowed { action, .. } => action.try_apply(backend, x, y),
            Self::Owned { matrix, .. } => matrix.try_apply(backend, x, y),
        }
    }

    pub(super) fn try_apply_colmajor(
        &self,
        backend: &B,
        rhs: usize,
        x: &B::Buffer<T>,
        y: &mut B::Buffer<T>,
    ) -> BackendResult<()> {
        self.ensure_valid()?;
        match self {
            Self::Borrowed { action, .. } => action.try_apply_colmajor(backend, rhs, x, y),
            Self::Owned { matrix, .. } => matrix.try_apply_colmajor(backend, rhs, x, y),
        }
    }

    pub(super) fn try_refresh(
        &mut self,
        backend: &B,
        values: &B::Buffer<T>,
        diagonal: &[T],
        inverse: &mut PersistentDeviceCsr<B, T>,
    ) -> BackendResult<()> {
        let invalid = |reason: &str| BackendError::InvalidArgument {
            operation: "polynomial_matrix_refresh",
            reason: reason.into(),
        };
        let Self::Owned { matrix, valid } = self else {
            return Err(invalid("numeric refresh requires an owned matrix"));
        };
        if diagonal.len() != matrix.m() || values.len() != matrix.nnz() {
            return Err(invalid("numeric data must match the retained matrix shape"));
        }
        let mut inv = Vec::with_capacity(diagonal.len());
        for &d in diagonal {
            if !d.is_finite() || d.norm_sqr().sqrt() <= T::Real::epsilon() {
                return Err(invalid("diagonal must be finite and invertible"));
            }
            let value = T::one() / d;
            if !value.is_finite() {
                return Err(invalid("inverse diagonal must be finite"));
            }
            inv.push(value);
        }
        *valid = false;
        matrix.try_copy_values_from_device(backend, values)?;
        backend.try_inject_preconditioner_failure(
            crate::PreconditionerFailureInjectionPoint::DiagonalUpload,
        )?;
        inverse.try_copy_values_from_host(backend, &inv)?;
        *valid = true;
        Ok(())
    }
}
