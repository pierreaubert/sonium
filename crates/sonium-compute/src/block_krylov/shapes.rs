//! Retained exact-width block buffers for opt-in deflation reuse.

// Rust guideline compliant 2026-02-21
use super::*;

pub(super) struct BlockShape<B: Backend, T: DeviceScalar> {
    pub(super) block_len: usize,
    resident_block_len: usize,
    block_x: Option<B::Buffer<T>>,
    block_v: Option<B::Buffer<T>>,
    block_z: Option<B::Buffer<T>>,
    block_w: Option<B::Buffer<T>>,
    block_ax: Option<B::Buffer<T>>,
    block_b: Option<B::Buffer<T>>,
    block_x_initial: Option<B::Buffer<T>>,
    block_residual: Option<B::Buffer<T>>,
    resident_v_flat: Option<B::Buffer<T>>,
    resident_z_flat: Option<B::Buffer<T>>,
    resident_projection: Option<B::Buffer<T>>,
    resident_coefficients: Option<B::Buffer<T>>,
    resident_basis: Vec<B::Buffer<T>>,
    resident_z: Vec<B::Buffer<T>>,
}

impl<B: Backend, T: DeviceScalar> BlockShape<B, T> {
    pub(super) fn buffer_count(&self) -> usize {
        [
            self.block_x.as_ref(),
            self.block_v.as_ref(),
            self.block_z.as_ref(),
            self.block_w.as_ref(),
            self.block_ax.as_ref(),
            self.block_b.as_ref(),
            self.block_x_initial.as_ref(),
            self.block_residual.as_ref(),
            self.resident_v_flat.as_ref(),
            self.resident_z_flat.as_ref(),
            self.resident_projection.as_ref(),
            self.resident_coefficients.as_ref(),
        ]
        .into_iter()
        .flatten()
        .count()
            + self.resident_basis.len()
            + self.resident_z.len()
    }

    pub(super) fn capacity(&self) -> usize {
        [
            self.block_x.as_ref(),
            self.block_v.as_ref(),
            self.block_z.as_ref(),
            self.block_w.as_ref(),
            self.block_ax.as_ref(),
            self.block_b.as_ref(),
            self.block_x_initial.as_ref(),
            self.block_residual.as_ref(),
            self.resident_v_flat.as_ref(),
            self.resident_z_flat.as_ref(),
            self.resident_projection.as_ref(),
            self.resident_coefficients.as_ref(),
        ]
        .into_iter()
        .flatten()
        .chain(self.resident_basis.iter())
        .chain(self.resident_z.iter())
        .map(DeviceBufferLike::len)
        .sum()
    }
}

impl<B: Backend, T: DeviceScalar> BlockGmresWorkspace<B, T> {
    pub(super) fn take_block_shape(&mut self) -> BlockShape<B, T> {
        BlockShape {
            block_len: std::mem::replace(&mut self.block_len, 0),
            resident_block_len: std::mem::replace(&mut self.resident_block_len, 0),
            block_x: self.block_x.take(),
            block_v: self.block_v.take(),
            block_z: self.block_z.take(),
            block_w: self.block_w.take(),
            block_ax: self.block_ax.take(),
            block_b: self.block_b.take(),
            block_x_initial: self.block_x_initial.take(),
            block_residual: self.block_residual.take(),
            resident_v_flat: self.resident_v_flat.take(),
            resident_z_flat: self.resident_z_flat.take(),
            resident_projection: self.resident_projection.take(),
            resident_coefficients: self.resident_coefficients.take(),
            resident_basis: std::mem::take(&mut self.resident_basis),
            resident_z: std::mem::take(&mut self.resident_z),
        }
    }

    pub(super) fn install_block_shape(&mut self, shape: BlockShape<B, T>) {
        self.block_len = shape.block_len;
        self.resident_block_len = shape.resident_block_len;
        self.block_x = shape.block_x;
        self.block_v = shape.block_v;
        self.block_z = shape.block_z;
        self.block_w = shape.block_w;
        self.block_ax = shape.block_ax;
        self.block_b = shape.block_b;
        self.block_x_initial = shape.block_x_initial;
        self.block_residual = shape.block_residual;
        self.resident_v_flat = shape.resident_v_flat;
        self.resident_z_flat = shape.resident_z_flat;
        self.resident_projection = shape.resident_projection;
        self.resident_coefficients = shape.resident_coefficients;
        self.resident_basis = shape.resident_basis;
        self.resident_z = shape.resident_z;
    }
}
