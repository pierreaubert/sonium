//! Tensor-product hexahedral partial assembly with complex symmetric quadrature physics.
//!
//! Connectivity is lexicographic (x fastest). Basis tables are row-major q-by-n.
//! Each quadrature point stores seven weighted entries: mass, xx, yy, zz, xy,
//! xz, yz. Diffusion entries include the transformed geometry metric; mass
//! includes the Jacobian determinant and quadrature weight. The caller prepares
//! this data once. No dense element matrices are retained.

// Rust guideline compliant 2026-02-21
use crate::{BackendError, BackendResult, Transpose};
use num_complex::Complex64;

/// Validated tensor basis, restriction and quadrature data for hex elements.
#[derive(Debug, Clone)]
pub struct HexPartialC64 {
    pub(crate) dimension: usize,
    pub(crate) constrained: Vec<u32>,
    pub(crate) n: usize,
    pub(crate) q: usize,
    pub(crate) nodes: Vec<u32>,
    pub(crate) basis: Vec<f64>,
    pub(crate) derivative: Vec<f64>,
    pub(crate) data: Vec<Complex64>,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) offsets: Vec<u32>,
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) slots: Vec<u32>,
}
pub(crate) fn invalid(reason: impl Into<String>) -> BackendError {
    BackendError::InvalidArgument {
        operation: "hex_partial_assembly",
        reason: reason.into(),
    }
}
impl HexPartialC64 {
    /// Prepare tensor restrictions, one-dimensional basis tables and weighted quadrature data.
    ///
    /// `nodes` contains n³ global DOFs per element; each element has q³ entries
    /// in `data`. The same basis is used in each axis. Orders n=2..8 and q=1..8
    /// are supported. Complex diffusion is symmetric, not necessarily Hermitian.
    /// Boundary facets are separate; prescribed values use `try_with_dirichlet`.
    ///
    /// # Errors
    /// Rejects invalid dimensions, counts, connectivity, non-finite data or index overflow.
    pub fn try_new(
        dimension: usize,
        n: usize,
        q: usize,
        nodes: Vec<u32>,
        basis: Vec<f64>,
        derivative: Vec<f64>,
        data: Vec<[Complex64; 7]>,
    ) -> BackendResult<Self> {
        if !(2..=8).contains(&n)
            || !(1..=8).contains(&q)
            || dimension == 0
            || dimension > u32::MAX as usize
        {
            return Err(invalid("invalid hex tensor dimensions"));
        }
        let local = n * n * n;
        let quadrature = q * q * q;
        if nodes.is_empty()
            || !nodes.len().is_multiple_of(local)
            || nodes.len() > u32::MAX as usize
            || basis.len() != n * q
            || derivative.len() != n * q
            || (nodes.len() / local).checked_mul(quadrature) != Some(data.len())
        {
            return Err(invalid("hex connectivity/basis/quadrature count mismatch"));
        }
        if basis.iter().chain(&derivative).any(|v| !v.is_finite())
            || data
                .iter()
                .flatten()
                .any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Err(invalid("hex basis and quadrature data must be finite"));
        }
        let mut offsets = vec![
            0_u32;
            dimension
                .checked_add(1)
                .ok_or_else(|| invalid("node offset overflow"))?
        ];
        for element in nodes.chunks_exact(local) {
            for (i, &node) in element.iter().enumerate() {
                if node as usize >= dimension || element[..i].contains(&node) {
                    return Err(invalid("invalid or repeated hex element DOF"));
                }
                offsets[node as usize + 1] += 1;
            }
        }
        for i in 0..dimension {
            offsets[i + 1] += offsets[i];
        }
        let mut next = offsets[..dimension].to_vec();
        let mut slots = vec![0; nodes.len()];
        for (slot, &node) in nodes.iter().enumerate() {
            slots[next[node as usize] as usize] = slot as u32;
            next[node as usize] += 1;
        }
        Ok(Self {
            dimension,
            constrained: vec![0; dimension],
            n,
            q,
            nodes,
            basis,
            derivative,
            data: data.into_iter().flatten().collect(),
            offsets,
            slots,
        })
    }
    /// Number of global degrees of freedom.
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    /// Return weighted quadrature coefficients in element-major, x-fastest order.
    ///
    /// Each point contains mass, xx, yy, zz, xy, xz, yz. Entries already include
    /// geometry transforms and integration weights; they are not raw material samples.
    pub fn quadrature_data(&self) -> &[Complex64] {
        &self.data
    }
    /// Number of hexahedral elements.
    pub fn element_count(&self) -> usize {
        self.nodes.len() / (self.n * self.n * self.n)
    }

    /// Replace prescribed-pressure DOFs, applying symmetric row and column elimination.
    ///
    /// The resulting operator is P A P + Q, with Q the prescribed-DOF projector
    /// and P = I - Q. Duplicate DOFs are harmless. Lift nonzero data separately.
    ///
    /// # Errors
    /// Rejects out-of-range DOFs.
    pub fn try_with_dirichlet(mut self, dofs: &[usize]) -> BackendResult<Self> {
        if dofs.iter().any(|&i| i >= self.dimension) {
            return Err(invalid("prescribed hex DOF outside operator"));
        }
        self.constrained.fill(0);
        for &i in dofs {
            self.constrained[i] = 1;
        }
        Ok(self)
    }

    /// Lift nonzero prescribed values into a volume RHS, preserving identity rows.
    ///
    /// Returns P(b - A Q g) + Q g. Values of g on free DOFs are ignored.
    ///
    /// # Errors
    /// Rejects incompatible vector lengths before modifying output.
    pub fn lift_rhs(
        &self,
        rhs: &[Complex64],
        prescribed: &[Complex64],
        output: &mut [Complex64],
    ) -> BackendResult<()> {
        if rhs.len() != self.dimension
            || prescribed.len() != self.dimension
            || output.len() != self.dimension
        {
            return Err(invalid("hex lifting vector length mismatch"));
        }
        self.apply_impl(prescribed, output, Transpose::NoTrans, true)?;
        for (i, value) in output.iter_mut().enumerate() {
            *value = if self.constrained[i] != 0 {
                prescribed[i]
            } else {
                rhs[i] - *value
            };
        }
        Ok(())
    }

    /// Extract the assembled diagonal directly from quadrature data during setup.
    ///
    /// This host preparation requires no dense element matrix. Retain the result
    /// for device Jacobi or smoother setup; it is not a repeated solve operation.
    pub fn diagonal(&self) -> Vec<Complex64> {
        let mut diagonal = vec![Complex64::default(); self.dimension];
        let n = self.n;
        let q = self.q;
        for (e, nodes) in self.nodes.chunks_exact(n * n * n).enumerate() {
            for (local, &node) in nodes.iter().enumerate() {
                let ip = [local % n, local / n % n, local / (n * n)];
                for point in 0..q * q * q {
                    let qp = [point % q, point / q % q, point / (q * q)];
                    let shape: [f64; 4] = std::array::from_fn(|component| {
                        (0..3)
                            .map(|axis| {
                                let table = if component == axis + 1 {
                                    &self.derivative
                                } else {
                                    &self.basis
                                };
                                table[qp[axis] * n + ip[axis]]
                            })
                            .product()
                    });
                    let d = &self.data[(e * q * q * q + point) * 7..][..7];
                    let [u, x, y, z] = shape;
                    diagonal[node as usize] += d[0] * u * u
                        + d[1] * x * x
                        + d[2] * y * y
                        + d[3] * z * z
                        + 2.0 * (d[4] * x * y + d[5] * x * z + d[6] * y * z);
                }
            }
        }
        for (i, value) in diagonal.iter_mut().enumerate() {
            if self.constrained[i] != 0 {
                *value = Complex64::new(1.0, 0.0);
            }
        }
        diagonal
    }

    /// Apply the partial-assembly operator through one-dimensional tensor contractions.
    ///
    /// # Errors
    /// Rejects incompatible input/output lengths before modifying output.
    pub fn apply(
        &self,
        x: &[Complex64],
        y: &mut [Complex64],
        action: Transpose,
    ) -> BackendResult<()> {
        self.apply_impl(x, y, action, false)
    }

    fn apply_impl(
        &self,
        x: &[Complex64],
        y: &mut [Complex64],
        action: Transpose,
        lifting: bool,
    ) -> BackendResult<()> {
        if x.len() != self.dimension || y.len() != self.dimension {
            return Err(invalid("hex vector length mismatch"));
        }
        y.fill(Complex64::default());
        let n = self.n;
        let q = self.q;
        let local = n * n * n;
        let quadrature = q * q * q;
        for (e, ids) in self.nodes.chunks_exact(local).enumerate() {
            let input: Vec<_> = ids
                .iter()
                .map(|&i| {
                    if (self.constrained[i as usize] != 0) == lifting {
                        x[i as usize]
                    } else {
                        Complex64::default()
                    }
                })
                .collect();
            let mut fields: [Vec<Complex64>; 4] = std::array::from_fn(|component| {
                let mut value = input.clone();
                let mut dims = [n; 3];
                for axis in 0..3 {
                    let table = if component == axis + 1 {
                        &self.derivative
                    } else {
                        &self.basis
                    };
                    value = contract(&value, dims, axis, q, table, false);
                    dims[axis] = q;
                }
                value
            });
            let coefficients = &self.data[e * quadrature * 7..(e + 1) * quadrature * 7];
            for (point, data) in coefficients.as_chunks::<7>().0.iter().enumerate() {
                let mut d: [Complex64; 7] = std::array::from_fn(|i| data[i]);
                if matches!(action, Transpose::ConjTrans) {
                    for v in &mut d {
                        *v = v.conj();
                    }
                }
                let [u, gx, gy, gz] = std::array::from_fn(|i| fields[i][point]);
                fields[0][point] = d[0] * u;
                fields[1][point] = d[1] * gx + d[4] * gy + d[5] * gz;
                fields[2][point] = d[4] * gx + d[2] * gy + d[6] * gz;
                fields[3][point] = d[5] * gx + d[6] * gy + d[3] * gz;
            }
            for (component, mut value) in fields.into_iter().enumerate() {
                let mut dims = [q; 3];
                for axis in (0..3).rev() {
                    let table = if component == axis + 1 {
                        &self.derivative
                    } else {
                        &self.basis
                    };
                    value = contract(&value, dims, axis, n, table, true);
                    dims[axis] = n;
                }
                for (&node, v) in ids.iter().zip(value) {
                    y[node as usize] += v;
                }
            }
        }
        if !lifting {
            for (i, value) in y.iter_mut().enumerate() {
                if self.constrained[i] != 0 {
                    *value = x[i];
                }
            }
        }
        Ok(())
    }
}
fn contract(
    input: &[Complex64],
    dims: [usize; 3],
    axis: usize,
    size: usize,
    table: &[f64],
    transpose: bool,
) -> Vec<Complex64> {
    let mut output_dims = dims;
    output_dims[axis] = size;
    let mut output = vec![Complex64::default(); output_dims.iter().product()];
    for (index, value) in output.iter_mut().enumerate() {
        let mut coord = [
            index % output_dims[0],
            index / output_dims[0] % output_dims[1],
            index / (output_dims[0] * output_dims[1]),
        ];
        let row = coord[axis];
        for j in 0..dims[axis] {
            coord[axis] = j;
            let weight = if transpose {
                table[j * size + row]
            } else {
                table[row * dims[axis] + j]
            };
            *value += weight * input[coord[0] + dims[0] * (coord[1] + dims[1] * coord[2])];
        }
    }
    output
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    // Independent nodal Lagrange tables, including unequal nodal/quadrature sizes.
    fn tables(n: usize, q: usize) -> (Vec<f64>, Vec<f64>) {
        let nodes: Vec<_> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let mut b = Vec::new();
        let mut d = Vec::new();
        for point in 0..q {
            let x = (point as f64 + 0.5) / q as f64;
            for i in 0..n {
                b.push(
                    (0..n)
                        .filter(|&j| j != i)
                        .map(|j| (x - nodes[j]) / (nodes[i] - nodes[j]))
                        .product(),
                );
                d.push(
                    (0..n)
                        .filter(|&k| k != i)
                        .map(|k| {
                            (0..n)
                                .filter(|&j| j != i && j != k)
                                .map(|j| (x - nodes[j]) / (nodes[i] - nodes[j]))
                                .product::<f64>()
                                / (nodes[i] - nodes[k])
                        })
                        .sum(),
                );
            }
        }
        (b, d)
    }

    pub(crate) fn fixture(n: usize, q: usize) -> HexPartialC64 {
        let width = 2 * n - 1;
        let mut nodes = Vec::new();
        for e in 0..2 {
            for k in 0..n {
                for j in 0..n {
                    for i in 0..n {
                        nodes.push((e * (n - 1) + i + width * (j + n * k)) as u32);
                    }
                }
            }
        }
        let (b, d) = tables(n, q);
        let data = (0..2 * q * q * q)
            .map(|p| {
                std::array::from_fn(|i| {
                    Complex64::new(0.2 + (p + i) as f64 * 0.01, 0.03 * (i + 1) as f64)
                })
            })
            .collect();
        HexPartialC64::try_new(width * n * n, n, q, nodes, b, d, data).unwrap()
    }

    // Direct quadrature evaluation uses no contraction implementation or scratch layout.
    fn reference(op: &HexPartialC64, x: &[Complex64], adjoint: bool) -> Vec<Complex64> {
        let mut y = vec![Complex64::default(); op.dimension];
        let n = op.n;
        let q = op.q;
        for (e, nodes) in op.nodes.chunks_exact(n * n * n).enumerate() {
            for point in 0..q * q * q {
                let qp = [point % q, point / q % q, point / (q * q)];
                let shapes: Vec<[f64; 4]> = (0..n * n * n)
                    .map(|i| {
                        let ip = [i % n, i / n % n, i / (n * n)];
                        std::array::from_fn(|component| {
                            (0..3)
                                .map(|axis| {
                                    let table = if component == axis + 1 {
                                        &op.derivative
                                    } else {
                                        &op.basis
                                    };
                                    table[qp[axis] * n + ip[axis]]
                                })
                                .product()
                        })
                    })
                    .collect();
                let field: [Complex64; 4] = std::array::from_fn(|c| {
                    nodes
                        .iter()
                        .zip(&shapes)
                        .map(|(&node, s)| x[node as usize] * s[c])
                        .sum()
                });
                let d: [Complex64; 7] = std::array::from_fn(|i| {
                    let v = op.data[(e * q * q * q + point) * 7 + i];
                    if adjoint { v.conj() } else { v }
                });
                let flux = [
                    d[0] * field[0],
                    d[1] * field[1] + d[4] * field[2] + d[5] * field[3],
                    d[4] * field[1] + d[2] * field[2] + d[6] * field[3],
                    d[5] * field[1] + d[6] * field[2] + d[3] * field[3],
                ];
                for (&node, s) in nodes.iter().zip(&shapes) {
                    y[node as usize] += (0..4).map(|c| s[c] * flux[c]).sum::<Complex64>();
                }
            }
        }
        y
    }

    #[test]
    fn sum_factorization_matches_independent_quadrature() {
        for (n, q) in [(2, 3), (3, 2), (4, 5), (8, 1), (2, 8)] {
            let op = fixture(n, q);
            let x: Vec<_> = (0..op.dimension)
                .map(|i| Complex64::new((i as f64 * 0.3).sin(), (i as f64 * 0.2).cos()))
                .collect();
            for action in [Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans] {
                let expected = reference(&op, &x, matches!(action, Transpose::ConjTrans));
                let mut actual = vec![Complex64::default(); op.dimension];
                op.apply(&x, &mut actual, action).unwrap();
                for (a, b) in actual.iter().zip(expected) {
                    assert!(
                        (*a - b).norm() < 2e-11 * (1.0 + b.norm()),
                        "n={n}, q={q}: {a} != {b}"
                    );
                }
            }
        }
    }

    #[test]
    fn diffusion_preserves_constant_nullspace() {
        let mut op = fixture(4, 5);
        for d in op.data.as_chunks_mut::<7>().0 {
            d[0] = Complex64::default();
        }
        let x = vec![Complex64::new(1.0, 0.5); op.dimension];
        let mut y = x.clone();
        op.apply(&x, &mut y, Transpose::NoTrans).unwrap();
        assert!(y.iter().all(|v| v.norm() < 1e-10));
    }

    #[test]
    fn rejects_invalid_input_before_changing_output() {
        let op = fixture(2, 3);
        let mut y = vec![Complex64::new(3.0, 4.0); op.dimension];
        let before = y.clone();
        assert!(op.apply(&[], &mut y, Transpose::NoTrans).is_err());
        assert_eq!(before, y);
        for case in 0..6 {
            let mut nodes = op.nodes.clone();
            let mut b = op.basis.clone();
            let mut d = op.derivative.clone();
            let mut data: Vec<[Complex64; 7]> =
                op.data.as_chunks::<7>().0.iter().copied().collect();
            match case {
                0 => nodes[0] = nodes[1],
                1 => nodes[0] = op.dimension as u32,
                2 => b[0] = f64::NAN,
                3 => {
                    d.pop();
                }
                4 => data[0][0].im = f64::INFINITY,
                _ => {
                    data.pop();
                }
            }
            assert!(HexPartialC64::try_new(op.dimension, 2, 3, nodes, b, d, data).is_err());
        }
    }
    #[test]
    fn dirichlet_projection_and_lifting_match_unconstrained_quadrature() {
        for (n, q) in [(2, 3), (3, 4), (4, 5)] {
            let volume = fixture(n, q);
            let size = volume.dimension();
            let x: Vec<_> = (0..size)
                .map(|i| Complex64::new((i as f64 * 0.3).sin(), 0.2))
                .collect();
            for dofs in [vec![], vec![0, size / 2, size - 1, 0], (0..size).collect()] {
                let op = volume.clone().try_with_dirichlet(&dofs).unwrap();
                let mut projected = x.clone();
                for &i in &dofs {
                    projected[i] = Complex64::default();
                }
                for action in [Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans] {
                    let mut expected =
                        reference(&volume, &projected, matches!(action, Transpose::ConjTrans));
                    for &i in &dofs {
                        expected[i] = x[i];
                    }
                    let mut actual = vec![Complex64::default(); size];
                    op.apply(&x, &mut actual, action).unwrap();
                    for (a, b) in actual.iter().zip(expected) {
                        assert!((*a - b).norm() < 2e-11 * (1.0 + b.norm()));
                    }
                }
                let mut g = vec![Complex64::new(f64::NAN, f64::NAN); size];
                let mut qg = vec![Complex64::default(); size];
                for &i in &dofs {
                    g[i] = x[i];
                    qg[i] = x[i];
                }
                let aqg = reference(&volume, &qg, false);
                let mut actual = vec![Complex64::default(); size];
                op.lift_rhs(&x, &g, &mut actual).unwrap();
                for i in 0..size {
                    let expected = if dofs.contains(&i) {
                        x[i]
                    } else {
                        x[i] - aqg[i]
                    };
                    assert!((actual[i] - expected).norm() < 2e-11 * (1.0 + expected.norm()));
                }
                let diagonal = op.diagonal();
                for &i in &dofs {
                    assert_eq!(diagonal[i], Complex64::new(1.0, 0.0));
                }
                let before = actual.clone();
                assert!(op.lift_rhs(&[], &g, &mut actual).is_err());
                assert_eq!(actual, before);
            }
            assert!(volume.try_with_dirichlet(&[size]).is_err());
        }
    }
}
