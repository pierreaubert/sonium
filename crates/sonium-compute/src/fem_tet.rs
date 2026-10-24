//! Affine tetrahedral P1/P2 mass/diffusion actions with complex element coefficients.
//!
//! The operator is the sum of `diffusion[e] * K_e + mass[e] * M_e`.
//! Geometry is prepared once on the host. No dense element matrices are retained.
//! Optional nodal Dirichlet elimination preserves symmetry and supports RHS
//! lifting. Boundary facets and curved geometry remain separate.

// Rust guideline compliant 2026-02-21
use crate::{BackendError, BackendResult, Transpose};
use num_complex::Complex64;

mod assembly;
#[doc(inline)]
pub use assembly::TetCsrPattern;

/// Affine P1 tetrahedral acoustic operator.
pub type TetP1AcousticC64 = TetAcousticC64<4>;
/// Affine P2 tetrahedral acoustic operator with canonical edge DOFs.
pub type TetP2AcousticC64 = TetAcousticC64<10>;

/// Validated affine tetrahedra with four (P1) or ten (P2) local DOFs.
#[derive(Debug, Clone)]
pub struct TetAcousticC64<const N: usize> {
    pub(crate) n: usize,
    pub(crate) dirichlet: Vec<u32>,
    pub(crate) nodes: Vec<u32>,
    /// Per element: volume, then four physical basis gradients (xyz).
    pub(crate) geometry: Vec<f64>,
    pub(crate) offsets: Vec<u32>,
    pub(crate) slots: Vec<u32>,
    /// Per element: diffusion, mass.
    pub(crate) coefficients: Vec<Complex64>,
}

fn invalid(reason: impl Into<String>) -> BackendError {
    BackendError::InvalidArgument {
        operation: "tet_acoustic",
        reason: reason.into(),
    }
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.into_iter().zip(b).map(|(x, y)| x * y).sum()
}

// Normalized barycentric moments: integral(prod l_i^a_i)/V =
// 6*prod(a_i!)/(3+sum(a_i))!. These exact P2 mass entries are shared by
// every affine element, rather than storing dense element tensors.
pub(crate) const P2_MASS: [[f64; 10]; 10] = [
    [
        36.0 / 2520.0,
        6.0 / 2520.0,
        6.0 / 2520.0,
        6.0 / 2520.0,
        -24.0 / 2520.0,
        -24.0 / 2520.0,
        -24.0 / 2520.0,
        -36.0 / 2520.0,
        -36.0 / 2520.0,
        -36.0 / 2520.0,
    ],
    [
        6.0 / 2520.0,
        36.0 / 2520.0,
        6.0 / 2520.0,
        6.0 / 2520.0,
        -24.0 / 2520.0,
        -36.0 / 2520.0,
        -36.0 / 2520.0,
        -24.0 / 2520.0,
        -24.0 / 2520.0,
        -36.0 / 2520.0,
    ],
    [
        6.0 / 2520.0,
        6.0 / 2520.0,
        36.0 / 2520.0,
        6.0 / 2520.0,
        -36.0 / 2520.0,
        -24.0 / 2520.0,
        -36.0 / 2520.0,
        -24.0 / 2520.0,
        -36.0 / 2520.0,
        -24.0 / 2520.0,
    ],
    [
        6.0 / 2520.0,
        6.0 / 2520.0,
        6.0 / 2520.0,
        36.0 / 2520.0,
        -36.0 / 2520.0,
        -36.0 / 2520.0,
        -24.0 / 2520.0,
        -36.0 / 2520.0,
        -24.0 / 2520.0,
        -24.0 / 2520.0,
    ],
    [
        -24.0 / 2520.0,
        -24.0 / 2520.0,
        -36.0 / 2520.0,
        -36.0 / 2520.0,
        192.0 / 2520.0,
        96.0 / 2520.0,
        96.0 / 2520.0,
        96.0 / 2520.0,
        96.0 / 2520.0,
        48.0 / 2520.0,
    ],
    [
        -24.0 / 2520.0,
        -36.0 / 2520.0,
        -24.0 / 2520.0,
        -36.0 / 2520.0,
        96.0 / 2520.0,
        192.0 / 2520.0,
        96.0 / 2520.0,
        96.0 / 2520.0,
        48.0 / 2520.0,
        96.0 / 2520.0,
    ],
    [
        -24.0 / 2520.0,
        -36.0 / 2520.0,
        -36.0 / 2520.0,
        -24.0 / 2520.0,
        96.0 / 2520.0,
        96.0 / 2520.0,
        192.0 / 2520.0,
        48.0 / 2520.0,
        96.0 / 2520.0,
        96.0 / 2520.0,
    ],
    [
        -36.0 / 2520.0,
        -24.0 / 2520.0,
        -24.0 / 2520.0,
        -36.0 / 2520.0,
        96.0 / 2520.0,
        96.0 / 2520.0,
        48.0 / 2520.0,
        192.0 / 2520.0,
        96.0 / 2520.0,
        96.0 / 2520.0,
    ],
    [
        -36.0 / 2520.0,
        -24.0 / 2520.0,
        -36.0 / 2520.0,
        -24.0 / 2520.0,
        96.0 / 2520.0,
        48.0 / 2520.0,
        96.0 / 2520.0,
        96.0 / 2520.0,
        192.0 / 2520.0,
        96.0 / 2520.0,
    ],
    [
        -36.0 / 2520.0,
        -36.0 / 2520.0,
        -24.0 / 2520.0,
        -24.0 / 2520.0,
        48.0 / 2520.0,
        96.0 / 2520.0,
        96.0 / 2520.0,
        96.0 / 2520.0,
        96.0 / 2520.0,
        192.0 / 2520.0,
    ],
];

const P2_EDGES: [(usize, usize); 6] = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
fn p2_gradient_coefficients(i: usize, g: &[f64]) -> [[f64; 3]; 4] {
    std::array::from_fn(|a| {
        std::array::from_fn(|d| {
            if i < 4 {
                (if a == i { 3.0 } else { -1.0 }) * g[1 + 3 * i + d]
            } else {
                let (u, v) = P2_EDGES[i - 4];
                4.0 * ((if a == u { g[1 + 3 * v + d] } else { 0.0 })
                    + (if a == v { g[1 + 3 * u + d] } else { 0.0 }))
            }
        })
    })
}

impl<const N: usize> TetAcousticC64<N> {
    /// Replace the prescribed-node set while retaining the volume geometry.
    ///
    /// The resulting action is `P*A*P + Q`, where P selects free nodes and Q
    /// selects prescribed nodes. Values are supplied separately to `lift_rhs`,
    /// allowing repeated RHS without rebuilding the operator. Duplicate indices
    /// are accepted and coalesced.
    ///
    /// # Errors
    /// Rejects out-of-range node indices before changing the constraint mask.
    pub fn try_with_dirichlet(mut self, dofs: &[usize]) -> BackendResult<Self> {
        if dofs.iter().any(|&i| i >= self.n) {
            return Err(invalid("Dirichlet node index out of range"));
        }
        self.dirichlet.fill(0);
        for &i in dofs {
            self.dirichlet[i] = 1;
        }
        Ok(self)
    }

    /// Lift prescribed values into the unconstrained RHS and set identity rows.
    ///
    /// `prescribed` has one entry per node; values at free nodes are ignored.
    /// Computes `P*(b - A*Q*prescribed) + Q*prescribed` with the original volume
    /// operator. Call once for each new RHS or changed boundary values.
    ///
    /// # Errors
    /// Rejects mismatched vector lengths before changing output.
    pub fn lift_rhs(
        &self,
        b: &[Complex64],
        prescribed: &[Complex64],
        out: &mut [Complex64],
    ) -> BackendResult<()> {
        if b.len() != self.n || prescribed.len() != self.n || out.len() != self.n {
            return Err(invalid(
                "RHS/prescribed/output length must equal node count",
            ));
        }
        self.apply_impl(prescribed, out, Transpose::NoTrans, true)?;
        for i in 0..self.n {
            out[i] = if self.dirichlet[i] != 0 {
                prescribed[i]
            } else {
                b[i] - out[i]
            };
        }
        Ok(())
    }

    /// Assemble the volume diagonal without constructing element or global matrices.
    ///
    /// Prescribed nodes have unit diagonal. Unconstrained isolated nodes retain
    /// zero entries. Facet terms are excluded.
    #[must_use]
    pub fn diagonal(&self) -> Vec<Complex64> {
        let mut diagonal = vec![Complex64::new(0.0, 0.0); self.n];
        for e in 0..self.element_count() {
            let geometry = &self.geometry[e * 13..e * 13 + 13];
            if N == 10 {
                for (i, mass_row) in P2_MASS.iter().enumerate() {
                    let g = p2_gradient_coefficients(i, geometry);
                    let sum: [f64; 3] = std::array::from_fn(|d| g.iter().map(|v| v[d]).sum());
                    let stiffness: f64 = g
                        .iter()
                        .map(|v| (0..3).map(|d| v[d] * (v[d] + sum[d])).sum::<f64>())
                        .sum();
                    diagonal[self.nodes[e * N + i] as usize] += self.coefficients[2 * e]
                        * (geometry[0] * stiffness / 20.0)
                        + self.coefficients[2 * e + 1] * (geometry[0] * mass_row[i]);
                }
                continue;
            }
            for i in 0..4 {
                let grad = &geometry[1 + 3 * i..1 + 3 * i + 3];
                let stiffness = geometry[0] * grad.iter().map(|g| g * g).sum::<f64>();
                // The consistent P1 mass diagonal is twice its off-diagonal: V/10.
                diagonal[self.nodes[e * N + i] as usize] += self.coefficients[2 * e] * stiffness
                    + self.coefficients[2 * e + 1] * (geometry[0] / 10.0);
            }
        }
        for (i, value) in diagonal.iter_mut().enumerate() {
            if self.dirichlet[i] != 0 {
                *value = Complex64::new(1.0, 0.0);
            }
        }
        diagonal
    }

    /// Prepare geometry, adjacency and complex mass/diffusion coefficients.
    ///
    /// Each element lists four vertices, followed for P2 by edge DOFs in order
    /// (01, 02, 03, 12, 13, 23). P2 coordinates must be edge midpoints. Either orientation is
    /// accepted. Coefficients are constant within each element; for Helmholtz,
    /// callers typically include the negative squared wavenumber in `mass`.
    ///
    /// # Errors
    /// Rejects empty meshes, inconsistent coefficient lengths, invalid or repeated
    /// node indices, non-finite data, degenerate geometry and index overflow.
    pub fn try_new(
        vertices: &[[f64; 3]],
        elements: &[[usize; N]],
        diffusion: &[Complex64],
        mass: &[Complex64],
    ) -> BackendResult<Self> {
        if N != 4 && N != 10 {
            return Err(invalid("only P1/P2 tetrahedra are supported"));
        }
        let n = vertices.len();
        let count = elements
            .len()
            .checked_mul(N)
            .ok_or_else(|| invalid("element count overflow"))?;
        if n == 0
            || elements.is_empty()
            || u32::try_from(n).is_err()
            || u32::try_from(count).is_err()
        {
            return Err(invalid(
                "nonempty mesh and 32-bit node/contribution counts required",
            ));
        }
        if diffusion.len() != elements.len() || mass.len() != elements.len() {
            return Err(invalid(
                "one diffusion and mass coefficient per element required",
            ));
        }
        if vertices.iter().flatten().any(|x| !x.is_finite())
            || diffusion
                .iter()
                .chain(mass)
                .any(|x| !x.re.is_finite() || !x.im.is_finite())
        {
            return Err(invalid("geometry and coefficients must be finite"));
        }
        let geometry_len = elements
            .len()
            .checked_mul(13)
            .ok_or_else(|| invalid("geometry size overflow"))?;
        let mut result = Self {
            n,
            dirichlet: vec![0; n],
            nodes: Vec::with_capacity(count),
            geometry: Vec::with_capacity(geometry_len),
            offsets: vec![
                0;
                n.checked_add(1)
                    .ok_or_else(|| invalid("node offset count overflow"))?
            ],
            slots: vec![0; count],
            coefficients: Vec::with_capacity(elements.len() * 2),
        };
        for (e, ids) in elements.iter().enumerate() {
            if ids.iter().any(|&i| i >= n) || (0..N).any(|i| ids[..i].contains(&ids[i])) {
                return Err(invalid("tetrahedron must reference distinct in-range DOFs"));
            }
            if N == 10 {
                for (edge, &(a, b)) in P2_EDGES.iter().enumerate() {
                    for ((&left, &right), &actual) in vertices[ids[a]]
                        .iter()
                        .zip(&vertices[ids[b]])
                        .zip(&vertices[ids[4 + edge]])
                    {
                        let expected = left * 0.5 + right * 0.5;
                        let scale = left.abs().max(right.abs());
                        // Allow midpoint arithmetic rounding without accepting
                        // geometry perturbations merely because the mesh is tiny.
                        if (actual - expected).abs() > 64.0 * f64::EPSILON * scale {
                            return Err(invalid(
                                "P2 edge DOFs must follow affine midpoint geometry",
                            ));
                        }
                    }
                }
            }
            let edges: [[f64; 3]; 3] = std::array::from_fn(|i| {
                std::array::from_fn(|d| vertices[ids[i + 1]][d] - vertices[ids[0]][d])
            });
            let cofactors = [
                cross(edges[1], edges[2]),
                cross(edges[2], edges[0]),
                cross(edges[0], edges[1]),
            ];
            let determinant = dot(edges[0], cofactors[0]);
            // The reference tetrahedron has volume 1/6. Orientation changes
            // the basis transform but never the integration measure.
            let volume = determinant.abs() / 6.0;
            if !determinant.is_finite() || !volume.is_finite() || volume <= 0.0 {
                return Err(invalid("degenerate or non-finite tetrahedron"));
            }
            let mut gradients = [[0.0; 3]; 4];
            for i in 0..3 {
                for d in 0..3 {
                    gradients[i + 1][d] = cofactors[i][d] / determinant;
                    gradients[0][d] -= gradients[i + 1][d];
                }
            }
            if gradients.iter().flatten().any(|x| !x.is_finite()) {
                return Err(invalid("non-finite physical basis gradient"));
            }
            result.geometry.push(volume);
            result.geometry.extend(gradients.into_iter().flatten());
            result.coefficients.extend([diffusion[e], mass[e]]);
            for &id in ids {
                result.nodes.push(id as u32);
                result.offsets[id + 1] += 1;
            }
        }
        for i in 0..n {
            result.offsets[i + 1] += result.offsets[i];
        }
        let mut next = result.offsets[..n].to_vec();
        for (slot, &node) in result.nodes.iter().enumerate() {
            result.slots[next[node as usize] as usize] = slot as u32;
            next[node as usize] += 1;
        }
        Ok(result)
    }

    /// Number of global nodal degrees of freedom.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.n
    }

    /// Number of affine tetrahedra.
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.nodes.len() / N
    }

    /// Apply the volume operator on host slices without forming element matrices.
    ///
    /// Algebraic transpose equals forward for these symmetric local forms.
    /// Hermitian action conjugates both complex element coefficients.
    ///
    /// # Errors
    /// Rejects incompatible vector lengths before modifying output.
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
        if x.len() != self.n || y.len() != self.n {
            return Err(invalid("input/output length must equal node count"));
        }
        let input = |i: usize| {
            if (self.dirichlet[i] != 0) == lifting {
                x[i]
            } else {
                Complex64::new(0.0, 0.0)
            }
        };
        y.fill(Complex64::new(0.0, 0.0));
        for e in 0..self.element_count() {
            let geometry = &self.geometry[e * 13..e * 13 + 13];
            if N == 10 {
                let values: [Complex64; 10] =
                    std::array::from_fn(|i| input(self.nodes[e * N + i] as usize));
                let gradients: [[[f64; 3]; 4]; 10] =
                    std::array::from_fn(|i| p2_gradient_coefficients(i, geometry));
                let mut gx = [[Complex64::new(0.0, 0.0); 3]; 4];
                for i in 0..10 {
                    for a in 0..4 {
                        for d in 0..3 {
                            gx[a][d] += values[i] * gradients[i][a][d];
                        }
                    }
                }
                let sum: [Complex64; 3] = std::array::from_fn(|d| gx.iter().map(|v| v[d]).sum());
                let mut diffusion = self.coefficients[2 * e];
                let mut mass = self.coefficients[2 * e + 1];
                if matches!(action, Transpose::ConjTrans) {
                    diffusion = diffusion.conj();
                    mass = mass.conj();
                }
                for i in 0..10 {
                    let mut stiffness = Complex64::new(0.0, 0.0);
                    for a in 0..4 {
                        for d in 0..3 {
                            stiffness += gradients[i][a][d] * (gx[a][d] + sum[d]);
                        }
                    }
                    let mass_action: Complex64 = (0..10).map(|j| values[j] * P2_MASS[i][j]).sum();
                    y[self.nodes[e * N + i] as usize] += diffusion
                        * (geometry[0] * stiffness / 20.0)
                        + mass * (geometry[0] * mass_action);
                }
                continue;
            }
            let mut gradient = [Complex64::new(0.0, 0.0); 3];
            let mut sum = Complex64::new(0.0, 0.0);
            for i in 0..4 {
                let value = input(self.nodes[e * N + i] as usize);
                sum += value;
                for d in 0..3 {
                    gradient[d] += value * geometry[1 + 3 * i + d];
                }
            }
            let mut diffusion = self.coefficients[2 * e];
            let mut mass = self.coefficients[2 * e + 1];
            if matches!(action, Transpose::ConjTrans) {
                diffusion = diffusion.conj();
                mass = mass.conj();
            }
            for i in 0..4 {
                let node = self.nodes[e * N + i] as usize;
                let stiffness: Complex64 =
                    (0..3).map(|d| gradient[d] * geometry[1 + 3 * i + d]).sum();
                // Exact consistent P1 mass: M_ij = volume * (1 + delta_ij) / 20.
                y[node] += diffusion * (geometry[0] * stiffness)
                    + mass * (geometry[0] / 20.0 * (sum + input(node)));
            }
        }
        if !lifting {
            for i in 0..self.n {
                if self.dirichlet[i] != 0 {
                    y[i] = x[i];
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn p2_affine_energy_nullspace_and_midpoint_validation() {
        let mut vertices = vec![
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [0.5, 0., 0.],
            [0., 0.5, 0.],
            [0., 0., 0.5],
            [0.5, 0.5, 0.],
            [0.5, 0., 0.5],
            [0., 0.5, 0.5],
        ];
        let elements = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]];
        let diffusion = [Complex64::new(1., 0.)];
        let mass = [Complex64::new(0., 0.)];
        let op = TetP2AcousticC64::try_new(&vertices, &elements, &diffusion, &mass).unwrap();
        let constant = vec![Complex64::new(1., 0.); 10];
        let mut out = vec![Complex64::default(); 10];
        op.apply(&constant, &mut out, Transpose::NoTrans).unwrap();
        assert!(out.iter().all(|v| v.norm() < 1e-14));
        let affine: Vec<_> = vertices
            .iter()
            .map(|p| Complex64::new(2. * p[0] - 3. * p[1] + p[2], 0.))
            .collect();
        op.apply(&affine, &mut out, Transpose::NoTrans).unwrap();
        let energy: Complex64 = affine.iter().zip(&out).map(|(x, y)| x.conj() * y).sum();
        assert!((energy.re - 14.0 / 6.0).abs() < 1e-13);
        vertices[4][1] += 0.01;
        assert!(TetP2AcousticC64::try_new(&vertices, &elements, &diffusion, &mass).is_err());
        assert!(TetAcousticC64::<0>::try_new(&vertices, &[[]], &diffusion, &mass).is_err());
    }

    #[test]
    fn dirichlet_empty_full_and_invalid_sets() {
        let volume = fixture(&[[0, 1, 2, 3], [1, 2, 3, 4]]);
        let b = vec![Complex64::new(1.0, -0.2); 5];
        let prescribed = vec![Complex64::new(0.7, 0.3); 5];
        let mut out = b.clone();
        volume.lift_rhs(&b, &prescribed, &mut out).unwrap();
        assert_eq!(out, b);
        assert!(volume.clone().try_with_dirichlet(&[5]).is_err());
        let all = volume
            .clone()
            .try_with_dirichlet(&[0, 1, 2, 3, 4, 4])
            .unwrap();
        for action in [Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans] {
            all.apply(&b, &mut out, action).unwrap();
            assert_eq!(out, b);
        }
        all.lift_rhs(&b, &prescribed, &mut out).unwrap();
        assert_eq!(out, prescribed);
        let sentinel = out.clone();
        assert!(all.lift_rhs(&b[..4], &prescribed, &mut out).is_err());
        assert_eq!(out, sentinel);
        let reset = all.try_with_dirichlet(&[]).unwrap();
        assert_eq!(reset.diagonal(), volume.diagonal());
    }

    fn fixture(elements: &[[usize; 4]]) -> TetP1AcousticC64 {
        TetP1AcousticC64::try_new(
            &[
                [0., 0., 0.],
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
                [1., 1., 1.],
            ],
            elements,
            &[Complex64::new(1.2, 0.4), Complex64::new(0.7, -0.2)],
            &[Complex64::new(-2.0, 0.3), Complex64::new(-1.0, 0.8)],
        )
        .unwrap()
    }
    /// Independent dense assembly using exact physical gradients and a degree-2
    /// quadrature rule for the mass integral, rather than the fused formula.
    fn assembled() -> [[Complex64; 5]; 5] {
        let mut a = [[Complex64::new(0., 0.); 5]; 5];
        let gradients = [
            [[-1., -1., -1.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
            [
                [0.5, -0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
            ],
        ];
        let elements = [[0, 1, 2, 3], [1, 2, 3, 4]];
        let high = (5.0 + 3.0 * 5.0_f64.sqrt()) / 20.0;
        let low = (5.0 - 5.0_f64.sqrt()) / 20.0;
        for e in 0..2 {
            let volume = [1.0 / 6.0, 1.0 / 3.0][e];
            let diffusion = [Complex64::new(1.2, 0.4), Complex64::new(0.7, -0.2)][e];
            let mass = [Complex64::new(-2.0, 0.3), Complex64::new(-1.0, 0.8)][e];
            for i in 0..4 {
                for j in 0..4 {
                    let k: f64 = (0..3)
                        .map(|d| gradients[e][i][d] * gradients[e][j][d] * volume)
                        .sum();
                    let m: f64 = (0..4)
                        .map(|q| {
                            (if i == q { high } else { low })
                                * (if j == q { high } else { low })
                                * volume
                                / 4.0
                        })
                        .sum();
                    a[elements[e][i]][elements[e][j]] += diffusion * k + mass * m;
                }
            }
        }
        a
    }
    #[test]
    fn matches_quadrature_assembly_with_shared_nodes_and_reversed_orientation() {
        let x = [
            Complex64::new(1., 2.),
            Complex64::new(-1., 0.5),
            Complex64::new(0.1, -0.7),
            Complex64::new(3., 1.),
            Complex64::new(-2., 1.4),
        ];
        let matrix = assembled();
        for elements in [[[0, 1, 2, 3], [1, 2, 3, 4]], [[0, 2, 1, 3], [1, 3, 2, 4]]] {
            let operator = fixture(&elements);
            for action in [Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans] {
                let mut y = [Complex64::new(0., 0.); 5];
                operator.apply(&x, &mut y, action).unwrap();
                for i in 0..5 {
                    let expected: Complex64 = (0..5)
                        .map(|j| {
                            (if matches!(action, Transpose::ConjTrans) {
                                matrix[j][i].conj()
                            } else {
                                matrix[i][j]
                            }) * x[j]
                        })
                        .sum();
                    assert!(
                        (y[i] - expected).norm() < 2e-14,
                        "row {i}: {} != {expected}",
                        y[i]
                    );
                }
            }
        }
    }
    #[test]
    fn affine_energy_and_constant_nullspace() {
        let vertices = [
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 1., 1.],
        ];
        let operator = TetP1AcousticC64::try_new(
            &vertices,
            &[[0, 1, 2, 3], [1, 2, 3, 4]],
            &[Complex64::new(1., 0.); 2],
            &[Complex64::new(0., 0.); 2],
        )
        .unwrap();
        let x: Vec<_> = vertices
            .iter()
            .map(|v| Complex64::new(2. * v[0] - 3. * v[1] + v[2], 0.))
            .collect();
        let mut y = vec![Complex64::new(0., 0.); 5];
        operator.apply(&x, &mut y, Transpose::NoTrans).unwrap();
        let energy: Complex64 = x.iter().zip(&y).map(|(a, b)| a.conj() * b).sum();
        assert!((energy.re - 7.).abs() < 1e-14);
        operator
            .apply(&[Complex64::new(1., 0.); 5], &mut y, Transpose::NoTrans)
            .unwrap();
        assert!(y.iter().all(|v| v.norm() < 1e-14));
    }
    #[test]
    fn invalid_geometry_and_lengths_fail_before_output_changes() {
        let vertices = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
        let one = [Complex64::new(1., 0.)];
        for element in [[0, 0, 2, 3], [0, 1, 2, 4]] {
            assert!(TetP1AcousticC64::try_new(&vertices, &[element], &one, &one).is_err());
        }
        let mut flat = vertices;
        flat[3] = [1., 1., 0.];
        assert!(TetP1AcousticC64::try_new(&flat, &[[0, 1, 2, 3]], &one, &one).is_err());
        assert!(TetP1AcousticC64::try_new(&vertices, &[[0, 1, 2, 3]], &[], &one).is_err());
        assert!(
            TetP1AcousticC64::try_new(
                &vertices,
                &[[0, 1, 2, 3]],
                &[Complex64::new(f64::NAN, 0.)],
                &one
            )
            .is_err()
        );
        let operator = fixture(&[[0, 1, 2, 3], [1, 2, 3, 4]]);
        let mut output = [Complex64::new(7., 8.); 5];
        assert!(
            operator
                .apply(&one, &mut output, Transpose::NoTrans)
                .is_err()
        );
        assert_eq!(output, [Complex64::new(7., 8.); 5]);
    }
}
