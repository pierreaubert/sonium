//! Element shape functions and geometry computations

use crate::core::array::array;
use crate::core::types::{BasisOrder, Element, ElementType};
use sonium_solvers::dense::DenseMatrix;

/// Shape functions for triangular and quadrilateral elements
impl Element {
    /// Compute shape functions at local coordinates (s, t)
    ///
    /// For triangles: Area coordinates with standard vertex mapping:
    /// - (0,0) -> node 0
    /// - (1,0) -> node 1
    /// - (0,1) -> node 2
    ///
    /// For quads: Bilinear functions on [-1,1]²
    pub fn shape_functions(&self, s: f64, t: f64) -> Vec<f64> {
        match self.element_type {
            ElementType::Tri3 => {
                // Triangle shape functions:
                // N0 = 1-s-t (vertex at (0,0))
                // N1 = s (vertex at (1,0))
                // N2 = t (vertex at (0,1))
                array![1.0 - s - t, s, t]
            }
            ElementType::Quad4 => {
                // Quad: bilinear on [-1,1]²
                let s1 = 0.25 * (s + 1.0);
                let s2 = 0.25 * (s - 1.0);
                let t1 = t + 1.0;
                let t2 = t - 1.0;
                array![s1 * t1, -s2 * t1, s2 * t2, -s1 * t2]
            }
        }
    }

    /// Compute shape function derivatives dN/ds and dN/dt
    pub fn shape_derivatives(&self, s: f64, t: f64) -> (Vec<f64>, Vec<f64>) {
        match self.element_type {
            ElementType::Tri3 => {
                // dN/ds = [-1, 1, 0], dN/dt = [-1, 0, 1]
                (array![-1.0, 1.0, 0.0], array![-1.0, 0.0, 1.0])
            }
            ElementType::Quad4 => {
                // dN/ds
                let dns = array![
                    0.25 * (t + 1.0),
                    -0.25 * (t + 1.0),
                    0.25 * (t - 1.0),
                    -0.25 * (t - 1.0)
                ];
                // dN/dt
                let dnt = array![
                    0.25 * (s + 1.0),
                    0.25 * (1.0 - s),
                    0.25 * (s - 1.0),
                    -0.25 * (s + 1.0)
                ];
                (dns, dnt)
            }
        }
    }

    /// Compute global coordinates from local coordinates (s, t)
    pub fn local_to_global(&self, s: f64, t: f64, nodes: &[[f64; 3]]) -> [f64; 3] {
        let n = self.shape_functions(s, t);
        let mut global = [0.0_f64; 3];

        for (i, &node_idx) in self.connectivity.iter().enumerate() {
            for j in 0..3 {
                global[j] += n[i] * nodes[node_idx][j];
            }
        }

        global
    }

    /// Get local collocation point coordinates for a given basis order
    ///
    /// For Constant: returns [(1/3, 1/3)] for triangles (centroid)
    /// For Linear: returns the 3/4 internal Gauss points used by the
    /// discontinuous linear panel path.
    /// For Quadratic: returns six interior triangle points or a 3x3 interior
    /// tensor-product rule for quadrilateral panels.
    pub fn collocation_local_coords(&self, basis: BasisOrder) -> Vec<(f64, f64)> {
        match basis {
            BasisOrder::Constant => match self.element_type {
                ElementType::Tri3 => vec![(1.0 / 3.0, 1.0 / 3.0)],
                ElementType::Quad4 => vec![(0.0, 0.0)],
            },
            BasisOrder::Linear => {
                match self.element_type {
                    ElementType::Tri3 => {
                        // Interior Gauss points for a triangle (avoids edge singularities)
                        vec![
                            (1.0 / 6.0, 1.0 / 6.0),
                            (2.0 / 3.0, 1.0 / 6.0),
                            (1.0 / 6.0, 2.0 / 3.0),
                        ]
                    }
                    ElementType::Quad4 => {
                        // Interior Gauss points for a quad (avoids edge singularities)
                        let g = 1.0 / 3.0_f64.sqrt();
                        vec![(-g, -g), (g, -g), (g, g), (-g, g)]
                    }
                }
            }
            BasisOrder::Quadratic => match self.element_type {
                ElementType::Tri3 => {
                    let a = 0.8168475729804585;
                    let b = 0.0915762135097715;
                    let c = 0.108103018168070;
                    let d = 0.445948490915965;
                    vec![(b, b), (a, b), (b, a), (d, d), (d, c), (c, d)]
                }
                ElementType::Quad4 => {
                    let g = (3.0_f64 / 5.0).sqrt();
                    [-g, 0.0, g]
                        .into_iter()
                        .flat_map(|s| [-g, 0.0, g].into_iter().map(move |t| (s, t)))
                        .collect()
                }
            },
        }
    }

    /// Compute collocation points in physical coordinates for a given basis order
    ///
    /// For Constant: returns just [self.center]
    /// For Linear: maps internal Gauss points to physical coordinates using shape functions
    pub fn collocation_points(&self, basis: BasisOrder, nodes: &[[f64; 3]]) -> Vec<[f64; 3]> {
        match basis {
            BasisOrder::Constant | BasisOrder::Linear | BasisOrder::Quadratic => {
                let has_curved = self
                    .node_normals
                    .iter()
                    .any(|v| v.iter().any(|x| x.abs() > 1e-15));
                self.collocation_local_coords(basis)
                    .iter()
                    .map(|&(s, t)| {
                        if has_curved {
                            let shape = self.shape_functions(s, t);
                            let (shape_ds, shape_dt) = self.shape_derivatives(s, t);
                            let coords = self.element_coords(nodes);
                            let (point, _, _) = quadrature_geometry(
                                &coords,
                                shape.as_slice(),
                                shape_ds.as_slice(),
                                shape_dt.as_slice(),
                                Some(self.node_normals.as_slice()),
                            );
                            point
                        } else {
                            self.local_to_global(s, t, nodes)
                        }
                    })
                    .collect()
            }
        }
    }

    /// Compute normals at collocation points for a given basis order
    ///
    /// Normals always come from the element's geometric mapping. Smoothed
    /// vertex normals are visualization data; interpolating them on affine
    /// Tri3 geometry would describe a different surface than the Jacobian and
    /// invalidate hypersingular jump terms.
    pub fn collocation_normals(&self, basis: BasisOrder, nodes: &[[f64; 3]]) -> Vec<[f64; 3]> {
        match basis {
            BasisOrder::Constant => vec![self.normal],
            BasisOrder::Linear | BasisOrder::Quadratic => self
                .collocation_local_coords(basis)
                .iter()
                .map(|&(s, t)| {
                    let (normal, _) = self.normal_and_jacobian(s, t, nodes);
                    normal
                })
                .collect(),
        }
    }

    fn element_coords(&self, nodes: &[[f64; 3]]) -> DenseMatrix<f64> {
        let mut coords = DenseMatrix::<f64>::zeros(self.connectivity.len(), 3);
        for (i, &node_idx) in self.connectivity.iter().enumerate() {
            for j in 0..3 {
                coords[[i, j]] = nodes[node_idx][j];
            }
        }
        coords
    }

    /// Compute unit normal vector and Jacobian at local coordinates (s, t)
    ///
    /// Returns (normal, jacobian) where jacobian is the surface element dS.
    pub fn normal_and_jacobian(&self, s: f64, t: f64, nodes: &[[f64; 3]]) -> ([f64; 3], f64) {
        let (dns, dnt) = self.shape_derivatives(s, t);

        let mut dxds = [0.0_f64; 3];
        let mut dxdt = [0.0_f64; 3];

        for (i, &node_idx) in self.connectivity.iter().enumerate() {
            for j in 0..3 {
                dxds[j] += dns[i] * nodes[node_idx][j];
                dxdt[j] += dnt[i] * nodes[node_idx][j];
            }
        }

        let n = cross_product(dxds, dxdt);
        let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();

        if length > 1e-15 {
            ([n[0] / length, n[1] / length, n[2] / length], length)
        } else {
            ([0.0; 3], 0.0)
        }
    }
}

/// Evaluate the discontinuous panel basis used by TBEM assembly and field
/// post-processing at reference coordinates `(s, t)`.
///
/// The geometry is still described by [`Element::shape_functions`]. A
/// quadratic basis therefore supplies six or nine field DOFs on the existing
/// Tri3 or Quad4 geometric panel; it does not imply globally shared
/// high-order mesh nodes.
/// Quadratic Quad4 slots follow the geometric connectivity vertices, then
/// the four consecutive edge midpoints and the panel center. This ordering
/// also matches the continuous quadratic global DOF map.
pub fn basis_values(element_type: ElementType, basis: BasisOrder, s: f64, t: f64) -> Vec<f64> {
    match basis {
        BasisOrder::Constant => vec![1.0],
        BasisOrder::Linear => match element_type {
            ElementType::Tri3 => {
                let n = [1.0 - s - t, s, t];
                n.into_iter().map(|value| 2.0 * value - 1.0 / 3.0).collect()
            }
            ElementType::Quad4 => {
                let g = 1.0 / 3.0_f64.sqrt();
                let left_s = (g - s) / (2.0 * g);
                let right_s = (s + g) / (2.0 * g);
                let low_t = (g - t) / (2.0 * g);
                let high_t = (t + g) / (2.0 * g);
                vec![
                    left_s * low_t,
                    right_s * low_t,
                    right_s * high_t,
                    left_s * high_t,
                ]
            }
        },
        BasisOrder::Quadratic => match element_type {
            ElementType::Tri3 => {
                let l0 = 1.0 - s - t;
                let l1 = s;
                let l2 = t;
                vec![
                    l0 * (2.0 * l0 - 1.0),
                    l1 * (2.0 * l1 - 1.0),
                    l2 * (2.0 * l2 - 1.0),
                    4.0 * l0 * l1,
                    4.0 * l1 * l2,
                    4.0 * l2 * l0,
                ]
            }
            ElementType::Quad4 => {
                fn lagrange2(x: f64) -> [f64; 3] {
                    [0.5 * x * (x - 1.0), 1.0 - x * x, 0.5 * x * (x + 1.0)]
                }
                let ls = lagrange2(s);
                let lt = lagrange2(t);
                // Connectivity convention: vertices (-,-),(+,-),(+,+),
                // (-,+), then consecutive edge midpoints and the center.
                // This matches the continuous quadratic DOF map and the
                // Galerkin `quad4_basis_values` ordering.
                vec![
                    ls[0] * lt[0],
                    ls[2] * lt[0],
                    ls[2] * lt[2],
                    ls[0] * lt[2],
                    ls[1] * lt[0],
                    ls[2] * lt[1],
                    ls[1] * lt[2],
                    ls[0] * lt[1],
                    ls[1] * lt[1],
                ]
            }
        },
    }
}

/// Cross product of two 3D vectors
pub fn cross_product(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Dot product of two 3D vectors
pub fn dot_product(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Normalize a 3D vector
pub fn normalize(v: [f64; 3]) -> ([f64; 3], f64) {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-15 {
        ([v[0] / len, v[1] / len, v[2] / len], len)
    } else {
        ([0.0; 3], 0.0)
    }
}

/// Compute element center (centroid)
pub fn compute_element_center(nodes: &[[f64; 3]], connectivity: &[usize]) -> [f64; 3] {
    let n = connectivity.len() as f64;
    let mut center = [0.0_f64; 3];
    for &node_idx in connectivity {
        for j in 0..3 {
            center[j] += nodes[node_idx][j];
        }
    }
    [center[0] / n, center[1] / n, center[2] / n]
}

/// Compute element area
pub fn compute_element_area(
    nodes: &[[f64; 3]],
    connectivity: &[usize],
    element_type: ElementType,
) -> f64 {
    let row = |idx: usize| -> [f64; 3] { [nodes[idx][0], nodes[idx][1], nodes[idx][2]] };
    let sub = |a: [f64; 3], b: [f64; 3]| -> [f64; 3] { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] };
    let cross_len = |a: [f64; 3], b: [f64; 3]| -> f64 {
        let c = cross_product(a, b);
        (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt()
    };
    match element_type {
        ElementType::Tri3 => {
            let p0 = row(connectivity[0]);
            let p1 = row(connectivity[1]);
            let p2 = row(connectivity[2]);
            0.5 * cross_len(sub(p1, p0), sub(p2, p0))
        }
        ElementType::Quad4 => {
            // Quad area via average of both diagonal splits — handles warped quads.
            let p0 = row(connectivity[0]);
            let p1 = row(connectivity[1]);
            let p2 = row(connectivity[2]);
            let p3 = row(connectivity[3]);

            let v1 = sub(p1, p0);
            let v2 = sub(p2, p0);
            let v3 = sub(p3, p0);
            let v4 = sub(p2, p1);
            let v5 = sub(p3, p1);
            let v6 = sub(p0, p1);

            let area1 = 0.5 * cross_len(v1, v2);
            let area2 = 0.5 * cross_len(v2, v3);
            let area3 = 0.5 * cross_len(v4, v5);
            let area4 = 0.5 * cross_len(v5, v6);
            0.5 * (area1 + area2 + area3 + area4)
        }
    }
}

/// Compute element normal at center
pub fn compute_element_normal(
    nodes: &[[f64; 3]],
    connectivity: &[usize],
    element_type: ElementType,
) -> [f64; 3] {
    let row = |idx: usize| -> [f64; 3] { [nodes[idx][0], nodes[idx][1], nodes[idx][2]] };
    let sub = |a: [f64; 3], b: [f64; 3]| -> [f64; 3] { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] };
    match element_type {
        ElementType::Tri3 => {
            let p0 = row(connectivity[0]);
            let p1 = row(connectivity[1]);
            let p2 = row(connectivity[2]);
            let (n, _) = normalize(cross_product(sub(p1, p0), sub(p2, p0)));
            n
        }
        ElementType::Quad4 => {
            let p0 = row(connectivity[0]);
            let p1 = row(connectivity[1]);
            let p2 = row(connectivity[2]);
            let p3 = row(connectivity[3]);
            let (n, _) = normalize(cross_product(sub(p2, p0), sub(p3, p1)));
            n
        }
    }
}

/// Interpolate a unit normal from per-node normals using shape function weights.
///
/// The interpolated vector is re-normalized to ensure unit length.
/// Returns a zero vector if the interpolated normal has negligible magnitude.
pub fn interpolate_normal(node_normals: &[[f64; 3]], shape_weights: &[f64]) -> [f64; 3] {
    let mut n = [0.0_f64; 3];
    for (i, &w) in shape_weights.iter().enumerate() {
        if i >= node_normals.len() {
            break;
        }
        for j in 0..3 {
            n[j] += w * node_normals[i][j];
        }
    }
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if len > 1e-15 {
        [n[0] / len, n[1] / len, n[2] / len]
    } else {
        [0.0; 3]
    }
}

/// Geometry at a quadrature point.
///
/// Node normals affect only the integration normal. Geometry and Jacobian remain
/// the flat isoparametric element mapping represented by `element_coords`.
pub fn quadrature_geometry(
    element_coords: &DenseMatrix<f64>,
    shape_fn: &[f64],
    shape_ds: &[f64],
    shape_dt: &[f64],
    node_normals: Option<&[[f64; 3]]>,
) -> ([f64; 3], f64, [f64; 3]) {
    let n_nodes = shape_fn.len();
    let mut q = [0.0_f64; 3];
    let mut dq_ds = [0.0_f64; 3];
    let mut dq_dt = [0.0_f64; 3];

    for n in 0..n_nodes {
        for d in 0..3 {
            q[d] += shape_fn[n] * element_coords[[n, d]];
            dq_ds[d] += shape_ds[n] * element_coords[[n, d]];
            dq_dt[d] += shape_dt[n] * element_coords[[n, d]];
        }
    }

    let fn_arr = cross_product(dq_ds, dq_dt);
    let flat_jacobian =
        (fn_arr[0] * fn_arr[0] + fn_arr[1] * fn_arr[1] + fn_arr[2] * fn_arr[2]).sqrt();
    if flat_jacobian < 1e-15 {
        return (q, 0.0, [0.0; 3]);
    }

    let flat_unit = [
        fn_arr[0] / flat_jacobian,
        fn_arr[1] / flat_jacobian,
        fn_arr[2] / flat_jacobian,
    ];
    let normal = if let Some(nn) = node_normals {
        let interp = interpolate_normal(nn, shape_fn);
        if interp[0] * interp[0] + interp[1] * interp[1] + interp[2] * interp[2] > 1e-30 {
            interp
        } else {
            flat_unit
        }
    } else {
        flat_unit
    };

    (q, flat_jacobian, normal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{BoundaryCondition, ElementProperty};
    use num_complex::Complex64;

    fn make_test_triangle() -> (Element, Vec<[f64; 3]>) {
        let nodes: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

        let element = Element {
            connectivity: vec![0, 1, 2],
            element_type: ElementType::Tri3,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0; 3]; 3],
            center: [1.0 / 3.0, 1.0 / 3.0, 0.0],
            area: 0.5,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses: vec![0],
        };

        (element, nodes)
    }

    #[test]
    fn test_triangle_shape_functions() {
        let (element, _nodes) = make_test_triangle();

        // At vertex 0 (s=0, t=0) -> N = [1, 0, 0]
        let n = element.shape_functions(0.0, 0.0);
        assert!((n[0] - 1.0).abs() < 1e-10);
        assert!(n[1].abs() < 1e-10);
        assert!(n[2].abs() < 1e-10);

        // At vertex 1 (s=1, t=0) -> N = [0, 1, 0]
        let n = element.shape_functions(1.0, 0.0);
        assert!(n[0].abs() < 1e-10);
        assert!((n[1] - 1.0).abs() < 1e-10);
        assert!(n[2].abs() < 1e-10);

        // At center (s=1/3, t=1/3) -> N = [1/3, 1/3, 1/3]
        let n = element.shape_functions(1.0 / 3.0, 1.0 / 3.0);
        assert!((n[0] - 1.0 / 3.0).abs() < 1e-10);
        assert!((n[1] - 1.0 / 3.0).abs() < 1e-10);
        assert!((n[2] - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_triangle_local_to_global() {
        let (element, nodes) = make_test_triangle();

        // At vertex 0 (s=0, t=0) -> node 0 at (0,0,0)
        let p = element.local_to_global(0.0, 0.0, &nodes);
        assert!((p[0] - 0.0).abs() < 1e-10);
        assert!((p[1] - 0.0).abs() < 1e-10);

        // At vertex 1 (s=1, t=0) -> node 1 at (1,0,0)
        let p = element.local_to_global(1.0, 0.0, &nodes);
        assert!((p[0] - 1.0).abs() < 1e-10);
        assert!((p[1] - 0.0).abs() < 1e-10);

        // At vertex 2 (s=0, t=1) -> node 2 at (0,1,0)
        let p = element.local_to_global(0.0, 1.0, &nodes);
        assert!((p[0] - 0.0).abs() < 1e-10);
        assert!((p[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_triangle_normal() {
        let (element, nodes) = make_test_triangle();

        let (normal, jacobian) = element.normal_and_jacobian(0.5, 0.25, &nodes);

        // Normal should be (0, 0, 1) for this XY-plane triangle
        assert!((normal[0]).abs() < 1e-10);
        assert!((normal[1]).abs() < 1e-10);
        assert!((normal[2] - 1.0).abs() < 1e-10);

        // Jacobian should be constant for linear triangle = 2 * area = 1
        assert!((jacobian - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quadrature_geometry_keeps_flat_mapping_with_node_normals() {
        let coords = DenseMatrix::from_row_major_vec(
            3,
            3,
            vec![
                1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0,
            ],
        );
        let node_normals: Vec<[f64; 3]> = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let shape_fn = [0.2, 0.3, 0.5];
        let shape_ds = [-1.0, 1.0, 0.0];
        let shape_dt = [-1.0, 0.0, 1.0];

        let (point, jacobian, normal) = quadrature_geometry(
            &coords,
            &shape_fn,
            &shape_ds,
            &shape_dt,
            Some(&node_normals),
        );

        assert!((point[0] - 0.2).abs() < 1e-12);
        assert!((point[1] - 0.3).abs() < 1e-12);
        assert!((point[2] - 0.5).abs() < 1e-12);
        assert!(jacobian > 0.0);
        assert!(
            (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2] - 1.0).abs()
                < 1e-12
        );
    }

    #[test]
    fn test_element_area() {
        let nodes: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

        let area = compute_element_area(&nodes, &[0, 1, 2], ElementType::Tri3);
        assert!((area - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_collocation_points_constant() {
        let (element, nodes) = make_test_triangle();
        let points = element.collocation_points(BasisOrder::Constant, &nodes);

        // Constant: single point at centroid
        assert_eq!(points.len(), 1);
        assert!((points[0][0] - 1.0 / 3.0).abs() < 1e-10);
        assert!((points[0][1] - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_collocation_points_linear() {
        let (element, nodes) = make_test_triangle();
        let points = element.collocation_points(BasisOrder::Linear, &nodes);

        // Linear: 3 interior Gauss points
        assert_eq!(points.len(), 3);

        // All points should be inside the triangle (all coordinates >= 0)
        for p in &points {
            assert!(p[0] >= -1e-10, "x={} not inside triangle", p[0]);
            assert!(p[1] >= -1e-10, "y={} not inside triangle", p[1]);
            // For reference triangle with nodes at (0,0), (1,0), (0,1):
            // s+t should be <= 1
            assert!(p[0] + p[1] <= 1.0 + 1e-10, "x+y={} exceeds 1", p[0] + p[1]);
        }

        // Points should be distinct
        for i in 0..3 {
            for j in (i + 1)..3 {
                let dist = ((points[i][0] - points[j][0]).powi(2)
                    + (points[i][1] - points[j][1]).powi(2))
                .sqrt();
                assert!(dist > 1e-10, "Points {i} and {j} are too close");
            }
        }
    }

    #[test]
    fn test_collocation_normals_linear() {
        let (element, nodes) = make_test_triangle();
        let normals = element.collocation_normals(BasisOrder::Linear, &nodes);

        // For a flat triangle, all normals should be the same
        assert_eq!(normals.len(), 3);
        for n in &normals {
            assert!(
                (n[2] - 1.0).abs() < 1e-10,
                "Normal z={} should be 1.0",
                n[2]
            );
        }
    }

    #[test]
    fn test_quadratic_basis_layout_and_collocation() {
        let (element, nodes) = make_test_triangle();
        let basis = basis_values(ElementType::Tri3, BasisOrder::Quadratic, 0.2, 0.3);
        assert_eq!(basis.len(), 6);
        assert!((basis.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);
        assert!(basis.iter().all(|value| value.is_finite()));

        let points = element.collocation_points(BasisOrder::Quadratic, &nodes);
        assert_eq!(points.len(), 6);
        for point in points {
            assert!(point[0] > 0.0);
            assert!(point[1] > 0.0);
            assert!(point[0] + point[1] < 1.0);
        }
        assert_eq!(element.num_dofs_for_basis(BasisOrder::Quadratic), 6);
    }

    #[test]
    fn test_quadratic_quad_basis_layout_and_collocation() {
        let element = Element {
            connectivity: vec![0, 1, 2, 3],
            element_type: ElementType::Quad4,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0; 3]; 4],
            center: [0.0, 0.0, 0.0],
            area: 4.0,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::default()]),
            group: 0,
            dof_addresses: (0..9).collect(),
        };
        let basis = basis_values(ElementType::Quad4, BasisOrder::Quadratic, 0.2, -0.4);
        assert_eq!(basis.len(), 9);
        assert!((basis.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);
        assert_eq!(element.num_dofs_for_basis(BasisOrder::Quadratic), 9);
        // Continuous DOFs follow connectivity vertices, then consecutive
        // edge midpoints and the center. Basis slots must represent those
        // same points under the geometry's local-coordinate convention.
        let local_nodes = [
            (-1.0, -1.0),
            (1.0, -1.0),
            (1.0, 1.0),
            (-1.0, 1.0),
            (0.0, -1.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (-1.0, 0.0),
            (0.0, 0.0),
        ];
        for (slot, &(s, t)) in local_nodes.iter().enumerate() {
            let values = basis_values(ElementType::Quad4, BasisOrder::Quadratic, s, t);
            for (index, &value) in values.iter().enumerate() {
                let expected = if index == slot { 1.0 } else { 0.0 };
                assert!(
                    (value - expected).abs() < 1e-12,
                    "geometry node {slot} must select basis slot {slot}, got slot {index}={value}"
                );
            }
        }
        assert_eq!(
            element
                .collocation_local_coords(BasisOrder::Quadratic)
                .len(),
            9
        );
    }

    #[test]
    fn test_affine_tri3_collocation_uses_flat_normal_with_smoothed_vertex_normals() {
        let (mut element, nodes) = make_test_triangle();
        element.node_normals = vec![[1.0, 0.0, 0.0]; 3];

        let normals = element.collocation_normals(BasisOrder::Linear, &nodes);

        assert_eq!(normals, vec![element.normal; 3]);
    }

    #[test]
    fn test_num_dofs_for_basis() {
        let (element, _) = make_test_triangle();
        assert_eq!(element.num_dofs_for_basis(BasisOrder::Constant), 1);
        assert_eq!(element.num_dofs_for_basis(BasisOrder::Linear), 3);
    }
}
