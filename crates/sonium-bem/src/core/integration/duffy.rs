//! Duffy transform and collocation singular integration helpers
//!
//! Implements analytically regularized integration for singular and near-singular
//! element integrals on triangles. The Duffy transform removes 1/r singularities
//! by introducing a Jacobian that cancels the singular kernel, giving exponential
//! convergence with quadrature order.
//!
//! ## Methods
//!
//! - **Duffy self-element**: For integrating over a triangle when source = field element.
//!   The triangle is split into 3 sub-triangles (one per vertex), and a coordinate
//!   transform regularizes the singularity at each vertex.
//!
//! - **Galerkin common edge/vertex helpers**: Integrate over source and field triangles
//!   with Sauter-Schwab-style charts for shared-edge/shared-vertex element pairs.
//!
//! - **Common edge collocation helper**: For two triangles sharing an edge (2 common nodes).
//!   Sauter-Schwab-style charts regularize the field integral around a fixed source point.
//!
//! - **Common vertex collocation helper**: For two triangles sharing a vertex (1 common node).
//!   Sauter-Schwab-style charts regularize the field integral around a fixed source point.
//!
//! ## References
//!
//! - S. Sauter, C. Schwab, "Boundary Element Methods", Springer, 2011.
//! - M. G. Duffy, "Quadrature over a pyramid or cube of integrands with a
//!   singularity at a vertex", SIAM J. Numer. Anal. 19(6), 1982.
//!
//! The legacy `sauter_schwab_common_edge` and `sauter_schwab_common_vertex`
//! entry points are collocation helpers. Use the `*_galerkin_*` functions when
//! both source and field basis functions must be integrated.

use num_complex::Complex64;
use sonium_solvers::dense::DenseMatrix;
use std::f64::consts::PI;

use crate::core::integration::gauss::gauss_legendre;
use crate::core::mesh::element::{cross_product, normalize};
use crate::core::types::{
    AdjacencyType, IntegrationResult, LinearIntegrationResult, PhysicsParams,
};

/// Galerkin source-by-field basis integration results.
///
/// `basis_results[i][j]` is the integral for source basis `i` and field basis `j`.
#[derive(Debug, Clone)]
pub struct GalerkinIntegrationResult {
    /// Source-basis by field-basis integration table.
    pub basis_results: Vec<Vec<IntegrationResult>>,
    /// Weak-form surface-curl products for Quad4 physical maps.
    ///
    /// Affine triangle paths leave this table zero because their
    /// hypersingular assembly computes curl products from panel gradients.
    pub curl_integrals: Vec<Vec<Complex64>>,
    /// Weak-form normal products for Quad4 physical maps.
    pub normal_mass_integrals: Vec<Vec<Complex64>>,
}

/// Regular separated-Quad4 data needed by the weak-form hypersingular
/// operator.  The two extra tables retain the physical-surface curl and
/// normal products pointwise under the same isoparametric quadrature as the
/// kernel tables.
#[derive(Debug, Clone)]
pub struct GalerkinQuad4WeakResult {
    /// Scalar and normal-derivative kernel integrals.
    pub kernels: GalerkinIntegrationResult,
    /// `∬ G curl_Γ(phi_source) · curl_Γ(phi_field)` entries.
    pub curl_integrals: Vec<Vec<Complex64>>,
    /// `∬ G (n_source · n_field) phi_source phi_field` entries.
    pub normal_mass_integrals: Vec<Vec<Complex64>>,
}

impl GalerkinIntegrationResult {
    fn new(num_source_basis: usize, num_field_basis: usize) -> Self {
        Self {
            basis_results: (0..num_source_basis)
                .map(|_| {
                    (0..num_field_basis)
                        .map(|_| IntegrationResult::default())
                        .collect()
                })
                .collect(),
            curl_integrals: vec![vec![Complex64::new(0.0, 0.0); num_field_basis]; num_source_basis],
            normal_mass_integrals: vec![
                vec![Complex64::new(0.0, 0.0); num_field_basis];
                num_source_basis
            ],
        }
    }

    /// Number of source basis functions represented by this result.
    pub fn num_source_basis(&self) -> usize {
        self.basis_results.len()
    }

    /// Number of field basis functions represented by this result.
    pub fn num_field_basis(&self) -> usize {
        self.basis_results.first().map_or(0, Vec::len)
    }
}

// ============================================================================
// Kernel computation helpers
// ============================================================================

/// Compute shape functions for a triangle at local coordinates (xi, eta)
///
/// Reference triangle: V0=(0,0), V1=(1,0), V2=(0,1)
/// N0 = 1 - xi - eta, N1 = xi, N2 = eta
#[inline]
fn tri_shape_functions(xi: f64, eta: f64) -> [f64; 3] {
    [1.0 - xi - eta, xi, eta]
}

/// Evaluate the affine Tri3 Lagrange basis for the requested local basis
/// count.  Quadratic traces use the standard six functions ordered as three
/// vertices followed by edges `(0,1)`, `(1,2)`, and `(2,0)`.
#[inline]
fn galerkin_basis_values(shape: [f64; 3], num_basis: usize) -> Vec<f64> {
    match num_basis {
        1 => vec![1.0],
        3 => shape.to_vec(),
        6 => {
            let [l0, l1, l2] = shape;
            vec![
                l0 * (2.0 * l0 - 1.0),
                l1 * (2.0 * l1 - 1.0),
                l2 * (2.0 * l2 - 1.0),
                4.0 * l0 * l1,
                4.0 * l1 * l2,
                4.0 * l2 * l0,
            ]
        }
        _ => panic!("Galerkin triangle basis count must be 1, 3, or 6"),
    }
}

/// Compute shape function derivatives for a triangle
///
/// dN/dxi = [-1, 1, 0], dN/deta = [-1, 0, 1]
#[inline]
fn tri_shape_derivatives() -> ([f64; 3], [f64; 3]) {
    ([-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0])
}

/// Compute position, normal, and Jacobian from local coordinates on a triangle
///
/// Returns (position, normal, jacobian, shape_functions)
fn tri_geometry(
    element_coords: &DenseMatrix<f64>,
    xi: f64,
    eta: f64,
) -> ([f64; 3], [f64; 3], f64, [f64; 3]) {
    let shape = tri_shape_functions(xi, eta);
    let (ds, dt) = tri_shape_derivatives();

    let mut pos = [0.0_f64; 3];
    let mut dx_dxi = [0.0_f64; 3];
    let mut dx_deta = [0.0_f64; 3];

    for i in 0..3 {
        for j in 0..3 {
            pos[j] += shape[i] * element_coords[[i, j]];
            dx_dxi[j] += ds[i] * element_coords[[i, j]];
            dx_deta[j] += dt[i] * element_coords[[i, j]];
        }
    }

    let nu = cross_product(dx_dxi, dx_deta);
    let jacobian = (nu[0] * nu[0] + nu[1] * nu[1] + nu[2] * nu[2]).sqrt();

    let normal = if jacobian > 1e-15 {
        [nu[0] / jacobian, nu[1] / jacobian, nu[2] / jacobian]
    } else {
        [0.0; 3]
    };

    (pos, normal, jacobian, shape)
}

/// Compute all four BEM kernels at a pair of points
///
/// Returns (G, dG/dn_y, dG/dn_x, d2G/dnx_dny) each multiplied by
/// the combined quadrature weight.
///
/// G = exp(ikr) / (4*pi*r) * weight
/// dG/dn_y = G * (ik - 1/r) * (r_hat . n_y)
/// dG/dn_x = -G * (ik - 1/r) * (r_hat . n_x)
/// d2G/dnx_dny = G * [ (3/r^2 - k^2)(r_hat.n_x)(r_hat.n_y) + (n_x.n_y)/r^2
///                      - ik/r * (3*(r_hat.n_x)(r_hat.n_y) + n_x.n_y) ]
///   (note: signs adjusted for r = y - x convention)
#[inline]
fn compute_kernels(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    field_point: &[f64; 3],
    field_normal: &[f64; 3],
    physics: &PhysicsParams,
    weight: f64,
) -> (Complex64, Complex64, Complex64, Complex64) {
    let wavruim = physics.harmonic_factor * physics.wave_number;
    let k2 = physics.wave_number * physics.wave_number;

    let diff = [
        field_point[0] - source_point[0],
        field_point[1] - source_point[1],
        field_point[2] - source_point[2],
    ];

    let (unit_r, r) = normalize(diff);

    if r < 1e-15 {
        return (
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        );
    }

    // G kernel: exp(ikr) / (4*pi*r) * weight
    let kr = wavruim * r;
    let g_scaled = weight / (4.0 * PI * r);
    let zg = sonium_fastmath::exp_i(kr) * g_scaled;

    // Gradient factor: (ik - 1/r)
    let z1 = Complex64::new(-1.0 / r, wavruim);
    let zhh_base = zg * z1;

    // dG/dn_y: directional derivative in field normal direction
    let r_dot_ny =
        unit_r[0] * field_normal[0] + unit_r[1] * field_normal[1] + unit_r[2] * field_normal[2];
    let zhh = zhh_base * r_dot_ny;

    // dG/dn_x: directional derivative in source normal direction
    let r_dot_nx = -(unit_r[0] * source_normal[0]
        + unit_r[1] * source_normal[1]
        + unit_r[2] * source_normal[2]);
    let zht = zhh_base * r_dot_nx;

    // Hypersingular kernel d2G/(dn_x dn_y)
    let rq = r_dot_ny * r_dot_nx;
    let nx_dot_ny = source_normal[0] * field_normal[0]
        + source_normal[1] * field_normal[1]
        + source_normal[2] * field_normal[2];
    let dq = r * r;

    let ze_factor = Complex64::new(
        (3.0 / dq - k2) * rq + nx_dot_ny / dq,
        -wavruim / r * (3.0 * rq + nx_dot_ny),
    );
    let ze = zg * ze_factor;

    (zg, zhh, zht, ze)
}

/// Compute the wavenumber derivatives of the four Helmholtz kernels.
///
/// Geometry, normals, and quadrature weight are held fixed.  The returned
/// tuple has the same ordering as [`compute_kernels`], but each entry is the
/// analytic derivative with respect to the medium wavenumber.
#[inline]
fn compute_kernel_wavenumber_derivatives(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    field_point: &[f64; 3],
    field_normal: &[f64; 3],
    physics: &PhysicsParams,
    weight: f64,
) -> (Complex64, Complex64, Complex64, Complex64) {
    let phase_sign = physics.harmonic_factor;
    let k = physics.wave_number;
    let diff = [
        field_point[0] - source_point[0],
        field_point[1] - source_point[1],
        field_point[2] - source_point[2],
    ];
    let (unit_r, distance) = normalize(diff);
    if distance < 1.0e-15 {
        return (
            Complex64::default(),
            Complex64::default(),
            Complex64::default(),
            Complex64::default(),
        );
    }

    let g = sonium_fastmath::exp_i(phase_sign * k * distance) * (weight / (4.0 * PI * distance));
    let dg_dk = g * Complex64::new(0.0, phase_sign * distance);
    let r_dot_ny =
        unit_r[0] * field_normal[0] + unit_r[1] * field_normal[1] + unit_r[2] * field_normal[2];
    let r_dot_nx = -(unit_r[0] * source_normal[0]
        + unit_r[1] * source_normal[1]
        + unit_r[2] * source_normal[2]);
    let kernel_factor = Complex64::new(-1.0 / distance, phase_sign * k);
    let d_kernel_factor = Complex64::new(0.0, phase_sign);
    let d_first = dg_dk * kernel_factor + g * d_kernel_factor;
    let dg_dn_y_dk = d_first * r_dot_ny;
    let dg_dn_x_dk = d_first * r_dot_nx;

    let normal_dot = source_normal[0] * field_normal[0]
        + source_normal[1] * field_normal[1]
        + source_normal[2] * field_normal[2];
    let product = r_dot_nx * r_dot_ny;
    let inv_distance_squared = 1.0 / (distance * distance);
    let second_factor = Complex64::new(
        (3.0 * inv_distance_squared - k * k) * product + normal_dot * inv_distance_squared,
        -phase_sign * k / distance * (3.0 * product + normal_dot),
    );
    let d_second_factor = Complex64::new(
        -2.0 * k * product,
        -phase_sign / distance * (3.0 * product + normal_dot),
    );
    let d2g_dnxdny_dk = dg_dk * second_factor + g * d_second_factor;

    (dg_dk, dg_dn_y_dk, dg_dn_x_dk, d2g_dnxdny_dk)
}

/// Get Gauss-Legendre points and weights mapped to [0, 1]
///
/// The standard gauss_legendre returns points on [-1, 1].
/// We transform: x_01 = (x+1)/2, w_01 = w/2.
fn gauss_legendre_01(order: usize) -> (Vec<f64>, Vec<f64>) {
    let (pts, wts) = gauss_legendre(order);
    let pts_01: Vec<f64> = pts.iter().map(|&x| (x + 1.0) / 2.0).collect();
    let wts_01: Vec<f64> = wts.iter().map(|&w| w / 2.0).collect();
    (pts_01, wts_01)
}

fn reorder_triangle(coords: &DenseMatrix<f64>, order: [usize; 3]) -> DenseMatrix<f64> {
    let mut reordered = DenseMatrix::<f64>::zeros(3, 3);
    for (new_idx, &old_idx) in order.iter().enumerate() {
        for d in 0..3 {
            reordered[[new_idx, d]] = coords[[old_idx, d]];
        }
    }
    reordered
}

fn squared_distance(a: &DenseMatrix<f64>, ia: usize, b: &DenseMatrix<f64>, ib: usize) -> f64 {
    (0..3)
        .map(|d| {
            let diff = a[[ia, d]] - b[[ib, d]];
            diff * diff
        })
        .sum()
}

fn shared_vertex_pairs(
    source_coords: &DenseMatrix<f64>,
    field_coords: &DenseMatrix<f64>,
) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    let mut max_extent = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            max_extent =
                max_extent.max(squared_distance(source_coords, i, source_coords, j).sqrt());
            max_extent = max_extent.max(squared_distance(field_coords, i, field_coords, j).sqrt());
        }
    }
    let tol2 = (1.0e-10 * max_extent.max(1.0)).powi(2);

    for i in 0..3 {
        for j in 0..3 {
            if squared_distance(source_coords, i, field_coords, j) <= tol2 {
                pairs.push((i, j));
            }
        }
    }
    pairs
}

fn reorder_common_edge_pair(
    source_coords: &DenseMatrix<f64>,
    field_coords: &DenseMatrix<f64>,
) -> (DenseMatrix<f64>, DenseMatrix<f64>, [usize; 3], [usize; 3]) {
    let shared = shared_vertex_pairs(source_coords, field_coords);
    assert_eq!(
        shared.len(),
        2,
        "common-edge Galerkin integration requires exactly two shared vertices"
    );

    let source_free = (0..3)
        .find(|idx| !shared.iter().any(|(source_idx, _)| source_idx == idx))
        .expect("source triangle must have one non-shared vertex");
    let field_free = (0..3)
        .find(|idx| !shared.iter().any(|(_, field_idx)| field_idx == idx))
        .expect("field triangle must have one non-shared vertex");

    let source_order = [shared[0].0, shared[1].0, source_free];
    let field_order = [shared[0].1, shared[1].1, field_free];
    (
        reorder_triangle(source_coords, source_order),
        reorder_triangle(field_coords, field_order),
        source_order,
        field_order,
    )
}

fn reorder_common_vertex_pair(
    source_coords: &DenseMatrix<f64>,
    field_coords: &DenseMatrix<f64>,
) -> (DenseMatrix<f64>, DenseMatrix<f64>, [usize; 3], [usize; 3]) {
    let shared = shared_vertex_pairs(source_coords, field_coords);
    assert_eq!(
        shared.len(),
        1,
        "common-vertex Galerkin integration requires exactly one shared vertex"
    );

    let source_shared = shared[0].0;
    let field_shared = shared[0].1;
    let mut source_order = [source_shared, 0, 0];
    let mut field_order = [field_shared, 0, 0];

    let mut s = 1;
    for idx in 0..3 {
        if idx != source_shared {
            source_order[s] = idx;
            s += 1;
        }
    }
    let mut f = 1;
    for idx in 0..3 {
        if idx != field_shared {
            field_order[f] = idx;
            f += 1;
        }
    }

    (
        reorder_triangle(source_coords, source_order),
        reorder_triangle(field_coords, field_order),
        source_order,
        field_order,
    )
}

// ============================================================================
// Duffy self-element integration
// ============================================================================

/// Duffy-transformed self-element integration for a triangle
///
/// Splits the reference triangle into 3 sub-triangles (one per vertex),
/// applies the Duffy coordinate transform to regularize the 1/r singularity,
/// and integrates with tensor-product Gauss-Legendre quadrature on [0,1]^2.
///
/// Returns one `IntegrationResult` per basis function (for linear elements).
///
/// # Arguments
/// * `source_point` - Source/collocation point (typically element centroid or Gauss point)
/// * `source_normal` - Unit normal at the source point
/// * `element_coords` - Triangle vertex coordinates (3x3)
/// * `physics` - Physics parameters (wave number, etc.)
/// * `num_basis` - Number of basis functions: 1 for constant, 3 for linear
/// * `gauss_order` - Gauss-Legendre quadrature order per dimension
pub fn duffy_self_triangle(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    physics: &PhysicsParams,
    num_basis: usize,
    gauss_order: usize,
) -> LinearIntegrationResult {
    let mut basis_results: Vec<IntegrationResult> = (0..num_basis)
        .map(|_| IntegrationResult::default())
        .collect();

    let (pts, wts) = gauss_legendre_01(gauss_order);

    // For self-element: integrate over the element with source_point inside
    // We split the reference triangle into 3 sub-triangles, each having the
    // singularity at one vertex.
    //
    // Sub-triangle k has vertices:
    //   V_k (singularity vertex), V_{k+1 mod 3}, V_{k+2 mod 3}
    //
    // The Duffy transform for each sub-triangle maps (eta1, eta2) in [0,1]^2
    // to the sub-triangle, with the singularity at eta1=0.

    // Reference triangle vertices
    let ref_verts = [(0.0_f64, 0.0_f64), (1.0, 0.0), (0.0, 1.0)];

    for k in 0..3 {
        // Singularity vertex V_k, opposite vertices V_a, V_b
        let (xk, yk) = ref_verts[k];
        let (xa, ya) = ref_verts[(k + 1) % 3];
        let (xb, yb) = ref_verts[(k + 2) % 3];

        // The Duffy transform for sub-triangle k maps (eta1, eta2) in [0,1]^2:
        //
        // The sub-triangle has V_k at the singular point.
        // We parameterize as:
        //   point = V_k + eta1 * [(1-eta2)*(V_a - V_k) + eta2*(V_b - V_k)]
        //
        // This maps:
        //   eta1=0 -> V_k (singular point)
        //   eta1=1, eta2=0 -> V_a
        //   eta1=1, eta2=1 -> V_b
        //
        // Jacobian of this transform = eta1 * |det(d(xi,eta)/d(eta1,eta2))|
        // For a linear triangle, this simplifies to:
        //   J_duffy = eta1 * area_subtri / area_ref_tri
        // But we compute it more carefully.

        let dxa = xa - xk;
        let dya = ya - yk;
        let dxb = xb - xk;
        let dyb = yb - yk;

        // The parametric mapping is:
        //   xi  = xk + eta1 * ((1-eta2)*dxa + eta2*dxb)
        //   eta = yk + eta1 * ((1-eta2)*dya + eta2*dyb)
        //
        // The Jacobian determinant of this 2D mapping is:
        //   |d(xi,eta)/d(eta1,eta2)| = eta1 * |dxa*dyb - dxb*dya|
        //
        // This eta1 factor is precisely what cancels the 1/r singularity.
        let sub_tri_det = (dxa * dyb - dxb * dya).abs();

        for (i, &eta1) in pts.iter().enumerate() {
            for (j, &eta2) in pts.iter().enumerate() {
                let w = wts[i] * wts[j];

                // Map to reference triangle coordinates
                let xi_ref = xk + eta1 * ((1.0 - eta2) * dxa + eta2 * dxb);
                let eta_ref = yk + eta1 * ((1.0 - eta2) * dya + eta2 * dyb);

                // Duffy Jacobian: eta1 * sub_tri_det
                let duffy_jacobian = eta1 * sub_tri_det;

                // Compute physical position, normal, and surface Jacobian
                let (field_point, field_normal, surface_jacobian, shape) =
                    tri_geometry(element_coords, xi_ref, eta_ref);

                // Total quadrature weight:
                // w * duffy_jacobian * surface_jacobian
                let total_weight = w * duffy_jacobian * surface_jacobian;

                // Compute BEM kernels
                let (zg, zhh, zht, ze) = compute_kernels(
                    source_point,
                    source_normal,
                    &field_point,
                    &field_normal,
                    physics,
                    total_weight,
                );

                // Accumulate weighted by basis functions
                for b in 0..num_basis {
                    let n_b = if num_basis == 1 { 1.0 } else { shape[b] };
                    basis_results[b].g_integral += zg * n_b;
                    basis_results[b].dg_dn_integral += zhh * n_b;
                    basis_results[b].dg_dnx_integral += zht * n_b;
                    basis_results[b].d2g_dnxdny_integral += ze * n_b;
                }
            }
        }
    }

    LinearIntegrationResult { basis_results }
}

/// Duffy-transformed self-element integration (constant elements, backward compatible)
///
/// Convenience wrapper that returns a single `IntegrationResult` for constant elements.
pub fn duffy_self_triangle_constant(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    physics: &PhysicsParams,
    gauss_order: usize,
) -> IntegrationResult {
    let lin_result = duffy_self_triangle(
        source_point,
        source_normal,
        element_coords,
        physics,
        1,
        gauss_order,
    );
    lin_result.basis_results.into_iter().next().unwrap()
}

// ============================================================================
/// Galerkin single-layer integral for two separated affine triangles.
///
/// Both the test/source triangle and trial/field triangle are integrated;
/// unlike the collocation helpers this is a genuine double surface integral.
/// The function is intentionally limited to the weakly-singular single-layer
/// kernel. Double-layer and hypersingular Galerkin terms require additional
/// principal-value/finite-part regularisation and are not silently treated as
/// regular integrals here.
pub fn galerkin_single_layer_regular_triangle(
    source_coords: &DenseMatrix<f64>,
    field_coords: &DenseMatrix<f64>,
    physics: &PhysicsParams,
    gauss_order: usize,
) -> Complex64 {
    assert!(
        gauss_order > 0,
        "Galerkin quadrature order must be positive"
    );
    let (points, weights) = gauss_legendre_01(gauss_order);
    let mut integral = Complex64::new(0.0, 0.0);
    for (source_u, &source_weight_u) in points.iter().zip(&weights) {
        for (source_v, &source_weight_v) in points.iter().zip(&weights) {
            let source_xi = *source_u;
            let source_eta = (1.0 - source_u) * source_v;
            let source_reference_weight = source_weight_u * source_weight_v * (1.0 - source_u);
            let (source_point, source_normal, source_jacobian, _) =
                tri_geometry(source_coords, source_xi, source_eta);
            for (field_u, &field_weight_u) in points.iter().zip(&weights) {
                for (field_v, &field_weight_v) in points.iter().zip(&weights) {
                    let field_xi = *field_u;
                    let field_eta = (1.0 - field_u) * field_v;
                    let field_reference_weight = field_weight_u * field_weight_v * (1.0 - field_u);
                    let (field_point, field_normal, field_jacobian, _) =
                        tri_geometry(field_coords, field_xi, field_eta);
                    let weight = source_reference_weight
                        * field_reference_weight
                        * source_jacobian
                        * field_jacobian;
                    integral += compute_kernels(
                        &source_point,
                        &source_normal,
                        &field_point,
                        &field_normal,
                        physics,
                        weight,
                    )
                    .0;
                }
            }
        }
    }
    integral
}

/// Galerkin single-layer integration for separated affine triangles with
/// constant, linear, or quadratic panel-local bases.
///
/// Both surfaces are integrated with tensor Gauss quadrature. The basis count
/// must be one (constant), three (the affine barycentric P1 basis), or six
/// (the affine Tri6 Lagrange basis).
pub fn galerkin_single_layer_regular_triangle_with_basis(
    source_coords: &DenseMatrix<f64>,
    field_coords: &DenseMatrix<f64>,
    physics: &PhysicsParams,
    num_source_basis: usize,
    num_field_basis: usize,
    gauss_order: usize,
) -> GalerkinIntegrationResult {
    assert!(
        matches!(num_source_basis, 1 | 3 | 6) && matches!(num_field_basis, 1 | 3 | 6),
        "Galerkin triangle basis count must be 1, 3, or 6"
    );
    assert!(
        gauss_order > 0,
        "Galerkin quadrature order must be positive"
    );

    let (points, weights) = gauss_legendre_01(gauss_order);
    let mut result = GalerkinIntegrationResult::new(num_source_basis, num_field_basis);

    for (source_u, &source_weight_u) in points.iter().zip(&weights) {
        for (source_v, &source_weight_v) in points.iter().zip(&weights) {
            let source_xi = *source_u;
            let source_eta = (1.0 - source_u) * source_v;
            let source_reference_weight = source_weight_u * source_weight_v * (1.0 - source_u);
            let (source_point, source_normal, source_jacobian, source_shape) =
                tri_geometry(source_coords, source_xi, source_eta);

            for (field_u, &field_weight_u) in points.iter().zip(&weights) {
                for (field_v, &field_weight_v) in points.iter().zip(&weights) {
                    let field_xi = *field_u;
                    let field_eta = (1.0 - field_u) * field_v;
                    let field_reference_weight = field_weight_u * field_weight_v * (1.0 - field_u);
                    let (field_point, field_normal, field_jacobian, field_shape) =
                        tri_geometry(field_coords, field_xi, field_eta);
                    let weight = source_reference_weight
                        * field_reference_weight
                        * source_jacobian
                        * field_jacobian;
                    let (zg, zhh, zht, ze) = compute_kernels(
                        &source_point,
                        &source_normal,
                        &field_point,
                        &field_normal,
                        physics,
                        weight,
                    );

                    let source_values = galerkin_basis_values(source_shape, num_source_basis);
                    let field_values = galerkin_basis_values(field_shape, num_field_basis);
                    for (source_basis, &source_value) in
                        source_values.iter().enumerate().take(num_source_basis)
                    {
                        for (field_basis, &field_value) in
                            field_values.iter().enumerate().take(num_field_basis)
                        {
                            let basis_weight = source_value * field_value;
                            let entry = &mut result.basis_results[source_basis][field_basis];
                            entry.g_integral += zg * basis_weight;
                            entry.dg_dn_integral += zhh * basis_weight;
                            entry.dg_dnx_integral += zht * basis_weight;
                            entry.d2g_dnxdny_integral += ze * basis_weight;
                        }
                    }
                }
            }
        }
    }

    result
}

/// Galerkin single-layer integral over one affine triangle.
///
/// The outer triangle is integrated with a Duffy map. For every outer test
/// point the trial triangle is split into three subtriangles having that point
/// as a vertex, so the inner `1/r` singularity is cancelled by the Duffy
/// Jacobian. This compatibility entry point returns the constant-basis block;
/// use [`galerkin_single_layer_self_triangle_with_basis`] for P1 products.
pub fn galerkin_single_layer_self_triangle(
    element_coords: &DenseMatrix<f64>,
    physics: &PhysicsParams,
    gauss_order: usize,
) -> Complex64 {
    assert!(
        gauss_order > 0,
        "Galerkin quadrature order must be positive"
    );
    let (points, weights) = gauss_legendre_01(gauss_order);
    let vertices = (0..3)
        .map(|index| {
            [
                element_coords[[index, 0]],
                element_coords[[index, 1]],
                element_coords[[index, 2]],
            ]
        })
        .collect::<Vec<_>>();
    let mut integral = Complex64::new(0.0, 0.0);

    for (source_u, &source_weight_u) in points.iter().zip(&weights) {
        for (source_v, &source_weight_v) in points.iter().zip(&weights) {
            let source_xi = *source_u;
            let source_eta = (1.0 - source_u) * source_v;
            let source_reference_weight = source_weight_u * source_weight_v * (1.0 - source_u);
            let (source_point, source_normal, source_jacobian, _) =
                tri_geometry(element_coords, source_xi, source_eta);
            for edge in 0..3 {
                let first_vertex = vertices[edge];
                let second_vertex = vertices[(edge + 1) % 3];
                let first_delta = [
                    first_vertex[0] - source_point[0],
                    first_vertex[1] - source_point[1],
                    first_vertex[2] - source_point[2],
                ];
                let second_delta = [
                    second_vertex[0] - source_point[0],
                    second_vertex[1] - source_point[1],
                    second_vertex[2] - source_point[2],
                ];
                let cross = cross_product(first_delta, second_delta);
                let subtriangle_jacobian =
                    (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
                if subtriangle_jacobian <= 1.0e-20 {
                    continue;
                }
                for (radial, &radial_weight) in points.iter().zip(&weights) {
                    for (edge_parameter, &edge_weight) in points.iter().zip(&weights) {
                        let field_point = [
                            source_point[0]
                                + radial
                                    * ((1.0 - edge_parameter) * first_delta[0]
                                        + edge_parameter * second_delta[0]),
                            source_point[1]
                                + radial
                                    * ((1.0 - edge_parameter) * first_delta[1]
                                        + edge_parameter * second_delta[1]),
                            source_point[2]
                                + radial
                                    * ((1.0 - edge_parameter) * first_delta[2]
                                        + edge_parameter * second_delta[2]),
                        ];
                        let field_weight =
                            radial_weight * edge_weight * radial * subtriangle_jacobian;
                        integral += compute_kernels(
                            &source_point,
                            &source_normal,
                            &field_point,
                            &source_normal,
                            physics,
                            source_reference_weight * source_jacobian * field_weight,
                        )
                        .0;
                    }
                }
            }
        }
    }
    integral
}

// Collocation common edge integration
// ============================================================================

/// Galerkin self-panel single-layer integration for constant, linear, or
/// quadratic bases.
///
/// The outer test point is integrated over the triangle with a Duffy map. The
/// trial triangle is split into three subtriangles rooted at that point, so
/// the inner weak singularity is cancelled by the radial Jacobian.
pub fn galerkin_single_layer_self_triangle_with_basis(
    element_coords: &DenseMatrix<f64>,
    physics: &PhysicsParams,
    num_source_basis: usize,
    num_field_basis: usize,
    gauss_order: usize,
) -> GalerkinIntegrationResult {
    assert!(
        matches!(num_source_basis, 1 | 3 | 6) && matches!(num_field_basis, 1 | 3 | 6),
        "Galerkin triangle basis count must be 1, 3, or 6"
    );
    assert!(
        gauss_order > 0,
        "Galerkin quadrature order must be positive"
    );
    let (points, weights) = gauss_legendre_01(gauss_order);
    let vertices = (0..3)
        .map(|index| {
            [
                element_coords[[index, 0]],
                element_coords[[index, 1]],
                element_coords[[index, 2]],
            ]
        })
        .collect::<Vec<_>>();
    let mut result = GalerkinIntegrationResult::new(num_source_basis, num_field_basis);

    for (source_u, &source_weight_u) in points.iter().zip(&weights) {
        for (source_v, &source_weight_v) in points.iter().zip(&weights) {
            let source_xi = *source_u;
            let source_eta = (1.0 - source_u) * source_v;
            let source_reference_weight = source_weight_u * source_weight_v * (1.0 - source_u);
            let (source_point, source_normal, source_jacobian, source_shape) =
                tri_geometry(element_coords, source_xi, source_eta);

            for edge in 0..3 {
                let first_vertex = vertices[edge];
                let second_vertex = vertices[(edge + 1) % 3];
                let first_delta = [
                    first_vertex[0] - source_point[0],
                    first_vertex[1] - source_point[1],
                    first_vertex[2] - source_point[2],
                ];
                let second_delta = [
                    second_vertex[0] - source_point[0],
                    second_vertex[1] - source_point[1],
                    second_vertex[2] - source_point[2],
                ];
                let cross = cross_product(first_delta, second_delta);
                let subtriangle_jacobian =
                    (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
                if subtriangle_jacobian <= 1.0e-20 {
                    continue;
                }

                for (radial, &radial_weight) in points.iter().zip(&weights) {
                    for (edge_parameter, &edge_weight) in points.iter().zip(&weights) {
                        let field_point = [
                            source_point[0]
                                + radial
                                    * ((1.0 - edge_parameter) * first_delta[0]
                                        + edge_parameter * second_delta[0]),
                            source_point[1]
                                + radial
                                    * ((1.0 - edge_parameter) * first_delta[1]
                                        + edge_parameter * second_delta[1]),
                            source_point[2]
                                + radial
                                    * ((1.0 - edge_parameter) * first_delta[2]
                                        + edge_parameter * second_delta[2]),
                        ];
                        let mut field_shape = [
                            (1.0 - radial) * source_shape[0],
                            (1.0 - radial) * source_shape[1],
                            (1.0 - radial) * source_shape[2],
                        ];
                        field_shape[edge] += radial * (1.0 - edge_parameter);
                        field_shape[(edge + 1) % 3] += radial * edge_parameter;
                        let field_weight =
                            radial_weight * edge_weight * radial * subtriangle_jacobian;
                        let (zg, zhh, zht, ze) = compute_kernels(
                            &source_point,
                            &source_normal,
                            &field_point,
                            &source_normal,
                            physics,
                            source_reference_weight * source_jacobian * field_weight,
                        );

                        let source_values = galerkin_basis_values(source_shape, num_source_basis);
                        let field_values = galerkin_basis_values(field_shape, num_field_basis);
                        for (source_basis, &source_value) in source_values.iter().enumerate() {
                            for (field_basis, &field_value) in field_values.iter().enumerate() {
                                let basis_weight = source_value * field_value;
                                let entry = &mut result.basis_results[source_basis][field_basis];
                                entry.g_integral += zg * basis_weight;
                                entry.dg_dn_integral += zhh * basis_weight;
                                entry.dg_dnx_integral += zht * basis_weight;
                                entry.d2g_dnxdny_integral += ze * basis_weight;
                            }
                        }
                    }
                }
            }
        }
    }

    result
}

const COMMON_EDGE_COLLOCATION_CHARTS: usize = 5;
const COMMON_VERTEX_COLLOCATION_CHARTS: usize = 2;
const COMMON_EDGE_GALERKIN_CHARTS: usize = 6;
const COMMON_VERTEX_GALERKIN_CHARTS: usize = 2;

// Curved Quad4 maps have smooth but sharply varying isoparametric Jacobians,
// especially after a singular chart removes the kernel pole. Orders below this
// floor are useful for cheap exploratory calls, but are not resolved enough to
// serve as an adjacent-order admission sample. Keep the floor explicit and
// advance the effective order with the caller's order so adjacent-order
// admission still compares distinct rules.
const QUAD4_MIN_ORDER: usize = 8;

#[inline]
fn quad4_effective_order(gauss_order: usize) -> usize {
    gauss_order.saturating_add(7).max(QUAD4_MIN_ORDER)
}

fn basis_value(
    shape: [f64; 3],
    reorder: [usize; 3],
    original_basis: usize,
    num_basis: usize,
) -> f64 {
    if num_basis == 1 {
        return 1.0;
    }
    if num_basis == 3 {
        return reorder
            .iter()
            .position(|&idx| idx == original_basis)
            .map_or(0.0, |reordered_idx| shape[reordered_idx]);
    }
    assert_eq!(
        num_basis, 6,
        "Galerkin triangle basis count must be 1, 3, or 6"
    );
    if original_basis < 3 {
        return reorder
            .iter()
            .position(|&idx| idx == original_basis)
            .map_or(0.0, |reordered_idx| {
                galerkin_basis_values(shape, 6)[reordered_idx]
            });
    }
    let edges = [(0usize, 1usize), (1, 2), (2, 0)];
    let (first, second) = edges[original_basis - 3];
    let first = reorder
        .iter()
        .position(|&idx| idx == first)
        .expect("quadratic edge endpoint must be present in reorder");
    let second = reorder
        .iter()
        .position(|&idx| idx == second)
        .expect("quadratic edge endpoint must be present in reorder");
    4.0 * shape[first] * shape[second]
}

/// Galerkin common-edge integration over both source and field triangles.
///
/// This routine integrates source and field basis products. It is intended for
/// Galerkin or verification paths; the production TBEM assembly remains a
/// collocation method.
pub fn sauter_schwab_galerkin_common_edge(
    source_coords: &DenseMatrix<f64>,
    field_coords: &DenseMatrix<f64>,
    physics: &PhysicsParams,
    num_source_basis: usize,
    num_field_basis: usize,
    gauss_order: usize,
) -> GalerkinIntegrationResult {
    let mut result = GalerkinIntegrationResult::new(num_source_basis, num_field_basis);
    let (source_reordered, field_reordered, source_order, field_order) =
        reorder_common_edge_pair(source_coords, field_coords);
    let (pts, wts) = gauss_legendre_01(gauss_order);

    for chart in 0..COMMON_EDGE_GALERKIN_CHARTS {
        for i1 in 0..pts.len() {
            for i2 in 0..pts.len() {
                for i3 in 0..pts.len() {
                    for i4 in 0..pts.len() {
                        let eta1 = pts[i1];
                        let eta2 = pts[i2];
                        let eta3 = pts[i3];
                        let eta4 = pts[i4];
                        let w = wts[i1] * wts[i2] * wts[i3] * wts[i4];
                        let (xi_s, eta_s, xi_f, eta_f, chart_jac) =
                            common_edge_galerkin_chart(chart, eta1, eta2, eta3, eta4);
                        if chart_jac < 1e-20 {
                            continue;
                        }

                        let (source_point, source_normal, source_jac, source_shape) =
                            tri_geometry(&source_reordered, xi_s, eta_s);
                        let (field_point, field_normal, field_jac, field_shape) =
                            tri_geometry(&field_reordered, xi_f, eta_f);
                        let total_weight = w * chart_jac * source_jac * field_jac;
                        let (zg, zhh, zht, ze) = compute_kernels(
                            &source_point,
                            &source_normal,
                            &field_point,
                            &field_normal,
                            physics,
                            total_weight,
                        );

                        for i in 0..num_source_basis {
                            let n_source =
                                basis_value(source_shape, source_order, i, num_source_basis);
                            for j in 0..num_field_basis {
                                let n_field =
                                    basis_value(field_shape, field_order, j, num_field_basis);
                                let weight = n_source * n_field;
                                let entry = &mut result.basis_results[i][j];
                                entry.g_integral += zg * weight;
                                entry.dg_dn_integral += zhh * weight;
                                entry.dg_dnx_integral += zht * weight;
                                entry.d2g_dnxdny_integral += ze * weight;
                            }
                        }
                    }
                }
            }
        }
    }

    result
}

/// Galerkin common-vertex integration over both source and field triangles.
pub fn sauter_schwab_galerkin_common_vertex(
    source_coords: &DenseMatrix<f64>,
    field_coords: &DenseMatrix<f64>,
    physics: &PhysicsParams,
    num_source_basis: usize,
    num_field_basis: usize,
    gauss_order: usize,
) -> GalerkinIntegrationResult {
    let mut result = GalerkinIntegrationResult::new(num_source_basis, num_field_basis);
    let (source_reordered, field_reordered, source_order, field_order) =
        reorder_common_vertex_pair(source_coords, field_coords);
    let (pts, wts) = gauss_legendre_01(gauss_order);

    for chart in 0..COMMON_VERTEX_GALERKIN_CHARTS {
        for i1 in 0..pts.len() {
            for i2 in 0..pts.len() {
                for i3 in 0..pts.len() {
                    for i4 in 0..pts.len() {
                        let eta1 = pts[i1];
                        let eta2 = pts[i2];
                        let eta3 = pts[i3];
                        let eta4 = pts[i4];
                        let w = wts[i1] * wts[i2] * wts[i3] * wts[i4];
                        let (xi_s, eta_s, xi_f, eta_f, chart_jac) =
                            common_vertex_chart(chart, eta1, eta2, eta3, eta4);
                        if chart_jac < 1e-20 {
                            continue;
                        }

                        let (source_point, source_normal, source_jac, source_shape) =
                            tri_geometry(&source_reordered, xi_s, eta_s);
                        let (field_point, field_normal, field_jac, field_shape) =
                            tri_geometry(&field_reordered, xi_f, eta_f);
                        let total_weight = w * chart_jac * source_jac * field_jac;
                        let (zg, zhh, zht, ze) = compute_kernels(
                            &source_point,
                            &source_normal,
                            &field_point,
                            &field_normal,
                            physics,
                            total_weight,
                        );

                        for i in 0..num_source_basis {
                            let n_source =
                                basis_value(source_shape, source_order, i, num_source_basis);
                            for j in 0..num_field_basis {
                                let n_field =
                                    basis_value(field_shape, field_order, j, num_field_basis);
                                let weight = n_source * n_field;
                                let entry = &mut result.basis_results[i][j];
                                entry.g_integral += zg * weight;
                                entry.dg_dn_integral += zhh * weight;
                                entry.dg_dnx_integral += zht * weight;
                                entry.d2g_dnxdny_integral += ze * weight;
                            }
                        }
                    }
                }
            }
        }
    }

    result
}

/// Collocation integration helper for two triangles sharing an edge
///
/// The two triangles share 2 nodes (an edge). This uses Sauter-Schwab-style
/// chart maps to regularize the field integration around a fixed source
/// collocation point.
///
/// This is not the six-chart Galerkin common-edge double integral from
/// Sauter-Schwab. The returned source reference coordinates from the chart maps
/// are deliberately unused, and the source triangle Jacobian is not included in
/// the quadrature weight.
///
/// The `common_edge` parameter gives the local indices (in the field element)
/// of the two shared nodes, defining which edge is common.
///
/// # Arguments
/// * `source_point` - Collocation point on the source element
/// * `source_normal` - Unit normal at source point
/// * `source_coords` - Source triangle vertices (3x3)
/// * `field_coords` - Field triangle vertices (3x3)
/// * `common_edge` - Local indices of shared nodes in field element
/// * `physics` - Physics parameters
/// * `num_basis` - Number of basis functions (1 for constant, 3 for linear)
/// * `gauss_order` - Quadrature order per dimension
pub fn sauter_schwab_common_edge(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    _source_coords: &DenseMatrix<f64>,
    field_coords: &DenseMatrix<f64>,
    common_edge: [usize; 2],
    physics: &PhysicsParams,
    num_basis: usize,
    gauss_order: usize,
) -> LinearIntegrationResult {
    let mut basis_results: Vec<IntegrationResult> = (0..num_basis)
        .map(|_| IntegrationResult::default())
        .collect();

    let (pts, wts) = gauss_legendre_01(gauss_order);

    // Identify the shared edge vertices and the non-shared vertex in field element
    let n0 = common_edge[0]; // first shared node (local index in field)
    let n1 = common_edge[1]; // second shared node (local index in field)

    // Find the non-shared vertex in field element
    let n_free = (0..3).find(|&i| i != n0 && i != n1).unwrap();

    // Reorder field triangle so that:
    //   field vertex 0 = common_edge[0]
    //   field vertex 1 = common_edge[1]
    //   field vertex 2 = non-shared vertex
    let field_reorder = [n0, n1, n_free];

    // Build reordered field coordinates
    let mut field_reordered = DenseMatrix::<f64>::zeros(3, 3);
    for (new_idx, &old_idx) in field_reorder.iter().enumerate() {
        for j in 0..3 {
            field_reordered[[new_idx, j]] = field_coords[[old_idx, j]];
        }
    }

    // The collocation common-edge helper uses 5 field-side charts. A full
    // Galerkin common-edge Sauter-Schwab rule would integrate over source and
    // field triangles and uses a different six-chart decomposition.
    for chart in 0..COMMON_EDGE_COLLOCATION_CHARTS {
        for i1 in 0..pts.len() {
            for i2 in 0..pts.len() {
                for i3 in 0..pts.len() {
                    for i4 in 0..pts.len() {
                        let eta1 = pts[i1];
                        let eta2 = pts[i2];
                        let eta3 = pts[i3];
                        let eta4 = pts[i4];
                        let w = wts[i1] * wts[i2] * wts[i3] * wts[i4];

                        // Map (eta1..eta4) to chart coordinates. The source
                        // coordinates are ignored for collocation assembly.
                        let (_xi_s, _eta_s, xi_f, eta_f, chart_jac) =
                            common_edge_chart(chart, eta1, eta2, eta3, eta4);

                        // Skip degenerate points
                        if chart_jac < 1e-20 {
                            continue;
                        }

                        // Compute geometry on source triangle (for field normal at y)
                        // We only need the field point and normal for the kernel
                        let (field_point, field_normal, field_jac, _field_shape) =
                            tri_geometry(&field_reordered, xi_f, eta_f);

                        // Source is a fixed collocation point, not integrated over.
                        // src_jac must NOT be included in the quadrature weight.

                        let total_weight = w * chart_jac * field_jac;

                        // Compute kernels using the actual source collocation point
                        // and the field quadrature point
                        let (zg, zhh, zht, ze) = compute_kernels(
                            source_point,
                            source_normal,
                            &field_point,
                            &field_normal,
                            physics,
                            total_weight,
                        );

                        // Map basis function indices back to original field ordering
                        let orig_shape = tri_shape_functions(xi_f, eta_f);
                        for (b, br) in basis_results.iter_mut().enumerate().take(num_basis) {
                            let n_b = if num_basis == 1 {
                                1.0
                            } else {
                                // We want shape function for original vertex b
                                // Find which reordered index maps to b
                                let reordered_idx = field_reorder.iter().position(|&x| x == b);
                                match reordered_idx {
                                    Some(ri) => orig_shape[ri],
                                    None => 0.0,
                                }
                            };
                            br.g_integral += zg * n_b;
                            br.dg_dn_integral += zhh * n_b;
                            br.dg_dnx_integral += zht * n_b;
                            br.d2g_dnxdny_integral += ze * n_b;
                        }
                    }
                }
            }
        }
    }

    LinearIntegrationResult { basis_results }
}

/// Common-edge collocation chart mapping
///
/// Maps (eta1, eta2, eta3, eta4) in [0,1]^4 to
/// (xi_src, eta_src, xi_fld, eta_fld) in the reference triangles,
/// returning (xi_s, eta_s, xi_f, eta_f, jacobian).
///
/// Five charts cover the field-side singular region near the shared edge for
/// the collocation helper. This intentionally is not the full Galerkin chart
/// count.
fn common_edge_chart(
    chart: usize,
    eta1: f64,
    eta2: f64,
    eta3: f64,
    eta4: f64,
) -> (f64, f64, f64, f64, f64) {
    // The common-edge transform uses eta1 as the
    // "distance from the edge" variable. The Jacobian includes an eta1^2
    // factor that regularizes the edge singularity.

    match chart {
        0 => {
            // Chart 1: both triangles parameterized from edge outward
            let xi_s = eta1;
            let eta_s = eta1 * eta2;
            let xi_f = eta1 * eta3;
            let eta_f = eta1 * eta3 * eta4;
            let jac = eta1 * eta1 * eta1 * eta3;
            (xi_s, eta_s, xi_f, eta_f, jac)
        }
        1 => {
            // Chart 2
            let xi_s = eta1 * eta2;
            let eta_s = eta1;
            let xi_f = eta1 * eta3;
            let eta_f = eta1 * eta3 * eta4;
            let jac = eta1 * eta1 * eta1 * eta3;
            (xi_s, eta_s, xi_f, eta_f, jac)
        }
        2 => {
            // Chart 3
            let xi_s = eta1;
            let eta_s = eta1 * eta2;
            let xi_f = eta1 * eta3 * eta4;
            let eta_f = eta1 * eta3;
            let jac = eta1 * eta1 * eta1 * eta3;
            (xi_s, eta_s, xi_f, eta_f, jac)
        }
        3 => {
            // Chart 4
            let xi_s = eta1 * eta2;
            let eta_s = eta1;
            let xi_f = eta1 * eta3 * eta4;
            let eta_f = eta1 * eta3;
            let jac = eta1 * eta1 * eta1 * eta3;
            (xi_s, eta_s, xi_f, eta_f, jac)
        }
        4 => {
            // Chart 5: central chart
            let xi_s = eta1 * eta3;
            let eta_s = eta1 * eta3 * eta4;
            let xi_f = eta1;
            let eta_f = eta1 * eta2;
            let jac = eta1 * eta1 * eta1 * eta3;
            (xi_s, eta_s, xi_f, eta_f, jac)
        }
        // No default case per CLAUDE.md: crash hard on unknown values
        _ => unreachable!("Invalid chart index: {chart}"),
    }
}

fn common_edge_galerkin_chart(
    chart: usize,
    eta1: f64,
    eta2: f64,
    eta3: f64,
    eta4: f64,
) -> (f64, f64, f64, f64, f64) {
    assert!(chart < COMMON_EDGE_GALERKIN_CHARTS);

    // Reorder each triangle so that the common edge is eta = 0.  Let x be
    // the shared-edge coordinate and normalize the two heights by their
    // available distances to the opposite vertex.  The singular diagonal is
    // then (d, p, q) = (0, 0, 0), where d is the edge-coordinate separation
    // and p/q are the normalized source/field heights.  Splitting the edge
    // square into the two orderings x_s >= x_f and x_f >= x_s, and splitting
    // the positive (d, p, q) octant into its three largest-coordinate cones,
    // gives six non-overlapping charts.  The three-cone Duffy map contributes
    // eta2^2; the physical coordinate map contributes
    // (1 - x) * (1 - xi_s) * (1 - xi_f).
    let x = eta1;
    let radial = eta2;
    let first_angle = eta3;
    let second_angle = eta4;
    let source_leads = chart < 3;
    let cone = chart % 3;

    let (separation, source_height, field_height) = match cone {
        0 => (radial, radial * first_angle, radial * second_angle),
        1 => (radial * first_angle, radial, radial * second_angle),
        2 => (radial * first_angle, radial * second_angle, radial),
        _ => unreachable!(),
    };

    let source_edge = if source_leads {
        x + (1.0 - x) * separation
    } else {
        x
    };
    let field_edge = if source_leads {
        x
    } else {
        x + (1.0 - x) * separation
    };
    let xi_s = source_edge;
    let eta_s = (1.0 - source_edge) * source_height;
    let xi_f = field_edge;
    let eta_f = (1.0 - field_edge) * field_height;
    let jac = (1.0 - x) * (1.0 - source_edge) * (1.0 - field_edge) * radial * radial;
    (xi_s, eta_s, xi_f, eta_f, jac)
}

// ============================================================================
// Collocation common vertex integration
// ============================================================================

/// Collocation integration helper for two triangles sharing a vertex
///
/// The two triangles share 1 node (a vertex). This keeps the source point fixed
/// at the collocation point and regularizes the field integration near the
/// shared vertex with Sauter-Schwab-style charts.
///
/// # Arguments
/// * `source_point` - Collocation point on the source element
/// * `source_normal` - Unit normal at source point
/// * `source_coords` - Source triangle vertices (3x3)
/// * `field_coords` - Field triangle vertices (3x3)
/// * `common_vertex` - Local index of shared node in field element
/// * `physics` - Physics parameters
/// * `num_basis` - Number of basis functions (1 for constant, 3 for linear)
/// * `gauss_order` - Quadrature order per dimension
pub fn sauter_schwab_common_vertex(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    _source_coords: &DenseMatrix<f64>,
    field_coords: &DenseMatrix<f64>,
    common_vertex: usize,
    physics: &PhysicsParams,
    num_basis: usize,
    gauss_order: usize,
) -> LinearIntegrationResult {
    let mut basis_results: Vec<IntegrationResult> = (0..num_basis)
        .map(|_| IntegrationResult::default())
        .collect();

    let (pts, wts) = gauss_legendre_01(gauss_order);

    // Reorder field triangle so that the common vertex is at index 0
    let v0 = common_vertex;
    let v1 = (common_vertex + 1) % 3;
    let v2 = (common_vertex + 2) % 3;
    let field_reorder = [v0, v1, v2];

    let mut field_reordered = DenseMatrix::<f64>::zeros(3, 3);
    for (new_idx, &old_idx) in field_reorder.iter().enumerate() {
        for j in 0..3 {
            field_reordered[[new_idx, j]] = field_coords[[old_idx, j]];
        }
    }

    for chart in 0..COMMON_VERTEX_COLLOCATION_CHARTS {
        for i1 in 0..pts.len() {
            for i2 in 0..pts.len() {
                for i3 in 0..pts.len() {
                    for i4 in 0..pts.len() {
                        let eta1 = pts[i1];
                        let eta2 = pts[i2];
                        let eta3 = pts[i3];
                        let eta4 = pts[i4];
                        let w = wts[i1] * wts[i2] * wts[i3] * wts[i4];

                        let (_xi_s, _eta_s, xi_f, eta_f, chart_jac) =
                            common_vertex_chart(chart, eta1, eta2, eta3, eta4);

                        if chart_jac < 1e-20 {
                            continue;
                        }

                        // Compute geometry
                        let (field_point, field_normal, field_jac, _field_shape) =
                            tri_geometry(&field_reordered, xi_f, eta_f);

                        // Source is a fixed collocation point — no src_jac in weight.

                        let total_weight = w * chart_jac * field_jac;

                        let (zg, zhh, zht, ze) = compute_kernels(
                            source_point,
                            source_normal,
                            &field_point,
                            &field_normal,
                            physics,
                            total_weight,
                        );

                        let orig_shape = tri_shape_functions(xi_f, eta_f);
                        for (b, br) in basis_results.iter_mut().enumerate().take(num_basis) {
                            let n_b = if num_basis == 1 {
                                1.0
                            } else {
                                let reordered_idx = field_reorder.iter().position(|&x| x == b);
                                match reordered_idx {
                                    Some(ri) => orig_shape[ri],
                                    None => 0.0,
                                }
                            };
                            br.g_integral += zg * n_b;
                            br.dg_dn_integral += zhh * n_b;
                            br.dg_dnx_integral += zht * n_b;
                            br.d2g_dnxdny_integral += ze * n_b;
                        }
                    }
                }
            }
        }
    }

    LinearIntegrationResult { basis_results }
}

/// Common-vertex collocation chart mapping
///
/// Maps (eta1, eta2, eta3, eta4) in [0,1]^4 to
/// (xi_src, eta_src, xi_fld, eta_fld) in the reference triangles.
///
/// For common vertex, the singularity is at the shared vertex (mapped to origin
/// of both reference triangles). The transform introduces an eta1^3 Jacobian
/// factor that regularizes the O(1/r) Green's function singularity for the
/// collocation field integral.
fn common_vertex_chart(
    chart: usize,
    eta1: f64,
    eta2: f64,
    eta3: f64,
    eta4: f64,
) -> (f64, f64, f64, f64, f64) {
    match chart {
        0 => {
            // Chart 1: parameterize source triangle from the common vertex
            let xi_s = eta1;
            let eta_s = eta1 * eta2;
            let xi_f = eta1 * eta3;
            let eta_f = eta1 * eta3 * eta4;
            // Jacobian: eta1^3 * eta3 (from the chain of substitutions)
            let jac = eta1 * eta1 * eta1 * eta3;
            (xi_s, eta_s, xi_f, eta_f, jac)
        }
        1 => {
            // Chart 2: swap roles of source and field
            let xi_s = eta1 * eta3;
            let eta_s = eta1 * eta3 * eta4;
            let xi_f = eta1;
            let eta_f = eta1 * eta2;
            let jac = eta1 * eta1 * eta1 * eta3;
            (xi_s, eta_s, xi_f, eta_f, jac)
        }
        _ => unreachable!("Invalid chart index: {chart}"),
    }
}

// ============================================================================
// Adaptive integration dispatch
// ============================================================================

/// Smart integration that selects Duffy/collocation-adjacent/regular based on adjacency
///
/// Automatically dispatches to the appropriate integration method based on the
/// topological relationship between source and field elements:
///
/// - **Self**: Duffy transform (exponentially convergent for 1/r singularity)
/// - **Common edge**: field-side collocation edge regularization
/// - **Common vertex**: field-side collocation vertex regularization
/// - **Separated**: Falls through to regular integration (caller handles this)
///
/// # Arguments
/// * `source_point` - Collocation point
/// * `source_normal` - Unit normal at source point
/// * `source_coords` - Source element vertex coordinates
/// * `field_coords` - Field element vertex coordinates
/// * `physics` - Physics parameters
/// * `adjacency` - Topological relationship between elements
/// * `num_basis` - Number of basis functions (1 for constant, 3 for linear)
/// * `gauss_order` - Quadrature order per dimension
///
/// # Returns
/// `Some(result)` for singular/near-singular cases, `None` for separated elements
/// (indicating the caller should use regular integration).
pub fn singular_integration_dispatch(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    source_coords: &DenseMatrix<f64>,
    field_coords: &DenseMatrix<f64>,
    physics: &PhysicsParams,
    adjacency: &AdjacencyType,
    num_basis: usize,
    gauss_order: usize,
) -> Option<LinearIntegrationResult> {
    match adjacency {
        AdjacencyType::SelfElement => Some(duffy_self_triangle(
            source_point,
            source_normal,
            field_coords,
            physics,
            num_basis,
            gauss_order,
        )),
        AdjacencyType::CommonEdge { common_nodes } => Some(sauter_schwab_common_edge(
            source_point,
            source_normal,
            source_coords,
            field_coords,
            *common_nodes,
            physics,
            num_basis,
            gauss_order,
        )),
        AdjacencyType::CommonVertex { common_node } => Some(sauter_schwab_common_vertex(
            source_point,
            source_normal,
            source_coords,
            field_coords,
            *common_node,
            physics,
            num_basis,
            gauss_order,
        )),
        AdjacencyType::Separated => None,
    }
}

// ============================================================================
// Tests
// ============================================================================

/// Galerkin self-pair integration on a bilinear `Quad4` panel.
///
/// For each outer tensor-product point, the inner square is split into four
/// parameter-space triangles rooted at that point.  The radial Duffy factor
/// cancels the `1/r` Green-function singularity while the production
/// isoparametric geometry and basis functions are evaluated at every mapped
/// point.  This admits self blocks without a tensor rule touching the pole.
pub fn galerkin_single_layer_self_quad4_with_basis(
    element_coords: &DenseMatrix<f64>,
    physics: &PhysicsParams,
    num_basis: usize,
    gauss_order: usize,
) -> GalerkinIntegrationResult {
    assert!(matches!(num_basis, 1 | 4 | 9));
    assert!(
        gauss_order > 0,
        "Galerkin quadrature order must be positive"
    );
    let effective_order = quad4_effective_order(gauss_order);
    let (outer_points, outer_weights) = gauss_legendre(effective_order);
    let (inner_points, inner_weights) = gauss_legendre_01(effective_order);
    let corners = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
    let mut result = GalerkinIntegrationResult::new(num_basis, num_basis);
    for (source_s, &source_weight_s) in outer_points.iter().zip(&outer_weights) {
        for (source_t, &source_weight_t) in outer_points.iter().zip(&outer_weights) {
            let source_s = *source_s;
            let source_t = *source_t;
            let (source_point, source_normal, source_jacobian) =
                quad4_geometry(element_coords, source_s, source_t);
            let source_values = quad4_basis_values(source_s, source_t, num_basis);
            let source_curls =
                quad4_surface_curls(element_coords, source_s, source_t, num_basis, source_normal);
            for edge in 0..4 {
                let first = corners[edge];
                let second = corners[(edge + 1) % 4];
                let first_delta = [first[0] - source_s, first[1] - source_t];
                let second_delta = [second[0] - source_s, second[1] - source_t];
                let triangle_jacobian =
                    (first_delta[0] * second_delta[1] - first_delta[1] * second_delta[0]).abs();
                if triangle_jacobian <= 1.0e-15 {
                    continue;
                }
                for (&radial, &radial_weight) in inner_points.iter().zip(&inner_weights) {
                    for (&edge_parameter, &edge_weight) in inner_points.iter().zip(&inner_weights) {
                        let field_s = source_s
                            + radial
                                * ((1.0 - edge_parameter) * first_delta[0]
                                    + edge_parameter * second_delta[0]);
                        let field_t = source_t
                            + radial
                                * ((1.0 - edge_parameter) * first_delta[1]
                                    + edge_parameter * second_delta[1]);
                        let (field_point, field_normal, field_jacobian) =
                            quad4_geometry(element_coords, field_s, field_t);
                        let field_curls = quad4_surface_curls(
                            element_coords,
                            field_s,
                            field_t,
                            num_basis,
                            field_normal,
                        );
                        let normal_dot = source_normal[0] * field_normal[0]
                            + source_normal[1] * field_normal[1]
                            + source_normal[2] * field_normal[2];
                        let weight = source_weight_s
                            * source_weight_t
                            * source_jacobian
                            * radial_weight
                            * edge_weight
                            * radial
                            * triangle_jacobian
                            * field_jacobian;
                        let (green, d_green_dn_y, d_green_dn_x, d2_green_dnxdny) = compute_kernels(
                            &source_point,
                            &source_normal,
                            &field_point,
                            &field_normal,
                            physics,
                            weight,
                        );
                        let field_values = quad4_basis_values(field_s, field_t, num_basis);
                        for (source_basis, &source_value) in source_values.iter().enumerate() {
                            for (field_basis, &field_value) in field_values.iter().enumerate() {
                                let basis_weight = source_value * field_value;
                                let entry = &mut result.basis_results[source_basis][field_basis];
                                entry.g_integral += green * basis_weight;
                                entry.dg_dn_integral += d_green_dn_y * basis_weight;
                                entry.dg_dnx_integral += d_green_dn_x * basis_weight;
                                entry.d2g_dnxdny_integral += d2_green_dnxdny * basis_weight;
                                let curl_dot = source_curls[source_basis][0]
                                    * field_curls[field_basis][0]
                                    + source_curls[source_basis][1] * field_curls[field_basis][1]
                                    + source_curls[source_basis][2] * field_curls[field_basis][2];
                                result.curl_integrals[source_basis][field_basis] +=
                                    green * curl_dot;
                                result.normal_mass_integrals[source_basis][field_basis] +=
                                    green * normal_dot * basis_weight;
                            }
                        }
                    }
                }
            }
        }
    }
    result
}

/// Analytic wavenumber derivative of the self-panel Quad4 Duffy integral.
///
/// The radial Duffy chart, isoparametric geometry, basis functions, and
/// physical surface-gradient tables are held fixed; only the four Helmholtz
/// kernels are differentiated.  This is the singular counterpart to
/// [`galerkin_quad4_regular_weak_form_wavenumber_derivative_with_basis`].
pub fn galerkin_single_layer_self_quad4_wavenumber_derivative_with_basis(
    element_coords: &DenseMatrix<f64>,
    physics: &PhysicsParams,
    num_basis: usize,
    gauss_order: usize,
) -> GalerkinIntegrationResult {
    assert!(matches!(num_basis, 1 | 4 | 9));
    assert!(
        gauss_order > 0,
        "Galerkin quadrature order must be positive"
    );
    let effective_order = quad4_effective_order(gauss_order);
    let (outer_points, outer_weights) = gauss_legendre(effective_order);
    let (inner_points, inner_weights) = gauss_legendre_01(effective_order);
    let corners = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
    let mut result = GalerkinIntegrationResult::new(num_basis, num_basis);

    for (source_s, &source_weight_s) in outer_points.iter().zip(&outer_weights) {
        for (source_t, &source_weight_t) in outer_points.iter().zip(&outer_weights) {
            let source_s = *source_s;
            let source_t = *source_t;
            let (source_point, source_normal, source_jacobian) =
                quad4_geometry(element_coords, source_s, source_t);
            let source_values = quad4_basis_values(source_s, source_t, num_basis);
            let source_curls =
                quad4_surface_curls(element_coords, source_s, source_t, num_basis, source_normal);

            for edge in 0..4 {
                let first = corners[edge];
                let second = corners[(edge + 1) % 4];
                let first_delta = [first[0] - source_s, first[1] - source_t];
                let second_delta = [second[0] - source_s, second[1] - source_t];
                let triangle_jacobian =
                    (first_delta[0] * second_delta[1] - first_delta[1] * second_delta[0]).abs();
                if triangle_jacobian <= 1.0e-15 {
                    continue;
                }
                for (&radial, &radial_weight) in inner_points.iter().zip(&inner_weights) {
                    for (&edge_parameter, &edge_weight) in inner_points.iter().zip(&inner_weights) {
                        let field_s = source_s
                            + radial
                                * ((1.0 - edge_parameter) * first_delta[0]
                                    + edge_parameter * second_delta[0]);
                        let field_t = source_t
                            + radial
                                * ((1.0 - edge_parameter) * first_delta[1]
                                    + edge_parameter * second_delta[1]);
                        let (field_point, field_normal, field_jacobian) =
                            quad4_geometry(element_coords, field_s, field_t);
                        let field_curls = quad4_surface_curls(
                            element_coords,
                            field_s,
                            field_t,
                            num_basis,
                            field_normal,
                        );
                        let normal_dot = source_normal[0] * field_normal[0]
                            + source_normal[1] * field_normal[1]
                            + source_normal[2] * field_normal[2];
                        let weight = source_weight_s
                            * source_weight_t
                            * source_jacobian
                            * radial_weight
                            * edge_weight
                            * radial
                            * triangle_jacobian
                            * field_jacobian;
                        let (dg_dk, ddg_dn_y_dk, ddg_dn_x_dk, dd2g_dnxdny_dk) =
                            compute_kernel_wavenumber_derivatives(
                                &source_point,
                                &source_normal,
                                &field_point,
                                &field_normal,
                                physics,
                                weight,
                            );
                        let field_values = quad4_basis_values(field_s, field_t, num_basis);
                        for (source_basis, &source_value) in source_values.iter().enumerate() {
                            for (field_basis, &field_value) in field_values.iter().enumerate() {
                                let basis_weight = source_value * field_value;
                                let entry = &mut result.basis_results[source_basis][field_basis];
                                entry.g_integral += dg_dk * basis_weight;
                                entry.dg_dn_integral += ddg_dn_y_dk * basis_weight;
                                entry.dg_dnx_integral += ddg_dn_x_dk * basis_weight;
                                entry.d2g_dnxdny_integral += dd2g_dnxdny_dk * basis_weight;
                                let curl_dot = source_curls[source_basis][0]
                                    * field_curls[field_basis][0]
                                    + source_curls[source_basis][1] * field_curls[field_basis][1]
                                    + source_curls[source_basis][2] * field_curls[field_basis][2];
                                result.curl_integrals[source_basis][field_basis] +=
                                    dg_dk * curl_dot;
                                result.normal_mass_integrals[source_basis][field_basis] +=
                                    dg_dk * normal_dot * basis_weight;
                            }
                        }
                    }
                }
            }
        }
    }
    result
}

/// Return the runtime-selected implementation for the regular Quad4 cell.
///
/// This reports only the separated regular-pair path.  Singular/touching
/// quadrilateral charts and the weak-form derivative path remain scalar and
/// are not silently promoted by this dispatch.
pub fn galerkin_single_layer_regular_quad4_dispatch() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if !quad4_force_scalar()
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
        {
            return "quad4-regular-avx2-fma";
        }
    }
    "quad4-regular-scalar"
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn quad4_force_scalar() -> bool {
    matches!(
        std::env::var("SONIUM_BEM_QUAD4_FORCE_SCALAR").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes" | "on")
    )
}

/// Production regular separated-Quad4 integration with runtime AVX2/FMA
/// dispatch on x86_64.
///
/// This path deliberately admits only separated pairs.  Self and touching
/// quadrilateral pairs need a dedicated four-sided Duffy/Sauter--Schwab chart;
/// callers must route those pairs through a singular quadrilateral
/// implementation instead of silently applying a tensor rule to the pole.
/// The returned block uses one, four, or nine constant/bilinear/biquadratic
/// tensor-product basis functions on each panel.
pub fn galerkin_single_layer_regular_quad4_with_basis(
    source_coords: &DenseMatrix<f64>,
    field_coords: &DenseMatrix<f64>,
    physics: &PhysicsParams,
    num_source_basis: usize,
    num_field_basis: usize,
    gauss_order: usize,
) -> GalerkinIntegrationResult {
    #[cfg(target_arch = "x86_64")]
    if galerkin_single_layer_regular_quad4_dispatch() == "quad4-regular-avx2-fma" {
        // SAFETY: the dispatch predicate above checks both AVX2 and FMA at
        // the call site; the SIMD implementation handles its scalar tail.
        return unsafe {
            quad4_avx2::galerkin_single_layer_regular_quad4_with_basis_avx2(
                source_coords,
                field_coords,
                physics,
                num_source_basis,
                num_field_basis,
                gauss_order,
            )
        };
    }
    galerkin_single_layer_regular_quad4_with_basis_scalar(
        source_coords,
        field_coords,
        physics,
        num_source_basis,
        num_field_basis,
        gauss_order,
    )
}

/// Scalar reference for the production regular separated-Quad4 cell.
///
/// Keeping this entry point explicit lets native-x86 evidence compare the
/// dispatched AVX2/FMA path with the exact pre-existing accumulation order.
pub fn galerkin_single_layer_regular_quad4_with_basis_scalar(
    source_coords: &DenseMatrix<f64>,
    field_coords: &DenseMatrix<f64>,
    physics: &PhysicsParams,
    num_source_basis: usize,
    num_field_basis: usize,
    gauss_order: usize,
) -> GalerkinIntegrationResult {
    assert!(
        matches!(num_source_basis, 1 | 4 | 9) && matches!(num_field_basis, 1 | 4 | 9),
        "Galerkin quadrilateral basis count must be 1, 4, or 9"
    );
    assert!(
        gauss_order > 0,
        "Galerkin quadrature order must be positive"
    );

    let effective_order = quad4_effective_order(gauss_order);
    let (points, weights) = gauss_legendre(effective_order);
    let mut result = GalerkinIntegrationResult::new(num_source_basis, num_field_basis);

    for (source_s, &source_weight_s) in points.iter().zip(&weights) {
        for (source_t, &source_weight_t) in points.iter().zip(&weights) {
            let (source_point, source_normal, source_jacobian) =
                quad4_geometry(source_coords, *source_s, *source_t);
            let source_values = quad4_basis_values(*source_s, *source_t, num_source_basis);
            let source_curls = quad4_surface_curls(
                source_coords,
                *source_s,
                *source_t,
                num_source_basis,
                source_normal,
            );
            for (field_s, &field_weight_s) in points.iter().zip(&weights) {
                for (field_t, &field_weight_t) in points.iter().zip(&weights) {
                    let (field_point, field_normal, field_jacobian) =
                        quad4_geometry(field_coords, *field_s, *field_t);
                    let field_values = quad4_basis_values(*field_s, *field_t, num_field_basis);
                    let field_curls = quad4_surface_curls(
                        field_coords,
                        *field_s,
                        *field_t,
                        num_field_basis,
                        field_normal,
                    );
                    let normal_dot = source_normal[0] * field_normal[0]
                        + source_normal[1] * field_normal[1]
                        + source_normal[2] * field_normal[2];
                    let weight = source_weight_s
                        * source_weight_t
                        * field_weight_s
                        * field_weight_t
                        * source_jacobian
                        * field_jacobian;
                    let (green, d_green_dn_y, d_green_dn_x, d2_green_dnxdny) = compute_kernels(
                        &source_point,
                        &source_normal,
                        &field_point,
                        &field_normal,
                        physics,
                        weight,
                    );
                    for (source_basis, &source_value) in source_values.iter().enumerate() {
                        for (field_basis, &field_value) in field_values.iter().enumerate() {
                            let basis_weight = source_value * field_value;
                            let entry = &mut result.basis_results[source_basis][field_basis];
                            entry.g_integral += green * basis_weight;
                            entry.dg_dn_integral += d_green_dn_y * basis_weight;
                            entry.dg_dnx_integral += d_green_dn_x * basis_weight;
                            entry.d2g_dnxdny_integral += d2_green_dnxdny * basis_weight;
                            let curl_dot = source_curls[source_basis][0]
                                * field_curls[field_basis][0]
                                + source_curls[source_basis][1] * field_curls[field_basis][1]
                                + source_curls[source_basis][2] * field_curls[field_basis][2];
                            result.curl_integrals[source_basis][field_basis] += green * curl_dot;
                            result.normal_mass_integrals[source_basis][field_basis] +=
                                green * normal_dot * basis_weight;
                        }
                    }
                }
            }
        }
    }

    result
}

#[cfg(target_arch = "x86_64")]
mod quad4_avx2 {
    use super::{
        GalerkinIntegrationResult, PhysicsParams, compute_kernels, quad4_basis_values,
        quad4_geometry, quad4_surface_curls,
    };
    use num_complex::Complex64;
    use sonium_solvers::dense::DenseMatrix;
    use std::arch::x86_64::*;
    use std::f64::consts::PI;

    struct Lane {
        source_point: [f64; 3],
        source_normal: [f64; 3],
        field_point: [f64; 3],
        field_normal: [f64; 3],
        weight: f64,
        normal_dot: f64,
        source_values: Vec<f64>,
        field_values: Vec<f64>,
        source_curls: Vec<[f64; 3]>,
        field_curls: Vec<[f64; 3]>,
    }

    fn lane(
        source_coords: &DenseMatrix<f64>,
        field_coords: &DenseMatrix<f64>,
        source_s: f64,
        source_t: f64,
        source_weight_s: f64,
        source_weight_t: f64,
        field_s: f64,
        field_t: f64,
        field_weight_s: f64,
        field_weight_t: f64,
        num_source_basis: usize,
        num_field_basis: usize,
    ) -> Lane {
        let (source_point, source_normal, source_jacobian) =
            quad4_geometry(source_coords, source_s, source_t);
        let (field_point, field_normal, field_jacobian) =
            quad4_geometry(field_coords, field_s, field_t);
        let normal_dot = source_normal[0] * field_normal[0]
            + source_normal[1] * field_normal[1]
            + source_normal[2] * field_normal[2];
        Lane {
            source_point,
            source_normal,
            field_point,
            field_normal,
            weight: source_weight_s
                * source_weight_t
                * field_weight_s
                * field_weight_t
                * source_jacobian
                * field_jacobian,
            normal_dot,
            source_values: quad4_basis_values(source_s, source_t, num_source_basis),
            field_values: quad4_basis_values(field_s, field_t, num_field_basis),
            source_curls: quad4_surface_curls(
                source_coords,
                source_s,
                source_t,
                num_source_basis,
                source_normal,
            ),
            field_curls: quad4_surface_curls(
                field_coords,
                field_s,
                field_t,
                num_field_basis,
                field_normal,
            ),
        }
    }

    #[inline]
    unsafe fn pack(values: [f64; 4]) -> __m256d {
        // SAFETY: the caller is compiled with AVX2/FMA and supplies four
        // initialized scalar lanes.
        unsafe { _mm256_set_pd(values[3], values[2], values[1], values[0]) }
    }

    #[inline]
    unsafe fn lanes(value: __m256d) -> [f64; 4] {
        let mut output = [0.0; 4];
        // SAFETY: output has exactly four f64 slots.
        unsafe { _mm256_storeu_pd(output.as_mut_ptr(), value) };
        output
    }

    /// Evaluate the complete regular Quad4 Helmholtz kernel tuple for four
    /// quadrature-point pairs. Geometry and basis tabulation stay scalar at
    /// the boundary; the distance, phase, normals, and complex kernel algebra
    /// use AVX2/FMA lanes. This keeps the production scalar semantics for
    /// mapped singular charts while accelerating the separated tensor cell.
    #[target_feature(enable = "avx2,fma")]
    unsafe fn kernels(
        batch: &[Lane; 4],
        physics: &PhysicsParams,
    ) -> [(Complex64, Complex64, Complex64, Complex64); 4] {
        // SAFETY: target_feature above guarantees the AVX2/FMA intrinsics and
        // the caller has checked the same feature set at runtime.
        unsafe {
            let source_x = pack(std::array::from_fn(|i| batch[i].source_point[0]));
            let source_y = pack(std::array::from_fn(|i| batch[i].source_point[1]));
            let source_z = pack(std::array::from_fn(|i| batch[i].source_point[2]));
            let field_x = pack(std::array::from_fn(|i| batch[i].field_point[0]));
            let field_y = pack(std::array::from_fn(|i| batch[i].field_point[1]));
            let field_z = pack(std::array::from_fn(|i| batch[i].field_point[2]));
            let source_nx = pack(std::array::from_fn(|i| batch[i].source_normal[0]));
            let source_ny = pack(std::array::from_fn(|i| batch[i].source_normal[1]));
            let source_nz = pack(std::array::from_fn(|i| batch[i].source_normal[2]));
            let field_nx = pack(std::array::from_fn(|i| batch[i].field_normal[0]));
            let field_ny = pack(std::array::from_fn(|i| batch[i].field_normal[1]));
            let field_nz = pack(std::array::from_fn(|i| batch[i].field_normal[2]));
            let weight = pack(std::array::from_fn(|i| batch[i].weight));

            let diff_x = _mm256_sub_pd(field_x, source_x);
            let diff_y = _mm256_sub_pd(field_y, source_y);
            let diff_z = _mm256_sub_pd(field_z, source_z);
            let r_sq = _mm256_fmadd_pd(
                diff_z,
                diff_z,
                _mm256_fmadd_pd(diff_y, diff_y, _mm256_mul_pd(diff_x, diff_x)),
            );
            let r = _mm256_sqrt_pd(r_sq);
            let zero = _mm256_setzero_pd();
            let one = _mm256_set1_pd(1.0);
            let epsilon = _mm256_set1_pd(1.0e-15);
            let valid = _mm256_cmp_pd(r, epsilon, _CMP_GT_OQ);
            let inv_r = _mm256_blendv_pd(zero, _mm256_div_pd(one, r), valid);
            let unit_x = _mm256_mul_pd(diff_x, inv_r);
            let unit_y = _mm256_mul_pd(diff_y, inv_r);
            let unit_z = _mm256_mul_pd(diff_z, inv_r);
            let source_field_normal_dot = _mm256_fmadd_pd(
                source_nz,
                field_nz,
                _mm256_fmadd_pd(source_ny, field_ny, _mm256_mul_pd(source_nx, field_nx)),
            );
            let r_dot_field = _mm256_fmadd_pd(
                unit_z,
                field_nz,
                _mm256_fmadd_pd(unit_y, field_ny, _mm256_mul_pd(unit_x, field_nx)),
            );
            let r_dot_source = _mm256_xor_pd(
                _mm256_fmadd_pd(
                    unit_z,
                    source_nz,
                    _mm256_fmadd_pd(unit_y, source_ny, _mm256_mul_pd(unit_x, source_nx)),
                ),
                _mm256_set1_pd(-0.0),
            );
            let r_squared_inverse = _mm256_mul_pd(inv_r, inv_r);
            let wavenumber = physics.wave_number;
            let wavenumber_squared = _mm256_set1_pd(wavenumber * wavenumber);
            let wavenumber_im = _mm256_set1_pd(physics.harmonic_factor * wavenumber);
            let phase = _mm256_mul_pd(wavenumber_im, r);
            let (sin_phase, cos_phase) = sonium_fastmath::sincos_f64x4(phase);
            let g_scale = _mm256_mul_pd(
                _mm256_mul_pd(weight, _mm256_set1_pd(1.0 / (4.0 * PI))),
                inv_r,
            );
            let g_re = _mm256_mul_pd(g_scale, cos_phase);
            let g_im = _mm256_mul_pd(g_scale, sin_phase);
            let neg_inv_r = _mm256_xor_pd(inv_r, _mm256_set1_pd(-0.0));
            let h_base_re = _mm256_fnmadd_pd(g_im, wavenumber_im, _mm256_mul_pd(g_re, neg_inv_r));
            let h_base_im = _mm256_fmadd_pd(g_im, neg_inv_r, _mm256_mul_pd(g_re, wavenumber_im));
            let h_re = _mm256_mul_pd(h_base_re, r_dot_field);
            let h_im = _mm256_mul_pd(h_base_im, r_dot_field);
            let ht_re = _mm256_mul_pd(h_base_re, r_dot_source);
            let ht_im = _mm256_mul_pd(h_base_im, r_dot_source);
            let rq = _mm256_mul_pd(r_dot_field, r_dot_source);
            let three_over_r2_minus_k2 =
                _mm256_fnmadd_pd(_mm256_set1_pd(3.0), r_squared_inverse, wavenumber_squared);
            let three_over_r2_minus_k2 =
                _mm256_xor_pd(three_over_r2_minus_k2, _mm256_set1_pd(-0.0));
            let e_factor_re = _mm256_fmadd_pd(
                three_over_r2_minus_k2,
                rq,
                _mm256_mul_pd(source_field_normal_dot, r_squared_inverse),
            );
            let three_rq_plus_normals =
                _mm256_fmadd_pd(_mm256_set1_pd(3.0), rq, source_field_normal_dot);
            let e_factor_im = _mm256_mul_pd(
                _mm256_mul_pd(_mm256_xor_pd(wavenumber_im, _mm256_set1_pd(-0.0)), inv_r),
                three_rq_plus_normals,
            );
            let e_re = _mm256_fnmadd_pd(g_im, e_factor_im, _mm256_mul_pd(g_re, e_factor_re));
            let e_im = _mm256_fmadd_pd(g_im, e_factor_re, _mm256_mul_pd(g_re, e_factor_im));
            let g_re = lanes(g_re);
            let g_im = lanes(g_im);
            let h_re = lanes(h_re);
            let h_im = lanes(h_im);
            let ht_re = lanes(ht_re);
            let ht_im = lanes(ht_im);
            let e_re = lanes(e_re);
            let e_im = lanes(e_im);
            std::array::from_fn(|i| {
                (
                    Complex64::new(g_re[i], g_im[i]),
                    Complex64::new(h_re[i], h_im[i]),
                    Complex64::new(ht_re[i], ht_im[i]),
                    Complex64::new(e_re[i], e_im[i]),
                )
            })
        }
    }

    fn accumulate(
        result: &mut GalerkinIntegrationResult,
        lane: &Lane,
        (green, d_green_dn_y, d_green_dn_x, d2_green_dnxdny): (
            Complex64,
            Complex64,
            Complex64,
            Complex64,
        ),
    ) {
        for (source_basis, &source_value) in lane.source_values.iter().enumerate() {
            for (field_basis, &field_value) in lane.field_values.iter().enumerate() {
                let basis_weight = source_value * field_value;
                let entry = &mut result.basis_results[source_basis][field_basis];
                entry.g_integral += green * basis_weight;
                entry.dg_dn_integral += d_green_dn_y * basis_weight;
                entry.dg_dnx_integral += d_green_dn_x * basis_weight;
                entry.d2g_dnxdny_integral += d2_green_dnxdny * basis_weight;
                let curl_dot = lane.source_curls[source_basis][0]
                    * lane.field_curls[field_basis][0]
                    + lane.source_curls[source_basis][1] * lane.field_curls[field_basis][1]
                    + lane.source_curls[source_basis][2] * lane.field_curls[field_basis][2];
                result.curl_integrals[source_basis][field_basis] += green * curl_dot;
                result.normal_mass_integrals[source_basis][field_basis] +=
                    green * lane.normal_dot * basis_weight;
            }
        }
    }

    /// AVX2/FMA implementation of the separated regular Quad4 production
    /// cell. Four field quadrature points are processed per SIMD call; a
    /// scalar tail preserves odd quadrature orders exactly.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn galerkin_single_layer_regular_quad4_with_basis_avx2(
        source_coords: &DenseMatrix<f64>,
        field_coords: &DenseMatrix<f64>,
        physics: &PhysicsParams,
        num_source_basis: usize,
        num_field_basis: usize,
        gauss_order: usize,
    ) -> GalerkinIntegrationResult {
        // SAFETY: every intrinsic below is guarded by the public dispatch
        // feature check; all scalar helper calls retain their existing input
        // validation and separated-pair semantics.
        unsafe {
            assert!(matches!(num_source_basis, 1 | 4 | 9) && matches!(num_field_basis, 1 | 4 | 9));
            assert!(gauss_order > 0);
            let effective_order = super::quad4_effective_order(gauss_order);
            let (points, weights) = super::gauss_legendre(effective_order);
            let mut result = GalerkinIntegrationResult::new(num_source_basis, num_field_basis);
            let field_count = points.len() * points.len();

            for (source_s, &source_weight_s) in points.iter().zip(&weights) {
                for (source_t, &source_weight_t) in points.iter().zip(&weights) {
                    let mut field_index = 0;
                    while field_index + 4 <= field_count {
                        let batch = std::array::from_fn(|offset| {
                            let index = field_index + offset;
                            let field_s_index = index / points.len();
                            let field_t_index = index % points.len();
                            lane(
                                source_coords,
                                field_coords,
                                *source_s,
                                *source_t,
                                source_weight_s,
                                source_weight_t,
                                points[field_s_index],
                                points[field_t_index],
                                weights[field_s_index],
                                weights[field_t_index],
                                num_source_basis,
                                num_field_basis,
                            )
                        });
                        let values = kernels(&batch, physics);
                        for (lane, value) in batch.iter().zip(values) {
                            accumulate(&mut result, lane, value);
                        }
                        field_index += 4;
                    }
                    while field_index < field_count {
                        let field_s_index = field_index / points.len();
                        let field_t_index = field_index % points.len();
                        let field_s = points[field_s_index];
                        let field_t = points[field_t_index];
                        let lane = lane(
                            source_coords,
                            field_coords,
                            *source_s,
                            *source_t,
                            source_weight_s,
                            source_weight_t,
                            field_s,
                            field_t,
                            weights[field_s_index],
                            weights[field_t_index],
                            num_source_basis,
                            num_field_basis,
                        );
                        let value = compute_kernels(
                            &lane.source_point,
                            &lane.source_normal,
                            &lane.field_point,
                            &lane.field_normal,
                            physics,
                            lane.weight,
                        );
                        accumulate(&mut result, &lane, value);
                        field_index += 1;
                    }
                }
            }
            result
        }
    }
}

#[inline]
/// Regular separated Quad4 Galerkin integration including the physical
/// surface-gradient factors required by the weak hypersingular form.
///
/// This intentionally has the same separated-pair contract as
/// [`galerkin_single_layer_regular_quad4_with_basis`].  Singular and touching
/// pairs must use a mapped chart before this routine is called.
pub fn galerkin_quad4_regular_weak_form_with_basis(
    source_coords: &DenseMatrix<f64>,
    field_coords: &DenseMatrix<f64>,
    physics: &PhysicsParams,
    num_source_basis: usize,
    num_field_basis: usize,
    gauss_order: usize,
) -> GalerkinQuad4WeakResult {
    assert!(
        matches!(num_source_basis, 1 | 4 | 9) && matches!(num_field_basis, 1 | 4 | 9),
        "Galerkin quadrilateral basis count must be 1, 4, or 9"
    );
    assert!(
        gauss_order > 0,
        "Galerkin quadrature order must be positive"
    );

    let effective_order = quad4_effective_order(gauss_order);
    let (points, weights) = gauss_legendre(effective_order);
    let mut kernels = GalerkinIntegrationResult::new(num_source_basis, num_field_basis);
    let mut curl_integrals =
        vec![vec![Complex64::new(0.0, 0.0); num_field_basis]; num_source_basis];
    let mut normal_mass_integrals =
        vec![vec![Complex64::new(0.0, 0.0); num_field_basis]; num_source_basis];

    for (source_s, &source_weight_s) in points.iter().zip(&weights) {
        for (source_t, &source_weight_t) in points.iter().zip(&weights) {
            let (source_point, source_normal, source_jacobian) =
                quad4_geometry(source_coords, *source_s, *source_t);
            let source_values = quad4_basis_values(*source_s, *source_t, num_source_basis);
            let source_curls = quad4_surface_curls(
                source_coords,
                *source_s,
                *source_t,
                num_source_basis,
                source_normal,
            );

            for (field_s, &field_weight_s) in points.iter().zip(&weights) {
                for (field_t, &field_weight_t) in points.iter().zip(&weights) {
                    let (field_point, field_normal, field_jacobian) =
                        quad4_geometry(field_coords, *field_s, *field_t);
                    let field_values = quad4_basis_values(*field_s, *field_t, num_field_basis);
                    let field_curls = quad4_surface_curls(
                        field_coords,
                        *field_s,
                        *field_t,
                        num_field_basis,
                        field_normal,
                    );
                    let weight = source_weight_s
                        * source_weight_t
                        * field_weight_s
                        * field_weight_t
                        * source_jacobian
                        * field_jacobian;
                    let (green, d_green_dn_y, d_green_dn_x, d2_green_dnxdny) = compute_kernels(
                        &source_point,
                        &source_normal,
                        &field_point,
                        &field_normal,
                        physics,
                        weight,
                    );
                    let normal_dot = source_normal[0] * field_normal[0]
                        + source_normal[1] * field_normal[1]
                        + source_normal[2] * field_normal[2];
                    for (source_basis, &source_value) in source_values.iter().enumerate() {
                        for (field_basis, &field_value) in field_values.iter().enumerate() {
                            let basis_weight = source_value * field_value;
                            let entry = &mut kernels.basis_results[source_basis][field_basis];
                            entry.g_integral += green * basis_weight;
                            entry.dg_dn_integral += d_green_dn_y * basis_weight;
                            entry.dg_dnx_integral += d_green_dn_x * basis_weight;
                            entry.d2g_dnxdny_integral += d2_green_dnxdny * basis_weight;
                            let curl_dot = source_curls[source_basis][0]
                                * field_curls[field_basis][0]
                                + source_curls[source_basis][1] * field_curls[field_basis][1]
                                + source_curls[source_basis][2] * field_curls[field_basis][2];
                            curl_integrals[source_basis][field_basis] += green * curl_dot;
                            normal_mass_integrals[source_basis][field_basis] +=
                                green * normal_dot * basis_weight;
                        }
                    }
                }
            }
        }
    }

    GalerkinQuad4WeakResult {
        kernels,
        curl_integrals,
        normal_mass_integrals,
    }
}

/// Regular separated Quad4 weak-form kernels differentiated with respect to
/// the medium wavenumber.
///
/// The isoparametric geometry, physical surface gradients, basis functions,
/// and quadrature weights are held fixed.  This is the analytic counterpart
/// of [`galerkin_quad4_regular_weak_form_with_basis`] for the separated-pair
/// path; singular and touching charts intentionally remain separate APIs.
pub fn galerkin_quad4_regular_weak_form_wavenumber_derivative_with_basis(
    source_coords: &DenseMatrix<f64>,
    field_coords: &DenseMatrix<f64>,
    physics: &PhysicsParams,
    num_source_basis: usize,
    num_field_basis: usize,
    gauss_order: usize,
) -> GalerkinQuad4WeakResult {
    assert!(
        matches!(num_source_basis, 1 | 4 | 9) && matches!(num_field_basis, 1 | 4 | 9),
        "Galerkin quadrilateral basis count must be 1, 4, or 9"
    );
    assert!(
        gauss_order > 0,
        "Galerkin quadrature order must be positive"
    );

    let effective_order = quad4_effective_order(gauss_order);
    let (points, weights) = gauss_legendre(effective_order);
    let mut kernels = GalerkinIntegrationResult::new(num_source_basis, num_field_basis);
    let mut curl_integrals =
        vec![vec![Complex64::new(0.0, 0.0); num_field_basis]; num_source_basis];
    let mut normal_mass_integrals =
        vec![vec![Complex64::new(0.0, 0.0); num_field_basis]; num_source_basis];

    for (source_s, &source_weight_s) in points.iter().zip(&weights) {
        for (source_t, &source_weight_t) in points.iter().zip(&weights) {
            let (source_point, source_normal, source_jacobian) =
                quad4_geometry(source_coords, *source_s, *source_t);
            let source_values = quad4_basis_values(*source_s, *source_t, num_source_basis);
            let source_curls = quad4_surface_curls(
                source_coords,
                *source_s,
                *source_t,
                num_source_basis,
                source_normal,
            );

            for (field_s, &field_weight_s) in points.iter().zip(&weights) {
                for (field_t, &field_weight_t) in points.iter().zip(&weights) {
                    let (field_point, field_normal, field_jacobian) =
                        quad4_geometry(field_coords, *field_s, *field_t);
                    let field_values = quad4_basis_values(*field_s, *field_t, num_field_basis);
                    let field_curls = quad4_surface_curls(
                        field_coords,
                        *field_s,
                        *field_t,
                        num_field_basis,
                        field_normal,
                    );
                    let weight = source_weight_s
                        * source_weight_t
                        * field_weight_s
                        * field_weight_t
                        * source_jacobian
                        * field_jacobian;
                    let (dg_dk, ddg_dn_y_dk, ddg_dn_x_dk, dd2g_dnxdny_dk) =
                        compute_kernel_wavenumber_derivatives(
                            &source_point,
                            &source_normal,
                            &field_point,
                            &field_normal,
                            physics,
                            weight,
                        );
                    let normal_dot = source_normal[0] * field_normal[0]
                        + source_normal[1] * field_normal[1]
                        + source_normal[2] * field_normal[2];
                    for (source_basis, &source_value) in source_values.iter().enumerate() {
                        for (field_basis, &field_value) in field_values.iter().enumerate() {
                            let basis_weight = source_value * field_value;
                            let entry = &mut kernels.basis_results[source_basis][field_basis];
                            entry.g_integral += dg_dk * basis_weight;
                            entry.dg_dn_integral += ddg_dn_y_dk * basis_weight;
                            entry.dg_dnx_integral += ddg_dn_x_dk * basis_weight;
                            entry.d2g_dnxdny_integral += dd2g_dnxdny_dk * basis_weight;
                            let curl_dot = source_curls[source_basis][0]
                                * field_curls[field_basis][0]
                                + source_curls[source_basis][1] * field_curls[field_basis][1]
                                + source_curls[source_basis][2] * field_curls[field_basis][2];
                            curl_integrals[source_basis][field_basis] += dg_dk * curl_dot;
                            normal_mass_integrals[source_basis][field_basis] +=
                                dg_dk * normal_dot * basis_weight;
                        }
                    }
                }
            }
        }
    }

    GalerkinQuad4WeakResult {
        kernels,
        curl_integrals,
        normal_mass_integrals,
    }
}

fn quad4_surface_curls(
    element_coords: &DenseMatrix<f64>,
    s: f64,
    t: f64,
    count: usize,
    normal: [f64; 3],
) -> Vec<[f64; 3]> {
    let ds = [
        -0.25 * (1.0 - t),
        0.25 * (1.0 - t),
        0.25 * (1.0 + t),
        -0.25 * (1.0 + t),
    ];
    let dt = [
        -0.25 * (1.0 - s),
        -0.25 * (1.0 + s),
        0.25 * (1.0 + s),
        0.25 * (1.0 - s),
    ];
    let mut dx_ds = [0.0; 3];
    let mut dx_dt = [0.0; 3];
    for node in 0..4 {
        for axis in 0..3 {
            dx_ds[axis] += ds[node] * element_coords[[node, axis]];
            dx_dt[axis] += dt[node] * element_coords[[node, axis]];
        }
    }
    let g11 = dx_ds[0] * dx_ds[0] + dx_ds[1] * dx_ds[1] + dx_ds[2] * dx_ds[2];
    let g12 = dx_ds[0] * dx_dt[0] + dx_ds[1] * dx_dt[1] + dx_ds[2] * dx_dt[2];
    let g22 = dx_dt[0] * dx_dt[0] + dx_dt[1] * dx_dt[1] + dx_dt[2] * dx_dt[2];
    let determinant = (g11 * g22 - g12 * g12).max(1.0e-30);

    let basis_derivatives = quad4_basis_derivatives(s, t, count);
    basis_derivatives
        .into_iter()
        .map(|(d_s, d_t)| {
            let scale_s = (g22 * d_s - g12 * d_t) / determinant;
            let scale_t = (-g12 * d_s + g11 * d_t) / determinant;
            let gradient = [
                scale_s * dx_ds[0] + scale_t * dx_dt[0],
                scale_s * dx_ds[1] + scale_t * dx_dt[1],
                scale_s * dx_ds[2] + scale_t * dx_dt[2],
            ];
            [
                normal[1] * gradient[2] - normal[2] * gradient[1],
                normal[2] * gradient[0] - normal[0] * gradient[2],
                normal[0] * gradient[1] - normal[1] * gradient[0],
            ]
        })
        .collect()
}

fn quad4_basis_derivatives(s: f64, t: f64, count: usize) -> Vec<(f64, f64)> {
    match count {
        1 => vec![(0.0, 0.0)],
        4 => vec![
            (-0.25 * (1.0 - t), -0.25 * (1.0 - s)),
            (0.25 * (1.0 - t), -0.25 * (1.0 + s)),
            (0.25 * (1.0 + t), 0.25 * (1.0 + s)),
            (-0.25 * (1.0 + t), 0.25 * (1.0 - s)),
        ],
        9 => {
            let lagrange = |x: f64| [0.5 * x * (x - 1.0), 1.0 - x * x, 0.5 * x * (x + 1.0)];
            let derivative = |x: f64| [x - 0.5, -2.0 * x, x + 0.5];
            let ls = lagrange(s);
            let lt = lagrange(t);
            let ds = derivative(s);
            let dt = derivative(t);
            vec![
                (ds[0] * lt[0], ls[0] * dt[0]),
                (ds[2] * lt[0], ls[2] * dt[0]),
                (ds[2] * lt[2], ls[2] * dt[2]),
                (ds[0] * lt[2], ls[0] * dt[2]),
                (ds[1] * lt[0], ls[1] * dt[0]),
                (ds[2] * lt[1], ls[2] * dt[1]),
                (ds[1] * lt[2], ls[1] * dt[2]),
                (ds[0] * lt[1], ls[0] * dt[1]),
                (ds[1] * lt[1], ls[1] * dt[1]),
            ]
        }
        _ => unreachable!("validated quadrilateral basis count"),
    }
}

pub(crate) fn quad4_basis_values(s: f64, t: f64, count: usize) -> Vec<f64> {
    match count {
        1 => vec![1.0],
        4 => vec![
            0.25 * (1.0 - s) * (1.0 - t),
            0.25 * (1.0 + s) * (1.0 - t),
            0.25 * (1.0 + s) * (1.0 + t),
            0.25 * (1.0 - s) * (1.0 + t),
        ],
        9 => {
            let lagrange = |x: f64| [0.5 * x * (x - 1.0), 1.0 - x * x, 0.5 * x * (x + 1.0)];
            let ls = lagrange(s);
            let lt = lagrange(t);
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
        _ => unreachable!("validated quadrilateral basis count"),
    }
}

pub(crate) fn quad4_geometry(
    element_coords: &DenseMatrix<f64>,
    s: f64,
    t: f64,
) -> ([f64; 3], [f64; 3], f64) {
    let shape = [
        0.25 * (1.0 - s) * (1.0 - t),
        0.25 * (1.0 + s) * (1.0 - t),
        0.25 * (1.0 + s) * (1.0 + t),
        0.25 * (1.0 - s) * (1.0 + t),
    ];
    let ds = [
        -0.25 * (1.0 - t),
        0.25 * (1.0 - t),
        0.25 * (1.0 + t),
        -0.25 * (1.0 + t),
    ];
    let dt = [
        -0.25 * (1.0 - s),
        -0.25 * (1.0 + s),
        0.25 * (1.0 + s),
        0.25 * (1.0 - s),
    ];
    let mut point = [0.0; 3];
    let mut dx_ds = [0.0; 3];
    let mut dx_dt = [0.0; 3];
    for node in 0..4 {
        for axis in 0..3 {
            point[axis] += shape[node] * element_coords[[node, axis]];
            dx_ds[axis] += ds[node] * element_coords[[node, axis]];
            dx_dt[axis] += dt[node] * element_coords[[node, axis]];
        }
    }
    let normal_unnormalized = cross_product(dx_ds, dx_dt);
    let jacobian = (normal_unnormalized[0] * normal_unnormalized[0]
        + normal_unnormalized[1] * normal_unnormalized[1]
        + normal_unnormalized[2] * normal_unnormalized[2])
        .sqrt();
    let normal = if jacobian > 1.0e-15 {
        [
            normal_unnormalized[0] / jacobian,
            normal_unnormalized[1] / jacobian,
            normal_unnormalized[2] / jacobian,
        ]
    } else {
        [0.0; 3]
    };
    (point, normal, jacobian)
}

#[derive(Clone, Copy)]
struct Quad4ReferenceTriangle {
    corners: [[f64; 2]; 3],
}

struct MappedGalerkinPoint {
    point: [f64; 3],
    normal: [f64; 3],
    jacobian: f64,
    basis: Vec<f64>,
    curls: Vec<[f64; 3]>,
}

fn map_quad4_reference_triangle(
    element_coords: &DenseMatrix<f64>,
    triangle: Quad4ReferenceTriangle,
    xi: f64,
    eta: f64,
    num_basis: usize,
) -> MappedGalerkinPoint {
    let reference = [
        triangle.corners[0][0]
            + xi * (triangle.corners[1][0] - triangle.corners[0][0])
            + eta * (triangle.corners[2][0] - triangle.corners[0][0]),
        triangle.corners[0][1]
            + xi * (triangle.corners[1][1] - triangle.corners[0][1])
            + eta * (triangle.corners[2][1] - triangle.corners[0][1]),
    ];
    let parameter_jacobian = ((triangle.corners[1][0] - triangle.corners[0][0])
        * (triangle.corners[2][1] - triangle.corners[0][1])
        - (triangle.corners[1][1] - triangle.corners[0][1])
            * (triangle.corners[2][0] - triangle.corners[0][0]))
        .abs();
    let (point, normal, jacobian) = quad4_geometry(element_coords, reference[0], reference[1]);
    MappedGalerkinPoint {
        point,
        normal,
        jacobian: jacobian * parameter_jacobian,
        basis: quad4_basis_values(reference[0], reference[1], num_basis),
        curls: quad4_surface_curls(
            element_coords,
            reference[0],
            reference[1],
            num_basis,
            normal,
        ),
    }
}

fn add_galerkin_result(target: &mut GalerkinIntegrationResult, source: GalerkinIntegrationResult) {
    let GalerkinIntegrationResult {
        basis_results,
        curl_integrals,
        normal_mass_integrals,
    } = source;
    for (target_row, source_row) in target.basis_results.iter_mut().zip(basis_results) {
        for (target_entry, source_entry) in target_row.iter_mut().zip(source_row) {
            target_entry.g_integral += source_entry.g_integral;
            target_entry.dg_dn_integral += source_entry.dg_dn_integral;
            target_entry.dg_dnx_integral += source_entry.dg_dnx_integral;
            target_entry.d2g_dnxdny_integral += source_entry.d2g_dnxdny_integral;
        }
    }
    for (target_row, source_row) in target.curl_integrals.iter_mut().zip(curl_integrals) {
        for (target_entry, source_entry) in target_row.iter_mut().zip(source_row) {
            *target_entry += source_entry;
        }
    }
    for (target_row, source_row) in target
        .normal_mass_integrals
        .iter_mut()
        .zip(normal_mass_integrals)
    {
        for (target_entry, source_entry) in target_row.iter_mut().zip(source_row) {
            *target_entry += source_entry;
        }
    }
}

#[inline]
fn add_mapped_weak_products(
    result: &mut GalerkinIntegrationResult,
    source_basis: usize,
    field_basis: usize,
    source: &MappedGalerkinPoint,
    field: &MappedGalerkinPoint,
    green: Complex64,
    basis_weight: f64,
) {
    let curl_dot = source.curls[source_basis][0] * field.curls[field_basis][0]
        + source.curls[source_basis][1] * field.curls[field_basis][1]
        + source.curls[source_basis][2] * field.curls[field_basis][2];
    let normal_dot = source.normal[0] * field.normal[0]
        + source.normal[1] * field.normal[1]
        + source.normal[2] * field.normal[2];
    result.curl_integrals[source_basis][field_basis] += green * curl_dot;
    result.normal_mass_integrals[source_basis][field_basis] += green * normal_dot * basis_weight;
}

fn transpose_galerkin_result(result: GalerkinIntegrationResult) -> GalerkinIntegrationResult {
    let source_count = result.num_source_basis();
    let field_count = result.num_field_basis();
    let mut transposed = GalerkinIntegrationResult::new(field_count, source_count);
    for source in 0..source_count {
        for field in 0..field_count {
            transposed.basis_results[field][source] = result.basis_results[source][field].clone();
            transposed.curl_integrals[field][source] = result.curl_integrals[source][field];
            transposed.normal_mass_integrals[field][source] =
                result.normal_mass_integrals[source][field];
        }
    }
    transposed
}

fn mapped_galerkin_common_edge(
    source_coords: &DenseMatrix<f64>,
    field_coords: &DenseMatrix<f64>,
    source_triangle: Quad4ReferenceTriangle,
    field_triangle: Quad4ReferenceTriangle,
    physics: &PhysicsParams,
    num_source_basis: usize,
    num_field_basis: usize,
    gauss_order: usize,
) -> GalerkinIntegrationResult {
    mapped_galerkin_common_edge_impl(
        source_coords,
        field_coords,
        source_triangle,
        field_triangle,
        physics,
        num_source_basis,
        num_field_basis,
        gauss_order,
        false,
    )
}

fn mapped_galerkin_common_edge_wavenumber_derivative(
    source_coords: &DenseMatrix<f64>,
    field_coords: &DenseMatrix<f64>,
    source_triangle: Quad4ReferenceTriangle,
    field_triangle: Quad4ReferenceTriangle,
    physics: &PhysicsParams,
    num_source_basis: usize,
    num_field_basis: usize,
    gauss_order: usize,
) -> GalerkinIntegrationResult {
    mapped_galerkin_common_edge_impl(
        source_coords,
        field_coords,
        source_triangle,
        field_triangle,
        physics,
        num_source_basis,
        num_field_basis,
        gauss_order,
        true,
    )
}

fn mapped_galerkin_common_edge_impl(
    source_coords: &DenseMatrix<f64>,
    field_coords: &DenseMatrix<f64>,
    source_triangle: Quad4ReferenceTriangle,
    field_triangle: Quad4ReferenceTriangle,
    physics: &PhysicsParams,
    num_source_basis: usize,
    num_field_basis: usize,
    gauss_order: usize,
    wavenumber_derivative: bool,
) -> GalerkinIntegrationResult {
    let mut result = GalerkinIntegrationResult::new(num_source_basis, num_field_basis);
    let (points, weights) = gauss_legendre_01(gauss_order);
    for chart in 0..COMMON_EDGE_GALERKIN_CHARTS {
        for (i1, &eta1) in points.iter().enumerate() {
            for (i2, &eta2) in points.iter().enumerate() {
                for (i3, &eta3) in points.iter().enumerate() {
                    for (i4, &eta4) in points.iter().enumerate() {
                        let (xi_s, eta_s, xi_f, eta_f, chart_jacobian) =
                            common_edge_galerkin_chart(chart, eta1, eta2, eta3, eta4);
                        if chart_jacobian <= 1.0e-20 {
                            continue;
                        }
                        let source = map_quad4_reference_triangle(
                            source_coords,
                            source_triangle,
                            xi_s,
                            eta_s,
                            num_source_basis,
                        );
                        let field = map_quad4_reference_triangle(
                            field_coords,
                            field_triangle,
                            xi_f,
                            eta_f,
                            num_field_basis,
                        );
                        let weight = weights[i1]
                            * weights[i2]
                            * weights[i3]
                            * weights[i4]
                            * chart_jacobian
                            * source.jacobian
                            * field.jacobian;
                        let (green, d_green_dn_y, d_green_dn_x, d2_green_dnxdny) =
                            if wavenumber_derivative {
                                compute_kernel_wavenumber_derivatives(
                                    &source.point,
                                    &source.normal,
                                    &field.point,
                                    &field.normal,
                                    physics,
                                    weight,
                                )
                            } else {
                                compute_kernels(
                                    &source.point,
                                    &source.normal,
                                    &field.point,
                                    &field.normal,
                                    physics,
                                    weight,
                                )
                            };
                        for (source_basis, &source_value) in source.basis.iter().enumerate() {
                            for (field_basis, &field_value) in field.basis.iter().enumerate() {
                                let entry = &mut result.basis_results[source_basis][field_basis];
                                let basis_weight = source_value * field_value;
                                entry.g_integral += green * basis_weight;
                                entry.dg_dn_integral += d_green_dn_y * basis_weight;
                                entry.dg_dnx_integral += d_green_dn_x * basis_weight;
                                entry.d2g_dnxdny_integral += d2_green_dnxdny * basis_weight;
                                add_mapped_weak_products(
                                    &mut result,
                                    source_basis,
                                    field_basis,
                                    &source,
                                    &field,
                                    green,
                                    basis_weight,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    result
}

fn mapped_galerkin_common_vertex(
    source_coords: &DenseMatrix<f64>,
    field_coords: &DenseMatrix<f64>,
    source_triangle: Quad4ReferenceTriangle,
    field_triangle: Quad4ReferenceTriangle,
    physics: &PhysicsParams,
    num_source_basis: usize,
    num_field_basis: usize,
    gauss_order: usize,
) -> GalerkinIntegrationResult {
    mapped_galerkin_common_vertex_impl(
        source_coords,
        field_coords,
        source_triangle,
        field_triangle,
        physics,
        num_source_basis,
        num_field_basis,
        gauss_order,
        false,
    )
}

fn mapped_galerkin_common_vertex_wavenumber_derivative(
    source_coords: &DenseMatrix<f64>,
    field_coords: &DenseMatrix<f64>,
    source_triangle: Quad4ReferenceTriangle,
    field_triangle: Quad4ReferenceTriangle,
    physics: &PhysicsParams,
    num_source_basis: usize,
    num_field_basis: usize,
    gauss_order: usize,
) -> GalerkinIntegrationResult {
    mapped_galerkin_common_vertex_impl(
        source_coords,
        field_coords,
        source_triangle,
        field_triangle,
        physics,
        num_source_basis,
        num_field_basis,
        gauss_order,
        true,
    )
}

fn mapped_galerkin_common_vertex_impl(
    source_coords: &DenseMatrix<f64>,
    field_coords: &DenseMatrix<f64>,
    source_triangle: Quad4ReferenceTriangle,
    field_triangle: Quad4ReferenceTriangle,
    physics: &PhysicsParams,
    num_source_basis: usize,
    num_field_basis: usize,
    gauss_order: usize,
    wavenumber_derivative: bool,
) -> GalerkinIntegrationResult {
    let mut result = GalerkinIntegrationResult::new(num_source_basis, num_field_basis);
    let (points, weights) = gauss_legendre_01(gauss_order);
    for chart in 0..COMMON_VERTEX_GALERKIN_CHARTS {
        for (i1, &eta1) in points.iter().enumerate() {
            for (i2, &eta2) in points.iter().enumerate() {
                for (i3, &eta3) in points.iter().enumerate() {
                    for (i4, &eta4) in points.iter().enumerate() {
                        let (xi_s, eta_s, xi_f, eta_f, chart_jacobian) =
                            common_vertex_chart(chart, eta1, eta2, eta3, eta4);
                        if chart_jacobian <= 1.0e-20 {
                            continue;
                        }
                        let source = map_quad4_reference_triangle(
                            source_coords,
                            source_triangle,
                            xi_s,
                            eta_s,
                            num_source_basis,
                        );
                        let field = map_quad4_reference_triangle(
                            field_coords,
                            field_triangle,
                            xi_f,
                            eta_f,
                            num_field_basis,
                        );
                        let weight = weights[i1]
                            * weights[i2]
                            * weights[i3]
                            * weights[i4]
                            * chart_jacobian
                            * source.jacobian
                            * field.jacobian;
                        let (green, d_green_dn_y, d_green_dn_x, d2_green_dnxdny) =
                            if wavenumber_derivative {
                                compute_kernel_wavenumber_derivatives(
                                    &source.point,
                                    &source.normal,
                                    &field.point,
                                    &field.normal,
                                    physics,
                                    weight,
                                )
                            } else {
                                compute_kernels(
                                    &source.point,
                                    &source.normal,
                                    &field.point,
                                    &field.normal,
                                    physics,
                                    weight,
                                )
                            };
                        for (source_basis, &source_value) in source.basis.iter().enumerate() {
                            for (field_basis, &field_value) in field.basis.iter().enumerate() {
                                let entry = &mut result.basis_results[source_basis][field_basis];
                                let basis_weight = source_value * field_value;
                                entry.g_integral += green * basis_weight;
                                entry.dg_dn_integral += d_green_dn_y * basis_weight;
                                entry.dg_dnx_integral += d_green_dn_x * basis_weight;
                                entry.d2g_dnxdny_integral += d2_green_dnxdny * basis_weight;
                                add_mapped_weak_products(
                                    &mut result,
                                    source_basis,
                                    field_basis,
                                    &source,
                                    &field,
                                    green,
                                    basis_weight,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    result
}

fn mapped_galerkin_regular(
    source_coords: &DenseMatrix<f64>,
    field_coords: &DenseMatrix<f64>,
    source_triangle: Quad4ReferenceTriangle,
    field_triangle: Quad4ReferenceTriangle,
    physics: &PhysicsParams,
    num_source_basis: usize,
    num_field_basis: usize,
    gauss_order: usize,
) -> GalerkinIntegrationResult {
    mapped_galerkin_regular_impl(
        source_coords,
        field_coords,
        source_triangle,
        field_triangle,
        physics,
        num_source_basis,
        num_field_basis,
        gauss_order,
        false,
    )
}

fn mapped_galerkin_regular_wavenumber_derivative(
    source_coords: &DenseMatrix<f64>,
    field_coords: &DenseMatrix<f64>,
    source_triangle: Quad4ReferenceTriangle,
    field_triangle: Quad4ReferenceTriangle,
    physics: &PhysicsParams,
    num_source_basis: usize,
    num_field_basis: usize,
    gauss_order: usize,
) -> GalerkinIntegrationResult {
    mapped_galerkin_regular_impl(
        source_coords,
        field_coords,
        source_triangle,
        field_triangle,
        physics,
        num_source_basis,
        num_field_basis,
        gauss_order,
        true,
    )
}

fn mapped_galerkin_regular_impl(
    source_coords: &DenseMatrix<f64>,
    field_coords: &DenseMatrix<f64>,
    source_triangle: Quad4ReferenceTriangle,
    field_triangle: Quad4ReferenceTriangle,
    physics: &PhysicsParams,
    num_source_basis: usize,
    num_field_basis: usize,
    gauss_order: usize,
    wavenumber_derivative: bool,
) -> GalerkinIntegrationResult {
    let mut result = GalerkinIntegrationResult::new(num_source_basis, num_field_basis);
    let (points, weights) = gauss_legendre_01(gauss_order);
    for (source_u, &source_weight_u) in points.iter().zip(&weights) {
        for (source_v, &source_weight_v) in points.iter().zip(&weights) {
            let source_xi = *source_u;
            let source_eta = (1.0 - source_u) * source_v;
            let source_weight = source_weight_u * source_weight_v * (1.0 - source_u);
            let source = map_quad4_reference_triangle(
                source_coords,
                source_triangle,
                source_xi,
                source_eta,
                num_source_basis,
            );
            for (field_u, &field_weight_u) in points.iter().zip(&weights) {
                for (field_v, &field_weight_v) in points.iter().zip(&weights) {
                    let field_xi = *field_u;
                    let field_eta = (1.0 - field_u) * field_v;
                    let field_weight = field_weight_u * field_weight_v * (1.0 - field_u);
                    let field = map_quad4_reference_triangle(
                        field_coords,
                        field_triangle,
                        field_xi,
                        field_eta,
                        num_field_basis,
                    );
                    let weight = source_weight * field_weight * source.jacobian * field.jacobian;
                    let (green, d_green_dn_y, d_green_dn_x, d2_green_dnxdny) =
                        if wavenumber_derivative {
                            compute_kernel_wavenumber_derivatives(
                                &source.point,
                                &source.normal,
                                &field.point,
                                &field.normal,
                                physics,
                                weight,
                            )
                        } else {
                            compute_kernels(
                                &source.point,
                                &source.normal,
                                &field.point,
                                &field.normal,
                                physics,
                                weight,
                            )
                        };
                    for (source_basis, &source_value) in source.basis.iter().enumerate() {
                        for (field_basis, &field_value) in field.basis.iter().enumerate() {
                            let entry = &mut result.basis_results[source_basis][field_basis];
                            let basis_weight = source_value * field_value;
                            entry.g_integral += green * basis_weight;
                            entry.dg_dn_integral += d_green_dn_y * basis_weight;
                            entry.dg_dnx_integral += d_green_dn_x * basis_weight;
                            entry.d2g_dnxdny_integral += d2_green_dnxdny * basis_weight;
                            add_mapped_weak_products(
                                &mut result,
                                source_basis,
                                field_basis,
                                &source,
                                &field,
                                green,
                                basis_weight,
                            );
                        }
                    }
                }
            }
        }
    }
    result
}

/// Galerkin single-layer integration for topologically touching Quad4 panels.
///
/// Each reference square is split into two triangles. Every subtriangle pair
/// is routed through a mapped six-chart common-edge, two-chart common-vertex,
/// or regular triangle rule, while the physical bilinear geometry and the
/// requested Quad4 basis are evaluated at every mapped quadrature point.
pub fn galerkin_single_layer_touching_quad4_with_basis(
    source_coords: &DenseMatrix<f64>,
    field_coords: &DenseMatrix<f64>,
    source_connectivity: &[usize],
    field_connectivity: &[usize],
    physics: &PhysicsParams,
    num_source_basis: usize,
    num_field_basis: usize,
    gauss_order: usize,
) -> GalerkinIntegrationResult {
    galerkin_single_layer_touching_quad4_with_basis_impl(
        source_coords,
        field_coords,
        source_connectivity,
        field_connectivity,
        physics,
        num_source_basis,
        num_field_basis,
        gauss_order,
        false,
    )
}

/// Analytic wavenumber derivative of the mapped touching-Quad4 charts.
pub fn galerkin_single_layer_touching_quad4_wavenumber_derivative_with_basis(
    source_coords: &DenseMatrix<f64>,
    field_coords: &DenseMatrix<f64>,
    source_connectivity: &[usize],
    field_connectivity: &[usize],
    physics: &PhysicsParams,
    num_source_basis: usize,
    num_field_basis: usize,
    gauss_order: usize,
) -> GalerkinIntegrationResult {
    galerkin_single_layer_touching_quad4_with_basis_impl(
        source_coords,
        field_coords,
        source_connectivity,
        field_connectivity,
        physics,
        num_source_basis,
        num_field_basis,
        gauss_order,
        true,
    )
}

fn galerkin_single_layer_touching_quad4_with_basis_impl(
    source_coords: &DenseMatrix<f64>,
    field_coords: &DenseMatrix<f64>,
    source_connectivity: &[usize],
    field_connectivity: &[usize],
    physics: &PhysicsParams,
    num_source_basis: usize,
    num_field_basis: usize,
    gauss_order: usize,
    wavenumber_derivative: bool,
) -> GalerkinIntegrationResult {
    assert_eq!(source_connectivity.len(), 4);
    assert_eq!(field_connectivity.len(), 4);
    assert!(matches!(num_source_basis, 1 | 4 | 9));
    assert!(matches!(num_field_basis, 1 | 4 | 9));
    assert!(
        gauss_order > 0,
        "Galerkin quadrature order must be positive"
    );
    let effective_order = quad4_effective_order(gauss_order);

    // The scalar single-layer kernel is reciprocal. Canonicalizing the panel
    // order makes both call directions use identical chart partitions and
    // avoids exposing chart-order roundoff as a matrix asymmetry.
    if source_connectivity > field_connectivity {
        return transpose_galerkin_result(galerkin_single_layer_touching_quad4_with_basis_impl(
            field_coords,
            source_coords,
            field_connectivity,
            source_connectivity,
            physics,
            num_field_basis,
            num_source_basis,
            gauss_order,
            wavenumber_derivative,
        ));
    }

    let reference_corners = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
    let subtriangle_indices = [[0usize, 1, 2], [0, 2, 3]];
    let mut result = GalerkinIntegrationResult::new(num_source_basis, num_field_basis);

    for source_indices in subtriangle_indices {
        for field_indices in subtriangle_indices {
            let mut shared = Vec::new();
            for &source_local in &source_indices {
                if let Some(field_position) = field_indices.iter().position(|&field_local| {
                    source_connectivity[source_local] == field_connectivity[field_local]
                }) {
                    shared.push((source_local, field_indices[field_position]));
                }
            }
            let (source_order, field_order, chart_kind) = match shared.len() {
                2 => {
                    let source_order = [
                        shared[0].0,
                        shared[1].0,
                        source_indices
                            .iter()
                            .copied()
                            .find(|&local| local != shared[0].0 && local != shared[1].0)
                            .expect("common-edge source subtriangle has a free vertex"),
                    ];
                    let field_order = [
                        shared[0].1,
                        shared[1].1,
                        field_indices
                            .iter()
                            .copied()
                            .find(|&local| local != shared[0].1 && local != shared[1].1)
                            .expect("common-edge field subtriangle has a free vertex"),
                    ];
                    (source_order, field_order, 2usize)
                }
                1 => {
                    let source_free = source_indices
                        .iter()
                        .copied()
                        .filter(|&local| local != shared[0].0)
                        .collect::<Vec<_>>();
                    let field_free = field_indices
                        .iter()
                        .copied()
                        .filter(|&local| local != shared[0].1)
                        .collect::<Vec<_>>();
                    (
                        [shared[0].0, source_free[0], source_free[1]],
                        [shared[0].1, field_free[0], field_free[1]],
                        1usize,
                    )
                }
                0 => (source_indices, field_indices, 0usize),
                _ => continue,
            };
            let source_triangle = Quad4ReferenceTriangle {
                corners: [
                    reference_corners[source_order[0]],
                    reference_corners[source_order[1]],
                    reference_corners[source_order[2]],
                ],
            };
            let field_triangle = Quad4ReferenceTriangle {
                corners: [
                    reference_corners[field_order[0]],
                    reference_corners[field_order[1]],
                    reference_corners[field_order[2]],
                ],
            };
            let subresult = match chart_kind {
                2 => {
                    if wavenumber_derivative {
                        mapped_galerkin_common_edge_wavenumber_derivative(
                            source_coords,
                            field_coords,
                            source_triangle,
                            field_triangle,
                            physics,
                            num_source_basis,
                            num_field_basis,
                            effective_order,
                        )
                    } else {
                        mapped_galerkin_common_edge(
                            source_coords,
                            field_coords,
                            source_triangle,
                            field_triangle,
                            physics,
                            num_source_basis,
                            num_field_basis,
                            effective_order,
                        )
                    }
                }
                1 => {
                    if wavenumber_derivative {
                        mapped_galerkin_common_vertex_wavenumber_derivative(
                            source_coords,
                            field_coords,
                            source_triangle,
                            field_triangle,
                            physics,
                            num_source_basis,
                            num_field_basis,
                            effective_order,
                        )
                    } else {
                        mapped_galerkin_common_vertex(
                            source_coords,
                            field_coords,
                            source_triangle,
                            field_triangle,
                            physics,
                            num_source_basis,
                            num_field_basis,
                            effective_order,
                        )
                    }
                }
                _ => {
                    if wavenumber_derivative {
                        mapped_galerkin_regular_wavenumber_derivative(
                            source_coords,
                            field_coords,
                            source_triangle,
                            field_triangle,
                            physics,
                            num_source_basis,
                            num_field_basis,
                            effective_order,
                        )
                    } else {
                        mapped_galerkin_regular(
                            source_coords,
                            field_coords,
                            source_triangle,
                            field_triangle,
                            physics,
                            num_source_basis,
                            num_field_basis,
                            effective_order,
                        )
                    }
                }
            };
            add_galerkin_result(&mut result, subresult);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_triangle() -> DenseMatrix<f64> {
        DenseMatrix::from_row_major_vec(3, 3, vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    }

    fn make_adjacent_triangle() -> DenseMatrix<f64> {
        // Shares edge (0,0,0)-(1,0,0) with test triangle
        DenseMatrix::from_row_major_vec(3, 3, vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, -0.866, 0.0])
    }

    fn make_vertex_adjacent_triangle() -> DenseMatrix<f64> {
        // Shares only vertex (0,0,0) with test triangle
        DenseMatrix::from_row_major_vec(3, 3, vec![0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 0.0])
    }

    #[test]
    fn quadratic_galerkin_basis_is_partition_of_unity() {
        let values = galerkin_basis_values([0.2, 0.3, 0.5], 6);
        assert_eq!(values.len(), 6);
        assert!((values.iter().sum::<f64>() - 1.0).abs() < 1.0e-14);
        assert!(values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn test_gauss_legendre_01() {
        let (pts, wts) = gauss_legendre_01(4);
        // All points should be in [0, 1]
        for &p in &pts {
            assert!((0.0..=1.0).contains(&p), "Point {p} not in [0,1]");
        }
        // Weights should sum to 1.0 (integral of 1 over [0,1])
        let sum: f64 = wts.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Weights sum to {sum}, expected 1.0"
        );
    }

    #[test]
    fn test_duffy_self_triangle_constant_finite() {
        let coords = make_test_triangle();
        let source = [1.0 / 3.0, 1.0 / 3.0, 0.0]; // centroid
        let normal = [0.0, 0.0, 1.0];

        let physics = PhysicsParams::new(10.0, 343.0, 1.21, false);

        let result = duffy_self_triangle_constant(&source, &normal, &coords, &physics, 6);

        // G integral should be finite and positive (at low frequency)
        assert!(
            result.g_integral.norm().is_finite(),
            "G integral is not finite: {:?}",
            result.g_integral
        );
        assert!(
            result.g_integral.re > 0.0,
            "G integral real part should be positive at low freq, got: {}",
            result.g_integral.re
        );

        // H integral should be ~0 for source on flat element with normal perpendicular to surface
        assert!(
            result.dg_dn_integral.norm() < 1e-6,
            "H integral should be ~0 for centroid on flat element, got: {:?}",
            result.dg_dn_integral
        );
    }

    #[test]
    fn test_duffy_self_triangle_linear() {
        let coords = make_test_triangle();
        let source = [1.0 / 6.0, 1.0 / 6.0, 0.0]; // first Gauss point
        let normal = [0.0, 0.0, 1.0];

        let physics = PhysicsParams::new(10.0, 343.0, 1.21, false);

        let result = duffy_self_triangle(&source, &normal, &coords, &physics, 3, 6);

        assert_eq!(result.basis_results.len(), 3);

        // All G integrals should be finite
        for (i, br) in result.basis_results.iter().enumerate() {
            assert!(
                br.g_integral.norm().is_finite(),
                "Basis {i}: G integral not finite"
            );
        }

        // Sum of basis-weighted G integrals should be positive
        let g_sum: Complex64 = result.basis_results.iter().map(|br| br.g_integral).sum();
        assert!(
            g_sum.re > 0.0,
            "Sum of G integrals should be positive, got: {}",
            g_sum.re
        );
    }

    #[test]
    fn test_duffy_convergence() {
        // Test that increasing quadrature order converges
        let coords = make_test_triangle();
        let source = [1.0 / 3.0, 1.0 / 3.0, 0.0];
        let normal = [0.0, 0.0, 1.0];

        let physics = PhysicsParams::new(10.0, 343.0, 1.21, false);

        let r4 = duffy_self_triangle_constant(&source, &normal, &coords, &physics, 4);
        let r6 = duffy_self_triangle_constant(&source, &normal, &coords, &physics, 6);
        let r8 = duffy_self_triangle_constant(&source, &normal, &coords, &physics, 8);

        // Differences should decrease (convergence)
        let diff_46 = (r4.g_integral - r6.g_integral).norm();
        let diff_68 = (r6.g_integral - r8.g_integral).norm();

        assert!(
            diff_68 < diff_46,
            "Not converging: diff(6-8)={diff_68:.6e} >= diff(4-6)={diff_46:.6e}"
        );
    }

    #[test]
    fn test_sauter_schwab_common_edge_finite() {
        let source_coords = make_test_triangle();
        let field_coords = make_adjacent_triangle();

        let source = [1.0 / 3.0, 1.0 / 3.0, 0.0];
        let normal = [0.0, 0.0, 1.0];

        let physics = PhysicsParams::new(10.0, 343.0, 1.21, false);

        let result = sauter_schwab_common_edge(
            &source,
            &normal,
            &source_coords,
            &field_coords,
            [0, 1], // nodes 0 and 1 are shared
            &physics,
            1,
            4,
        );

        assert_eq!(result.basis_results.len(), 1);
        assert!(
            result.basis_results[0].g_integral.norm().is_finite(),
            "G integral not finite: {:?}",
            result.basis_results[0].g_integral
        );
    }

    #[test]
    fn test_sauter_schwab_common_vertex_finite() {
        let source_coords = make_test_triangle();
        let field_coords = make_vertex_adjacent_triangle();

        let source = [1.0 / 3.0, 1.0 / 3.0, 0.0];
        let normal = [0.0, 0.0, 1.0];

        let physics = PhysicsParams::new(10.0, 343.0, 1.21, false);

        let result = sauter_schwab_common_vertex(
            &source,
            &normal,
            &source_coords,
            &field_coords,
            0, // node 0 is shared
            &physics,
            1,
            4,
        );

        assert_eq!(result.basis_results.len(), 1);
        assert!(
            result.basis_results[0].g_integral.norm().is_finite(),
            "G integral not finite: {:?}",
            result.basis_results[0].g_integral
        );
    }

    #[test]
    fn test_galerkin_common_edge_constant_finite_and_symmetric() {
        let source_coords = make_test_triangle();
        let field_coords = make_adjacent_triangle();
        let physics = PhysicsParams::new(10.0, 343.0, 1.21, false);

        let result_low =
            sauter_schwab_galerkin_common_edge(&source_coords, &field_coords, &physics, 1, 1, 3);
        let reversed_low =
            sauter_schwab_galerkin_common_edge(&field_coords, &source_coords, &physics, 1, 1, 3);
        let result =
            sauter_schwab_galerkin_common_edge(&source_coords, &field_coords, &physics, 1, 1, 4);
        let reversed =
            sauter_schwab_galerkin_common_edge(&field_coords, &source_coords, &physics, 1, 1, 4);

        assert_eq!(result.num_source_basis(), 1);
        assert_eq!(result.num_field_basis(), 1);
        let g = result.basis_results[0][0].g_integral;
        let g_reversed = reversed.basis_results[0][0].g_integral;
        assert!(
            g.norm().is_finite(),
            "Galerkin common-edge G integral not finite: {g:?}"
        );
        let low_g = result_low.basis_results[0][0].g_integral;
        let low_g_reversed = reversed_low.basis_results[0][0].g_integral;
        let low_rel = (low_g - low_g_reversed).norm() / low_g.norm().max(1e-15);
        let rel = (g - g_reversed).norm() / g.norm().max(1e-15);
        assert!(
            rel <= low_rel * 10.0 + 1.0e-12 && rel < 5e-3,
            "Galerkin common-edge G symmetry should converge: order3={low_rel:.3e}, order4={rel:.3e}"
        );
    }

    #[test]
    fn test_galerkin_common_edge_linear_partition_of_unity() {
        let source_coords = make_test_triangle();
        let field_coords = make_adjacent_triangle();
        let physics = PhysicsParams::new(10.0, 343.0, 1.21, false);

        let constant =
            sauter_schwab_galerkin_common_edge(&source_coords, &field_coords, &physics, 1, 1, 4);
        let linear =
            sauter_schwab_galerkin_common_edge(&source_coords, &field_coords, &physics, 3, 3, 4);

        let mut linear_sum = Complex64::new(0.0, 0.0);
        for row in &linear.basis_results {
            for entry in row {
                linear_sum += entry.g_integral;
            }
        }

        let constant_g = constant.basis_results[0][0].g_integral;
        assert!(
            (linear_sum - constant_g).norm() / constant_g.norm().max(1e-15) < 1e-10,
            "linear source/field basis functions should form a partition of unity"
        );
    }

    #[test]
    fn test_galerkin_common_vertex_constant_finite() {
        let source_coords = make_test_triangle();
        let field_coords = make_vertex_adjacent_triangle();
        let physics = PhysicsParams::new(10.0, 343.0, 1.21, false);

        let result =
            sauter_schwab_galerkin_common_vertex(&source_coords, &field_coords, &physics, 1, 1, 4);

        assert_eq!(result.num_source_basis(), 1);
        assert_eq!(result.num_field_basis(), 1);
        assert!(
            result.basis_results[0][0].g_integral.norm().is_finite(),
            "Galerkin common-vertex G integral not finite"
        );
    }

    #[test]
    fn test_singular_integration_dispatch_self() {
        let coords = make_test_triangle();
        let source = [1.0 / 3.0, 1.0 / 3.0, 0.0];
        let normal = [0.0, 0.0, 1.0];

        let physics = PhysicsParams::new(10.0, 343.0, 1.21, false);

        let result = singular_integration_dispatch(
            &source,
            &normal,
            &coords,
            &coords,
            &physics,
            &AdjacencyType::SelfElement,
            1,
            6,
        );

        assert!(result.is_some(), "Should return Some for self-element");
        let result = result.unwrap();
        assert!(result.basis_results[0].g_integral.norm() > 0.0);
    }

    #[test]
    fn test_singular_integration_dispatch_separated() {
        let coords = make_test_triangle();
        let source = [10.0, 10.0, 0.0];
        let normal = [0.0, 0.0, 1.0];

        let physics = PhysicsParams::new(10.0, 343.0, 1.21, false);

        let result = singular_integration_dispatch(
            &source,
            &normal,
            &coords,
            &coords,
            &physics,
            &AdjacencyType::Separated,
            1,
            4,
        );

        assert!(
            result.is_none(),
            "Should return None for separated elements"
        );
    }

    #[test]
    fn test_tri_geometry_at_centroid() {
        let coords = make_test_triangle();
        let (pos, normal, jac, shape) = tri_geometry(&coords, 1.0 / 3.0, 1.0 / 3.0);

        // Position should be centroid
        assert!((pos[0] - 1.0 / 3.0).abs() < 1e-10);
        assert!((pos[1] - 1.0 / 3.0).abs() < 1e-10);
        assert!(pos[2].abs() < 1e-10);

        // Normal should be (0, 0, 1)
        assert!(normal[2].abs() > 0.99);

        // Jacobian should be 1.0 (2 * area of unit right triangle)
        assert!((jac - 1.0).abs() < 1e-10);

        // Shape functions should all be 1/3
        for &s in &shape {
            assert!((s - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn quad4_reference_map_matches_affine_triangle() {
        let source_coords = DenseMatrix::from_row_major_vec(
            4,
            3,
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        );
        let triangle = Quad4ReferenceTriangle {
            corners: [[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0]],
        };
        let mapped = map_quad4_reference_triangle(&source_coords, triangle, 0.2, 0.3, 1);
        let triangle_coords = DenseMatrix::from_row_major_vec(
            3,
            3,
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        );
        let (point, _, jacobian, _) = tri_geometry(&triangle_coords, 0.2, 0.3);
        assert!(
            mapped
                .point
                .iter()
                .zip(point)
                .all(|(a, b)| (a - b).abs() < 1e-12)
        );
        assert!((mapped.jacobian - jacobian).abs() < 1e-12);
    }

    #[test]
    fn mapped_touching_charts_match_triangle_partition() {
        let source_coords = DenseMatrix::from_row_major_vec(
            4,
            3,
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        );
        let field_coords = DenseMatrix::from_row_major_vec(
            4,
            3,
            vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        );
        let source_triangle = Quad4ReferenceTriangle {
            corners: [[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0]],
        };
        let field_triangle = Quad4ReferenceTriangle {
            corners: [[-1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]],
        };
        let physics = PhysicsParams::new(420.0, 343.0, 1.21, false);
        let order = quad4_effective_order(8);
        let mapped = mapped_galerkin_common_edge(
            &source_coords,
            &field_coords,
            source_triangle,
            field_triangle,
            &physics,
            1,
            1,
            order,
        )
        .basis_results[0][0]
            .g_integral;
        let source_tri = DenseMatrix::from_row_major_vec(
            3,
            3,
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        );
        let field_tri = DenseMatrix::from_row_major_vec(
            3,
            3,
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 0.0],
        );
        let expected =
            sauter_schwab_galerkin_common_edge(&source_tri, &field_tri, &physics, 1, 1, order)
                .basis_results[0][0]
                .g_integral;
        assert!((mapped - expected).norm() < 1e-12);

        let mapped_vertex = mapped_galerkin_common_vertex(
            &source_coords,
            &field_coords,
            Quad4ReferenceTriangle {
                corners: [[1.0, -1.0], [-1.0, -1.0], [1.0, 1.0]],
            },
            Quad4ReferenceTriangle {
                corners: [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0]],
            },
            &physics,
            1,
            1,
            order,
        )
        .basis_results[0][0]
            .g_integral;
        let expected_vertex = sauter_schwab_galerkin_common_vertex(
            &DenseMatrix::from_row_major_vec(
                3,
                3,
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            ),
            &DenseMatrix::from_row_major_vec(
                3,
                3,
                vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.0, 0.0],
            ),
            &physics,
            1,
            1,
            order,
        )
        .basis_results[0][0]
            .g_integral;
        assert!((mapped_vertex - expected_vertex).norm() < 1e-12);

        let mapped_regular = mapped_galerkin_regular(
            &source_coords,
            &field_coords,
            Quad4ReferenceTriangle {
                corners: [[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
            },
            Quad4ReferenceTriangle {
                corners: [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0]],
            },
            &physics,
            1,
            1,
            order,
        )
        .basis_results[0][0]
            .g_integral;
        let expected_regular = galerkin_single_layer_regular_triangle(
            &DenseMatrix::from_row_major_vec(
                3,
                3,
                vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            ),
            &DenseMatrix::from_row_major_vec(
                3,
                3,
                vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.0, 0.0],
            ),
            &physics,
            order,
        );
        assert!((mapped_regular - expected_regular).norm() < 1e-12);

        let mapped_vertex_b = mapped_galerkin_common_vertex(
            &source_coords,
            &field_coords,
            Quad4ReferenceTriangle {
                corners: [[1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0]],
            },
            Quad4ReferenceTriangle {
                corners: [[-1.0, 1.0], [-1.0, -1.0], [1.0, 1.0]],
            },
            &physics,
            1,
            1,
            order,
        )
        .basis_results[0][0]
            .g_integral;
        let expected_vertex_b = sauter_schwab_galerkin_common_vertex(
            &DenseMatrix::from_row_major_vec(
                3,
                3,
                vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ),
            &DenseMatrix::from_row_major_vec(
                3,
                3,
                vec![1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0],
            ),
            &physics,
            1,
            1,
            order,
        )
        .basis_results[0][0]
            .g_integral;
        assert!((mapped_vertex_b - expected_vertex_b).norm() < 1e-12);
        let total = galerkin_single_layer_touching_quad4_with_basis(
            &source_coords,
            &field_coords,
            &[0, 1, 2, 3],
            &[1, 4, 5, 2],
            &physics,
            1,
            1,
            8,
        )
        .basis_results[0][0]
            .g_integral;
        assert!(
            (total - (mapped_vertex + mapped + mapped_regular + mapped_vertex_b)).norm() < 1e-12,
            "mapped partition residual = {:e}; total={:?}; vertex={:?}; edge={:?}; regular={:?}; vertex_b={:?}",
            (total - (mapped_vertex + mapped + mapped_regular + mapped_vertex_b)).norm(),
            total,
            mapped_vertex,
            mapped,
            mapped_regular,
            mapped_vertex_b
        );
    }

    #[test]
    fn galerkin_common_edge_charts_partition_triangle_product() {
        let (points, weights) = gauss_legendre_01(6);
        let mut volume = 0.0;
        for chart in 0..COMMON_EDGE_GALERKIN_CHARTS {
            for (i1, &eta1) in points.iter().enumerate() {
                for (i2, &eta2) in points.iter().enumerate() {
                    for (i3, &eta3) in points.iter().enumerate() {
                        for (i4, &eta4) in points.iter().enumerate() {
                            let (_, _, _, _, jacobian) =
                                common_edge_galerkin_chart(chart, eta1, eta2, eta3, eta4);
                            volume +=
                                weights[i1] * weights[i2] * weights[i3] * weights[i4] * jacobian;
                        }
                    }
                }
            }
        }
        assert!((volume - 0.25).abs() < 1.0e-12, "mapped volume = {volume}");
    }
}
