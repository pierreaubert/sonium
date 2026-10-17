//! Singular integration handling
//!
//! Adaptive subdivision for nearly-singular and singular element integrals.

use crate::core::array::array;
use num_complex::Complex64;
use sonium_solvers::dense::DenseMatrix;
use std::f64::consts::PI;

use crate::core::integration::gauss::gauss_legendre;
use crate::core::mesh::element::{basis_values, cross_product, normalize, quadrature_geometry};
use crate::core::types::{
    BasisOrder, ElementType, IntegrationEstimate, IntegrationResult, LinearIntegrationResult,
    PhysicsParams, QuadratureConvergenceError, QuadratureOrderError,
};

/// Maximum number of subelements for adaptive integration
pub const MAX_SUBELEMENTS: usize = 110;

/// Quadrature parameters for singular integration
///
/// Controls the accuracy of numerical integration. Higher values give
/// better accuracy at higher computational cost.
#[derive(Debug, Clone, Copy)]
pub struct QuadratureParams {
    /// Number of Gauss points for 1D edge integration (default: 3)
    pub edge_gauss_order: usize,
    /// Number of Gauss points per dimension for 2D subelement integration (default: 4)
    pub subelement_gauss_order: usize,
    /// Number of edge sections for hypersingular integration (default: 4)
    pub edge_sections: usize,
    /// Number of subtriangles per edge section (default: 2)
    pub subtriangles_per_section: usize,
}

impl Default for QuadratureParams {
    fn default() -> Self {
        Self {
            edge_gauss_order: 3,
            subelement_gauss_order: 4,
            edge_sections: 4,
            subtriangles_per_section: 2,
        }
    }
}

impl QuadratureParams {
    /// Create quadrature parameters optimized for the given ka (wave number × element size)
    ///
    /// Higher ka values require more integration points for accuracy.
    pub fn for_ka(ka: f64) -> Self {
        if ka < 0.3 {
            // Very low frequency: coarse quadrature is fine
            Self {
                edge_gauss_order: 3,
                subelement_gauss_order: 4,
                edge_sections: 4,
                subtriangles_per_section: 2,
            }
        } else if ka < 1.0 {
            // Low-medium frequency: increase quadrature
            Self {
                edge_gauss_order: 4,
                subelement_gauss_order: 5,
                edge_sections: 6,
                subtriangles_per_section: 2,
            }
        } else if ka < 2.0 {
            // Medium frequency: further increase
            Self {
                edge_gauss_order: 5,
                subelement_gauss_order: 6,
                edge_sections: 8,
                subtriangles_per_section: 3,
            }
        } else {
            // High frequency: maximum quadrature
            Self {
                edge_gauss_order: 6,
                subelement_gauss_order: 7,
                edge_sections: 10,
                subtriangles_per_section: 4,
            }
        }
    }

    /// High-accuracy quadrature for validation/debugging
    pub fn high_accuracy() -> Self {
        Self {
            edge_gauss_order: 8,
            subelement_gauss_order: 8,
            edge_sections: 12,
            subtriangles_per_section: 4,
        }
    }

    fn is_strictly_finer_than(&self, coarse: &Self) -> bool {
        self.edge_gauss_order >= coarse.edge_gauss_order
            && self.subelement_gauss_order >= coarse.subelement_gauss_order
            && self.edge_sections >= coarse.edge_sections
            && self.subtriangles_per_section >= coarse.subtriangles_per_section
            && (self.edge_gauss_order > coarse.edge_gauss_order
                || self.subelement_gauss_order > coarse.subelement_gauss_order
                || self.edge_sections > coarse.edge_sections
                || self.subtriangles_per_section > coarse.subtriangles_per_section)
    }

    fn is_valid(&self) -> bool {
        (1..=crate::core::integration::gauss::N_GAUORDER).contains(&self.edge_gauss_order)
            && (1..=crate::core::integration::gauss::N_GAUORDER)
                .contains(&self.subelement_gauss_order)
            && self.edge_sections > 0
            && self.subtriangles_per_section > 0
    }
}

/// Local coordinates for 6-point triangle subdivision (vertices + midpoints)
/// Standard vertex mapping: v0 at (0,0), v1 at (1,0), v2 at (0,1)
/// Midpoints: m3 = mid(v0,v1) at (0.5,0), m4 = mid(v1,v2) at (0.5,0.5), m5 = mid(v2,v0) at (0,0.5)
const CSI6: [f64; 6] = [0.0, 1.0, 0.0, 0.5, 0.5, 0.0];
const ETA6: [f64; 6] = [0.0, 0.0, 1.0, 0.0, 0.5, 0.5];

/// Local coordinates for 8-point quad subdivision (vertices + midpoints)
const CSI8: [f64; 8] = [1.0, -1.0, -1.0, 1.0, 0.0, -1.0, 0.0, 1.0];
const ETA8: [f64; 8] = [1.0, 1.0, -1.0, -1.0, 1.0, 0.0, -1.0, 0.0];

/// Perform singular integration for self-element
///
/// This implements NC_SingularIntegration from the C++ code.
/// Uses edge-based integration for the hypersingular kernel and
/// subelement triangulation for accurate integration near singularity.
///
/// # Arguments
/// * `source_point` - Source point (collocation point)
/// * `source_normal` - Unit normal at source point
/// * `element_coords` - Node coordinates of the field element (num_nodes × 3)
/// * `element_type` - Triangle or quad element
/// * `physics` - Physics parameters (wave number, etc.)
/// * `bc_values` - Boundary condition values at element nodes (for RHS)
/// * `bc_type` - Boundary condition type (0=velocity, 1=pressure)
/// * `compute_rhs` - Whether to compute RHS contribution
///
/// # Returns
/// IntegrationResult with G, H, H^T, E integrals and RHS contribution
pub fn singular_integration(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    physics: &PhysicsParams,
    bc_values: Option<&[Complex64]>,
    bc_type: i32,
    compute_rhs: bool,
) -> IntegrationResult {
    singular_integration_curved(
        source_point,
        source_normal,
        element_coords,
        element_type,
        physics,
        bc_values,
        bc_type,
        compute_rhs,
        None,
    )
}

/// Analytic wavenumber derivative using the production frequency-dependent
/// singular quadrature schedule.
pub fn singular_integration_wavenumber_derivative(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    physics: &PhysicsParams,
    bc_values: Option<&[Complex64]>,
    bc_type: i32,
    compute_rhs: bool,
    node_normals: Option<&[[f64; 3]]>,
) -> IntegrationResult {
    let element_size = estimate_element_size(element_coords, element_type);
    let ka = physics.wave_number * element_size;
    let quadrature = QuadratureParams::for_ka(ka);

    singular_integration_wavenumber_derivative_with_params(
        source_point,
        source_normal,
        element_coords,
        element_type,
        physics,
        bc_values,
        bc_type,
        compute_rhs,
        &quadrature,
        node_normals,
    )
}

/// Perform singular integration with optional isoparametric curved normals
pub fn singular_integration_curved(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    physics: &PhysicsParams,
    bc_values: Option<&[Complex64]>,
    bc_type: i32,
    compute_rhs: bool,
    node_normals: Option<&[[f64; 3]]>,
) -> IntegrationResult {
    // Estimate element size for frequency-dependent quadrature
    let element_size = estimate_element_size(element_coords, element_type);
    let ka = physics.wave_number * element_size;
    let quadrature = QuadratureParams::for_ka(ka);

    singular_integration_with_params(
        source_point,
        source_normal,
        element_coords,
        element_type,
        physics,
        bc_values,
        bc_type,
        compute_rhs,
        &quadrature,
        node_normals,
    )
}

/// Perform singular integration with explicit quadrature parameters
///
/// This version allows fine-grained control over integration accuracy.
pub fn singular_integration_with_params(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    physics: &PhysicsParams,
    bc_values: Option<&[Complex64]>,
    bc_type: i32,
    compute_rhs: bool,
    quadrature: &QuadratureParams,
    node_normals: Option<&[[f64; 3]]>,
) -> IntegrationResult {
    singular_integration_with_params_mode(
        source_point,
        source_normal,
        element_coords,
        element_type,
        physics,
        bc_values,
        bc_type,
        compute_rhs,
        quadrature,
        node_normals,
        false,
    )
}

/// Analytic wavenumber derivative of singular/curved integration.
///
/// Geometry, element normals, quadrature parameters, and boundary values are
/// held fixed. The derivative covers the regularised edge and Duffy paths,
/// including the hypersingular `E` channel and the Burton--Miller beta factor
/// in the optional RHS. This is the singular counterpart to
/// `regular_integration_fixed_order_wavenumber_derivative`.
pub fn singular_integration_wavenumber_derivative_with_params(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    physics: &PhysicsParams,
    bc_values: Option<&[Complex64]>,
    bc_type: i32,
    compute_rhs: bool,
    quadrature: &QuadratureParams,
    node_normals: Option<&[[f64; 3]]>,
) -> IntegrationResult {
    singular_integration_with_params_mode(
        source_point,
        source_normal,
        element_coords,
        element_type,
        physics,
        bc_values,
        bc_type,
        compute_rhs,
        quadrature,
        node_normals,
        true,
    )
}

/// Shared singular integration kernel, optionally evaluating its analytic
/// wavenumber derivative. Keeping both paths in one quadrature loop prevents
/// the derivative from drifting from production geometry mappings.
fn singular_integration_with_params_mode(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    physics: &PhysicsParams,
    bc_values: Option<&[Complex64]>,
    bc_type: i32,
    compute_rhs: bool,
    quadrature: &QuadratureParams,
    node_normals: Option<&[[f64; 3]]>,
    wavenumber_derivative: bool,
) -> IntegrationResult {
    let num_nodes = element_type.num_nodes();
    let wavruim = physics.harmonic_factor * physics.wave_number;
    let k2 = physics.wave_number * physics.wave_number;

    let mut result = IntegrationResult::default();

    // Gauss-Legendre for edge integration
    let ngpo1 = quadrature.edge_gauss_order;
    let (coord_gau, weit_gau) = gauss_legendre(ngpo1);
    let nsec1 = quadrature.edge_sections;
    let nsec2 = quadrature.subtriangles_per_section;

    // Loop over edges of the element
    for ieg in 0..num_nodes {
        let ig1 = (ieg + 1) % num_nodes;
        let ig2 = ieg + num_nodes;

        // Compute edge length and direction
        let mut diff_poi = [0.0_f64; 3];
        let mut leneg = 0.0_f64;
        for i in 0..3 {
            diff_poi[i] = element_coords[[ig1, i]] - element_coords[[ieg, i]];
            leneg += diff_poi[i] * diff_poi[i];
        }
        leneg = leneg.sqrt();

        // Unit vector along edge
        let diff_poo = [
            diff_poi[0] / leneg,
            diff_poi[1] / leneg,
            diff_poi[2] / leneg,
        ];
        let leneg_scaled = leneg / (2.0 * nsec1 as f64);

        // Edge integration for hypersingular kernel (E integral)
        let mut zre = Complex64::new(0.0, 0.0);

        // Each edge is divided into nsec1 sections
        let delsec = 2.0 / nsec1 as f64;
        let mut secmid = -1.0 - delsec / 2.0;

        for _isec in 0..nsec1 {
            secmid += delsec;

            // Loop over Gauss points
            for ig in 0..ngpo1 {
                let sga = secmid + coord_gau[ig] / nsec1 as f64;
                let wga = weit_gau[ig] * leneg_scaled;

                // Gauss point coordinates
                let mut crd_gp = [0.0_f64; 3];
                let mut diff_fsp = [0.0_f64; 3];
                for i in 0..3 {
                    crd_gp[i] = element_coords[[ieg, i]] + diff_poi[i] * (sga + 1.0) / 2.0;
                    diff_fsp[i] = crd_gp[i] - source_point[i];
                }

                // Normalize and get distance
                let (unit_r, dis_fsp) = normalize(diff_fsp);

                if dis_fsp < 1e-15 {
                    continue;
                }

                // Green's function and gradient factor. For the derivative
                // path differentiate both the phase and `(ik - 1/r)` factor.
                let re1 = wavruim * dis_fsp;
                let re2 = 4.0 * PI * dis_fsp;
                let zg = sonium_fastmath::exp_i(re1) / re2;

                // Factor (ik - 1/r)
                let z1 = Complex64::new(-1.0 / dis_fsp, wavruim);
                let zg_factor = if wavenumber_derivative {
                    let dzg = zg * Complex64::new(0.0, physics.harmonic_factor * dis_fsp);
                    let dz1 = Complex64::new(0.0, physics.harmonic_factor);
                    dzg * z1 + zg * dz1
                } else {
                    zg * z1
                };

                // Compute ∇G × edge_direction
                let mut zdg_dy = [Complex64::new(0.0, 0.0); 3];
                for i in 0..3 {
                    zdg_dy[i] = zg_factor * unit_r[i];
                }

                // Cross product: (∇G) × (edge direction)
                let zwk = array![
                    zdg_dy[1] * diff_poo[2] - zdg_dy[2] * diff_poo[1],
                    zdg_dy[2] * diff_poo[0] - zdg_dy[0] * diff_poo[2],
                    zdg_dy[0] * diff_poo[1] - zdg_dy[1] * diff_poo[0]
                ];

                // Dot with source normal
                zre += (zwk[0] * source_normal[0]
                    + zwk[1] * source_normal[1]
                    + zwk[2] * source_normal[2])
                    * wga;
            }
        }
        result.d2g_dnxdny_integral += zre;

        // Subelement integration for G, H, H^T kernels
        for isec in 0..nsec2 {
            // Compute subtriangle vertices in local coordinates
            let (ssub, tsub, aresub) = match element_type {
                ElementType::Tri3 => {
                    let aresub = 1.0 / 24.0 / nsec2 as f64;
                    let mut ssub = [0.0; 3];
                    let mut tsub = [0.0; 3];

                    ssub[0] = 1.0 / 3.0;
                    tsub[0] = 1.0 / 3.0;

                    if isec == 0 {
                        ssub[1] = CSI6[ieg];
                        ssub[2] = CSI6[ig2];
                        tsub[1] = ETA6[ieg];
                        tsub[2] = ETA6[ig2];
                    } else {
                        ssub[1] = CSI6[ig2];
                        ssub[2] = CSI6[ig1];
                        tsub[1] = ETA6[ig2];
                        tsub[2] = ETA6[ig1];
                    }
                    (ssub, tsub, aresub)
                }
                ElementType::Quad4 => {
                    let aresub = 0.25 / nsec2 as f64;
                    let mut ssub = [0.0; 3];
                    let mut tsub = [0.0; 3];

                    ssub[0] = 0.0;
                    tsub[0] = 0.0;

                    if isec == 0 {
                        ssub[1] = CSI8[ieg];
                        ssub[2] = CSI8[ig2];
                        tsub[1] = ETA8[ieg];
                        tsub[2] = ETA8[ig2];
                    } else {
                        ssub[1] = CSI8[ig2];
                        ssub[2] = CSI8[ig1];
                        tsub[1] = ETA8[ig2];
                        tsub[2] = ETA8[ig1];
                    }
                    (ssub, tsub, aresub)
                }
            };

            // Get quadrature points for subtriangle integration
            // The mapping uses a tensor-product domain [-1,1]² with Duffy-type transformation
            let ngausin = quadrature.subelement_gauss_order;
            let quad_points = gauss_legendre(ngausin);
            let (gau_coords, gau_weights) = quad_points;

            for (i, &sga) in gau_coords.iter().enumerate() {
                for (j, &tga) in gau_coords.iter().enumerate() {
                    let wei_gau = gau_weights[i] * gau_weights[j];

                    // Map from subtriangle to original element coordinates
                    let sgg = 0.5 * (1.0 - sga) * ssub[0]
                        + 0.25 * (1.0 + sga) * ((1.0 - tga) * ssub[1] + (1.0 + tga) * ssub[2]);
                    let tgg = 0.5 * (1.0 - sga) * tsub[0]
                        + 0.25 * (1.0 + sga) * ((1.0 - tga) * tsub[1] + (1.0 + tga) * tsub[2]);

                    // Compute shape functions and Jacobian
                    let (shape_fn, shape_ds, shape_dt, jacobian, el_norm_flat, crd_poi) =
                        compute_shape_and_jacobian(element_coords, element_type, sgg, tgg);

                    let (crd_poi, jacobian, el_norm) = if node_normals.is_some() {
                        quadrature_geometry(
                            element_coords,
                            &shape_fn,
                            &shape_ds,
                            &shape_dt,
                            node_normals,
                        )
                    } else {
                        (crd_poi, jacobian, el_norm_flat)
                    };

                    let wga = wei_gau * (1.0 + sga) * aresub * jacobian;

                    // Compute kernel functions
                    let diff_fsp = [
                        crd_poi[0] - source_point[0],
                        crd_poi[1] - source_point[1],
                        crd_poi[2] - source_point[2],
                    ];

                    let (unit_r, dis_fsp) = normalize(diff_fsp);

                    if dis_fsp < 1e-15 {
                        continue;
                    }

                    // G kernel and its fixed-geometry wavenumber derivative.
                    let re1 = wavruim * dis_fsp;
                    let re2 = wga / (4.0 * PI * dis_fsp);
                    let zg = sonium_fastmath::exp_i(re1) * re2;
                    let dzg = zg * Complex64::new(0.0, physics.harmonic_factor * dis_fsp);

                    // H and H^T kernels
                    let z1 = Complex64::new(-1.0 / dis_fsp, wavruim);
                    let zhh_base = zg * z1;
                    let dzhh_base = dzg * z1 + zg * Complex64::new(0.0, physics.harmonic_factor);
                    let re1_h =
                        unit_r[0] * el_norm[0] + unit_r[1] * el_norm[1] + unit_r[2] * el_norm[2];
                    let re2_h = -(unit_r[0] * source_normal[0]
                        + unit_r[1] * source_normal[1]
                        + unit_r[2] * source_normal[2]);
                    let zhh = if wavenumber_derivative {
                        dzhh_base * re1_h
                    } else {
                        zhh_base * re1_h
                    };
                    let zht = if wavenumber_derivative {
                        dzhh_base * re2_h
                    } else {
                        zhh_base * re2_h
                    };

                    // Accumulate integrals
                    result.g_integral += if wavenumber_derivative { dzg } else { zg };
                    result.dg_dn_integral += zhh;
                    result.dg_dnx_integral += zht;

                    // E kernel contribution
                    let normal_dot = source_normal[0] * el_norm[0]
                        + source_normal[1] * el_norm[1]
                        + source_normal[2] * el_norm[2];
                    result.d2g_dnxdny_integral += if wavenumber_derivative {
                        (dzg * k2 + zg * (2.0 * physics.wave_number)) * normal_dot
                    } else {
                        zg * k2 * normal_dot
                    };

                    // RHS contribution if needed
                    if compute_rhs && bc_type == 0 {
                        // Velocity BC
                        if let Some(bc) = bc_values {
                            let mut zbgao = Complex64::new(0.0, 0.0);
                            for (ii, &n) in shape_fn.iter().enumerate() {
                                if ii < bc.len() {
                                    zbgao += bc[ii] * n;
                                }
                            }
                            let gamma = physics.gamma();
                            let tau = physics.tau;
                            let beta = physics.burton_miller_beta();
                            let beta_term = if wavenumber_derivative {
                                zht * beta
                                    + (dzg * re2_h)
                                        * physics.burton_miller_beta_wavenumber_derivative()
                            } else {
                                zht * beta
                            };
                            let g_term = if wavenumber_derivative {
                                dzg * gamma * tau
                            } else {
                                zg * gamma * tau
                            };
                            result.rhs_contribution += (g_term + beta_term) * zbgao;
                        }
                    }
                }
            }
        }
    }

    // RHS for pressure BC (Dirichlet): known p contributes through K and E kernels
    if compute_rhs
        && bc_type == 1
        && let Some(bc) = bc_values
    {
        let zbgao = bc.iter().sum::<Complex64>() / bc.len() as f64;
        let gamma = physics.gamma();
        let tau = physics.tau;
        let beta = physics.burton_miller_beta();
        result.rhs_contribution = if wavenumber_derivative {
            (result.dg_dn_integral * gamma * tau
                + result.d2g_dnxdny_integral * beta
                + singular_integration_with_params(
                    source_point,
                    source_normal,
                    element_coords,
                    element_type,
                    physics,
                    None,
                    bc_type,
                    false,
                    quadrature,
                    node_normals,
                )
                .d2g_dnxdny_integral
                    * physics.burton_miller_beta_wavenumber_derivative())
                * zbgao
        } else {
            (result.dg_dn_integral * gamma * tau + result.d2g_dnxdny_integral * beta) * zbgao
        };
    }

    result
}

/// Compare two parameterized singular evaluations, including hypersingular and
/// curved-normal contributions.
pub fn singular_integration_error_estimate(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    physics: &PhysicsParams,
    bc_values: Option<&[Complex64]>,
    bc_type: i32,
    compute_rhs: bool,
    coarse_params: &QuadratureParams,
    fine_params: &QuadratureParams,
    node_normals: Option<&[[f64; 3]]>,
) -> Result<IntegrationEstimate, QuadratureOrderError> {
    if !coarse_params.is_valid()
        || !fine_params.is_valid()
        || !fine_params.is_strictly_finer_than(coarse_params)
    {
        return Err(QuadratureOrderError {
            coarse_order: coarse_params.subelement_gauss_order,
            fine_order: fine_params.subelement_gauss_order,
        });
    }

    let coarse = singular_integration_with_params(
        source_point,
        source_normal,
        element_coords,
        element_type,
        physics,
        bc_values,
        bc_type,
        compute_rhs,
        coarse_params,
        node_normals,
    );
    let fine = singular_integration_with_params(
        source_point,
        source_normal,
        element_coords,
        element_type,
        physics,
        bc_values,
        bc_type,
        compute_rhs,
        fine_params,
        node_normals,
    );
    Ok(IntegrationEstimate::from_results(coarse, fine))
}

/// Evaluate a singular panel with an explicit adjacent-parameter
/// convergence gate and return the finer result only when every kernel
/// channel satisfies the requested mixed error budget.
pub fn singular_integration_checked(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    physics: &PhysicsParams,
    bc_values: Option<&[Complex64]>,
    bc_type: i32,
    compute_rhs: bool,
    coarse_params: &QuadratureParams,
    fine_params: &QuadratureParams,
    node_normals: Option<&[[f64; 3]]>,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) -> Result<IntegrationResult, QuadratureConvergenceError> {
    let estimate = singular_integration_error_estimate(
        source_point,
        source_normal,
        element_coords,
        element_type,
        physics,
        bc_values,
        bc_type,
        compute_rhs,
        coarse_params,
        fine_params,
        node_normals,
    )
    .map_err(QuadratureConvergenceError::InvalidOrder)?;
    estimate.require_converged(absolute_tolerance, relative_tolerance)?;
    Ok(estimate.fine)
}

/// Perform singular integration for linear elements, returning per-basis-function results
///
/// This integrates the singular kernels (G, H, H^T, E) over the self-element,
/// weighting each by the basis functions. For discontinuous linear elements,
/// the collocation point is NOT at the centroid but at an internal Gauss point,
/// so the singularity handling via edge integrals and Duffy-type subtriangulation
/// still applies.
///
/// # Arguments
/// * `source_point` - Collocation point (an internal Gauss point of this element)
/// * `source_normal` - Unit normal at the collocation point
/// * `element_coords` - Node coordinates of the element
/// * `element_type` - Triangle or quad element
/// * `physics` - Physics parameters
/// * `num_basis` - Number of basis functions (3 for Tri3, 4 for Quad4)
pub fn singular_integration_linear(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    physics: &PhysicsParams,
    num_basis: usize,
) -> LinearIntegrationResult {
    singular_integration_linear_curved(
        source_point,
        source_normal,
        element_coords,
        element_type,
        physics,
        num_basis,
        None,
    )
}

/// Perform singular integration for linear elements with optional curved normals
pub fn singular_integration_linear_curved(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    physics: &PhysicsParams,
    num_basis: usize,
    node_normals: Option<&[[f64; 3]]>,
) -> LinearIntegrationResult {
    let element_size = estimate_element_size(element_coords, element_type);
    let ka = physics.wave_number * element_size;
    let quadrature = QuadratureParams::for_ka(ka);

    singular_integration_linear_with_params(
        source_point,
        source_normal,
        element_coords,
        element_type,
        physics,
        num_basis,
        &quadrature,
        node_normals,
    )
}

/// Perform singular linear-element integration with the analytic wavenumber
/// derivative and the production frequency-dependent quadrature schedule.
pub(crate) fn singular_integration_linear_wavenumber_derivative(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    physics: &PhysicsParams,
    num_basis: usize,
) -> LinearIntegrationResult {
    let element_size = estimate_element_size(element_coords, element_type);
    let ka = physics.wave_number * element_size;
    let quadrature = QuadratureParams::for_ka(ka);
    singular_integration_linear_wavenumber_derivative_with_params(
        source_point,
        source_normal,
        element_coords,
        element_type,
        physics,
        num_basis,
        &quadrature,
        None,
    )
}

/// Analytic wavenumber derivative for singular linear panels with curved normals.
pub(crate) fn singular_integration_linear_curved_wavenumber_derivative(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    physics: &PhysicsParams,
    num_basis: usize,
    node_normals: Option<&[[f64; 3]]>,
) -> LinearIntegrationResult {
    let element_size = estimate_element_size(element_coords, element_type);
    let ka = physics.wave_number * element_size;
    let quadrature = QuadratureParams::for_ka(ka);
    singular_integration_linear_wavenumber_derivative_with_params(
        source_point,
        source_normal,
        element_coords,
        element_type,
        physics,
        num_basis,
        &quadrature,
        node_normals,
    )
}

/// Perform singular integration for linear elements with explicit quadrature parameters
pub fn singular_integration_linear_with_params(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    physics: &PhysicsParams,
    num_basis: usize,
    quadrature: &QuadratureParams,
    node_normals: Option<&[[f64; 3]]>,
) -> LinearIntegrationResult {
    singular_integration_linear_with_params_mode(
        source_point,
        source_normal,
        element_coords,
        element_type,
        physics,
        num_basis,
        quadrature,
        node_normals,
        false,
    )
}

/// Perform singular linear-element integration while differentiating the
/// Helmholtz kernels with respect to acoustic wavenumber.
///
/// The returned channels retain one result per panel basis function.  This
/// is the higher-order counterpart to
/// [`singular_integration_wavenumber_derivative`] and keeps self blocks on the
/// same Duffy/edge quadrature path as the production value operator.
pub(crate) fn singular_integration_linear_wavenumber_derivative_with_params(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    physics: &PhysicsParams,
    num_basis: usize,
    quadrature: &QuadratureParams,
    node_normals: Option<&[[f64; 3]]>,
) -> LinearIntegrationResult {
    singular_integration_linear_with_params_mode(
        source_point,
        source_normal,
        element_coords,
        element_type,
        physics,
        num_basis,
        quadrature,
        node_normals,
        true,
    )
}

fn singular_integration_linear_with_params_mode(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    physics: &PhysicsParams,
    num_basis: usize,
    quadrature: &QuadratureParams,
    node_normals: Option<&[[f64; 3]]>,
    wavenumber_derivative: bool,
) -> LinearIntegrationResult {
    let num_nodes = element_type.num_nodes();
    let wavruim = physics.harmonic_factor * physics.wave_number;
    let k2 = physics.wave_number * physics.wave_number;

    let mut basis_results: Vec<IntegrationResult> = (0..num_basis)
        .map(|_| IntegrationResult::default())
        .collect();

    // Gauss-Legendre for edge integration
    let ngpo1 = quadrature.edge_gauss_order;
    let (coord_gau, weit_gau) = gauss_legendre(ngpo1);
    let nsec1 = quadrature.edge_sections;
    let nsec2 = quadrature.subtriangles_per_section;

    // Loop over edges of the element
    for ieg in 0..num_nodes {
        let ig1 = (ieg + 1) % num_nodes;
        let ig2 = ieg + num_nodes;

        // Compute edge length and direction
        let mut diff_poi = [0.0_f64; 3];
        let mut leneg = 0.0_f64;
        for i in 0..3 {
            diff_poi[i] = element_coords[[ig1, i]] - element_coords[[ieg, i]];
            leneg += diff_poi[i] * diff_poi[i];
        }
        leneg = leneg.sqrt();

        // Unit vector along edge
        let diff_poo = [
            diff_poi[0] / leneg,
            diff_poi[1] / leneg,
            diff_poi[2] / leneg,
        ];
        let leneg_scaled = leneg / (2.0 * nsec1 as f64);

        // Edge integration for hypersingular kernel (E integral)
        // For linear elements, the edge integral also needs basis-function weighting
        // We compute the edge position and evaluate shape functions there
        let delsec = 2.0 / nsec1 as f64;
        let mut secmid = -1.0 - delsec / 2.0;

        for _isec in 0..nsec1 {
            secmid += delsec;

            for ig in 0..ngpo1 {
                let sga = secmid + coord_gau[ig] / nsec1 as f64;
                let wga = weit_gau[ig] * leneg_scaled;

                // Gauss point on the edge in physical coords
                let mut crd_gp = [0.0_f64; 3];
                let mut diff_fsp = [0.0_f64; 3];
                for i in 0..3 {
                    crd_gp[i] = element_coords[[ieg, i]] + diff_poi[i] * (sga + 1.0) / 2.0;
                    diff_fsp[i] = crd_gp[i] - source_point[i];
                }

                let (unit_r, dis_fsp) = normalize(diff_fsp);

                if dis_fsp < 1e-15 {
                    continue;
                }

                // Green's function and gradient factor
                let re1 = wavruim * dis_fsp;
                let re2 = 4.0 * PI * dis_fsp;
                let zg = sonium_fastmath::exp_i(re1) / re2;

                let z1 = Complex64::new(-1.0 / dis_fsp, wavruim);
                let zg_factor = if wavenumber_derivative {
                    let dzg = zg * Complex64::new(0.0, physics.harmonic_factor * dis_fsp);
                    dzg * z1 + zg * Complex64::new(0.0, physics.harmonic_factor)
                } else {
                    zg * z1
                };

                let mut zdg_dy = [Complex64::new(0.0, 0.0); 3];
                for i in 0..3 {
                    zdg_dy[i] = zg_factor * unit_r[i];
                }

                let zwk = array![
                    zdg_dy[1] * diff_poo[2] - zdg_dy[2] * diff_poo[1],
                    zdg_dy[2] * diff_poo[0] - zdg_dy[0] * diff_poo[2],
                    zdg_dy[0] * diff_poo[1] - zdg_dy[1] * diff_poo[0]
                ];

                let edge_contrib = (zwk[0] * source_normal[0]
                    + zwk[1] * source_normal[1]
                    + zwk[2] * source_normal[2])
                    * wga;

                // Evaluate shape functions at the edge point (map physical to local)
                // For the edge integral, the point is on the edge between vertices ieg and ig1.
                // Parametric position along the edge: t_edge = (sga + 1) / 2
                // Local coords: interpolate between vertex ieg and ig1 local coords
                let t_edge = (sga + 1.0) / 2.0;
                let (s_local, t_local) = match element_type {
                    ElementType::Tri3 => {
                        let s_l = CSI6[ieg] * (1.0 - t_edge) + CSI6[ig1] * t_edge;
                        let t_l = ETA6[ieg] * (1.0 - t_edge) + ETA6[ig1] * t_edge;
                        (s_l, t_l)
                    }
                    ElementType::Quad4 => {
                        let s_l = CSI8[ieg] * (1.0 - t_edge) + CSI8[ig1] * t_edge;
                        let t_l = ETA8[ieg] * (1.0 - t_edge) + ETA8[ig1] * t_edge;
                        (s_l, t_l)
                    }
                };
                let _vertex_shape_at_edge = match element_type {
                    ElementType::Tri3 => vec![1.0 - s_local - t_local, s_local, t_local],
                    ElementType::Quad4 => {
                        let s1 = 0.25 * (s_local + 1.0);
                        let s2 = 0.25 * (s_local - 1.0);
                        let t1 = t_local + 1.0;
                        let t2 = t_local - 1.0;
                        vec![s1 * t1, -s2 * t1, s2 * t2, -s1 * t2]
                    }
                };
                let shape_at_edge = panel_basis_values(element_type, s_local, t_local, num_basis);

                for b in 0..num_basis {
                    basis_results[b].d2g_dnxdny_integral += edge_contrib * shape_at_edge[b];
                }
            }
        }

        // The legacy subelement fan below stays anchored at the panel
        // centre for the G, H, and H^T channels and the hypersingular
        // area term. The hypersingular density-gradient term is
        // integrated separately on a fan anchored at the singular
        // source point: higher-order collocation points sit off-centre,
        // and the Duffy factor (1 + sga) only cancels that stronger
        // singularity when the fan apex is the source.
        let centre_apex = match element_type {
            ElementType::Tri3 => (1.0 / 3.0, 1.0 / 3.0),
            ElementType::Quad4 => (0.0, 0.0),
        };
        let source_apex = panel_local_coordinates(source_point, element_coords, element_type);

        // Subelement integration for G, H, H^T kernels
        for isec in 0..nsec2 {
            let (ssub, tsub) = match element_type {
                ElementType::Tri3 => {
                    let mut ssub = [0.0; 3];
                    let mut tsub = [0.0; 3];

                    ssub[0] = centre_apex.0;
                    tsub[0] = centre_apex.1;

                    if isec == 0 {
                        ssub[1] = CSI6[ieg];
                        ssub[2] = CSI6[ig2];
                        tsub[1] = ETA6[ieg];
                        tsub[2] = ETA6[ig2];
                    } else {
                        ssub[1] = CSI6[ig2];
                        ssub[2] = CSI6[ig1];
                        tsub[1] = ETA6[ig2];
                        tsub[2] = ETA6[ig1];
                    }
                    (ssub, tsub)
                }
                ElementType::Quad4 => {
                    let mut ssub = [0.0; 3];
                    let mut tsub = [0.0; 3];

                    ssub[0] = centre_apex.0;
                    tsub[0] = centre_apex.1;

                    if isec == 0 {
                        ssub[1] = CSI8[ieg];
                        ssub[2] = CSI8[ig2];
                        tsub[1] = ETA8[ieg];
                        tsub[2] = ETA8[ig2];
                    } else {
                        ssub[1] = CSI8[ig2];
                        ssub[2] = CSI8[ig1];
                        tsub[1] = ETA8[ig2];
                        tsub[2] = ETA8[ig1];
                    }
                    (ssub, tsub)
                }
            };
            // Reference-area factor of the affine fan map: with
            // u = (1 + sga) / 2 the map Jacobian is u * D / 4 where
            // D is the subtriangle determinant below, and the weight
            // already carries (1 + sga), leaving D / 8 per section.
            // The legacy loop repeats the second section
            // (nsec2 - 1) times, so sections share the historical
            // 2 / nsec2 split; for a centre apex this reduces to the
            // historical constants.
            let aresub = ((ssub[1] - ssub[0]) * (tsub[2] - tsub[0])
                - (ssub[2] - ssub[0]) * (tsub[1] - tsub[0]))
                .abs()
                / (4.0 * nsec2 as f64);

            let ngausin = quadrature.subelement_gauss_order;
            let quad_points = gauss_legendre(ngausin);
            let (gau_coords, gau_weights) = quad_points;

            for (i, &sga) in gau_coords.iter().enumerate() {
                for (j, &tga) in gau_coords.iter().enumerate() {
                    let wei_gau = gau_weights[i] * gau_weights[j];

                    let sgg = 0.5 * (1.0 - sga) * ssub[0]
                        + 0.25 * (1.0 + sga) * ((1.0 - tga) * ssub[1] + (1.0 + tga) * ssub[2]);
                    let tgg = 0.5 * (1.0 - sga) * tsub[0]
                        + 0.25 * (1.0 + sga) * ((1.0 - tga) * tsub[1] + (1.0 + tga) * tsub[2]);

                    let (shape_fn, shape_ds, shape_dt, jacobian, el_norm_flat, crd_poi) =
                        compute_shape_and_jacobian(element_coords, element_type, sgg, tgg);

                    let (crd_poi, jacobian, el_norm) = if node_normals.is_some() {
                        quadrature_geometry(
                            element_coords,
                            &shape_fn,
                            &shape_ds,
                            &shape_dt,
                            node_normals,
                        )
                    } else {
                        (crd_poi, jacobian, el_norm_flat)
                    };

                    let wga = wei_gau * (1.0 + sga) * aresub * jacobian;

                    let diff_fsp = [
                        crd_poi[0] - source_point[0],
                        crd_poi[1] - source_point[1],
                        crd_poi[2] - source_point[2],
                    ];

                    let (unit_r, dis_fsp) = normalize(diff_fsp);

                    if dis_fsp < 1e-15 {
                        continue;
                    }

                    // G kernel
                    let re1 = wavruim * dis_fsp;
                    let re2 = wga / (4.0 * PI * dis_fsp);
                    let zg = sonium_fastmath::exp_i(re1) * re2;
                    let dzg = zg * Complex64::new(0.0, physics.harmonic_factor * dis_fsp);

                    // H and H^T kernels
                    let z1 = Complex64::new(-1.0 / dis_fsp, wavruim);
                    let zhh_base = zg * z1;
                    let dzhh_base = dzg * z1 + zg * Complex64::new(0.0, physics.harmonic_factor);
                    let re1_h =
                        unit_r[0] * el_norm[0] + unit_r[1] * el_norm[1] + unit_r[2] * el_norm[2];
                    let re2_h = -(unit_r[0] * source_normal[0]
                        + unit_r[1] * source_normal[1]
                        + unit_r[2] * source_normal[2]);
                    let zhh = if wavenumber_derivative {
                        dzhh_base * re1_h
                    } else {
                        zhh_base * re1_h
                    };
                    let zht = if wavenumber_derivative {
                        dzhh_base * re2_h
                    } else {
                        zhh_base * re2_h
                    };

                    // E kernel contribution from area integral
                    let normal_dot = source_normal[0] * el_norm[0]
                        + source_normal[1] * el_norm[1]
                        + source_normal[2] * el_norm[2];
                    let e_area_contrib = if wavenumber_derivative {
                        (dzg * k2 + zg * (2.0 * physics.wave_number)) * normal_dot
                    } else {
                        zg * k2 * normal_dot
                    };

                    let basis_values = panel_basis_values(element_type, sgg, tgg, num_basis);

                    // Weight by collocation-basis functions
                    for b in 0..num_basis {
                        let n_b = basis_values[b];
                        basis_results[b].g_integral += if wavenumber_derivative {
                            dzg * n_b
                        } else {
                            zg * n_b
                        };
                        basis_results[b].dg_dn_integral += zhh * n_b;
                        basis_results[b].dg_dnx_integral += zht * n_b;
                        basis_results[b].d2g_dnxdny_integral += e_area_contrib * n_b;
                    }
                }
            }
        }

        // Hypersingular density-gradient (Maue) term on a fan anchored
        // at the source point. Stokes' theorem for a weighted
        // hypersingular kernel also differentiates the density; the
        // edge term alone is valid only for a constant basis.
        for isec in 0..nsec2 {
            let (ssub, tsub) = match element_type {
                ElementType::Tri3 => {
                    let mut ssub = [0.0; 3];
                    let mut tsub = [0.0; 3];
                    ssub[0] = source_apex.0;
                    tsub[0] = source_apex.1;
                    if isec == 0 {
                        ssub[1] = CSI6[ieg];
                        ssub[2] = CSI6[ig2];
                        tsub[1] = ETA6[ieg];
                        tsub[2] = ETA6[ig2];
                    } else {
                        ssub[1] = CSI6[ig2];
                        ssub[2] = CSI6[ig1];
                        tsub[1] = ETA6[ig2];
                        tsub[2] = ETA6[ig1];
                    }
                    (ssub, tsub)
                }
                ElementType::Quad4 => {
                    let mut ssub = [0.0; 3];
                    let mut tsub = [0.0; 3];
                    ssub[0] = source_apex.0;
                    tsub[0] = source_apex.1;
                    if isec == 0 {
                        ssub[1] = CSI8[ieg];
                        ssub[2] = CSI8[ig2];
                        tsub[1] = ETA8[ieg];
                        tsub[2] = ETA8[ig2];
                    } else {
                        ssub[1] = CSI8[ig2];
                        ssub[2] = CSI8[ig1];
                        tsub[1] = ETA8[ig2];
                        tsub[2] = ETA8[ig1];
                    }
                    (ssub, tsub)
                }
            };
            let aresub = ((ssub[1] - ssub[0]) * (tsub[2] - tsub[0])
                - (ssub[2] - ssub[0]) * (tsub[1] - tsub[0]))
                .abs()
                / (4.0 * nsec2 as f64);
            let ngausin = quadrature.subelement_gauss_order;
            let quad_points = gauss_legendre(ngausin);
            let (gau_coords, gau_weights) = quad_points;
            for (i, &sga) in gau_coords.iter().enumerate() {
                for (j, &tga) in gau_coords.iter().enumerate() {
                    let wei_gau = gau_weights[i] * gau_weights[j];
                    let sgg = 0.5 * (1.0 - sga) * ssub[0]
                        + 0.25 * (1.0 + sga) * ((1.0 - tga) * ssub[1] + (1.0 + tga) * ssub[2]);
                    let tgg = 0.5 * (1.0 - sga) * tsub[0]
                        + 0.25 * (1.0 + sga) * ((1.0 - tga) * tsub[1] + (1.0 + tga) * tsub[2]);
                    let (shape_fn, shape_ds, shape_dt, jacobian, el_norm_flat, crd_poi) =
                        compute_shape_and_jacobian(element_coords, element_type, sgg, tgg);
                    let (crd_poi, jacobian, el_norm) = if node_normals.is_some() {
                        quadrature_geometry(
                            element_coords,
                            &shape_fn,
                            &shape_ds,
                            &shape_dt,
                            node_normals,
                        )
                    } else {
                        (crd_poi, jacobian, el_norm_flat)
                    };
                    let wga = wei_gau * (1.0 + sga) * aresub * jacobian;
                    let diff_fsp = [
                        crd_poi[0] - source_point[0],
                        crd_poi[1] - source_point[1],
                        crd_poi[2] - source_point[2],
                    ];
                    let (unit_r, dis_fsp) = normalize(diff_fsp);
                    if dis_fsp < 1e-15 {
                        continue;
                    }
                    let re1 = wavruim * dis_fsp;
                    let re2 = wga / (4.0 * PI * dis_fsp);
                    let zg = sonium_fastmath::exp_i(re1) * re2;
                    let dzg = zg * Complex64::new(0.0, physics.harmonic_factor * dis_fsp);
                    let z1 = Complex64::new(-1.0 / dis_fsp, wavruim);
                    let zhh_base = zg * z1;
                    let dzhh_base = dzg * z1 + zg * Complex64::new(0.0, physics.harmonic_factor);
                    let re1_h =
                        unit_r[0] * el_norm[0] + unit_r[1] * el_norm[1] + unit_r[2] * el_norm[2];
                    let normal_dot = source_normal[0] * el_norm[0]
                        + source_normal[1] * el_norm[1]
                        + source_normal[2] * el_norm[2];
                    let differentiated_green = if wavenumber_derivative {
                        dzhh_base
                    } else {
                        zhh_base
                    };
                    let basis_gradients =
                        panel_basis_gradients(element_coords, element_type, sgg, tgg, num_basis);
                    for (b, gradient) in basis_gradients.iter().enumerate() {
                        let gradient_dot_r =
                            gradient.iter().zip(unit_r).map(|(a, b)| a * b).sum::<f64>();
                        let gradient_dot_source = gradient
                            .iter()
                            .zip(source_normal)
                            .map(|(a, b)| a * b)
                            .sum::<f64>();
                        let gradient_factor =
                            -normal_dot * gradient_dot_r + re1_h * gradient_dot_source;
                        basis_results[b].d2g_dnxdny_integral +=
                            differentiated_green * gradient_factor;
                    }
                }
            }
        }
    }

    LinearIntegrationResult { basis_results }
}

/// Invert the flat panel map at `source_point`, returning the local
/// coordinates of the singular point. Falls back to the panel centre
/// (the historical fan apex) when the map is degenerate or the source
/// projects outside the panel.
fn panel_local_coordinates(
    source_point: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
) -> (f64, f64) {
    let fallback = match element_type {
        ElementType::Tri3 => (1.0 / 3.0, 1.0 / 3.0),
        ElementType::Quad4 => (0.0, 0.0),
    };
    let solve_tangent_system = |tangent_s: [f64; 3], tangent_t: [f64; 3], rhs: [f64; 3]| {
        let ss = tangent_s.iter().map(|a| a * a).sum::<f64>();
        let tt = tangent_t.iter().map(|a| a * a).sum::<f64>();
        let st = tangent_s
            .iter()
            .zip(tangent_t)
            .map(|(a, b)| a * b)
            .sum::<f64>();
        let determinant = ss * tt - st * st;
        if determinant.abs() < 1e-30 {
            return None;
        }
        let rs = tangent_s
            .iter()
            .zip(rhs)
            .map(|(a, b)| a * b)
            .sum::<f64>();
        let rt = tangent_t
            .iter()
            .zip(rhs)
            .map(|(a, b)| a * b)
            .sum::<f64>();
        Some(((tt * rs - st * rt) / determinant, (ss * rt - st * rs) / determinant))
    };
    let (mut s, mut t) = fallback;
    for _ in 0..50 {
        let (_, shape_ds, shape_dt, _, _, position) =
            compute_shape_and_jacobian(element_coords, element_type, s, t);
        let residual = [
            source_point[0] - position[0],
            source_point[1] - position[1],
            source_point[2] - position[2],
        ];
        if residual.iter().map(|a| a * a).sum::<f64>().sqrt() < 1e-13 {
            break;
        }
        let mut tangent_s = [0.0; 3];
        let mut tangent_t = [0.0; 3];
        for node in 0..element_type.num_nodes() {
            for axis in 0..3 {
                tangent_s[axis] += shape_ds[node] * element_coords[[node, axis]];
                tangent_t[axis] += shape_dt[node] * element_coords[[node, axis]];
            }
        }
        let Some((ds, dt)) = solve_tangent_system(tangent_s, tangent_t, residual) else {
            return fallback;
        };
        s += ds;
        t += dt;
    }
    let inside = match element_type {
        ElementType::Tri3 => s >= -1e-9 && t >= -1e-9 && s + t <= 1.0 + 1e-9,
        ElementType::Quad4 => s.abs() <= 1.0 + 1e-9 && t.abs() <= 1.0 + 1e-9,
    };
    if inside { (s, t) } else { fallback }
}

fn panel_basis_gradients(
    coordinates: &DenseMatrix<f64>,
    element_type: ElementType,
    s: f64,
    t: f64,
    num_basis: usize,
) -> Vec<[f64; 3]> {
    let derivatives: Vec<[f64; 2]> = match (element_type, num_basis) {
        (ElementType::Tri3, 3) => vec![[-2.0, -2.0], [2.0, 0.0], [0.0, 2.0]],
        (ElementType::Tri3, 6) => {
            let l = 1.0 - s - t;
            vec![
                [1.0 - 4.0 * l, 1.0 - 4.0 * l],
                [4.0 * s - 1.0, 0.0],
                [0.0, 4.0 * t - 1.0],
                [4.0 * (l - s), -4.0 * s],
                [4.0 * t, 4.0 * s],
                [-4.0 * t, 4.0 * (l - t)],
            ]
        }
        (ElementType::Quad4, 4) => {
            let g = 1.0 / 3.0_f64.sqrt();
            let a = 1.0 / (2.0 * g);
            let ls = [(g - s) * a, (g + s) * a];
            let lt = [(g - t) * a, (g + t) * a];
            vec![
                [-a * lt[0], -a * ls[0]],
                [a * lt[0], -a * ls[1]],
                [a * lt[1], a * ls[1]],
                [-a * lt[1], a * ls[0]],
            ]
        }
        (ElementType::Quad4, 9) => {
            let lagrange = |x: f64| [0.5 * x * (x - 1.0), 1.0 - x * x, 0.5 * x * (x + 1.0)];
            let derivative = |x: f64| [x - 0.5, -2.0 * x, x + 0.5];
            let ls = lagrange(s);
            let lt = lagrange(t);
            let ds = derivative(s);
            let dt = derivative(t);
            // Same slot order as `panel_basis_values` (connectivity
            // vertices, consecutive edge midpoints, center).
            [
                (0, 0),
                (2, 0),
                (2, 2),
                (0, 2),
                (1, 0),
                (2, 1),
                (1, 2),
                (0, 1),
                (1, 1),
            ]
            .into_iter()
            .map(|(i, j)| [ds[i] * lt[j], ls[i] * dt[j]])
            .collect()
        }
        _ => panic!("unsupported discontinuous panel basis layout"),
    };
    let (_, shape_ds, shape_dt, _, _, _) =
        compute_shape_and_jacobian(coordinates, element_type, s, t);
    let mut tangent_s = [0.0; 3];
    let mut tangent_t = [0.0; 3];
    for node in 0..element_type.num_nodes() {
        for axis in 0..3 {
            tangent_s[axis] += shape_ds[node] * coordinates[[node, axis]];
            tangent_t[axis] += shape_dt[node] * coordinates[[node, axis]];
        }
    }
    let ss = tangent_s.iter().map(|a| a * a).sum::<f64>();
    let tt = tangent_t.iter().map(|a| a * a).sum::<f64>();
    let st = tangent_s
        .iter()
        .zip(tangent_t)
        .map(|(a, b)| a * b)
        .sum::<f64>();
    let determinant = ss * tt - st * st;
    derivatives
        .into_iter()
        .map(|[ds, dt]| {
            let a = (tt * ds - st * dt) / determinant;
            let b = (ss * dt - st * ds) / determinant;
            std::array::from_fn(|axis| a * tangent_s[axis] + b * tangent_t[axis])
        })
        .collect()
}

fn panel_basis_values(element_type: ElementType, s: f64, t: f64, num_basis: usize) -> Vec<f64> {
    let basis = match (element_type, num_basis) {
        (ElementType::Tri3, 3) | (ElementType::Quad4, 4) => BasisOrder::Linear,
        (ElementType::Tri3, 6) | (ElementType::Quad4, 9) => BasisOrder::Quadratic,
        _ => panic!("unsupported discontinuous panel basis layout"),
    };
    basis_values(element_type, basis, s, t)
}

/// Compute shape functions, derivatives, Jacobian and position at local coordinates
#[allow(clippy::type_complexity)]
fn compute_shape_and_jacobian(
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    s: f64,
    t: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64, [f64; 3], [f64; 3]) {
    let num_nodes = element_type.num_nodes();

    let (shape_fn, shape_ds, shape_dt) = match element_type {
        ElementType::Tri3 => {
            // Triangle area coordinates:
            // N0 = 1-s-t (vertex at (0,0))
            // N1 = s (vertex at (1,0))
            // N2 = t (vertex at (0,1))
            let shape_fn = vec![1.0 - s - t, s, t];
            let shape_ds = vec![-1.0, 1.0, 0.0];
            let shape_dt = vec![-1.0, 0.0, 1.0];
            (shape_fn, shape_ds, shape_dt)
        }
        ElementType::Quad4 => {
            // Bilinear quad
            let s1 = 0.25 * (s + 1.0);
            let s2 = 0.25 * (s - 1.0);
            let t1 = t + 1.0;
            let t2 = t - 1.0;

            let shape_fn = vec![s1 * t1, -s2 * t1, s2 * t2, -s1 * t2];
            let shape_ds = vec![
                0.25 * (t + 1.0),
                -0.25 * (t + 1.0),
                0.25 * (t - 1.0),
                -0.25 * (t - 1.0),
            ];
            let shape_dt = vec![
                0.25 * (s + 1.0),
                0.25 * (1.0 - s),
                0.25 * (s - 1.0),
                -0.25 * (s + 1.0),
            ];
            (shape_fn, shape_ds, shape_dt)
        }
    };

    // Compute global position and tangent vectors
    let mut crd_poi = [0.0_f64; 3];
    let mut dx_ds = [0.0_f64; 3];
    let mut dx_dt = [0.0_f64; 3];

    for i in 0..num_nodes {
        for j in 0..3 {
            crd_poi[j] += shape_fn[i] * element_coords[[i, j]];
            dx_ds[j] += shape_ds[i] * element_coords[[i, j]];
            dx_dt[j] += shape_dt[i] * element_coords[[i, j]];
        }
    }

    // Normal = dx_ds × dx_dt
    let n_arr = cross_product(dx_ds, dx_dt);
    let jacobian = (n_arr[0] * n_arr[0] + n_arr[1] * n_arr[1] + n_arr[2] * n_arr[2]).sqrt();

    let el_norm = if jacobian > 1e-15 {
        [
            n_arr[0] / jacobian,
            n_arr[1] / jacobian,
            n_arr[2] / jacobian,
        ]
    } else {
        [0.0; 3]
    };

    (shape_fn, shape_ds, shape_dt, jacobian, el_norm, crd_poi)
}

/// Subelement data for adaptive integration
#[derive(Debug, Clone)]
pub struct Subelement {
    /// Center local coordinate (xi) - used for quads
    pub xi_center: f64,
    /// Center local coordinate (eta) - used for quads
    pub eta_center: f64,
    /// Dimension factor (2^-j for subdivision level j) - used for quads
    pub factor: f64,
    /// Gauss order for this subelement
    pub gauss_order: usize,
    /// Vertex local coordinates for triangular subelements (for affine mapping)
    /// Format: [(xi0, eta0), (xi1, eta1), (xi2, eta2)]
    pub tri_vertices: Option<[(f64, f64); 3]>,
}

/// Generate adaptive subelements based on source point proximity
///
/// This implements NC_GenerateSubelements from the C++ code.
/// Elements are recursively subdivided until the distance from the
/// source point to subelement center exceeds a tolerance.
///
/// # Arguments
/// * `source_point` - Source point coordinates
/// * `element_coords` - Node coordinates of the element
/// * `element_type` - Triangle or quad element
/// * `element_area` - Area of the element
///
/// # Returns
/// Vector of Subelements with their centers, scale factors, and Gauss orders
pub fn generate_subelements(
    source_point: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    element_area: f64,
) -> Vec<Subelement> {
    const MAX_NSE: usize = 60;
    const NSE: usize = 4; // Subdivision factor
    // NumCalc uses ~3.0 for quasi-singular threshold
    // Increased from 1.5 to improve near-singular integration accuracy
    const TOL_F: f64 = 3.0; // Distance tolerance factor (was 1.5)
    const GAU_MAX: usize = 7; // Maximum Gauss order
    const GAU_MIN: usize = 4; // Minimum Gauss order (was 3, increased for better accuracy)
    const GAU_ACCU: f64 = 0.0005; // Accuracy tolerance

    let num_vertices = element_type.num_nodes();
    let mut result = Vec::new();

    // Local coordinates of vertices
    let (csi_nodes, eta_nodes): (Vec<f64>, Vec<f64>) = match element_type {
        ElementType::Tri3 => (CSI6[0..3].to_vec(), ETA6[0..3].to_vec()),
        ElementType::Quad4 => (CSI8[0..4].to_vec(), ETA8[0..4].to_vec()),
    };

    // Working arrays for subelement vertices
    let mut xi_sfp: Vec<Vec<f64>> = vec![vec![0.0; num_vertices]; MAX_NSE];
    let mut et_sfp: Vec<Vec<f64>> = vec![vec![0.0; num_vertices]; MAX_NSE];

    // Initialize with original element
    xi_sfp[0][..num_vertices].copy_from_slice(&csi_nodes[..num_vertices]);
    et_sfp[0][..num_vertices].copy_from_slice(&eta_nodes[..num_vertices]);

    let mut nsfl = 1; // Number of subelements at current level
    let mut faclin = 2.0; // Relative edge length factor

    loop {
        let mut ndie = 0; // Number of elements to subdivide
        faclin *= 0.5;
        let arels = element_area * faclin * faclin;
        let nsel = nsfl;

        // Copy current level
        let xi_sep: Vec<Vec<f64>> = xi_sfp[..nsel].to_vec();
        let et_sep: Vec<Vec<f64>> = et_sfp[..nsel].to_vec();

        for idi in 0..nsel {
            // Compute center of current subelement
            let scent: f64 =
                xi_sep[idi].iter().take(num_vertices).sum::<f64>() / num_vertices as f64;
            let tcent: f64 =
                et_sep[idi].iter().take(num_vertices).sum::<f64>() / num_vertices as f64;

            // Global coordinates of center
            let crd_poip = local_to_global(element_coords, element_type, scent, tcent);

            // Distance ratio
            let dist = distance(&crd_poip, source_point);
            let ratdis = dist / arels.sqrt();

            if ratdis < TOL_F {
                // Need to subdivide
                ndie += 1;
                if ndie > 15 {
                    // Too many subdivisions, stop
                    break;
                }

                nsfl = ndie * NSE;
                let nsf0 = nsfl - NSE;

                // Generate midside points
                let mut xisp = vec![0.0; num_vertices * 2];
                let mut etsp = vec![0.0; num_vertices * 2];

                for j in 0..num_vertices {
                    let j1 = (j + 1) % num_vertices;
                    xisp[j] = xi_sep[idi][j];
                    xisp[j + num_vertices] = (xi_sep[idi][j] + xi_sep[idi][j1]) / 2.0;
                    etsp[j] = et_sep[idi][j];
                    etsp[j + num_vertices] = (et_sep[idi][j] + et_sep[idi][j1]) / 2.0;
                }

                // Generate 4 (or 3 for triangle) subelements
                for j in 0..num_vertices {
                    let nsu = nsf0 + j;
                    let j1 = j + num_vertices;
                    // For the second midpoint index:
                    // - For vertex j, we need the midpoint between vertex j and vertex (j+1)%num_vertices
                    // - This is stored at index j + num_vertices in the midpoint arrays
                    // - The "previous" midpoint (between vertex (j-1)%num_vertices and j) is at
                    //   index ((j + num_vertices - 1) % num_vertices) + num_vertices
                    let j2 = if j == 0 {
                        num_vertices + num_vertices - 1 // Last midpoint (between last vertex and vertex 0)
                    } else {
                        j - 1 + num_vertices // Previous midpoint
                    };

                    match element_type {
                        ElementType::Quad4 => {
                            xi_sfp[nsu] = vec![xisp[j], xisp[j1], scent, xisp[j2]];
                            et_sfp[nsu] = vec![etsp[j], etsp[j1], tcent, etsp[j2]];
                        }
                        ElementType::Tri3 => {
                            xi_sfp[nsu] = vec![xisp[j], xisp[j1], xisp[j2]];
                            et_sfp[nsu] = vec![etsp[j], etsp[j1], etsp[j2]];

                            // Central triangle for Tri3
                            if j == num_vertices - 1 {
                                let nsu_center = nsf0 + NSE - 1;
                                xi_sfp[nsu_center] = vec![
                                    xisp[num_vertices],
                                    xisp[num_vertices + 1],
                                    xisp[num_vertices + 2],
                                ];
                                et_sfp[nsu_center] = vec![
                                    etsp[num_vertices],
                                    etsp[num_vertices + 1],
                                    etsp[num_vertices + 2],
                                ];
                            }
                        }
                    }
                }
            } else {
                // Store subelement result
                let (xi_center, eta_center, scale, tri_verts) = match element_type {
                    ElementType::Quad4 => {
                        let xc = xi_sep[idi].iter().sum::<f64>() / 4.0;
                        let ec = et_sep[idi].iter().sum::<f64>() / 4.0;
                        (xc, ec, faclin, None)
                    }
                    ElementType::Tri3 => {
                        // Store the actual triangle vertices for proper affine mapping
                        let xc = (xi_sep[idi][0] + xi_sep[idi][1] + xi_sep[idi][2]) / 3.0;
                        let ec = (et_sep[idi][0] + et_sep[idi][1] + et_sep[idi][2]) / 3.0;
                        let vertices = [
                            (xi_sep[idi][0], et_sep[idi][0]),
                            (xi_sep[idi][1], et_sep[idi][1]),
                            (xi_sep[idi][2], et_sep[idi][2]),
                        ];
                        (xc, ec, faclin, Some(vertices))
                    }
                };

                // Compute Gauss order based on distance
                let disfac = 0.5 / ratdis;
                let gauss_order = compute_gauss_order(disfac, GAU_MIN, GAU_MAX, GAU_ACCU);

                result.push(Subelement {
                    xi_center,
                    eta_center,
                    factor: scale,
                    gauss_order,
                    tri_vertices: tri_verts,
                });

                if result.len() >= MAX_SUBELEMENTS {
                    return result;
                }
            }
        }

        if ndie == 0 {
            break;
        }
    }

    result
}

/// Compute optimal Gauss order for given distance factor
fn compute_gauss_order(disfac: f64, gau_min: usize, gau_max: usize, accuracy: f64) -> usize {
    for order in gau_min..=gau_max {
        let err_g = estimate_error_g(order, disfac);
        let err_h = estimate_error_h(order, disfac);
        let err_e = estimate_error_e(order, disfac);

        if err_g < accuracy && err_h < accuracy && err_e < accuracy {
            return order;
        }
    }
    gau_max
}

/// Error estimate for G kernel (1/r) integration
fn estimate_error_g(order: usize, disfac: f64) -> f64 {
    // Empirical error formula from NC_IntegrationConstants.h
    let n = order as f64;
    (disfac / (2.0 * n + 1.0)).powi(2 * order as i32 + 1)
}

/// Error estimate for H kernel (1/r²) integration
fn estimate_error_h(order: usize, disfac: f64) -> f64 {
    let n = order as f64;
    (disfac / (2.0 * n + 1.0)).powi(2 * order as i32 + 2)
}

/// Error estimate for E kernel (1/r³) integration
fn estimate_error_e(order: usize, disfac: f64) -> f64 {
    let n = order as f64;
    (disfac / (2.0 * n + 1.0)).powi(2 * order as i32 + 3)
}

/// Convert local coordinates to global coordinates
fn local_to_global(
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    s: f64,
    t: f64,
) -> [f64; 3] {
    let shape_fn = match element_type {
        ElementType::Tri3 => vec![1.0 - s - t, s, t],
        ElementType::Quad4 => {
            let s1 = 0.25 * (s + 1.0);
            let s2 = 0.25 * (s - 1.0);
            let t1 = t + 1.0;
            let t2 = t - 1.0;
            vec![s1 * t1, -s2 * t1, s2 * t2, -s1 * t2]
        }
    };

    let num_nodes = element_type.num_nodes();
    let mut result = [0.0_f64; 3];
    for i in 0..num_nodes {
        for j in 0..3 {
            result[j] += shape_fn[i] * element_coords[[i, j]];
        }
    }
    result
}

/// Euclidean distance between two points
fn distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Estimate characteristic element size (square root of area for 2D elements)
///
/// For accuracy in quasi-singular integration, we use sqrt(area) as it
/// provides a consistent measure regardless of element shape.
fn estimate_element_size(element_coords: &DenseMatrix<f64>, element_type: ElementType) -> f64 {
    let _num_nodes = element_type.num_nodes();

    // Compute element area using cross product
    match element_type {
        ElementType::Tri3 => {
            // Triangle area = 0.5 * |v1 × v2|
            let v1: [f64; 3] = [
                element_coords[[1, 0]] - element_coords[[0, 0]],
                element_coords[[1, 1]] - element_coords[[0, 1]],
                element_coords[[1, 2]] - element_coords[[0, 2]],
            ];
            let v2: [f64; 3] = [
                element_coords[[2, 0]] - element_coords[[0, 0]],
                element_coords[[2, 1]] - element_coords[[0, 1]],
                element_coords[[2, 2]] - element_coords[[0, 2]],
            ];
            let cross = [
                v1[1] * v2[2] - v1[2] * v2[1],
                v1[2] * v2[0] - v1[0] * v2[2],
                v1[0] * v2[1] - v1[1] * v2[0],
            ];
            let area =
                0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
            area.sqrt().max(1e-10)
        }
        ElementType::Quad4 => {
            // Quad area via two triangles (split along diagonal 0-2)
            let v1: [f64; 3] = [
                element_coords[[1, 0]] - element_coords[[0, 0]],
                element_coords[[1, 1]] - element_coords[[0, 1]],
                element_coords[[1, 2]] - element_coords[[0, 2]],
            ];
            let v2: [f64; 3] = [
                element_coords[[2, 0]] - element_coords[[0, 0]],
                element_coords[[2, 1]] - element_coords[[0, 1]],
                element_coords[[2, 2]] - element_coords[[0, 2]],
            ];
            let cross1 = [
                v1[1] * v2[2] - v1[2] * v2[1],
                v1[2] * v2[0] - v1[0] * v2[2],
                v1[0] * v2[1] - v1[1] * v2[0],
            ];

            let v3: [f64; 3] = [
                element_coords[[3, 0]] - element_coords[[0, 0]],
                element_coords[[3, 1]] - element_coords[[0, 1]],
                element_coords[[3, 2]] - element_coords[[0, 2]],
            ];
            let cross2 = [
                v2[1] * v3[2] - v2[2] * v3[1],
                v2[2] * v3[0] - v2[0] * v3[2],
                v2[0] * v3[1] - v2[1] * v3[0],
            ];

            let area1 = 0.5
                * (cross1[0] * cross1[0] + cross1[1] * cross1[1] + cross1[2] * cross1[2]).sqrt();
            let area2 = 0.5
                * (cross2[0] * cross2[0] + cross2[1] * cross2[1] + cross2[2] * cross2[2]).sqrt();
            (area1 + area2).sqrt().max(1e-10)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighted_hypersingular_matches_integrable_quadratic_density() {
        for (a, b, angle) in [(1.0_f64, 1.0_f64, 0.0_f64), (2.0, 0.5, 0.7)] {
            let source = [0.2, -0.5, 0.3];
            let normal = [0.0, -angle.sin(), angle.cos()];
            let coordinates = DenseMatrix::from_row_major_vec(
                4,
                3,
                [(a, b), (-a, b), (-a, -b), (a, -b)]
                    .into_iter()
                    .flat_map(|(x, y)| {
                        [
                            source[0] + x,
                            source[1] + y * angle.cos(),
                            source[2] + y * angle.sin(),
                        ]
                    })
                    .collect(),
            );
            let mut physics = PhysicsParams::new(1.0, 343.0, 1.21, false);
            physics.wave_number = 0.0;
            let result = singular_integration_linear_with_params(
                &source,
                &normal,
                &coordinates,
                ElementType::Quad4,
                &physics,
                9,
                &QuadratureParams {
                    edge_gauss_order: 16,
                    subelement_gauss_order: 16,
                    edge_sections: 4,
                    subtriangles_per_section: 2,
                },
                None,
            );
            // Density x^2 removes the hypersingularity. Direct integration
            // of x^2/(4*pi*r^3) on [-a,a]x[-b,b] gives b*asinh(a/b)/pi.
            let density = [a * a, a * a, a * a, a * a, 0.0, a * a, 0.0, a * a, 0.0];
            let actual = result
                .basis_results
                .iter()
                .zip(density)
                .map(|(basis, value)| basis.d2g_dnxdny_integral * value)
                .sum::<Complex64>();
            let expected = b * (a / b).asinh() / PI;
            assert!(
                (actual - expected).norm() < 1e-8,
                "weighted W={actual}, exact={expected}, half-widths=({a},{b}), angle={angle}"
            );
        }
    }

    fn make_test_triangle() -> DenseMatrix<f64> {
        DenseMatrix::from_row_major_vec(3, 3, vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    }

    #[test]
    fn test_local_to_global_triangle() {
        let coords = make_test_triangle();

        // At center (1/3, 1/3)
        let center = local_to_global(&coords, ElementType::Tri3, 1.0 / 3.0, 1.0 / 3.0);
        assert!((center[0] - 1.0 / 3.0).abs() < 1e-10);
        assert!((center[1] - 1.0 / 3.0).abs() < 1e-10);
        assert!(center[2].abs() < 1e-10);
    }

    #[test]
    fn test_generate_subelements_far_point() {
        let coords = make_test_triangle();
        let source = [10.0, 10.0, 0.0]; // Far from element

        let subelements = generate_subelements(&source, &coords, ElementType::Tri3, 0.5);

        // Far point should not require many subdivisions
        assert!(subelements.len() <= 4);
    }

    #[test]
    fn test_singular_integration_basic() {
        let coords = make_test_triangle();
        let source = [1.0 / 3.0, 1.0 / 3.0, 0.0]; // At center
        let normal = [0.0, 0.0, 1.0];

        // Use low frequency so kr << 1 and cos(kr) ≈ 1 (positive real part)
        let physics = PhysicsParams::new(10.0, 343.0, 1.21, false);

        let result = singular_integration(
            &source,
            &normal,
            &coords,
            ElementType::Tri3,
            &physics,
            None,
            0,
            false,
        );

        // G integral should be finite and non-zero
        assert!(result.g_integral.norm().is_finite());
        assert!(result.g_integral.norm() > 0.0, "G integral was zero");

        // At low frequency (small kr), the real part should be positive
        assert!(
            result.g_integral.re > 0.0,
            "G integral real part was: {} (expected positive at low freq)",
            result.g_integral.re
        );

        // H integral should be zero (source normal perpendicular to element)
        assert!(
            result.dg_dn_integral.norm() < 1e-10,
            "H integral should be zero, was: {:?}",
            result.dg_dn_integral
        );
    }

    #[test]
    fn test_singular_quadrature_error_estimate_includes_hypersingular_channel() {
        let coords = make_test_triangle();
        let source = [1.0 / 3.0, 1.0 / 3.0, 0.0];
        let normal = [0.0, 0.0, 1.0];
        let physics = PhysicsParams::new(10.0, 343.0, 1.21, false);
        let coarse = QuadratureParams::default();
        let fine = QuadratureParams {
            edge_gauss_order: coarse.edge_gauss_order + 1,
            subelement_gauss_order: coarse.subelement_gauss_order + 1,
            edge_sections: coarse.edge_sections + 1,
            subtriangles_per_section: coarse.subtriangles_per_section,
        };

        let estimate = singular_integration_error_estimate(
            &source,
            &normal,
            &coords,
            ElementType::Tri3,
            &physics,
            None,
            0,
            false,
            &coarse,
            &fine,
            None,
        )
        .expect("finer singular quadrature should be accepted");

        assert!(estimate.error.max_absolute.is_finite());
        assert!(estimate.error.d2g_dnxdny_absolute.is_finite());
        assert!(
            singular_integration_error_estimate(
                &source,
                &normal,
                &coords,
                ElementType::Tri3,
                &physics,
                None,
                0,
                false,
                &fine,
                &coarse,
                None,
            )
            .is_err()
        );

        let accepted = singular_integration_checked(
            &source,
            &normal,
            &coords,
            ElementType::Tri3,
            &physics,
            None,
            0,
            false,
            &coarse,
            &fine,
            None,
            1.0,
            1.0,
        )
        .expect("broad mixed budget should accept the singular estimate");
        assert!(accepted.d2g_dnxdny_integral.norm().is_finite());

        let invalid_tolerance = singular_integration_checked(
            &source,
            &normal,
            &coords,
            ElementType::Tri3,
            &physics,
            None,
            0,
            false,
            &coarse,
            &fine,
            None,
            f64::NAN,
            1.0,
        )
        .expect_err("non-finite tolerance must fail before accepting a result");
        assert!(matches!(
            invalid_tolerance,
            QuadratureConvergenceError::InvalidTolerance { .. }
        ));

        let invalid_parameters = QuadratureParams {
            edge_gauss_order: 0,
            ..coarse
        };
        assert!(
            singular_integration_error_estimate(
                &source,
                &normal,
                &coords,
                ElementType::Tri3,
                &physics,
                None,
                0,
                false,
                &invalid_parameters,
                &fine,
                None,
            )
            .is_err()
        );
    }

    #[test]
    fn singular_wavenumber_derivative_matches_central_difference() {
        let coords = make_test_triangle();
        let source = [1.0 / 3.0, 1.0 / 3.0, 0.0];
        let normal = [0.0, 0.0, 1.0];
        let speed_of_sound = 343.0;
        let density = 1.21;
        let base = PhysicsParams::new(120.0, speed_of_sound, density, false);
        let step = 1e-4;
        let plus_physics = PhysicsParams::new(
            (base.wave_number + step) * speed_of_sound / std::f64::consts::TAU,
            speed_of_sound,
            density,
            false,
        );
        let minus_physics = PhysicsParams::new(
            (base.wave_number - step) * speed_of_sound / std::f64::consts::TAU,
            speed_of_sound,
            density,
            false,
        );
        let quadrature = QuadratureParams {
            edge_gauss_order: 5,
            subelement_gauss_order: 6,
            edge_sections: 8,
            subtriangles_per_section: 3,
        };

        let derivative = singular_integration_wavenumber_derivative_with_params(
            &source,
            &normal,
            &coords,
            ElementType::Tri3,
            &base,
            None,
            0,
            false,
            &quadrature,
            None,
        );
        let plus = singular_integration_with_params(
            &source,
            &normal,
            &coords,
            ElementType::Tri3,
            &plus_physics,
            None,
            0,
            false,
            &quadrature,
            None,
        );
        let minus = singular_integration_with_params(
            &source,
            &normal,
            &coords,
            ElementType::Tri3,
            &minus_physics,
            None,
            0,
            false,
            &quadrature,
            None,
        );
        let central = |plus: Complex64, minus: Complex64| (plus - minus) / (2.0 * step);
        let assert_close = |analytic: Complex64, finite_difference: Complex64| {
            let scale = 1.0 + analytic.norm().max(finite_difference.norm());
            assert!(
                (analytic - finite_difference).norm() < 2e-5 * scale,
                "analytic={analytic} finite_difference={finite_difference}"
            );
        };
        assert_close(
            derivative.g_integral,
            central(plus.g_integral, minus.g_integral),
        );
        assert_close(
            derivative.dg_dn_integral,
            central(plus.dg_dn_integral, minus.dg_dn_integral),
        );
        assert_close(
            derivative.dg_dnx_integral,
            central(plus.dg_dnx_integral, minus.dg_dnx_integral),
        );
        assert_close(
            derivative.d2g_dnxdny_integral,
            central(plus.d2g_dnxdny_integral, minus.d2g_dnxdny_integral),
        );
    }

    #[test]
    fn test_compute_shape_and_jacobian() {
        let coords = make_test_triangle();

        let (shape, _, _, jac, normal, _pos) =
            compute_shape_and_jacobian(&coords, ElementType::Tri3, 0.5, 0.25);

        // Shape functions should sum to 1
        let sum: f64 = shape.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Jacobian should be 2 * area = 1 for this triangle
        assert!((jac - 1.0).abs() < 1e-10);

        // Normal should be (0, 0, 1)
        assert!(normal[2].abs() > 0.99);
    }
}
