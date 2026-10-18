//! Gauss-Legendre quadrature points and weights
//!
//! Pre-computed points and weights for 1D, 2D (triangle, quad), and 3D (tet, hex) integration.

/// A single quadrature point with weight
#[derive(Debug, Clone, Copy)]
pub struct QuadraturePoint {
    /// Reference coordinates (xi, eta, zeta)
    pub coords: [f64; 3],
    /// Integration weight
    pub weight: f64,
}

impl QuadraturePoint {
    pub fn new_1d(xi: f64, weight: f64) -> Self {
        Self {
            coords: [xi, 0.0, 0.0],
            weight,
        }
    }

    pub fn new_2d(xi: f64, eta: f64, weight: f64) -> Self {
        Self {
            coords: [xi, eta, 0.0],
            weight,
        }
    }

    pub fn new_3d(xi: f64, eta: f64, zeta: f64, weight: f64) -> Self {
        Self {
            coords: [xi, eta, zeta],
            weight,
        }
    }

    #[inline]
    pub fn xi(&self) -> f64 {
        self.coords[0]
    }

    #[inline]
    pub fn eta(&self) -> f64 {
        self.coords[1]
    }

    #[inline]
    pub fn zeta(&self) -> f64 {
        self.coords[2]
    }
}

/// 1D Gauss-Legendre quadrature on [-1, 1]
pub fn gauss_legendre_1d(order: usize) -> Vec<QuadraturePoint> {
    match order {
        1 => vec![QuadraturePoint::new_1d(0.0, 2.0)],
        2 => {
            let x = 1.0 / 3.0_f64.sqrt();
            vec![
                QuadraturePoint::new_1d(-x, 1.0),
                QuadraturePoint::new_1d(x, 1.0),
            ]
        }
        3 => {
            let x = (3.0 / 5.0_f64).sqrt();
            vec![
                QuadraturePoint::new_1d(-x, 5.0 / 9.0),
                QuadraturePoint::new_1d(0.0, 8.0 / 9.0),
                QuadraturePoint::new_1d(x, 5.0 / 9.0),
            ]
        }
        4 => {
            let a = (3.0 / 7.0 - 2.0 / 7.0 * (6.0 / 5.0_f64).sqrt()).sqrt();
            let b = (3.0 / 7.0 + 2.0 / 7.0 * (6.0 / 5.0_f64).sqrt()).sqrt();
            let wa = (18.0 + 30.0_f64.sqrt()) / 36.0;
            let wb = (18.0 - 30.0_f64.sqrt()) / 36.0;
            vec![
                QuadraturePoint::new_1d(-b, wb),
                QuadraturePoint::new_1d(-a, wa),
                QuadraturePoint::new_1d(a, wa),
                QuadraturePoint::new_1d(b, wb),
            ]
        }
        5 => {
            let a = (5.0 - 2.0 * (10.0 / 7.0_f64).sqrt()).sqrt() / 3.0;
            let b = (5.0 + 2.0 * (10.0 / 7.0_f64).sqrt()).sqrt() / 3.0;
            let wa = (322.0 + 13.0 * 70.0_f64.sqrt()) / 900.0;
            let wb = (322.0 - 13.0 * 70.0_f64.sqrt()) / 900.0;
            vec![
                QuadraturePoint::new_1d(-b, wb),
                QuadraturePoint::new_1d(-a, wa),
                QuadraturePoint::new_1d(0.0, 128.0 / 225.0),
                QuadraturePoint::new_1d(a, wa),
                QuadraturePoint::new_1d(b, wb),
            ]
        }
        _ => {
            // Generate higher-order rules instead of silently reusing the
            // five-point rule.  Newton's method is stable for the modest
            // orders used by the FEM assembly paths.
            let n = order.max(1);
            let mut points = Vec::with_capacity(n);
            let half = n.div_ceil(2);
            for i in 0..half {
                let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
                for _ in 0..32 {
                    let mut p0 = 1.0;
                    let mut p1 = z;
                    for degree in 2..=n {
                        let degree = degree as f64;
                        let p2 = ((2.0 * degree - 1.0) * z * p1 - (degree - 1.0) * p0) / degree;
                        p0 = p1;
                        p1 = p2;
                    }
                    let derivative = n as f64 * (z * p1 - p0) / (z * z - 1.0);
                    let next = z - p1 / derivative;
                    if (next - z).abs() <= 4.0 * f64::EPSILON * next.abs().max(1.0) {
                        z = next;
                        break;
                    }
                    z = next;
                }

                let mut p0 = 1.0;
                let mut p1 = z;
                for degree in 2..=n {
                    let degree = degree as f64;
                    let p2 = ((2.0 * degree - 1.0) * z * p1 - (degree - 1.0) * p0) / degree;
                    p0 = p1;
                    p1 = p2;
                }
                let derivative = n as f64 * (z * p1 - p0) / (z * z - 1.0);
                let weight = 2.0 / ((1.0 - z * z) * derivative * derivative);
                points.push(QuadraturePoint::new_1d(-z, weight));
                if points.len() < n {
                    points.push(QuadraturePoint::new_1d(z, weight));
                }
            }
            points.sort_by(|left, right| left.xi().total_cmp(&right.xi()));
            points
        }
    }
}

/// Error returned when a quadrature order cannot be honoured exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnsupportedQuadratureOrder {
    /// Requested polynomial exactness degree.
    pub order: usize,
}

impl std::fmt::Display for UnsupportedQuadratureOrder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "quadrature order {} is not supported (order must be >= 1)",
            self.order
        )
    }
}

impl std::error::Error for UnsupportedQuadratureOrder {}

/// A quadrature rule together with its guaranteed exactness degree.
#[derive(Debug, Clone)]
pub struct CheckedQuadrature {
    /// Quadrature points and weights.
    pub points: Vec<QuadraturePoint>,
    /// Polynomial degree the rule integrates exactly.
    pub exactness_degree: usize,
}

/// Returns the triangle rule for `order`, reporting its exactness degree.
///
/// # Errors
/// Returns [`UnsupportedQuadratureOrder`] when `order` is zero.
pub fn gauss_triangle_checked(
    order: usize,
) -> Result<CheckedQuadrature, UnsupportedQuadratureOrder> {
    if order == 0 {
        return Err(UnsupportedQuadratureOrder { order });
    }
    let exactness_degree = match order {
        1 => 1,
        2 => 2,
        3 | 4 => 4,
        5 => 5,
        6 => 6,
        _ => order,
    };
    Ok(CheckedQuadrature {
        points: gauss_triangle(order),
        exactness_degree,
    })
}

/// Gauss quadrature for triangles (area coordinates)
/// Reference triangle: (0,0), (1,0), (0,1)
///
/// The returned rule integrates polynomials of degree at least `order`
/// exactly for every positive `order`; use [`gauss_triangle_checked`] to
/// read the guaranteed exactness degree or reject order zero explicitly.
pub fn gauss_triangle(order: usize) -> Vec<QuadraturePoint> {
    match order {
        1 => {
            // 1-point rule (degree 1)
            vec![QuadraturePoint::new_2d(1.0 / 3.0, 1.0 / 3.0, 0.5)]
        }
        2 => {
            // 3-point rule (degree 2)
            let a = 1.0 / 6.0;
            let b = 2.0 / 3.0;
            let w = 1.0 / 6.0;
            vec![
                QuadraturePoint::new_2d(a, a, w),
                QuadraturePoint::new_2d(b, a, w),
                QuadraturePoint::new_2d(a, b, w),
            ]
        }
        3 | 4 => {
            // 6-point Dunavant rule (degree 4)
            let a1 = 0.445_948_490_915_965;
            let b1 = 0.108_103_018_168_070;
            let w1 = 0.111_690_794_839_006;
            let a2 = 0.091_576_213_509_771;
            let b2 = 0.816_847_572_980_459;
            let w2 = 0.054_975_871_827_661;
            vec![
                QuadraturePoint::new_2d(a1, a1, w1),
                QuadraturePoint::new_2d(a1, b1, w1),
                QuadraturePoint::new_2d(b1, a1, w1),
                QuadraturePoint::new_2d(a2, a2, w2),
                QuadraturePoint::new_2d(a2, b2, w2),
                QuadraturePoint::new_2d(b2, a2, w2),
            ]
        }
        5 => {
            // 7-point rule (degree 5)
            let a1 = 1.0 / 3.0;
            let w1 = 9.0 / 80.0;

            let a2 = (6.0 - 15.0_f64.sqrt()) / 21.0;
            let b2 = (9.0 + 2.0 * 15.0_f64.sqrt()) / 21.0;
            let w2 = (155.0 - 15.0_f64.sqrt()) / 2400.0;

            let a3 = (6.0 + 15.0_f64.sqrt()) / 21.0;
            let b3 = (9.0 - 2.0 * 15.0_f64.sqrt()) / 21.0;
            let w3 = (155.0 + 15.0_f64.sqrt()) / 2400.0;

            vec![
                QuadraturePoint::new_2d(a1, a1, w1),
                QuadraturePoint::new_2d(a2, a2, w2),
                QuadraturePoint::new_2d(b2, a2, w2),
                QuadraturePoint::new_2d(a2, b2, w2),
                QuadraturePoint::new_2d(a3, a3, w3),
                QuadraturePoint::new_2d(b3, a3, w3),
                QuadraturePoint::new_2d(a3, b3, w3),
            ]
        }
        6 => {
            // 12-point rule (degree 6)
            let a1 = 0.063089014491502;
            let b1 = 0.873821971016996;
            let w1 = 0.025422453185103;

            let a2 = 0.249286745170910;
            let b2 = 0.501426509658179;
            let w2 = 0.058393137863189;

            let a3 = 0.310352451033785;
            let b3 = 0.053145049844816;
            let c3 = 0.636502499121399;
            let w3 = 0.041425537809187;

            vec![
                QuadraturePoint::new_2d(a1, a1, w1),
                QuadraturePoint::new_2d(b1, a1, w1),
                QuadraturePoint::new_2d(a1, b1, w1),
                QuadraturePoint::new_2d(a2, a2, w2),
                QuadraturePoint::new_2d(b2, a2, w2),
                QuadraturePoint::new_2d(a2, b2, w2),
                QuadraturePoint::new_2d(a3, b3, w3),
                QuadraturePoint::new_2d(b3, a3, w3),
                QuadraturePoint::new_2d(a3, c3, w3),
                QuadraturePoint::new_2d(c3, a3, w3),
                QuadraturePoint::new_2d(b3, c3, w3),
                QuadraturePoint::new_2d(c3, b3, w3),
            ]
        }
        _ => {
            // Duffy-mapped tensor-product rule for orders above six:
            // x = u, y = (1-u)*v maps [0,1]^2 onto the reference triangle
            // with Jacobian (1-u), so no silent exactness ceiling remains.
            let one_d = gauss_legendre_1d(order.max(2));
            let unit = |point: &QuadraturePoint| ((point.xi() + 1.0) * 0.5, point.weight * 0.5);
            let mut result = Vec::with_capacity(one_d.len() * one_d.len());
            for u_point in &one_d {
                let (u, wu) = unit(u_point);
                for v_point in &one_d {
                    let (v, wv) = unit(v_point);
                    result.push(QuadraturePoint::new_2d(
                        u,
                        (1.0 - u) * v,
                        wu * wv * (1.0 - u),
                    ));
                }
            }
            result
        }
    }
}

/// Gauss quadrature for quadrilaterals
/// Reference quad: [-1,1] x [-1,1]
pub fn gauss_quadrilateral(order: usize) -> Vec<QuadraturePoint> {
    let pts_1d = gauss_legendre_1d(order);
    let mut result = Vec::with_capacity(pts_1d.len() * pts_1d.len());

    for pi in &pts_1d {
        for pj in &pts_1d {
            result.push(QuadraturePoint::new_2d(
                pi.xi(),
                pj.xi(),
                pi.weight * pj.weight,
            ));
        }
    }

    result
}

/// Gauss quadrature for tetrahedra
/// Reference tet: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
pub fn gauss_tetrahedron(order: usize) -> Vec<QuadraturePoint> {
    match order {
        1 => {
            // 1-point rule
            vec![QuadraturePoint::new_3d(0.25, 0.25, 0.25, 1.0 / 6.0)]
        }
        2 => {
            // 4-point rule (degree 2)
            let a = (5.0 - 5.0_f64.sqrt()) / 20.0;
            let b = (5.0 + 3.0 * 5.0_f64.sqrt()) / 20.0;
            let w = 1.0 / 24.0;
            vec![
                QuadraturePoint::new_3d(a, a, a, w),
                QuadraturePoint::new_3d(b, a, a, w),
                QuadraturePoint::new_3d(a, b, a, w),
                QuadraturePoint::new_3d(a, a, b, w),
            ]
        }
        3 => {
            // 5-point rule (degree 3)
            let w1 = -4.0 / 30.0;
            let w2 = 9.0 / 120.0;
            vec![
                QuadraturePoint::new_3d(0.25, 0.25, 0.25, w1),
                QuadraturePoint::new_3d(1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, w2),
                QuadraturePoint::new_3d(0.5, 1.0 / 6.0, 1.0 / 6.0, w2),
                QuadraturePoint::new_3d(1.0 / 6.0, 0.5, 1.0 / 6.0, w2),
                QuadraturePoint::new_3d(1.0 / 6.0, 1.0 / 6.0, 0.5, w2),
            ]
        }
        4 => {
            // Keast 11-point rule (degree 4). Weights are scaled to the
            // reference tetrahedron volume 1/6.
            let centroid_weight = -74.0 / 5625.0;
            let vertex_weight = 343.0 / 45000.0;
            let edge_weight = 28.0 / 1125.0;
            let a = 11.0 / 14.0;
            let b = 1.0 / 14.0;
            let root = (5.0_f64 / 14.0).sqrt();
            let c = (1.0 + root) / 4.0;
            let d = (1.0 - root) / 4.0;
            vec![
                QuadraturePoint::new_3d(0.25, 0.25, 0.25, centroid_weight),
                QuadraturePoint::new_3d(b, b, b, vertex_weight),
                QuadraturePoint::new_3d(a, b, b, vertex_weight),
                QuadraturePoint::new_3d(b, a, b, vertex_weight),
                QuadraturePoint::new_3d(b, b, a, vertex_weight),
                QuadraturePoint::new_3d(c, c, d, edge_weight),
                QuadraturePoint::new_3d(c, d, c, edge_weight),
                QuadraturePoint::new_3d(d, c, c, edge_weight),
                QuadraturePoint::new_3d(c, d, d, edge_weight),
                QuadraturePoint::new_3d(d, c, d, edge_weight),
                QuadraturePoint::new_3d(d, d, c, edge_weight),
            ]
        }
        _ => {
            // Duffy-mapped tensor-product Gauss-Legendre rule.  Mapping
            // [0,1]^3 -> {x,y,z >= 0, x+y+z <= 1} gives Jacobian
            // (1-u)^2(1-v); the three one-dimensional rules therefore
            // provide arbitrary polynomial order without a hidden cap.
            let one_d = gauss_legendre_1d(order.max(3));
            let unit = |point: &QuadraturePoint| ((point.xi() + 1.0) * 0.5, point.weight * 0.5);
            let mut result = Vec::with_capacity(one_d.len().pow(3));
            for u_point in &one_d {
                let (u, wu) = unit(u_point);
                for v_point in &one_d {
                    let (v, wv) = unit(v_point);
                    for w_point in &one_d {
                        let (w, ww) = unit(w_point);
                        let one_minus_u = 1.0 - u;
                        let one_minus_v = 1.0 - v;
                        result.push(QuadraturePoint::new_3d(
                            u,
                            one_minus_u * v,
                            one_minus_u * one_minus_v * w,
                            wu * wv * ww * one_minus_u * one_minus_u * one_minus_v,
                        ));
                    }
                }
            }
            result
        }
    }
}

/// Gauss quadrature for hexahedra
/// Reference hex: [-1,1] x [-1,1] x [-1,1]
pub fn gauss_hexahedron(order: usize) -> Vec<QuadraturePoint> {
    let pts_1d = gauss_legendre_1d(order);
    let mut result = Vec::with_capacity(pts_1d.len().pow(3));

    for pi in &pts_1d {
        for pj in &pts_1d {
            for pk in &pts_1d {
                result.push(QuadraturePoint::new_3d(
                    pi.xi(),
                    pj.xi(),
                    pk.xi(),
                    pi.weight * pj.weight * pk.weight,
                ));
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gauss_1d_integrates_polynomials() {
        // 2-point rule should exactly integrate up to degree 3
        let pts = gauss_legendre_1d(2);

        // Integrate x^2 from -1 to 1 = 2/3
        let integral: f64 = pts.iter().map(|p| p.xi().powi(2) * p.weight).sum();
        assert!((integral - 2.0 / 3.0).abs() < 1e-14);

        // Integrate x^3 from -1 to 1 = 0
        let integral: f64 = pts.iter().map(|p| p.xi().powi(3) * p.weight).sum();
        assert!(integral.abs() < 1e-14);
    }

    #[test]
    fn test_triangle_centroid_rule() {
        let pts = gauss_triangle(1);
        assert_eq!(pts.len(), 1);
        assert!((pts[0].xi() - 1.0 / 3.0).abs() < 1e-14);
        assert!((pts[0].eta() - 1.0 / 3.0).abs() < 1e-14);
        assert!((pts[0].weight - 0.5).abs() < 1e-14); // Area of reference triangle
    }

    #[test]
    fn test_quad_weights_sum() {
        for order in 1..=5 {
            let pts = gauss_quadrilateral(order);
            let sum: f64 = pts.iter().map(|p| p.weight).sum();
            // Weights should sum to 4 (area of [-1,1]^2)
            assert!(
                (sum - 4.0).abs() < 1e-14,
                "Order {} failed: sum = {}",
                order,
                sum
            );
        }
    }

    #[test]
    fn test_tet_weights_sum() {
        let pts = gauss_tetrahedron(1);
        let sum: f64 = pts.iter().map(|p| p.weight).sum();
        // Weights should sum to 1/6 (volume of reference tet)
        assert!((sum - 1.0 / 6.0).abs() < 1e-14);
    }

    #[test]
    fn tetrahedron_degree_four_rule_integrates_quartics() {
        let pts = gauss_tetrahedron(4);
        let volume: f64 = pts.iter().map(|p| p.weight).sum();
        let x4: f64 = pts.iter().map(|p| p.xi().powi(4) * p.weight).sum();
        let x2y2: f64 = pts
            .iter()
            .map(|p| p.xi().powi(2) * p.eta().powi(2) * p.weight)
            .sum();
        assert!((volume - 1.0 / 6.0).abs() < 1e-14);
        assert!((x4 - 1.0 / 210.0).abs() < 1e-14);
        assert!((x2y2 - 1.0 / 1260.0).abs() < 1e-14);
    }

    #[test]
    fn tetrahedron_degree_five_rule_integrates_quintics() {
        let pts = gauss_tetrahedron(5);
        let factorial = |n: usize| (1..=n).product::<usize>() as f64;
        for a in 0..=5 {
            for b in 0..=5 - a {
                for c in 0..=5 - a - b {
                    let numerical: f64 = pts
                        .iter()
                        .map(|p| {
                            p.xi().powi(a as i32)
                                * p.eta().powi(b as i32)
                                * p.zeta().powi(c as i32)
                                * p.weight
                        })
                        .sum();
                    let exact =
                        factorial(a) * factorial(b) * factorial(c) / factorial(a + b + c + 3);
                    assert!(
                        (numerical - exact).abs() < 1e-14,
                        "monomial x^{a} y^{b} z^{c}: {numerical:e} != {exact:e}"
                    );
                }
            }
        }
        assert!(pts.iter().all(|point| {
            point.xi() >= 0.0
                && point.eta() >= 0.0
                && point.zeta() >= 0.0
                && point.xi() + point.eta() + point.zeta() <= 1.0
        }));
    }

    #[test]
    fn generated_gauss_legendre_rule_integrates_high_order_polynomials() {
        let points = gauss_legendre_1d(8);
        let integral = points
            .iter()
            .map(|point| point.xi().powi(14) * point.weight)
            .sum::<f64>();
        assert!((integral - 2.0 / 15.0).abs() < 1.0e-13);
        assert_eq!(points.len(), 8);
    }

    #[test]
    fn generated_duffy_tetrahedron_rule_integrates_high_order_monomials() {
        let points = gauss_tetrahedron(8);
        let integral = points
            .iter()
            .map(|point| {
                point.xi().powi(4) * point.eta().powi(2) * point.zeta().powi(2) * point.weight
            })
            .sum::<f64>();
        // Integral over the reference tetrahedron of x^a y^b z^c is
        // a! b! c! / (a+b+c+3)!.
        let expected = 96.0 / 39_916_800.0;
        assert!((integral - expected).abs() < 1.0e-13);
        let volume = points.iter().map(|point| point.weight).sum::<f64>();
        assert!((volume - 1.0 / 6.0).abs() < 1.0e-13);
    }

    #[test]
    fn triangle_rule_integrates_monomials_through_advertised_degree() {
        // Exact integral of x^a*y^b over the reference triangle is
        // a!*b!/(a+b+2)!.
        let factorial = |n: usize| (1..=n).product::<usize>() as f64;
        for order in 1..=8usize {
            let pts = gauss_triangle(order);
            for a in 0..=order {
                for b in 0..=order - a {
                    let numerical: f64 = pts
                        .iter()
                        .map(|p| p.xi().powi(a as i32) * p.eta().powi(b as i32) * p.weight)
                        .sum();
                    let exact = factorial(a) * factorial(b) / factorial(a + b + 2);
                    assert!(
                        (numerical - exact).abs() < 1e-12,
                        "order {order} monomial x^{a} y^{b}: {numerical:e} != {exact:e}"
                    );
                }
            }
        }
    }

    #[test]
    fn triangle_degree_three_request_integrates_cubic_counterexample() {
        // Audit counterexample: integral of 2*x^3 over the reference
        // triangle is exactly 0.1; the old degree-two rule gave ~0.10185.
        let pts = gauss_triangle(3);
        let numerical: f64 = pts.iter().map(|p| 2.0 * p.xi().powi(3) * p.weight).sum();
        assert!((numerical - 0.1).abs() < 1e-12, "got {numerical}");
    }

    #[test]
    fn checked_triangle_rule_reports_exactness_degree() {
        for order in 1..=8usize {
            let rule = gauss_triangle_checked(order).expect("supported order");
            assert!(rule.exactness_degree >= order);
            assert_eq!(rule.points.len(), gauss_triangle(order).len());
        }
        assert!(gauss_triangle_checked(0).is_err());
    }

    #[test]
    fn test_hex_weights_sum() {
        for order in 1..=3 {
            let pts = gauss_hexahedron(order);
            let sum: f64 = pts.iter().map(|p| p.weight).sum();
            // Weights should sum to 8 (volume of [-1,1]^3)
            assert!(
                (sum - 8.0).abs() < 1e-14,
                "Order {} failed: sum = {}",
                order,
                sum
            );
        }
    }
}
