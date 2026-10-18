//! Quadrature rule selection based on element type and polynomial degree

use super::gauss::{
    QuadraturePoint, UnsupportedQuadratureOrder, gauss_hexahedron, gauss_quadrilateral,
    gauss_tetrahedron, gauss_triangle, gauss_triangle_checked,
};
use crate::mesh::ElementType;

/// Quadrature rule for a specific element type
#[derive(Debug, Clone)]
pub struct QuadratureRule {
    /// Element type this rule is for
    pub element_type: ElementType,
    /// Quadrature order (polynomial degree exactly integrated)
    pub order: usize,
    /// Quadrature points and weights
    pub points: Vec<QuadraturePoint>,
}

impl QuadratureRule {
    /// Create a quadrature rule for the given element type and order
    pub fn new(element_type: ElementType, order: usize) -> Self {
        let points = match element_type {
            ElementType::Triangle => gauss_triangle(order),
            ElementType::Quadrilateral => gauss_quadrilateral(order),
            ElementType::Tetrahedron => gauss_tetrahedron(order),
            ElementType::Hexahedron => gauss_hexahedron(order),
        };

        Self {
            element_type,
            order,
            points,
        }
    }

    /// Creates a rule, rejecting orders that cannot be honoured exactly.
    ///
    /// # Errors
    /// Returns [`UnsupportedQuadratureOrder`] for order zero, or when the
    /// element rule's guaranteed exactness is below the requested degree.
    pub fn try_new(
        element_type: ElementType,
        order: usize,
    ) -> Result<Self, UnsupportedQuadratureOrder> {
        if order == 0 {
            return Err(UnsupportedQuadratureOrder { order });
        }
        if element_type == ElementType::Triangle {
            let checked = gauss_triangle_checked(order)?;
            if checked.exactness_degree < order {
                return Err(UnsupportedQuadratureOrder { order });
            }
            return Ok(Self {
                element_type,
                order,
                points: checked.points,
            });
        }
        Ok(Self::new(element_type, order))
    }

    /// Number of quadrature points
    pub fn num_points(&self) -> usize {
        self.points.len()
    }

    /// Iterator over quadrature points
    pub fn iter(&self) -> impl Iterator<Item = &QuadraturePoint> {
        self.points.iter()
    }
}

/// Determine minimum quadrature order needed for given polynomial degree
///
/// For FEM with polynomial degree p:
/// - Stiffness matrix: integrate grad(phi) * grad(phi) ~ degree 2(p-1)
/// - Mass matrix: integrate phi * phi ~ degree 2p
///
/// The returned value is the polynomial degree that the element rule must
/// integrate. Tensor-product rules may use fewer one-dimensional points.
pub fn required_order_for_stiffness(polynomial_degree: usize) -> usize {
    (2 * polynomial_degree.saturating_sub(1)).max(1)
}

pub fn required_order_for_mass(polynomial_degree: usize) -> usize {
    2 * polynomial_degree.max(1)
}

/// Returns the degree needed for a value-times-gradient drift integrand.
///
/// The test value (degree `p`) times a trial gradient (degree `p - 1`)
/// gives degree `2p - 1` for a constant coefficient on an affine element.
pub fn required_order_for_drift(polynomial_degree: usize) -> usize {
    (2 * polynomial_degree.max(1)).saturating_sub(1)
}

/// Create quadrature rule for stiffness matrix integration
pub fn for_stiffness(element_type: ElementType, polynomial_degree: usize) -> QuadratureRule {
    QuadratureRule::new(
        element_type,
        required_order_for_stiffness(polynomial_degree),
    )
}

/// Create quadrature rule for mass matrix integration
pub fn for_mass(element_type: ElementType, polynomial_degree: usize) -> QuadratureRule {
    QuadratureRule::new(element_type, required_order_for_mass(polynomial_degree))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quadrature_rule_creation() {
        let rule = QuadratureRule::new(ElementType::Triangle, 2);
        assert_eq!(rule.element_type, ElementType::Triangle);
        assert_eq!(rule.order, 2);
        assert!(!rule.points.is_empty());
    }

    #[test]
    fn test_required_orders() {
        // P1 elements
        assert_eq!(required_order_for_stiffness(1), 1);
        assert_eq!(required_order_for_mass(1), 2);

        // P2 elements
        assert_eq!(required_order_for_stiffness(2), 2);
        assert_eq!(required_order_for_mass(2), 4);

        // P3 elements
        assert_eq!(required_order_for_stiffness(3), 4);
        assert_eq!(required_order_for_mass(3), 6);
    }

    #[test]
    fn drift_order_is_value_times_gradient_degree() {
        assert_eq!(required_order_for_drift(1), 1);
        assert_eq!(required_order_for_drift(2), 3);
        assert_eq!(required_order_for_drift(3), 5);
    }

    #[test]
    fn try_new_rejects_zero_and_honours_triangle_orders() {
        assert!(QuadratureRule::try_new(ElementType::Triangle, 0).is_err());
        for order in 1..=8 {
            let rule = QuadratureRule::try_new(ElementType::Triangle, order).unwrap();
            assert_eq!(rule.order, order);
            assert!(!rule.points.is_empty());
        }
    }

    #[test]
    fn test_for_stiffness_and_mass() {
        let stiffness_rule = for_stiffness(ElementType::Triangle, 2);
        let mass_rule = for_mass(ElementType::Triangle, 2);

        // Mass matrix needs higher order
        assert!(mass_rule.num_points() >= stiffness_rule.num_points());
    }
}
