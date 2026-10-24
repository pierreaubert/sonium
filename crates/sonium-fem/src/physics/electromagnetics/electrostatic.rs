//! Electrostatic transducer physics
//!
//! Models the electrostatic force, capacitance, and electromechanical
//! coupling for condenser microphones and electrostatic loudspeakers.
//!
//! # Parallel plate model
//!
//! A charged parallel-plate capacitor with gap d and voltage V exerts a
//! pressure `p_e = εV²/(2d²)` on the electrodes. Linearised around a
//! bias point (V₀, d₀) this gives:
//!
//! - **Bias pressure**: p₀ = εV₀²/(2d₀²)
//! - **Coupling factor**: η = εV₀/d₀² (Pa/V, sensitivity)
//! - **Electrostatic negative spring**: k_e = εV₀²/d₀³ (Pa/m, destabilising)
//!
//! # Pull-in instability
//!
//! When the bias voltage exceeds `V_pi = √(8 k_mech d₀³ / (27ε))`,
//! the electrostatic force overcomes the mechanical restoring force and
//! the gap collapses. This is the fundamental limit for condenser
//! microphone design.
//!
//! # References
//!
//! - Senturia (2001), "Microsystem Design", Springer.
//! - Beranek & Mellow (2012), "Acoustics: Sound Fields, Transducers and
//!   Vibration", Academic Press, Ch. 6 (electrostatic transducers).

use crate::mesh::Mesh;
use crate::spaces::{BasisTabulation, FunctionSpace};
use num_complex::Complex64;
use sonium_solvers::CsrMatrix;
use thiserror::Error;

#[cfg(feature = "cuda")]
#[path = "electrostatic_cuda.rs"]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::{CudaElectrostaticCentroidField, CudaElectrostaticObservables};

/// Vacuum permittivity ε₀ \[F/m\].
pub const EPSILON_0: f64 = 8.854_187_817e-12;

/// Errors reported by electrostatic solved-field postprocessing.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ElectrostaticFieldError {
    /// The supplied nodal solution does not match the function space.
    #[error("potential has length {actual}, expected {expected}")]
    PotentialLength { expected: usize, actual: usize },
    /// The energy matrix does not match the solved field.
    #[error("unit Laplacian is {rows}x{columns}, expected {expected}x{expected}")]
    MatrixDimension {
        rows: usize,
        columns: usize,
        expected: usize,
    },
    /// An element index was outside the solved mesh.
    #[error("element index {element} is outside a mesh with {count} elements")]
    ElementIndex { element: usize, count: usize },
    /// A capacitor voltage difference must be finite and nonzero.
    #[error("voltage difference must be finite and nonzero, got {0}")]
    InvalidVoltage(f64),
    /// Absolute permittivity must be finite and positive.
    #[error("permittivity must be finite and positive, got {0}")]
    InvalidPermittivity(f64),
    /// Every solved potential value must be finite.
    #[error("potential contains a non-finite value at degree of freedom {0}")]
    NonFinitePotential(usize),
}

/// Integral observables for a two-conductor electrostatic solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CapacitorObservables {
    /// Stored electrostatic field energy, `1/2 ε ∫ |∇V|² dΩ`.
    pub energy_j: f64,
    /// Capacitance inferred from `2 energy / ΔV²`.
    pub capacitance_f: f64,
    /// Charge on the conductor at the positive side of `voltage_difference`.
    pub charge_c: f64,
}

/// Public postprocessor for a solved scalar electrostatic potential field.
///
/// The supplied solution may come from any Sonium scalar [`FunctionSpace`].
/// Electric fields are recovered as `-∇V` from the basis gradients. Integral
/// energy/capacitance methods accept the unconstrained unit-coefficient
/// Laplacian assembled on the same mesh and space; absolute permittivity is
/// applied by this postprocessor exactly once.
pub struct ElectrostaticSolvedField<'a, S: FunctionSpace> {
    mesh: &'a Mesh,
    space: &'a S,
    potential: &'a [f64],
    permittivity: f64,
}

impl<'a, S: FunctionSpace> ElectrostaticSolvedField<'a, S> {
    /// Construct a checked solved-field view.
    pub fn new(
        mesh: &'a Mesh,
        space: &'a S,
        potential: &'a [f64],
        permittivity: f64,
    ) -> Result<Self, ElectrostaticFieldError> {
        let expected = space.num_global_dofs(mesh);
        if potential.len() != expected {
            return Err(ElectrostaticFieldError::PotentialLength {
                expected,
                actual: potential.len(),
            });
        }
        if !permittivity.is_finite() || permittivity <= 0.0 {
            return Err(ElectrostaticFieldError::InvalidPermittivity(permittivity));
        }
        if let Some((index, _)) = potential
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(ElectrostaticFieldError::NonFinitePotential(index));
        }
        Ok(Self {
            mesh,
            space,
            potential,
            permittivity,
        })
    }

    /// Borrow the solved nodal potential.
    pub fn potential(&self) -> &'a [f64] {
        self.potential
    }

    /// Recover `E = -∇V` at reference coordinates in one element.
    pub fn electric_field_at_reference(
        &self,
        element: usize,
        xi: f64,
        eta: f64,
        zeta: f64,
    ) -> Result<[f64; 3], ElectrostaticFieldError> {
        if element >= self.mesh.num_elements() {
            return Err(ElectrostaticFieldError::ElementIndex {
                element,
                count: self.mesh.num_elements(),
            });
        }
        let tabulation: BasisTabulation = self.space.tabulate(self.mesh, element, xi, eta, zeta);
        let dofs = self.space.local_to_global(self.mesh, element);
        let mut field = [0.0; 3];
        for (local, &dof) in dofs.iter().enumerate() {
            for (dimension, component) in field.iter_mut().enumerate().take(self.mesh.dimension) {
                *component -= self.potential[dof] * tabulation.grad(local, dimension);
            }
        }
        Ok(field)
    }

    /// Stored electrostatic energy from an unconstrained unit Laplacian.
    pub fn energy(&self, unit_laplacian: &CsrMatrix<f64>) -> Result<f64, ElectrostaticFieldError> {
        let expected = self.potential.len();
        if unit_laplacian.num_rows != expected || unit_laplacian.num_cols != expected {
            return Err(ElectrostaticFieldError::MatrixDimension {
                rows: unit_laplacian.num_rows,
                columns: unit_laplacian.num_cols,
                expected,
            });
        }
        let action = unit_laplacian.matvec(self.potential);
        Ok(0.5
            * self.permittivity
            * self
                .potential
                .iter()
                .zip(action)
                .map(|(potential, value)| potential * value)
                .sum::<f64>())
    }

    /// Energy, capacitance, and conductor charge for a two-potential solve.
    pub fn capacitor_observables(
        &self,
        unit_laplacian: &CsrMatrix<f64>,
        voltage_difference: f64,
    ) -> Result<CapacitorObservables, ElectrostaticFieldError> {
        if !voltage_difference.is_finite() || voltage_difference == 0.0 {
            return Err(ElectrostaticFieldError::InvalidVoltage(voltage_difference));
        }
        let energy_j = self.energy(unit_laplacian)?;
        let capacitance_f = 2.0 * energy_j / voltage_difference.powi(2);
        Ok(CapacitorObservables {
            energy_j,
            capacitance_f,
            charge_c: capacitance_f * voltage_difference,
        })
    }
}

// ---------------------------------------------------------------------------
// Parallel plate gap model
// ---------------------------------------------------------------------------

/// Parallel-plate electrostatic gap model.
///
/// Suitable for condenser microphones, electrostatic speakers, and MEMS
/// capacitive transducers. The gap is characterised by a bias voltage,
/// a rest gap distance, and the permittivity of the gap medium.
#[derive(Debug, Clone, Copy)]
pub struct ParallelPlateGap {
    /// Bias (polarisation) voltage V₀ \[V\].
    pub bias_voltage: f64,
    /// Rest gap distance d₀ \[m\] (no displacement).
    pub gap_distance: f64,
    /// Absolute permittivity of the gap medium ε = ε₀ε_r \[F/m\].
    pub permittivity: f64,
}

impl ParallelPlateGap {
    /// Air-gap model with given bias voltage and gap, εr = 1.
    pub fn air_gap(bias_voltage: f64, gap_distance: f64) -> Self {
        Self {
            bias_voltage,
            gap_distance,
            permittivity: EPSILON_0,
        }
    }

    /// Capacitance per unit area at the rest gap: C₀/A = ε/d₀ \[F/m²\].
    pub fn capacitance_per_area(&self) -> f64 {
        self.permittivity / self.gap_distance
    }

    /// Capacitance of a plate of area A at rest gap \[F\].
    pub fn capacitance(&self, area: f64) -> f64 {
        self.capacitance_per_area() * area
    }

    /// Static bias pressure (per unit area): p₀ = εV₀²/(2d₀²) \[Pa\].
    pub fn bias_pressure(&self) -> f64 {
        self.permittivity * self.bias_voltage * self.bias_voltage
            / (2.0 * self.gap_distance * self.gap_distance)
    }

    /// Linearised electromechanical coupling factor: η = εV₀/d₀² \[Pa/V\].
    ///
    /// Small-signal acoustic pressure from signal voltage: Δp ≈ η · v_signal.
    pub fn coupling_factor(&self) -> f64 {
        self.permittivity * self.bias_voltage / (self.gap_distance * self.gap_distance)
    }

    /// Electrostatic negative spring per unit area: k_e = εV₀²/d₀³ \[Pa/m\].
    ///
    /// This softening effect must be subtracted from the mechanical stiffness.
    /// If k_e > k_mech, pull-in occurs.
    pub fn electrostatic_spring_per_area(&self) -> f64 {
        self.permittivity * self.bias_voltage * self.bias_voltage
            / (self.gap_distance * self.gap_distance * self.gap_distance)
    }

    /// Pull-in voltage for a membrane with mechanical spring k_mech \[N/m\]
    /// and electrode area A \[m²\]: V_pi = √(8 k_mech d₀³ / (27 ε A)).
    pub fn pull_in_voltage(&self, mechanical_spring: f64, area: f64) -> f64 {
        (8.0 * mechanical_spring * self.gap_distance.powi(3) / (27.0 * self.permittivity * area))
            .sqrt()
    }

    /// Nonlinear electrostatic pressure at arbitrary voltage and gap.
    ///
    /// p_e(V, d) = ε V² / (2 d²). Positive = attractive.
    ///
    /// # Panics
    /// Panics if `gap <= 0` (physical contact / collapse).
    pub fn pressure(&self, voltage: f64, gap: f64) -> f64 {
        assert!(
            gap > 0.0,
            "gap must be positive (contact/collapse), got {gap}"
        );
        self.permittivity * voltage * voltage / (2.0 * gap * gap)
    }

    /// Nonlinear electrostatic force on a plate of area A.
    ///
    /// # Panics
    /// Panics if `gap <= 0`.
    pub fn force(&self, voltage: f64, gap: f64, area: f64) -> f64 {
        self.pressure(voltage, gap) * area
    }

    /// Capacitance at displacement u (gap = d₀ − u), per unit area.
    ///
    /// u > 0 means the gap is closing.
    ///
    /// # Panics
    /// Panics if `u >= gap_distance` (contact).
    pub fn capacitance_at_displacement_per_area(&self, u: f64) -> f64 {
        let effective_gap = self.gap_distance - u;
        assert!(
            effective_gap > 0.0,
            "displacement {u} exceeds gap {}, contact occurred",
            self.gap_distance
        );
        self.permittivity / effective_gap
    }
}

// ---------------------------------------------------------------------------
// Linearised electrostatic coupling for FEM
// ---------------------------------------------------------------------------

/// Linearised electrostatic coupling coefficients for the FEM system.
///
/// Around the operating point (V₀, d₀), the coupled system is:
///
/// ```text
/// (K_mech − k_e B_membrane) u = η v_signal B_membrane 1 + F_acoustic
/// ```
///
/// where:
/// - `K_mech`: mechanical stiffness matrix
/// - `B_membrane`: boundary mass matrix on the membrane surface
/// - `k_e`: electrostatic spring per unit area (softening, subtract)
/// - `η`: coupling factor (driving pressure per volt)
/// - `v_signal`: applied signal voltage
#[derive(Debug, Clone, Copy)]
pub struct LinearisedCoupling {
    /// Electrostatic spring per unit area k_e \[Pa/m\].
    pub spring_per_area: f64,
    /// Coupling factor η \[Pa/V\].
    pub coupling_factor: f64,
    /// Static bias pressure p₀ \[Pa\].
    pub bias_pressure: f64,
}

impl LinearisedCoupling {
    pub fn from_gap(gap: &ParallelPlateGap) -> Self {
        Self {
            spring_per_area: gap.electrostatic_spring_per_area(),
            coupling_factor: gap.coupling_factor(),
            bias_pressure: gap.bias_pressure(),
        }
    }
}

// ---------------------------------------------------------------------------
// Electrostatic circuit model (small signal)
// ---------------------------------------------------------------------------

/// Small-signal electrical impedance of the electrostatic transducer.
///
/// At the bias point: Z_e = 1 / (jωC₀) for a blocked membrane.
/// With membrane motion: the motional impedance contributes.
pub fn blocked_electrical_impedance(capacitance: f64, omega: f64) -> Complex64 {
    Complex64::new(1.0, 0.0) / (Complex64::new(0.0, omega * capacitance))
}

/// Motional current from membrane displacement.
///
/// For a condenser microphone with bias V₀ and capacitance C₀:
/// i_motional = V₀ dC/dt = V₀ (ε A / d₀²) du/dt = C₀ V₀ / d₀ × jωu
///
/// Returns the transduction coefficient: i = Γ × jω × u_avg
/// where Γ = C₀ V₀ / d₀.
pub fn motional_transduction_coefficient(
    capacitance: f64,
    bias_voltage: f64,
    gap_distance: f64,
) -> f64 {
    capacitance * bias_voltage / gap_distance
}

// ---------------------------------------------------------------------------
// Singular electrostatic sources
// ---------------------------------------------------------------------------

/// Distributional point charge for scalar electrostatic potential.
///
/// In weak form, `-div(eps grad phi) = rho` contributes `q v(x0)` to the RHS.
#[derive(Debug, Clone, Copy)]
pub struct PointChargeSource {
    pub position: [f64; 3],
    pub charge: f64,
}

impl PointChargeSource {
    pub fn new(position: [f64; 3], charge: f64) -> Self {
        Self { position, charge }
    }
}

/// Assemble a point-charge RHS over the element containing the source.
///
/// Supports affine triangle and tetrahedron meshes with nodal scalar spaces.
/// Evaluates the test basis at the source, including higher-order basis values.
/// A source on a shared facet contributes once, in the first containing element.
/// Returns a zero vector when the source lies outside the supported elements.
pub fn assemble_point_charge_source<S>(
    source: &PointChargeSource,
    mesh: &Mesh,
    space: &S,
) -> Vec<f64>
where
    S: FunctionSpace,
{
    let mut rhs = vec![0.0; space.num_global_dofs(mesh)];

    for elem_idx in 0..mesh.elements.len() {
        let elem = &mesh.elements[elem_idx];
        let nodes = elem.vertices();
        let reference = match elem.element_type {
            crate::mesh::ElementType::Triangle => reference_coords_in_triangle(
                source.position,
                mesh.nodes[nodes[0]],
                mesh.nodes[nodes[1]],
                mesh.nodes[nodes[2]],
            )
            .map(|(xi, eta)| [xi, eta, 0.0]),
            crate::mesh::ElementType::Tetrahedron => {
                let coordinates: Vec<[f64; 3]> = nodes
                    .iter()
                    .map(|&node| {
                        let p = mesh.nodes[node];
                        [p.x, p.y, p.z]
                    })
                    .collect();
                let shape = crate::basis::evaluate_shape(
                    crate::mesh::ElementType::Tetrahedron,
                    crate::basis::PolynomialDegree::P1,
                    0.0,
                    0.0,
                    0.0,
                );
                let jacobian = crate::basis::Jacobian::from_3d(&shape.gradients, 3, &coordinates);
                let offset: [f64; 3] =
                    std::array::from_fn(|d| source.position[d] - coordinates[0][d]);
                let reference: [f64; 3] = std::array::from_fn(|d| {
                    (0..3)
                        .map(|j| jacobian.inverse[3 * d + j] * offset[j])
                        .sum()
                });
                // Permit roundoff at shared facets, but never admit a
                // degenerate tetrahedron or a point beyond its barycentric hull.
                const CONTAINMENT_TOLERANCE: f64 = 1e-12;
                if reference
                    .iter()
                    .all(|x| x.is_finite() && *x >= -CONTAINMENT_TOLERANCE)
                    && reference.iter().sum::<f64>() <= 1.0 + CONTAINMENT_TOLERANCE
                {
                    Some(reference)
                } else {
                    None
                }
            }
            _ => None,
        };
        if let Some([xi, eta, zeta]) = reference {
            let tab = space.tabulate(mesh, elem_idx, xi, eta, zeta);
            let dofs = space.local_to_global(mesh, elem_idx);
            for (local, &dof) in dofs.iter().enumerate() {
                rhs[dof] += source.charge * tab.value(local);
            }
            break;
        }
    }

    rhs
}

/// Image-method potential of a point charge above a grounded plane `z = 0`.
pub fn grounded_plane_image_potential(
    point: [f64; 3],
    charge_position: [f64; 3],
    charge: f64,
    epsilon: f64,
) -> f64 {
    let image = [charge_position[0], charge_position[1], -charge_position[2]];
    let r = distance(point, charge_position).max(1e-30);
    let ri = distance(point, image).max(1e-30);
    charge / (4.0 * std::f64::consts::PI * epsilon) * (1.0 / r - 1.0 / ri)
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn reference_coords_in_triangle(
    p: [f64; 3],
    a: crate::mesh::Point,
    b: crate::mesh::Point,
    c: crate::mesh::Point,
) -> Option<(f64, f64)> {
    let v0 = [b.x - a.x, b.y - a.y];
    let v1 = [c.x - a.x, c.y - a.y];
    let v2 = [p[0] - a.x, p[1] - a.y];
    let den = v0[0] * v1[1] - v1[0] * v0[1];
    if den.abs() < 1e-30 {
        return None;
    }
    let xi = (v2[0] * v1[1] - v1[0] * v2[1]) / den;
    let eta = (v0[0] * v2[1] - v2[0] * v0[1]) / den;
    let l0 = 1.0 - xi - eta;
    if xi >= -1e-12 && eta >= -1e-12 && l0 >= -1e-12 {
        Some((xi, eta))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Push-pull electrostatic speaker
// ---------------------------------------------------------------------------

/// Push-pull electrostatic speaker configuration.
///
/// Two stators (front and back) with the membrane between them. Each stator
/// has a DC bias V₀ of opposite polarity. The signal voltage ±v(t) is applied
/// differentially:
///
/// ```text
///   Stator A: V_A = V₀ + v(t),  gap d₀ − u
///   Stator B: V_B = V₀ − v(t),  gap d₀ + u
/// ```
///
/// The net force is linearised:
///
/// ```text
/// F_net = ε A V₀ v(t) / d₀²   (to first order)
/// ```
///
/// The static bias forces cancel, and the negative electrostatic springs
/// partially cancel (the net spring is reduced compared to single-ended).
///
/// # Advantages over single-ended
///
/// - No static deflection (bias forces cancel)
/// - Linear force-voltage relationship (even-order harmonics cancel)
/// - Higher drive capability before pull-in
#[derive(Debug, Clone, Copy)]
pub struct PushPullConfig {
    /// Front gap (stator A).
    pub gap_front: ParallelPlateGap,
    /// Back gap (stator B), same dimensions but opposite polarity signal.
    pub gap_back: ParallelPlateGap,
    /// Membrane active area (m²).
    pub area: f64,
}

impl PushPullConfig {
    /// Create a symmetric push-pull configuration.
    ///
    /// Both gaps have the same distance and bias voltage.
    pub fn symmetric(bias_voltage: f64, gap_distance: f64, area: f64) -> Self {
        let gap = ParallelPlateGap::air_gap(bias_voltage, gap_distance);
        Self {
            gap_front: gap,
            gap_back: gap,
            area,
        }
    }

    /// Net linearised coupling factor: η_pp = 2 × ε V₀ / d₀² (Pa/V).
    ///
    /// This is twice the single-ended coupling factor because both gaps
    /// contribute in the same direction for the signal.
    pub fn coupling_factor(&self) -> f64 {
        self.gap_front.coupling_factor() + self.gap_back.coupling_factor()
    }

    /// Net electrostatic spring per unit area (Pa/m).
    ///
    /// In push-pull, the springs from both gaps add (both destabilise),
    /// so the total negative spring is the sum.
    pub fn electrostatic_spring_per_area(&self) -> f64 {
        self.gap_front.electrostatic_spring_per_area()
            + self.gap_back.electrostatic_spring_per_area()
    }

    /// Net static force on the membrane (N). Should be zero for a symmetric config.
    pub fn net_static_force(&self) -> f64 {
        let f_front = self.gap_front.bias_pressure() * self.area;
        let f_back = self.gap_back.bias_pressure() * self.area;
        // Front pushes membrane toward back, back pushes toward front → cancel
        f_front - f_back
    }

    /// Linearised coupling for the FEM system.
    pub fn linearised_coupling(&self) -> LinearisedCoupling {
        LinearisedCoupling {
            coupling_factor: self.coupling_factor(),
            spring_per_area: self.electrostatic_spring_per_area(),
            bias_pressure: 0.0, // push-pull: no net static pressure
        }
    }

    /// Nonlinear force on the membrane for a given signal voltage and displacement.
    ///
    /// `v_signal`: signal voltage (V). Applied as +v to front, −v to back.
    /// `u`: membrane displacement (m). Positive = toward front stator.
    ///
    /// Returns net force (N). Positive = toward front stator.
    pub fn force(&self, v_signal: f64, u: f64) -> f64 {
        let d_front = self.gap_front.gap_distance - u;
        let d_back = self.gap_back.gap_distance + u;
        let v_front = self.gap_front.bias_voltage + v_signal;
        let v_back = self.gap_back.bias_voltage - v_signal;
        let f_front = instantaneous_force(self.gap_front.permittivity, v_front, d_front, self.area);
        let f_back = instantaneous_force(self.gap_back.permittivity, v_back, d_back, self.area);
        // Front gap pulls membrane forward, back gap pulls it backward
        f_front - f_back
    }

    /// Pull-in voltage for the push-pull configuration.
    ///
    /// Higher than single-ended because the restoring spring from the opposite
    /// gap partially compensates the collapsing gap's negative spring.
    /// For a symmetric config: V_pi_pp ≈ √(2) × V_pi_single.
    pub fn pull_in_voltage(&self, mechanical_spring: f64) -> f64 {
        // Approximate: use single-gap formula with effective spring
        // k_eff = k_mech − k_e_other_gap (the other gap provides restoring)
        let k_e_one = self.gap_front.electrostatic_spring_per_area() * self.area;
        let k_eff = mechanical_spring + k_e_one; // other gap's spring helps
        self.gap_front.pull_in_voltage(k_eff, self.area)
    }

    /// Total blocked capacitance (both gaps in series for signal, parallel for bias).
    pub fn total_capacitance(&self) -> f64 {
        self.gap_front.capacitance(self.area) + self.gap_back.capacitance(self.area)
    }
}

// ---------------------------------------------------------------------------
// Nonlinear large-signal dynamics
// ---------------------------------------------------------------------------

/// Compute the electrostatic force for a time-domain step.
///
/// Given the instantaneous voltage V(t) and gap d(t) = d₀ − u(t):
/// F = ε A V² / (2 d²)
///
/// For large-signal analysis, the voltage may include both the bias
/// and the signal: V(t) = V₀ + v(t).
/// # Panics
/// Panics if `gap <= 0`.
pub fn instantaneous_force(permittivity: f64, voltage: f64, gap: f64, area: f64) -> f64 {
    assert!(gap > 0.0, "gap must be positive, got {gap}");
    permittivity * voltage * voltage * area / (2.0 * gap * gap)
}

/// Charge on the capacitor: Q = C(u) × V = ε A V / (d₀ − u).
///
/// # Panics
/// Panics if `gap <= 0`.
pub fn instantaneous_charge(permittivity: f64, voltage: f64, gap: f64, area: f64) -> f64 {
    assert!(gap > 0.0, "gap must be positive, got {gap}");
    permittivity * area * voltage / gap
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        assembly::assemble_bilinear, basis::PolynomialDegree, forms::WeightedLaplacianForm,
        mesh::unit_square_triangles, spaces::H1Space,
    };
    use std::f64::consts::PI;

    #[test]
    fn tetrahedral_point_charge_reproduces_test_polynomials() {
        use crate::mesh::unit_cube_tetrahedra;
        // A sheared/translated mesh prevents an axis-aligned inverse from
        // accidentally satisfying the source-location contract.
        let transform = |p: [f64; 3]| {
            [
                3.0 + 2.0 * p[0] + 0.2 * p[1] + 0.1 * p[2],
                -2.0 + 0.1 * p[0] + 1.4 * p[1] + 0.3 * p[2],
                0.4 + 0.2 * p[0] + 0.1 * p[1] + 0.9 * p[2],
            ]
        };
        let mut mesh = unit_cube_tetrahedra(2);
        for node in &mut mesh.nodes {
            let p = transform([node.x, node.y, node.z]);
            node.x = p[0];
            node.y = p[1];
            node.z = p[2];
        }
        for degree in [PolynomialDegree::P1, PolynomialDegree::P2] {
            let space = H1Space::new(degree);
            for position in [transform([0.23, 0.27, 0.31]), transform([0.5, 0.5, 0.5])] {
                let charge = 2.5;
                let rhs = assemble_point_charge_source(
                    &PointChargeSource::new(position, charge),
                    &mesh,
                    &space,
                );
                assert!(
                    (rhs.iter().sum::<f64>() - charge).abs() < 1e-12,
                    "shared source must contribute once"
                );
                let coordinates = space.dof_coordinates(&mesh).unwrap();
                for dimension in 0..3 {
                    let moment: f64 = rhs
                        .iter()
                        .zip(&coordinates)
                        .map(|(load, p)| load * p[dimension])
                        .sum();
                    assert!((moment - charge * position[dimension]).abs() < 1e-11);
                    if degree == PolynomialDegree::P2 {
                        let moment: f64 = rhs
                            .iter()
                            .zip(&coordinates)
                            .map(|(load, p)| load * p[dimension].powi(2))
                            .sum();
                        assert!((moment - charge * position[dimension].powi(2)).abs() < 1e-10);
                    }
                }
            }
            let outside = assemble_point_charge_source(
                &PointChargeSource::new(transform([2.0, 2.0, 2.0]), 1.0),
                &mesh,
                &space,
            );
            assert!(outside.iter().all(|x| *x == 0.0));
        }
    }

    #[test]
    fn solved_field_recovers_linear_gradient_and_energy() {
        let mesh = unit_square_triangles(2);
        let space = H1Space::new(PolynomialDegree::P1);
        let potential: Vec<_> = mesh
            .nodes
            .iter()
            .map(|node| node.x + 2.0 * node.y)
            .collect();
        let solved = ElectrostaticSolvedField::new(&mesh, &space, &potential, 3.0).unwrap();
        for element in 0..mesh.num_elements() {
            let field = solved
                .electric_field_at_reference(element, 1.0 / 3.0, 1.0 / 3.0, 0.0)
                .unwrap();
            assert!((field[0] + 1.0).abs() < 1.0e-12);
            assert!((field[1] + 2.0).abs() < 1.0e-12);
            assert!(field[2].abs() < 1.0e-12);
        }
        let unit_laplacian = assemble_bilinear(
            &WeightedLaplacianForm { coefficient: 1.0 },
            &space,
            &space,
            &mesh,
            2,
        )
        .to_csr();
        let observables = solved.capacitor_observables(&unit_laplacian, 1.0).unwrap();
        assert!((observables.energy_j - 7.5).abs() < 1.0e-12);
        assert!((observables.capacitance_f - 15.0).abs() < 1.0e-12);
        assert!((observables.charge_c - 15.0).abs() < 1.0e-12);
    }

    #[test]
    fn test_parallel_plate_capacitance() {
        // 1 cm² plate, 25 μm gap in air
        let gap = ParallelPlateGap::air_gap(200.0, 25e-6);
        let c = gap.capacitance(1e-4);
        let expected = EPSILON_0 * 1e-4 / 25e-6; // ~35.4 pF
        assert!(
            (c - expected).abs() / expected < 1e-10,
            "C = {c:.3e}, expected {expected:.3e}"
        );
    }

    #[test]
    fn test_bias_pressure() {
        let gap = ParallelPlateGap::air_gap(200.0, 25e-6);
        let p = gap.bias_pressure();
        let expected = EPSILON_0 * 200.0 * 200.0 / (2.0 * 25e-6 * 25e-6);
        assert!((p - expected).abs() < 1e-6);
        // Should be ~283 Pa for typical condenser mic
        assert!(p > 100.0 && p < 1000.0, "p₀ = {p:.1} Pa");
    }

    #[test]
    fn test_coupling_factor() {
        let gap = ParallelPlateGap::air_gap(200.0, 25e-6);
        let eta = gap.coupling_factor();
        // η = εV₀/d₀² = 8.854e-12 × 200 / (25e-6)² ≈ 2.83 Pa/V
        assert!(eta > 1.0 && eta < 10.0, "η = {eta:.3} Pa/V");
    }

    #[test]
    fn test_electrostatic_spring() {
        let gap = ParallelPlateGap::air_gap(200.0, 25e-6);
        let k_e = gap.electrostatic_spring_per_area();
        // k_e = εV₀²/d₀³ = 8.854e-12 × 40000 / 1.5625e-14 ≈ 2.27e7 Pa/m
        assert!(k_e > 1e6, "k_e = {k_e:.3e} Pa/m");
    }

    #[test]
    fn test_pull_in_voltage() {
        let gap = ParallelPlateGap::air_gap(0.0, 25e-6); // no bias for this calc
        // For a membrane: k_mech = 1000 N/m, A = 1e-4 m²
        let v_pi = gap.pull_in_voltage(1000.0, 1e-4);
        // V_pi = √(8×1000×(25e-6)³ / (27×8.854e-12×1e-4))
        //      = √(8×1000×1.5625e-14 / 2.39e-14) = √(5.23) ≈ 2.29 ... hmm
        // Actually: 8*1000*(25e-6)^3 = 1.25e-10
        //           27*8.854e-12*1e-4 = 2.39e-14
        //           ratio = 5230, sqrt = 72.3 V
        assert!(v_pi > 10.0 && v_pi < 500.0, "V_pi = {v_pi:.1} V");
    }

    #[test]
    fn test_nonlinear_force_reduces_to_bias() {
        let gap = ParallelPlateGap::air_gap(200.0, 25e-6);
        let area = 1e-4;
        let f = gap.force(200.0, 25e-6, area);
        let expected = gap.bias_pressure() * area;
        assert!((f - expected).abs() / expected < 1e-10);
    }

    #[test]
    fn test_linearised_coupling() {
        let gap = ParallelPlateGap::air_gap(200.0, 25e-6);
        let lc = LinearisedCoupling::from_gap(&gap);
        assert!(lc.bias_pressure > 0.0);
        assert!(lc.coupling_factor > 0.0);
        assert!(lc.spring_per_area > 0.0);
    }

    #[test]
    fn test_blocked_impedance_capacitive() {
        let c = 35e-12; // 35 pF
        let omega = 2.0 * PI * 1000.0;
        let z = blocked_electrical_impedance(c, omega);
        // |Z| = 1/(ωC) ≈ 4.55 MΩ
        let expected_mag = 1.0 / (omega * c);
        assert!(
            (z.norm() - expected_mag).abs() / expected_mag < 1e-10,
            "|Z| = {}, expected {}",
            z.norm(),
            expected_mag
        );
        // Should be purely imaginary with negative imaginary part (capacitive)
        assert!(z.re.abs() < 1e-5 * z.norm());
        assert!(z.im < 0.0, "Capacitive impedance should have Im < 0");
    }

    #[test]
    fn test_motional_transduction() {
        let c0 = 35e-12;
        let v0 = 200.0;
        let d0 = 25e-6;
        let gamma = motional_transduction_coefficient(c0, v0, d0);
        // Γ = C₀V₀/d₀ = 35e-12 × 200 / 25e-6 = 2.8e-4 A·s/m
        let expected = c0 * v0 / d0;
        assert!((gamma - expected).abs() < 1e-15);
    }

    // -- Push-pull tests --

    #[test]
    fn test_push_pull_zero_static_force() {
        let pp = PushPullConfig::symmetric(200.0, 25e-6, 1e-4);
        let f_static = pp.net_static_force();
        assert!(
            f_static.abs() < 1e-15,
            "Symmetric push-pull should have zero static force: {f_static}"
        );
    }

    #[test]
    fn test_push_pull_double_coupling() {
        let single = ParallelPlateGap::air_gap(200.0, 25e-6);
        let pp = PushPullConfig::symmetric(200.0, 25e-6, 1e-4);

        let eta_single = single.coupling_factor();
        let eta_pp = pp.coupling_factor();

        assert!(
            (eta_pp - 2.0 * eta_single).abs() < 1e-10,
            "Push-pull coupling should be 2× single: pp={eta_pp:.3}, 2×single={:.3}",
            2.0 * eta_single
        );
    }

    #[test]
    fn test_push_pull_linearised_zero_bias_pressure() {
        let pp = PushPullConfig::symmetric(200.0, 25e-6, 1e-4);
        let lc = pp.linearised_coupling();
        assert!(lc.bias_pressure.abs() < 1e-15);
        assert!(lc.coupling_factor > 0.0);
        assert!(lc.spring_per_area > 0.0);
    }

    #[test]
    fn test_push_pull_linear_force() {
        // At u=0, force should be proportional to signal voltage
        let pp = PushPullConfig::symmetric(200.0, 25e-6, 1e-4);

        let f1 = pp.force(1.0, 0.0);
        let f2 = pp.force(2.0, 0.0);

        // Linear: F(2v) ≈ 2×F(v)
        assert!(
            (f2 - 2.0 * f1).abs() / f1.abs() < 0.01,
            "Force should be linear: F(1V)={f1:.6e}, F(2V)={f2:.6e}"
        );
    }

    #[test]
    fn test_push_pull_zero_force_at_zero_signal() {
        let pp = PushPullConfig::symmetric(200.0, 25e-6, 1e-4);
        let f = pp.force(0.0, 0.0);
        assert!(
            f.abs() < 1e-15,
            "Zero signal should give zero net force: {f}"
        );
    }

    #[test]
    fn test_push_pull_higher_pull_in() {
        let single = ParallelPlateGap::air_gap(200.0, 25e-6);
        let pp = PushPullConfig::symmetric(200.0, 25e-6, 1e-4);
        let k_mech = 1000.0;

        let vpi_single = single.pull_in_voltage(k_mech, 1e-4);
        let vpi_pp = pp.pull_in_voltage(k_mech);

        assert!(
            vpi_pp > vpi_single,
            "Push-pull should have higher pull-in: pp={vpi_pp:.1}V, single={vpi_single:.1}V"
        );
    }

    // -- Regression: gap=0 must panic, not produce inf --

    #[test]
    #[should_panic(expected = "gap must be positive")]
    fn test_pressure_panics_at_zero_gap() {
        let gap = ParallelPlateGap::air_gap(200.0, 25e-6);
        gap.pressure(200.0, 0.0);
    }

    #[test]
    #[should_panic(expected = "displacement")]
    fn test_capacitance_panics_at_contact() {
        let gap = ParallelPlateGap::air_gap(200.0, 25e-6);
        gap.capacitance_at_displacement_per_area(25e-6); // u == gap_distance
    }
}
