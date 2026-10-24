//! Maxwell curl-curl bilinear form
//!
//! Time-harmonic Maxwell's equations in the frequency domain reduce to:
//!
//! ```text
//! curl((1/mu) curl E) - omega^2 eps E = -i omega J
//! ```
//!
//! Weak form: `a(E,F) = (1/mu) integral (curl E).(curl F) dOmega
//!                     - omega^2 eps integral E.F dOmega`
//!
//! Uses H(curl)-conforming Nédélec edge elements via [`HcurlSpace`](crate::spaces::HcurlSpace).

use crate::boundary::PmlRegion;
use crate::forms::{BilinearForm, LinearForm, LocalMatrix, LocalVector};
use crate::spaces::BasisTabulation;
use num_complex::Complex64;

// Re-export standard curl-curl and edge mass forms for backward compatibility.
pub use crate::forms::{CurlCurlForm, EdgeMassForm};

/// Electromagnetic material properties.
#[derive(Debug, Clone, Copy)]
pub struct EmMaterial {
    /// Relative permittivity (epsilon_r). Vacuum = 1.0.
    pub epsilon_r: f64,
    /// Relative permeability (mu_r). Vacuum = 1.0.
    pub mu_r: f64,
    /// Electrical conductivity (sigma) in S/m. Lossless = 0.0.
    pub conductivity: f64,
}

impl EmMaterial {
    /// Vacuum.
    pub fn vacuum() -> Self {
        Self {
            epsilon_r: 1.0,
            mu_r: 1.0,
            conductivity: 0.0,
        }
    }

    /// Dielectric with given relative permittivity.
    pub fn dielectric(epsilon_r: f64) -> Self {
        Self {
            epsilon_r,
            mu_r: 1.0,
            conductivity: 0.0,
        }
    }

    /// Complex permittivity including conductivity loss: eps_c = eps - i*sigma/omega
    pub fn complex_permittivity(&self, omega: f64) -> Complex64 {
        let eps0 = 8.854187817e-12;
        let eps = self.epsilon_r * eps0;
        if omega > 0.0 && self.conductivity > 0.0 {
            Complex64::new(eps, -self.conductivity / omega)
        } else {
            Complex64::new(eps, 0.0)
        }
    }

    /// Permeability: mu = mu_r * mu_0
    pub fn permeability(&self) -> f64 {
        let mu0 = 4.0 * std::f64::consts::PI * 1e-7;
        self.mu_r * mu0
    }
}

/// Maxwell curl-curl bilinear form.
///
/// `a(E,F) = (1/mu) integral (curl E).(curl F) dOmega - omega^2 eps integral E.F dOmega`
///
/// For one-shot assembly at a fixed frequency. For frequency sweeps, use
/// [`PrebakedAssembler<2>`](crate::assembly::PrebakedAssembler) with separate
/// curl-curl and mass forms.
#[derive(Debug, Clone)]
pub struct MaxwellForm {
    /// Angular frequency omega (rad/s).
    pub omega: f64,
    /// 1/mu coefficient for curl-curl term.
    pub inv_mu: f64,
    /// Complex permittivity for mass term (may include conductivity loss).
    pub epsilon: Complex64,
}

impl MaxwellForm {
    /// Create from material properties and frequency.
    pub fn new(material: &EmMaterial, omega: f64) -> Self {
        Self {
            omega,
            inv_mu: 1.0 / material.permeability(),
            epsilon: material.complex_permittivity(omega),
        }
    }

    /// Create with explicit coefficients (for normalized/dimensionless problems).
    ///
    /// Assembles `curl_curl_coeff * integral (curl E).(curl F)
    /// + mass_coeff * integral E.F`; the mass coefficient includes its sign.
    pub fn from_coefficients(curl_curl_coeff: f64, mass_coeff: Complex64) -> Self {
        Self {
            // The integration path factors the mass term as `-omega^2 *
            // epsilon`.  Use normalized omega = 1 so that the stored
            // epsilon = -mass_coeff realizes the explicit coefficient.
            omega: 1.0,
            inv_mu: curl_curl_coeff,
            epsilon: -mass_coeff, // stored as epsilon so -omega^2*eps = mass_coeff
        }
    }
}

impl BilinearForm<Complex64> for MaxwellForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<Complex64>,
    ) {
        let n = test.n_dofs;
        let dim = test.spatial_dim;
        let w = weight * test.det_j;
        let omega_sq = self.omega * self.omega;

        match dim {
            2 => integrate_2d(self, test, trial, n, w, omega_sq, local_matrix),
            3 => integrate_3d(self, test, trial, n, w, omega_sq, local_matrix),
            _ => unreachable!(),
        }
    }
}

/// Maxwell curl-curl form with element-wise anisotropic PML stretching.
///
/// This is a standalone PML assembly hook for propagation/scattering problems.
/// It uses the existing [`PmlRegion`] stretching factor to scale the curl-curl
/// and mass terms at quadrature points inside a PML region. Outside the PML it
/// reduces exactly to [`MaxwellForm`].
#[derive(Debug, Clone)]
pub struct PmlMaxwellForm {
    pub base: MaxwellForm,
    pub regions: Vec<PmlRegion>,
}

impl PmlMaxwellForm {
    pub fn new(base: MaxwellForm, regions: Vec<PmlRegion>) -> Self {
        Self { base, regions }
    }

    fn coefficients(&self, coords: [f64; 3]) -> PmlMaxwellCoefficients {
        let point = crate::mesh::Point::new_3d(coords[0], coords[1], coords[2]);
        self.regions
            .iter()
            .find(|region| region.contains(&point))
            .map(|region| PmlMaxwellCoefficients {
                curl: [
                    region.stiffness_coefficient_x(&point),
                    region.stiffness_coefficient_y(&point),
                    region.stiffness_coefficient_z(&point),
                ],
                mass: region.mass_coefficient(&point),
            })
            .unwrap_or_default()
    }
}

#[derive(Debug, Clone, Copy)]
struct PmlMaxwellCoefficients {
    curl: [Complex64; 3],
    mass: Complex64,
}

impl Default for PmlMaxwellCoefficients {
    fn default() -> Self {
        Self {
            curl: [Complex64::new(1.0, 0.0); 3],
            mass: Complex64::new(1.0, 0.0),
        }
    }
}

impl BilinearForm<Complex64> for PmlMaxwellForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<Complex64>,
    ) {
        let dim = test.spatial_dim;
        let w = weight * test.det_j;
        let omega_sq = self.base.omega * self.base.omega;
        let coeffs = self.coefficients(test.physical_coords);

        match dim {
            2 => integrate_2d_pml(&self.base, &coeffs, test, trial, w, omega_sq, local_matrix),
            3 => integrate_3d_pml(&self.base, &coeffs, test, trial, w, omega_sq, local_matrix),
            _ => unreachable!(),
        }
    }
}

/// Current-density source for time-harmonic Maxwell problems.
///
/// Assembles the RHS `-i omega ∫ J · F dΩ`.
pub struct MaxwellCurrentSourceForm<F: Fn(f64, f64, f64) -> [Complex64; 3] + Send + Sync> {
    pub omega: f64,
    pub current_fn: F,
}

impl<F: Fn(f64, f64, f64) -> [Complex64; 3] + Send + Sync> MaxwellCurrentSourceForm<F> {
    pub fn new(omega: f64, current_fn: F) -> Self {
        Self { omega, current_fn }
    }
}

impl<F: Fn(f64, f64, f64) -> [Complex64; 3] + Send + Sync> LinearForm<Complex64>
    for MaxwellCurrentSourceForm<F>
{
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<Complex64>,
    ) {
        let [x, y, z] = test.physical_coords;
        let j = (self.current_fn)(x, y, z);
        let dim = test.spatial_dim;
        let coeff = Complex64::new(0.0, -self.omega) * weight * test.det_j;
        for i in 0..test.n_dofs {
            let mut dot = Complex64::new(0.0, 0.0);
            for (c, jc) in j.iter().enumerate().take(dim) {
                dot += *jc * test.vector_value(i, c);
            }
            local_vector.add(i, coeff * dot);
        }
    }
}

/// Incident electric plane wave helper.
pub fn incident_plane_wave_e(
    amplitude: [Complex64; 3],
    direction: [f64; 3],
    wavenumber: f64,
) -> impl Fn(f64, f64, f64) -> [Complex64; 3] + Send + Sync {
    move |x, y, z| {
        let phase = wavenumber * (direction[0] * x + direction[1] * y + direction[2] * z);
        let factor = Complex64::from_polar(1.0, phase);
        [
            amplitude[0] * factor,
            amplitude[1] * factor,
            amplitude[2] * factor,
        ]
    }
}

/// 2D Maxwell: scalar curl, 2D vector values.
///
/// curl-curl: `(1/mu) * curl_i * curl_j` (scalar product)
/// mass: `eps * (N_i . N_j)` (2D dot product)
fn integrate_2d(
    form: &MaxwellForm,
    test: &BasisTabulation,
    trial: &BasisTabulation,
    n: usize,
    w: f64,
    omega_sq: f64,
    local_matrix: &mut LocalMatrix<Complex64>,
) {
    for i in 0..n {
        let curl_i = test.derivatives[i]; // scalar curl in 2D

        for j in 0..n {
            let curl_j = trial.derivatives[j];

            // Curl-curl term: (1/mu) * curl_i * curl_j
            let cc = form.inv_mu * curl_i * curl_j;

            // Mass term: E_i . E_j (2D dot product of vector basis functions)
            let dot: f64 = (0..2)
                .map(|d| test.values[i * 2 + d] * trial.values[j * 2 + d])
                .sum();

            let val = Complex64::new(cc * w, 0.0)
                - omega_sq * form.epsilon * Complex64::new(dot * w, 0.0);

            local_matrix.add(i, j, val);
        }
    }
}

fn integrate_2d_pml(
    form: &MaxwellForm,
    coeffs: &PmlMaxwellCoefficients,
    test: &BasisTabulation,
    trial: &BasisTabulation,
    w: f64,
    omega_sq: f64,
    local_matrix: &mut LocalMatrix<Complex64>,
) {
    let n = test.n_dofs;
    for i in 0..n {
        let curl_i = test.derivatives[i]; // scalar z-curl in 2D

        for j in 0..n {
            let curl_j = trial.derivatives[j];
            let cc = coeffs.curl[2] * Complex64::new(form.inv_mu * curl_i * curl_j * w, 0.0);

            let dot: f64 = (0..2)
                .map(|d| test.values[i * 2 + d] * trial.values[j * 2 + d])
                .sum();
            let mass = coeffs.mass * omega_sq * form.epsilon * Complex64::new(dot * w, 0.0);

            local_matrix.add(i, j, cc - mass);
        }
    }
}

/// 3D Maxwell: vector curl, 3D vector values.
///
/// curl-curl: `(1/mu) * curl_i . curl_j` (3D dot product)
/// mass: `eps * (N_i . N_j)` (3D dot product)
fn integrate_3d(
    form: &MaxwellForm,
    test: &BasisTabulation,
    trial: &BasisTabulation,
    n: usize,
    w: f64,
    omega_sq: f64,
    local_matrix: &mut LocalMatrix<Complex64>,
) {
    for i in 0..n {
        for j in 0..n {
            // Curl-curl: (1/mu) * curl_i . curl_j
            let curl_dot: f64 = (0..3)
                .map(|d| test.derivatives[i * 3 + d] * trial.derivatives[j * 3 + d])
                .sum();
            let cc = form.inv_mu * curl_dot;

            // Mass: eps * N_i . N_j
            let dot: f64 = (0..3)
                .map(|d| test.values[i * 3 + d] * trial.values[j * 3 + d])
                .sum();

            let val = Complex64::new(cc * w, 0.0)
                - omega_sq * form.epsilon * Complex64::new(dot * w, 0.0);

            local_matrix.add(i, j, val);
        }
    }
}

fn integrate_3d_pml(
    form: &MaxwellForm,
    coeffs: &PmlMaxwellCoefficients,
    test: &BasisTabulation,
    trial: &BasisTabulation,
    w: f64,
    omega_sq: f64,
    local_matrix: &mut LocalMatrix<Complex64>,
) {
    let n = test.n_dofs;
    for i in 0..n {
        for j in 0..n {
            let mut cc = Complex64::new(0.0, 0.0);
            for (d, &curl_coeff) in coeffs.curl.iter().enumerate() {
                let curl_dot = test.derivatives[i * 3 + d] * trial.derivatives[j * 3 + d];
                cc += curl_coeff * Complex64::new(form.inv_mu * curl_dot * w, 0.0);
            }

            let dot: f64 = (0..3)
                .map(|d| test.values[i * 3 + d] * trial.values[j * 3 + d])
                .sum();
            let mass = coeffs.mass * omega_sq * form.epsilon * Complex64::new(dot * w, 0.0);

            local_matrix.add(i, j, cc - mass);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::PmlRegion;
    use crate::mesh::unit_square_triangles;
    use crate::spaces::{FunctionSpace, HcurlSpace};

    #[test]
    fn test_curl_curl_form_symmetry_2d() {
        let mesh = unit_square_triangles(2);
        let space = HcurlSpace::new();
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let mut mat = LocalMatrix::zeros(n);
        CurlCurlForm.integrate_point(&tab, &tab, 1.0, &mut mat);

        for i in 0..n {
            for j in i + 1..n {
                let diff = (mat.get(i, j) - mat.get(j, i)).abs();
                assert!(diff < 1e-14, "Not symmetric at ({i},{j})");
            }
        }
    }

    #[test]
    fn test_edge_mass_form_positive_diagonal_2d() {
        let mesh = unit_square_triangles(2);
        let space = HcurlSpace::new();
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        let mut mat = LocalMatrix::zeros(n);
        EdgeMassForm.integrate_point(&tab, &tab, 1.0, &mut mat);

        for i in 0..n {
            assert!(mat.get(i, i) > 0.0, "Diagonal {i} should be positive");
        }
    }

    #[test]
    fn test_maxwell_form_symmetry_3d() {
        let mesh = crate::mesh::unit_cube_tetrahedra(2);
        let space = HcurlSpace::new();
        let tab = space.tabulate(&mesh, 0, 0.25, 0.25, 0.25);
        let n = tab.n_dofs;

        let form = MaxwellForm::from_coefficients(1.0, Complex64::new(-1.0, 0.0));
        let mut mat = LocalMatrix::zeros(n);
        form.integrate_point(&tab, &tab, 1.0, &mut mat);

        for i in 0..n {
            for j in i + 1..n {
                let diff = (mat.get(i, j) - mat.get(j, i)).norm();
                assert!(diff < 1e-14, "Not symmetric at ({i},{j})");
            }
        }
    }

    #[test]
    fn test_curl_curl_at_zero_freq_equals_curl_curl_only() {
        let mesh = unit_square_triangles(2);
        let space = HcurlSpace::new();
        let tab = space.tabulate(&mesh, 0, 1.0 / 3.0, 1.0 / 3.0, 0.0);
        let n = tab.n_dofs;

        // Maxwell at omega=0 should equal pure curl-curl
        let maxwell = MaxwellForm::from_coefficients(1.0, Complex64::new(0.0, 0.0));
        let mut mat_maxwell = LocalMatrix::zeros(n);
        maxwell.integrate_point(&tab, &tab, 1.0, &mut mat_maxwell);

        let mut mat_cc = LocalMatrix::zeros(n);
        CurlCurlForm.integrate_point(&tab, &tab, 1.0, &mut mat_cc);

        for i in 0..n {
            for j in 0..n {
                let m = mat_maxwell.get(i, j);
                let c = mat_cc.get(i, j);
                assert!(
                    (m.re - c).abs() < 1e-14 && m.im.abs() < 1e-14,
                    "Mismatch at ({i},{j}): maxwell={m}, curl_curl={c}"
                );
            }
        }
    }

    #[test]
    fn test_pml_maxwell_applies_anisotropic_curl_and_mass_coefficients() {
        let tab = BasisTabulation {
            n_dofs: 1,
            spatial_dim: 3,
            values: vec![1.0, 0.0, 0.0],
            derivatives: vec![2.0, 0.0, 0.0],
            hessians: Vec::new(),
            det_j: 1.0,
            physical_coords: [1.0, 0.0, 0.0],
        };
        let base = MaxwellForm {
            omega: 2.0,
            inv_mu: 3.0,
            epsilon: Complex64::new(5.0, 0.0),
        };
        let region = PmlRegion::x_positive(0.0, 1.0, 1.0, 1.0);
        let form = PmlMaxwellForm::new(base, vec![region]);

        let mut mat = LocalMatrix::zeros(1);
        form.integrate_point(&tab, &tab, 1.0, &mut mat);

        let sx = Complex64::new(1.0, 1.0);
        let expected = Complex64::new(12.0, 0.0) / sx - Complex64::new(20.0, 0.0) * sx;
        let old_uniform_rescale = Complex64::new(-8.0, 0.0) / sx;

        assert!((mat.get(0, 0) - expected).norm() < 1e-12);
        assert!((mat.get(0, 0) - old_uniform_rescale).norm() > 1.0);
    }
}
