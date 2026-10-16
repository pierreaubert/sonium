//! Dirichlet-to-Neumann (DtN) condensation of the exterior BEM problem.
//!
//! The exterior BEM model is the closed outer cabinet/baffle surface plus
//! the aperture faces. The aperture elements carry prescribed (outward)
//! normal velocity, every other element is rigid. The DtN map `D(ω)` maps
//! aperture element velocities to element-average aperture pressures:
//!
//! ```text
//! p̄ = D(ω) · v        (m × m dense complex, m = aperture element count)
//! ```
//!
//! # Factorization reuse
//!
//! The stabilized TBEM system matrix is independent of the boundary-condition
//! *values* (they enter only the right-hand side, see
//! `build_tbem_rhs_stabilized`), so per frequency the dense matrix is
//! assembled once, LU-factorized once ([`lu_factorize`]), and each of the
//! `m` unit-velocity right-hand sides is a back-solve
//! ([`LuFactorization::solve`]). The same factorization also serves
//! prescribed-velocity solves for far-field evaluation. This is the direct
//! counterpart of the `solve_forward`/`BemSolverState::Lu` pattern in
//! `sonium-sim-speaker`.
//!
//! # Convention
//!
//! The map lives in the shared `exp(+ikr)` phasor convention documented in
//! [`crate::fem_bem_coupling`]: outward normal velocity `v` and pressure
//! `p̄` are linked by `∂p/∂n = iωρ vₙ`, and outgoing waves satisfy
//! `Re(v* · D · v) ≥ 0` (passivity), verified by the pulsating-sphere test
//! below.

use num_complex::Complex64;
use sonium_bem::core::assembly::tbem::{build_tbem_rhs_stabilized, build_tbem_system_stabilized};
use sonium_bem::core::types::{
    BoundaryCondition, Element as BemElement, Mesh as BemMesh, PhysicsParams,
};
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::direct::{LuFactorization, lu_factorize};

/// Exterior stabilized-assembly formulation sign, matching
/// `sonium-sim-speaker`'s `EXTERIOR_TBEM_SIGN`.
const EXTERIOR_TBEM_SIGN: f64 = 1.0;
const MAX_DTN_INVERSE_CONDITION: f64 = 1.0e12;

/// Errors raised during DtN construction or application.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum DtnError {
    #[error("aperture element list is empty")]
    EmptyAperture,
    #[error("velocity vector length {got} does not match aperture element count {expected}")]
    VelocityLength { got: usize, expected: usize },
    #[error("BEM assembly produced {got} DOFs, expected {expected}")]
    DofMismatch { got: usize, expected: usize },
    #[error("LU factorization failed: {0}")]
    LuFactor(String),
    #[error("LU solve failed: {0}")]
    LuSolve(String),
    #[error("DtN solve produced a non-finite pressure")]
    NonFinitePressure,
    #[error("DtN map inverse is ill-conditioned (estimated condition {condition:.3e})")]
    IllConditioned { condition: f64 },
}

/// Builder for the exterior DtN map: owns the BEM surface, the aperture
/// element list, and the medium.
#[derive(Debug, Clone)]
pub struct DtnBuilder {
    mesh: BemMesh,
    aperture_elements: Vec<usize>,
    speed_of_sound: f64,
    density: f64,
    /// Burton-Miller β scale factor (matches `BemSolver::default`, 0.2).
    beta_scale: f64,
}

impl DtnBuilder {
    /// Create a builder from a BEM mesh and its aperture element indices.
    ///
    /// Every element must already carry `dof_addresses` matching a constant
    /// per-element basis (as produced by
    /// [`crate::fem_bem_coupling::mesh_bridge::build_exterior_bem_mesh`] or
    /// the BEM mesh generators).
    pub fn new(
        mesh: BemMesh,
        aperture_elements: Vec<usize>,
        speed_of_sound: f64,
        density: f64,
    ) -> Result<Self, DtnError> {
        if aperture_elements.is_empty() {
            return Err(DtnError::EmptyAperture);
        }
        Ok(Self {
            mesh,
            aperture_elements,
            speed_of_sound,
            density,
            beta_scale: 0.2,
        })
    }

    /// Override the Burton-Miller β scale factor.
    pub fn with_beta_scale(mut self, beta_scale: f64) -> Self {
        self.beta_scale = beta_scale;
        self
    }

    /// Number of aperture elements (DtN dimension `m`).
    pub fn num_aperture_elements(&self) -> usize {
        self.aperture_elements.len()
    }

    /// Total aperture area.
    pub fn aperture_area(&self) -> f64 {
        self.aperture_elements
            .iter()
            .map(|&i| self.mesh.elements[i].area)
            .sum()
    }

    /// Assemble the TBEM operator at `frequency_hz` and LU-factorize it.
    pub fn factor(&self, frequency_hz: f64) -> Result<DtnFactors, DtnError> {
        let physics = PhysicsParams::new(frequency_hz, self.speed_of_sound, self.density, false);
        let elements = self.mesh.elements.clone();
        let system = build_tbem_system_stabilized(
            &elements,
            &self.mesh.nodes,
            &physics,
            EXTERIOR_TBEM_SIGN,
            self.beta_scale,
        );
        if system.num_dofs != elements.len() {
            return Err(DtnError::DofMismatch {
                got: system.num_dofs,
                expected: elements.len(),
            });
        }
        let lu = lu_factorize(&system.matrix).map_err(|e| DtnError::LuFactor(e.to_string()))?;
        Ok(DtnFactors {
            frequency_hz,
            physics,
            lu,
            elements,
            nodes: self.mesh.nodes.clone(),
            aperture_elements: self.aperture_elements.clone(),
            beta_scale: self.beta_scale,
        })
    }
}

/// Per-frequency state: the LU-factorized TBEM operator, reused across all
/// `m` unit-velocity right-hand sides and any prescribed-velocity solves.
pub struct DtnFactors {
    frequency_hz: f64,
    physics: PhysicsParams,
    lu: LuFactorization<Complex64>,
    elements: Vec<BemElement>,
    nodes: Vec<[f64; 3]>,
    aperture_elements: Vec<usize>,
    beta_scale: f64,
}

impl DtnFactors {
    pub fn frequency_hz(&self) -> f64 {
        self.frequency_hz
    }

    /// Physics parameters at this frequency.
    pub fn physics(&self) -> &PhysicsParams {
        &self.physics
    }

    /// BEM elements (with the rigid/zero-velocity base boundary conditions).
    pub fn elements(&self) -> &[BemElement] {
        &self.elements
    }

    /// BEM nodes.
    pub fn nodes(&self) -> &[[f64; 3]] {
        &self.nodes
    }

    /// Solve the exterior problem for one prescribed aperture velocity
    /// pattern (`velocity[j]` = outward normal velocity on aperture element
    /// `j`, rigid elsewhere).
    ///
    /// Returns `(surface_pressure, surface_velocity)` over all BEM elements,
    /// suitable for far-field evaluation with
    /// `sonium_bem::core::postprocess::pressure::compute_scattered_field`.
    pub fn solve_prescribed_velocity(
        &self,
        velocity: &[Complex64],
    ) -> Result<(Vec<Complex64>, Vec<Complex64>), DtnError> {
        if velocity.len() != self.aperture_elements.len() {
            return Err(DtnError::VelocityLength {
                got: velocity.len(),
                expected: self.aperture_elements.len(),
            });
        }
        let mut elements = self.elements.clone();
        let mut surface_velocity = vec![Complex64::new(0.0, 0.0); elements.len()];
        for (pattern, &elem_idx) in self.aperture_elements.iter().enumerate() {
            elements[elem_idx].boundary_condition =
                BoundaryCondition::Velocity(vec![velocity[pattern]]);
            surface_velocity[elem_idx] = velocity[pattern];
        }
        let rhs = build_tbem_rhs_stabilized(
            &elements,
            &self.nodes,
            &self.physics,
            EXTERIOR_TBEM_SIGN,
            self.beta_scale,
        );
        let pressure = self
            .lu
            .solve(&rhs)
            .map_err(|e| DtnError::LuSolve(e.to_string()))?;
        if pressure.iter().any(|p| !p.is_finite()) {
            return Err(DtnError::NonFinitePressure);
        }
        Ok((pressure, surface_velocity))
    }

    /// Build the DtN map: `m` unit-velocity back-solves against the retained
    /// LU factorization.
    pub fn build_dtn(&self) -> Result<DtnMap, DtnError> {
        let m = self.aperture_elements.len();
        let mut matrix = vec![Complex64::new(0.0, 0.0); m * m];
        for (j, _) in self.aperture_elements.iter().enumerate() {
            let mut unit = vec![Complex64::new(0.0, 0.0); m];
            unit[j] = Complex64::new(1.0, 0.0);
            let (pressure, _) = self.solve_prescribed_velocity(&unit)?;
            for (i, &elem_idx) in self.aperture_elements.iter().enumerate() {
                matrix[i * m + j] = pressure[elem_idx];
            }
        }
        Ok(DtnMap {
            frequency_hz: self.frequency_hz,
            n: m,
            matrix,
        })
    }
}

/// The condensed DtN map `D(ω)`: aperture element velocities →
/// element-average aperture pressures, row-major `m × m`.
#[derive(Debug, Clone)]
pub struct DtnMap {
    pub frequency_hz: f64,
    /// Map dimension `m`.
    pub n: usize,
    /// Row-major dense matrix: `matrix[i * n + j]` is the average pressure
    /// on element `i` from unit outward velocity on element `j`.
    pub matrix: Vec<Complex64>,
}

impl DtnMap {
    /// Apply the map: `p̄ = D v`.
    pub fn apply(&self, velocity: &[Complex64]) -> Vec<Complex64> {
        let mut out = vec![Complex64::new(0.0, 0.0); self.n];
        self.apply_into(velocity, &mut out);
        out
    }

    /// No-allocation map application into a caller-owned buffer (hot path).
    ///
    /// Rows are independent, so large maps are evaluated with Rayon lanes;
    /// `velocity` and `out` must both have length `n` (checked in debug
    /// builds, truncated in release builds by the row slicing).
    pub fn apply_into(&self, velocity: &[Complex64], out: &mut [Complex64]) {
        use rayon::prelude::*;
        let n = self.n;
        debug_assert_eq!(velocity.len(), n);
        debug_assert_eq!(out.len(), n);
        if n >= 128 {
            out.par_iter_mut().enumerate().for_each(|(i, out_i)| {
                let row = &self.matrix[i * n..(i + 1) * n];
                *out_i = row
                    .iter()
                    .zip(velocity.iter())
                    .map(|(d, v)| *d * *v)
                    .sum();
            });
        } else {
            for (i, out_i) in out.iter_mut().enumerate() {
                let row = &self.matrix[i * n..(i + 1) * n];
                *out_i = row
                    .iter()
                    .zip(velocity.iter())
                    .map(|(d, v)| *d * *v)
                    .sum();
            }
        }
    }

    /// Time-averaged power crossing the aperture into the exterior,
    /// `Re(Σ_j conj(v_j) · (D v)_j · area_j)`, in watts for the given
    /// element areas. Non-negative for a passive exterior.
    pub fn radiated_power(&self, velocity: &[Complex64], element_areas: &[f64]) -> f64 {
        let pressure = self.apply(velocity);
        velocity
            .iter()
            .zip(pressure.iter())
            .zip(element_areas.iter())
            .map(|((v, p), a)| (v.conj() * p).re * a)
            .sum()
    }

    /// Explicit inverse `G = D⁻¹` (mobility map), computed by LU with
    /// partial pivoting on the `m × m` map.
    pub fn inverse(&self) -> Result<Vec<Complex64>, DtnError> {
        let n = self.n;
        if self.matrix.len() != n.saturating_mul(n) || n == 0 {
            return Err(DtnError::EmptyAperture);
        }
        let matrix_norm = (0..n)
            .map(|row| {
                self.matrix[row * n..(row + 1) * n]
                    .iter()
                    .map(|value| value.norm())
                    .sum::<f64>()
            })
            .fold(0.0, f64::max);
        if !matrix_norm.is_finite() || matrix_norm == 0.0 {
            return Err(DtnError::IllConditioned {
                condition: f64::INFINITY,
            });
        }
        let mut dense = DenseMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                dense[[i, j]] = self.matrix[i * n + j];
            }
        }
        let lu = lu_factorize(&dense).map_err(|e| DtnError::LuFactor(e.to_string()))?;
        let mut inv = vec![Complex64::new(0.0, 0.0); n * n];
        for col in 0..n {
            let mut e = vec![Complex64::new(0.0, 0.0); n];
            e[col] = Complex64::new(1.0, 0.0);
            let x = lu
                .solve(&e)
                .map_err(|e2| DtnError::LuSolve(e2.to_string()))?;
            for (row, value) in x.iter().enumerate() {
                inv[row * n + col] = *value;
            }
        }
        let inverse_norm = (0..n)
            .map(|row| {
                inv[row * n..(row + 1) * n]
                    .iter()
                    .map(|value| value.norm())
                    .sum::<f64>()
            })
            .fold(0.0, f64::max);
        let condition = matrix_norm * inverse_norm;
        if !condition.is_finite() || condition > MAX_DTN_INVERSE_CONDITION {
            return Err(DtnError::IllConditioned { condition });
        }
        Ok(inv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_bem::core::mesh::generators::generate_icosphere_mesh;

    /// Sign/passivity validation: the DtN of a uniformly pulsating sphere
    /// must reproduce the analytical radiation impedance (same `exp(+ikr)`
    /// convention as `sonium-wave`) and absorb non-negative power.
    #[test]
    fn pulsating_sphere_dtn_matches_analytical_radiation_impedance() {
        let radius = 0.05;
        let c = 343.0;
        let rho = 1.204;
        // ka = 1: k = 20 1/m, f = k c / 2π ≈ 1092 Hz.
        let k = 20.0;
        let frequency = k * c / (2.0 * std::f64::consts::PI);

        let mesh = generate_icosphere_mesh(radius, 2);
        let aperture: Vec<usize> = (0..mesh.elements.len()).collect();
        let builder = DtnBuilder::new(mesh, aperture, c, rho).unwrap();
        let factors = builder.factor(frequency).unwrap();
        let dtn = factors.build_dtn().unwrap();

        // Uniform unit outward velocity.
        let v = vec![Complex64::new(1.0, 0.0); dtn.n];
        let p = dtn.apply(&v);
        let p_avg = p.iter().sum::<Complex64>() / dtn.n as f64;

        let analytical =
            sonium_wave::analytical::pulsating_sphere_3d(k, radius, 1.0, c, rho, vec![radius]);
        let p_exact = analytical.pressure[0];

        let rel = (p_avg - p_exact).norm() / p_exact.norm();
        println!("pulsating sphere DtN: p_avg = {p_avg}, exact = {p_exact}, rel = {rel:.3e}");
        assert!(rel < 0.10, "DtN pressure off by {rel:.3e}");

        // Passivity: Re(Z) ≥ 0 for the radiation load.
        assert!(
            p_avg.re > 0.0,
            "radiation resistance must be positive, got {}",
            p_avg.re
        );
        // Reactance sign: mass-like (negative imaginary part in the
        // exp(+ikr) convention, cf. conj of electroacoustics_pulsating_sphere_impedance).
        assert!(
            p_avg.im < 0.0,
            "radiation reactance must be mass-like, got {}",
            p_avg.im
        );

        // Power form: Re(v* D v) > 0 for a non-uniform pattern too.
        let mut v2 = v.clone();
        for (i, vi) in v2.iter_mut().enumerate() {
            *vi = Complex64::new(1.0 + 0.1 * (i % 7) as f64, 0.0);
        }
        let areas: Vec<f64> = {
            // The builder consumed the mesh; recover uniform area weighting
            // from the map dimension (icosphere elements are near-uniform).
            vec![1.0; dtn.n]
        };
        let power = dtn.radiated_power(&v2, &areas);
        assert!(power > 0.0, "radiated power must be positive, got {power}");
    }

    #[test]
    fn inverse_rejects_rank_deficient_dtn_map() {
        let map = DtnMap {
            frequency_hz: 100.0,
            n: 2,
            matrix: vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(4.0, 0.0),
            ],
        };
        assert!(map.inverse().is_err());
    }
}
