//! Planar magnetic materials for rotating-machine `A_z` problems.
//!
//! Three region kinds share one constitutive interface `H(B)` with a
//! consistent tangent `dH/dB`: linear media, nonlinear iron on a measured
//! B-H curve (reusing [`MagneticConstitutiveLaw`]), and permanent magnets
//! on a recoil line.
//!
//! The magnet law is `B = mu_0 * mu_rec * H + B_r`, i.e.
//! `H = nu * B - H_c` with `nu = 1 / (mu_0 * mu_rec)` and coercivity
//! `H_c = B_r / (mu_0 * mu_rec)`. Assembling `M = B_r / mu_0` instead of
//! `H_c` models a magnet of remanence `mu_rec * B_r` (about 5 % strong
//! for NdFeB); this type only ever exposes `H_c` so that error cannot be
//! made at a call site.

// Rust guideline compliant 2026-02-21

use super::super::magnetostatics::{BhCurve, MU_0, MagneticConstitutiveLaw};

/// Permanent-magnet recoil-line law in the cross-section plane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MagnetLaw {
    /// Recoil relative permeability (NdFeB is typically 1.05).
    pub recoil_mu_r: f64,
    /// Remanent flux-density vector `[B_rx, B_ry]` in tesla, at full strength.
    pub remanence_t: [f64; 2],
    /// Irreversible-demagnetisation factor in `[0, 1]`; 1 is undamaged.
    pub demag_factor: f64,
}

impl MagnetLaw {
    /// Creates an undamaged magnet law.
    ///
    /// # Errors
    ///
    /// Returns [`AzMaterialError`] when a value is non-finite, the recoil
    /// permeability is not positive, or the remanence is zero.
    pub fn new(recoil_mu_r: f64, remanence_t: [f64; 2]) -> Result<Self, AzMaterialError> {
        let law = Self {
            recoil_mu_r,
            remanence_t,
            demag_factor: 1.0,
        };
        law.validate()?;
        Ok(law)
    }

    /// Validates every field.
    ///
    /// # Errors
    ///
    /// Returns [`AzMaterialError`] describing the first invalid field.
    pub fn validate(&self) -> Result<(), AzMaterialError> {
        if !self.recoil_mu_r.is_finite() || self.recoil_mu_r <= 0.0 {
            return Err(AzMaterialError::InvalidRecoilPermeability);
        }
        if !self.remanence_t.iter().all(|value| value.is_finite())
            || self.remanence_t.iter().all(|value| *value == 0.0)
        {
            return Err(AzMaterialError::InvalidRemanence);
        }
        if !self.demag_factor.is_finite() || !(0.0..=1.0).contains(&self.demag_factor) {
            return Err(AzMaterialError::InvalidDemagFactor);
        }
        Ok(())
    }

    /// Reluctivity `1 / (mu_0 * mu_rec)` in m/H.
    pub fn reluctivity(&self) -> f64 {
        1.0 / (MU_0 * self.recoil_mu_r)
    }

    /// Coercivity vector `H_c = demag * B_r / (mu_0 * mu_rec)` in A/m.
    pub fn coercivity_a_per_m(&self) -> [f64; 2] {
        let scale = self.demag_factor / (MU_0 * self.recoil_mu_r);
        [scale * self.remanence_t[0], scale * self.remanence_t[1]]
    }

    /// Effective remanence after demagnetisation, in tesla.
    pub fn effective_remanence_t(&self) -> [f64; 2] {
        [
            self.demag_factor * self.remanence_t[0],
            self.demag_factor * self.remanence_t[1],
        ]
    }
}

/// One planar region material for an `A_z` problem.
#[derive(Debug, Clone, PartialEq)]
pub struct AzMaterial {
    law: AzLaw,
    conductivity_s_per_m: f64,
}

#[derive(Debug, Clone, PartialEq)]
enum AzLaw {
    Soft(MagneticConstitutiveLaw),
    Magnet(MagnetLaw),
}

impl AzMaterial {
    /// Air or vacuum (`mu_r = 1`, non-conducting).
    pub fn air() -> Self {
        Self::linear_mu_r(1.0).expect("mu_r = 1 is valid")
    }

    /// Linear soft material from a relative permeability.
    ///
    /// # Errors
    ///
    /// Returns [`AzMaterialError::InvalidPermeability`] for non-positive
    /// or non-finite `mu_r`.
    pub fn linear_mu_r(mu_r: f64) -> Result<Self, AzMaterialError> {
        if !mu_r.is_finite() || mu_r <= 0.0 {
            return Err(AzMaterialError::InvalidPermeability);
        }
        Ok(Self {
            law: AzLaw::Soft(
                MagneticConstitutiveLaw::linear(mu_r * MU_0).expect("positive finite permeability"),
            ),
            conductivity_s_per_m: 0.0,
        })
    }

    /// Linear soft material from an absolute permeability in H/m.
    ///
    /// # Errors
    ///
    /// Returns [`AzMaterialError::InvalidPermeability`] for non-positive
    /// or non-finite permeability.
    pub fn linear_permeability(permeability_h_per_m: f64) -> Result<Self, AzMaterialError> {
        Ok(Self {
            law: AzLaw::Soft(MagneticConstitutiveLaw::linear(permeability_h_per_m)?),
            conductivity_s_per_m: 0.0,
        })
    }

    /// Nonlinear iron on a first-quadrant B-H curve.
    pub fn nonlinear(curve: BhCurve) -> Self {
        Self {
            law: AzLaw::Soft(MagneticConstitutiveLaw::Nonlinear(curve)),
            conductivity_s_per_m: 0.0,
        }
    }

    /// Permanent magnet on its recoil line.
    ///
    /// # Errors
    ///
    /// Propagates [`MagnetLaw::validate`] failures.
    pub fn magnet(law: MagnetLaw) -> Result<Self, AzMaterialError> {
        law.validate()?;
        Ok(Self {
            law: AzLaw::Magnet(law),
            conductivity_s_per_m: 0.0,
        })
    }

    /// Sets the electrical conductivity used by transient eddy terms.
    ///
    /// # Errors
    ///
    /// Returns [`AzMaterialError::InvalidConductivity`] for negative or
    /// non-finite conductivity.
    pub fn with_conductivity(mut self, conductivity_s_per_m: f64) -> Result<Self, AzMaterialError> {
        if !conductivity_s_per_m.is_finite() || conductivity_s_per_m < 0.0 {
            return Err(AzMaterialError::InvalidConductivity);
        }
        self.conductivity_s_per_m = conductivity_s_per_m;
        Ok(self)
    }

    /// Electrical conductivity in S/m (zero unless set).
    pub fn conductivity_s_per_m(&self) -> f64 {
        self.conductivity_s_per_m
    }

    /// The magnet law, when this region is a permanent magnet.
    pub fn magnet_law(&self) -> Option<&MagnetLaw> {
        match &self.law {
            AzLaw::Magnet(law) => Some(law),
            AzLaw::Soft(_) => None,
        }
    }

    /// Whether `H(B)` is affine, so one linear solve is exact.
    pub fn is_linear(&self) -> bool {
        match &self.law {
            AzLaw::Magnet(_) => true,
            AzLaw::Soft(MagneticConstitutiveLaw::Linear { .. }) => true,
            AzLaw::Soft(MagneticConstitutiveLaw::Nonlinear(_)) => false,
        }
    }

    /// Secant reluctivity `nu(|B|)` in m/H.
    pub fn reluctivity(&self, flux_density_t: f64) -> f64 {
        match &self.law {
            AzLaw::Magnet(law) => law.reluctivity(),
            AzLaw::Soft(law) => law.reluctivity(flux_density_t),
        }
    }

    /// Differential reluctivity `dH/dB` in m/H.
    pub fn differential_reluctivity(&self, flux_density_t: f64) -> f64 {
        match &self.law {
            AzLaw::Magnet(law) => law.reluctivity(),
            AzLaw::Soft(law) => law.differential_reluctivity(flux_density_t),
        }
    }

    /// Field strength `H(B)` and its consistent tangent `dH/dB`.
    ///
    /// For isotropic soft media the tangent is
    /// `nu * I + (nu_d - nu) * (B x B) / |B|^2`. For magnets it is `nu * I`
    /// and `H` additionally carries `-load_factor * H_c`.
    pub fn h_and_tangent(&self, b: [f64; 2], load_factor: f64) -> ([f64; 2], [[f64; 2]; 2]) {
        match &self.law {
            AzLaw::Magnet(law) => {
                let nu = law.reluctivity();
                let hc = law.coercivity_a_per_m();
                (
                    [
                        nu * b[0] - load_factor * hc[0],
                        nu * b[1] - load_factor * hc[1],
                    ],
                    [[nu, 0.0], [0.0, nu]],
                )
            }
            AzLaw::Soft(law) => {
                let magnitude = b[0].hypot(b[1]);
                let nu = law.reluctivity(magnitude);
                let differential = law.differential_reluctivity(magnitude);
                let h = [nu * b[0], nu * b[1]];
                // Below this magnitude the radial direction is undefined and
                // the tangent degenerates to the differential reluctivity.
                if magnitude <= 1.0e-14 {
                    return (h, [[differential, 0.0], [0.0, differential]]);
                }
                let radial = [b[0] / magnitude, b[1] / magnitude];
                let correction = differential - nu;
                (
                    h,
                    [
                        [
                            nu + correction * radial[0] * radial[0],
                            correction * radial[0] * radial[1],
                        ],
                        [
                            correction * radial[1] * radial[0],
                            nu + correction * radial[1] * radial[1],
                        ],
                    ],
                )
            }
        }
    }

    /// Magnetic energy density `integral H dB` in J/m^3.
    ///
    /// For a magnet on its recoil line this is
    /// `0.5 * nu * |B|^2 - H_c . B` (up to an irrelevant constant).
    pub fn energy_density_j_per_m3(&self, b: [f64; 2]) -> f64 {
        match &self.law {
            AzLaw::Magnet(law) => {
                let hc = law.coercivity_a_per_m();
                0.5 * law.reluctivity() * (b[0] * b[0] + b[1] * b[1]) - hc[0] * b[0] - hc[1] * b[1]
            }
            AzLaw::Soft(law) => law.energy_density(b[0].hypot(b[1])),
        }
    }

    /// Magnetic coenergy density `B . H - energy` in J/m^3.
    pub fn coenergy_density_j_per_m3(&self, b: [f64; 2]) -> f64 {
        let (h, _) = self.h_and_tangent(b, 1.0);
        b[0] * h[0] + b[1] * h[1] - self.energy_density_j_per_m3(b)
    }
}

/// Invalid planar magnetic material data.
#[derive(Debug, thiserror::Error)]
pub enum AzMaterialError {
    /// Permeability was non-positive or non-finite.
    #[error("magnetic permeability must be finite and positive")]
    InvalidPermeability,
    /// Recoil permeability was non-positive or non-finite.
    #[error("recoil permeability must be finite and positive")]
    InvalidRecoilPermeability,
    /// Remanence was zero or non-finite.
    #[error("remanence must be finite and non-zero")]
    InvalidRemanence,
    /// Demagnetisation factor lay outside `[0, 1]`.
    #[error("demagnetisation factor must lie in [0, 1]")]
    InvalidDemagFactor,
    /// Conductivity was negative or non-finite.
    #[error("conductivity must be finite and non-negative")]
    InvalidConductivity,
    /// A soft-material law rejected its input.
    #[error(transparent)]
    Law(#[from] super::super::magnetostatics::NonlinearMagneticError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magnet_coercivity_uses_recoil_permeability() {
        let law = MagnetLaw::new(1.05, [0.0, 1.26]).unwrap();
        let hc = law.coercivity_a_per_m();
        let expected = 1.26 / (MU_0 * 1.05);
        assert!((hc[1] - expected).abs() / expected < 1.0e-12);
        // The M-convention would give a value 5 % larger; guard against it.
        assert!((hc[1] - 1.26 / MU_0).abs() / expected > 0.04);
    }

    #[test]
    fn magnet_energy_and_coenergy_are_conjugate() {
        let material = AzMaterial::magnet(MagnetLaw::new(1.05, [0.3, 1.2]).unwrap()).unwrap();
        let b = [0.2, -0.7];
        let (h, _) = material.h_and_tangent(b, 1.0);
        let sum = material.energy_density_j_per_m3(b) + material.coenergy_density_j_per_m3(b);
        assert!((sum - (b[0] * h[0] + b[1] * h[1])).abs() < 1.0e-6);
    }

    #[test]
    fn invalid_materials_are_rejected() {
        assert!(AzMaterial::linear_mu_r(0.0).is_err());
        assert!(MagnetLaw::new(1.05, [0.0, 0.0]).is_err());
        assert!(AzMaterial::air().with_conductivity(-1.0).is_err());
    }
}
