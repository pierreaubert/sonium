//! Frequency-dependent viscoelastic complex-modulus models.
//!
//! Production loudspeaker suspensions (surround, spider) are not
//! frequency-independent `1 + i·η` materials: their storage modulus `E′(ω)`
//! rises and their loss modulus `E″(ω)` peaks across the audio band. This
//! module provides one fail-closed contract for that behaviour:
//!
//! * [`TabulatedComplexModulus`] — measured `(f, E′, E″)` points with
//!   log-frequency interpolation and **no extrapolation**;
//! * [`PronyComplexModulus`] — a generalized-Maxwell (Prony) series
//!   `E*(ω) = E∞ + Σᵢ Eᵢ·(iωτᵢ)/(1 + iωτᵢ)`;
//! * [`ConstantLossModulus`] — the legacy frequency-independent limit.
//!
//! All types implement [`ComplexModulusModel`], are `Debug`, validate at
//! construction, and return [`ViscoelasticError`] instead of panicking on
//! bad input (panics are reserved for programmer bugs). Evaluation is
//! allocation-free and branch-light, so it is safe on assembly hot paths:
//! one binary search (tabulated) or one pass over the Prony branches.

use num_complex::Complex64;

/// Errors for frequency-dependent viscoelastic models.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ViscoelasticError {
    /// Frequency was non-finite or non-positive.
    #[error("frequency must be finite and positive, got {0} Hz")]
    InvalidFrequency(f64),
    /// Tabulated query fell outside the measured frequency range.
    #[error(
        "frequency {frequency_hz} Hz is outside the tabulated range [{min_hz}, {max_hz}] Hz; extrapolation is refused"
    )]
    OutsideTabulatedRange {
        /// Queried frequency in Hz.
        frequency_hz: f64,
        /// Lowest tabulated frequency in Hz.
        min_hz: f64,
        /// Highest tabulated frequency in Hz.
        max_hz: f64,
    },
    /// A tabulated or Prony parameter failed validation.
    #[error("invalid viscoelastic parameter {field}: {detail}")]
    InvalidParameter {
        /// Offending field name.
        field: &'static str,
        /// Human-readable detail.
        detail: String,
    },
}

/// Frequency-dependent complex Young's modulus `E*(f) = E′ + i·E″` in Pa.
///
/// Implementations must be passive (`E″ ≥ 0`) and fail closed outside
/// their domain of validity.
pub trait ComplexModulusModel: std::fmt::Debug + Send + Sync {
    /// Complex Young's modulus at `frequency_hz` (RMS phasor, `exp(+iωt)`).
    fn complex_youngs_modulus_pa(&self, frequency_hz: f64) -> Result<Complex64, ViscoelasticError>;

    /// Loss factor `η(f) = E″/E′` at `frequency_hz`.
    fn loss_factor(&self, frequency_hz: f64) -> Result<f64, ViscoelasticError> {
        let modulus = self.complex_youngs_modulus_pa(frequency_hz)?;
        Ok(modulus.im / modulus.re)
    }

    /// Dimensionless stiffness factor relative to a real reference modulus,
    /// for scaling an already-assembled elastic stiffness matrix.
    fn stiffness_factor(
        &self,
        frequency_hz: f64,
        reference_modulus_pa: f64,
    ) -> Result<Complex64, ViscoelasticError> {
        if !reference_modulus_pa.is_finite() || reference_modulus_pa <= 0.0 {
            return Err(ViscoelasticError::InvalidParameter {
                field: "reference_modulus_pa",
                detail: "must be finite and positive".to_string(),
            });
        }
        Ok(self.complex_youngs_modulus_pa(frequency_hz)? / reference_modulus_pa)
    }
}

fn check_frequency(frequency_hz: f64) -> Result<f64, ViscoelasticError> {
    if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        return Err(ViscoelasticError::InvalidFrequency(frequency_hz));
    }
    Ok(frequency_hz)
}

fn check_poisson(poisson_ratio: f64) -> Result<(), ViscoelasticError> {
    if !(-1.0..0.5).contains(&poisson_ratio) {
        return Err(ViscoelasticError::InvalidParameter {
            field: "poisson_ratio",
            detail: format!("must lie in (-1, 0.5), got {poisson_ratio}"),
        });
    }
    Ok(())
}

/// Frequency-independent complex modulus `E·(1 + i·η)` (legacy limit).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstantLossModulus {
    /// Real Young's modulus in Pa.
    pub youngs_modulus_pa: f64,
    /// Constant loss factor `η ≥ 0`.
    pub loss_factor: f64,
    /// Poisson ratio.
    pub poisson_ratio: f64,
}

impl ConstantLossModulus {
    /// Build and validate a constant-loss modulus.
    pub fn new(
        youngs_modulus_pa: f64,
        loss_factor: f64,
        poisson_ratio: f64,
    ) -> Result<Self, ViscoelasticError> {
        if !youngs_modulus_pa.is_finite() || youngs_modulus_pa <= 0.0 {
            return Err(ViscoelasticError::InvalidParameter {
                field: "youngs_modulus_pa",
                detail: "must be finite and positive".to_string(),
            });
        }
        if !loss_factor.is_finite() || loss_factor < 0.0 {
            return Err(ViscoelasticError::InvalidParameter {
                field: "loss_factor",
                detail: "must be finite and non-negative".to_string(),
            });
        }
        check_poisson(poisson_ratio)?;
        Ok(Self {
            youngs_modulus_pa,
            loss_factor,
            poisson_ratio,
        })
    }
}

impl ComplexModulusModel for ConstantLossModulus {
    fn complex_youngs_modulus_pa(&self, frequency_hz: f64) -> Result<Complex64, ViscoelasticError> {
        check_frequency(frequency_hz)?;
        Ok(Complex64::new(
            self.youngs_modulus_pa,
            self.youngs_modulus_pa * self.loss_factor,
        ))
    }
}

/// One tabulated point: frequency in Hz, storage modulus `E′` and loss
/// modulus `E″` in Pa.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModulusPoint {
    /// Frequency in Hz.
    pub frequency_hz: f64,
    /// Storage modulus `E′` in Pa.
    pub storage_pa: f64,
    /// Loss modulus `E″` in Pa (`≥ 0` for a passive material).
    pub loss_pa: f64,
}

/// Measured tabulated complex modulus with log-frequency interpolation.
///
/// Frequencies must be strictly increasing, `E′ > 0`, and `E″ ≥ 0`.
/// Queries outside the tabulated range fail closed
/// ([`ViscoelasticError::OutsideTabulatedRange`]); no extrapolation,
/// clamping, or silent nearest-point fallback is performed.
#[derive(Debug, Clone, PartialEq)]
pub struct TabulatedComplexModulus {
    points: Vec<ModulusPoint>,
    /// Poisson ratio (frequency-independent in this model).
    pub poisson_ratio: f64,
}

impl TabulatedComplexModulus {
    /// Build and validate a tabulated modulus (at least two points).
    pub fn new(points: Vec<ModulusPoint>, poisson_ratio: f64) -> Result<Self, ViscoelasticError> {
        check_poisson(poisson_ratio)?;
        if points.len() < 2 {
            return Err(ViscoelasticError::InvalidParameter {
                field: "points",
                detail: "at least two frequency points are required".to_string(),
            });
        }
        for (index, point) in points.iter().enumerate() {
            if !point.frequency_hz.is_finite() || point.frequency_hz <= 0.0 {
                return Err(ViscoelasticError::InvalidParameter {
                    field: "points.frequency_hz",
                    detail: format!("point {index} must be finite and positive"),
                });
            }
            if !point.storage_pa.is_finite() || point.storage_pa <= 0.0 {
                return Err(ViscoelasticError::InvalidParameter {
                    field: "points.storage_pa",
                    detail: format!("point {index} storage modulus must be finite and positive"),
                });
            }
            if !point.loss_pa.is_finite() || point.loss_pa < 0.0 {
                return Err(ViscoelasticError::InvalidParameter {
                    field: "points.loss_pa",
                    detail: format!("point {index} loss modulus must be finite and non-negative"),
                });
            }
            if index > 0 && point.frequency_hz <= points[index - 1].frequency_hz {
                return Err(ViscoelasticError::InvalidParameter {
                    field: "points.frequency_hz",
                    detail: "frequencies must be strictly increasing".to_string(),
                });
            }
        }
        Ok(Self {
            points,
            poisson_ratio,
        })
    }

    /// Tabulated points in increasing-frequency order.
    pub fn points(&self) -> &[ModulusPoint] {
        &self.points
    }

    /// Covered frequency range `(min_hz, max_hz)`.
    pub fn frequency_range_hz(&self) -> (f64, f64) {
        (
            self.points
                .first()
                .expect("validated non-empty")
                .frequency_hz,
            self.points
                .last()
                .expect("validated non-empty")
                .frequency_hz,
        )
    }
}

impl ComplexModulusModel for TabulatedComplexModulus {
    fn complex_youngs_modulus_pa(&self, frequency_hz: f64) -> Result<Complex64, ViscoelasticError> {
        check_frequency(frequency_hz)?;
        let (min_hz, max_hz) = self.frequency_range_hz();
        if frequency_hz < min_hz || frequency_hz > max_hz {
            return Err(ViscoelasticError::OutsideTabulatedRange {
                frequency_hz,
                min_hz,
                max_hz,
            });
        }
        let upper = self
            .points
            .partition_point(|point| point.frequency_hz <= frequency_hz);
        if upper == 0 {
            let point = self.points[0];
            return Ok(Complex64::new(point.storage_pa, point.loss_pa));
        }
        if upper >= self.points.len() {
            let point = self.points[self.points.len() - 1];
            return Ok(Complex64::new(point.storage_pa, point.loss_pa));
        }
        let lo = self.points[upper - 1];
        let hi = self.points[upper];
        if (frequency_hz - lo.frequency_hz).abs() <= f64::EPSILON * lo.frequency_hz.max(1.0) {
            return Ok(Complex64::new(lo.storage_pa, lo.loss_pa));
        }
        // Log-frequency interpolation; log-log for E′ (strictly positive)
        // and log-frequency/linear for E″ (may legitimately touch zero).
        let t = (frequency_hz.ln() - lo.frequency_hz.ln())
            / (hi.frequency_hz.ln() - lo.frequency_hz.ln());
        let storage = (lo.storage_pa.ln() + t * (hi.storage_pa.ln() - lo.storage_pa.ln())).exp();
        let loss = lo.loss_pa + t * (hi.loss_pa - lo.loss_pa);
        Ok(Complex64::new(storage, loss))
    }
}

/// One Prony (generalized-Maxwell) branch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PronyBranch {
    /// Branch modulus `Eᵢ` in Pa (`≥ 0`).
    pub modulus_pa: f64,
    /// Relaxation time `τᵢ` in seconds (`> 0`).
    pub relaxation_time_s: f64,
}

/// Prony-series complex modulus
/// `E*(ω) = E∞ + Σᵢ Eᵢ·(iωτᵢ)/(1 + iωτᵢ)`.
///
/// Low-frequency limit `E∞` (rubbery plateau), high-frequency limit
/// `E∞ + Σᵢ Eᵢ` (glassy plateau); passive for non-negative branch moduli.
#[derive(Debug, Clone, PartialEq)]
pub struct PronyComplexModulus {
    /// Long-term (equilibrium) modulus `E∞` in Pa.
    pub equilibrium_modulus_pa: f64,
    branches: Vec<PronyBranch>,
    /// Poisson ratio.
    pub poisson_ratio: f64,
}

impl PronyComplexModulus {
    /// Build and validate a Prony modulus (at least one branch).
    pub fn new(
        equilibrium_modulus_pa: f64,
        branches: Vec<PronyBranch>,
        poisson_ratio: f64,
    ) -> Result<Self, ViscoelasticError> {
        check_poisson(poisson_ratio)?;
        if !equilibrium_modulus_pa.is_finite() || equilibrium_modulus_pa <= 0.0 {
            return Err(ViscoelasticError::InvalidParameter {
                field: "equilibrium_modulus_pa",
                detail: "must be finite and positive".to_string(),
            });
        }
        if branches.is_empty() {
            return Err(ViscoelasticError::InvalidParameter {
                field: "branches",
                detail: "at least one Prony branch is required".to_string(),
            });
        }
        for (index, branch) in branches.iter().enumerate() {
            if !branch.modulus_pa.is_finite() || branch.modulus_pa < 0.0 {
                return Err(ViscoelasticError::InvalidParameter {
                    field: "branches.modulus_pa",
                    detail: format!("branch {index} modulus must be finite and non-negative"),
                });
            }
            if !branch.relaxation_time_s.is_finite() || branch.relaxation_time_s <= 0.0 {
                return Err(ViscoelasticError::InvalidParameter {
                    field: "branches.relaxation_time_s",
                    detail: format!("branch {index} relaxation time must be finite and positive"),
                });
            }
        }
        Ok(Self {
            equilibrium_modulus_pa,
            branches,
            poisson_ratio,
        })
    }

    /// Prony branches.
    pub fn branches(&self) -> &[PronyBranch] {
        &self.branches
    }

    /// High-frequency (glassy) modulus `E∞ + Σᵢ Eᵢ` in Pa.
    pub fn glassy_modulus_pa(&self) -> f64 {
        self.equilibrium_modulus_pa
            + self
                .branches
                .iter()
                .map(|branch| branch.modulus_pa)
                .sum::<f64>()
    }
}

impl ComplexModulusModel for PronyComplexModulus {
    fn complex_youngs_modulus_pa(&self, frequency_hz: f64) -> Result<Complex64, ViscoelasticError> {
        check_frequency(frequency_hz)?;
        let omega = 2.0 * std::f64::consts::PI * frequency_hz;
        let mut modulus = Complex64::new(self.equilibrium_modulus_pa, 0.0);
        for branch in &self.branches {
            let x = omega * branch.relaxation_time_s;
            // Eᵢ·(i x)/(1 + i x) = Eᵢ·(x² + i x)/(1 + x²).
            let denom = 1.0 + x * x;
            modulus += Complex64::new(
                branch.modulus_pa * x * x / denom,
                branch.modulus_pa * x / denom,
            );
        }
        Ok(modulus)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tabulated() -> TabulatedComplexModulus {
        TabulatedComplexModulus::new(
            vec![
                ModulusPoint {
                    frequency_hz: 20.0,
                    storage_pa: 2.0e6,
                    loss_pa: 0.6e6,
                },
                ModulusPoint {
                    frequency_hz: 200.0,
                    storage_pa: 4.0e6,
                    loss_pa: 1.6e6,
                },
                ModulusPoint {
                    frequency_hz: 2_000.0,
                    storage_pa: 8.0e6,
                    loss_pa: 2.4e6,
                },
            ],
            0.45,
        )
        .unwrap()
    }

    #[test]
    fn tabulated_hits_nodes_and_interpolates_in_log_frequency() {
        let model = tabulated();
        let at_node = model.complex_youngs_modulus_pa(200.0).unwrap();
        assert!((at_node.re - 4.0e6).abs() < 1.0);
        assert!((at_node.im - 1.6e6).abs() < 1.0);
        // Geometric midpoint of 20..200 Hz: E′ is the geometric mean.
        let mid_hz = (20.0_f64 * 200.0).sqrt();
        let mid = model.complex_youngs_modulus_pa(mid_hz).unwrap();
        assert!((mid.re - (2.0e6_f64 * 4.0e6).sqrt()).abs() / mid.re < 1.0e-12);
        assert!(mid.im > 0.6e6 && mid.im < 1.6e6);
        assert!(model.loss_factor(200.0).unwrap() > 0.0);
    }

    #[test]
    fn tabulated_fails_closed_outside_range_and_on_bad_tables() {
        let model = tabulated();
        assert!(matches!(
            model.complex_youngs_modulus_pa(10.0),
            Err(ViscoelasticError::OutsideTabulatedRange { .. })
        ));
        assert!(matches!(
            model.complex_youngs_modulus_pa(20_000.0),
            Err(ViscoelasticError::OutsideTabulatedRange { .. })
        ));
        assert!(
            TabulatedComplexModulus::new(
                vec![
                    ModulusPoint {
                        frequency_hz: 100.0,
                        storage_pa: 1.0e6,
                        loss_pa: 0.0
                    },
                    ModulusPoint {
                        frequency_hz: 100.0,
                        storage_pa: 1.0e6,
                        loss_pa: 0.0
                    },
                ],
                0.4,
            )
            .is_err()
        );
        assert!(
            TabulatedComplexModulus::new(
                vec![
                    ModulusPoint {
                        frequency_hz: 100.0,
                        storage_pa: 1.0e6,
                        loss_pa: -1.0
                    },
                    ModulusPoint {
                        frequency_hz: 200.0,
                        storage_pa: 1.0e6,
                        loss_pa: 0.0
                    },
                ],
                0.4,
            )
            .is_err()
        );
    }

    #[test]
    fn prony_recovers_plateaus_and_is_passive() {
        let model = PronyComplexModulus::new(
            2.0e6,
            vec![
                PronyBranch {
                    modulus_pa: 3.0e6,
                    relaxation_time_s: 1.0e-3,
                },
                PronyBranch {
                    modulus_pa: 5.0e6,
                    relaxation_time_s: 1.0e-5,
                },
            ],
            0.45,
        )
        .unwrap();
        let low = model.complex_youngs_modulus_pa(1.0).unwrap();
        assert!((low.re - 2.0e6).abs() / 2.0e6 < 0.01);
        let high = model.complex_youngs_modulus_pa(1.0e6).unwrap();
        assert!((high.re - model.glassy_modulus_pa()).abs() / high.re < 0.01);
        for frequency in [10.0, 100.0, 1_000.0, 10_000.0] {
            let modulus = model.complex_youngs_modulus_pa(frequency).unwrap();
            assert!(modulus.im >= 0.0, "passive loss at {frequency} Hz");
            assert!(modulus.re >= 2.0e6 && modulus.re <= model.glassy_modulus_pa());
        }
        assert!(PronyComplexModulus::new(1.0e6, vec![], 0.4).is_err());
        assert!(
            PronyComplexModulus::new(
                1.0e6,
                vec![PronyBranch {
                    modulus_pa: 1.0e6,
                    relaxation_time_s: 0.0
                }],
                0.4,
            )
            .is_err()
        );
    }

    #[test]
    fn constant_loss_matches_legacy_factor() {
        let model = ConstantLossModulus::new(5.0e6, 0.4, 0.45).unwrap();
        let modulus = model.complex_youngs_modulus_pa(440.0).unwrap();
        assert_eq!(modulus, Complex64::new(5.0e6, 2.0e6));
        assert!(
            (model.stiffness_factor(440.0, 5.0e6).unwrap() - Complex64::new(1.0, 0.4)).norm()
                < 1e-15
        );
        assert!(ConstantLossModulus::new(5.0e6, -0.1, 0.45).is_err());
        assert!(model.complex_youngs_modulus_pa(0.0).is_err());
    }

    #[test]
    fn unlisted_synonym_free_variant_still_fails_closed() {
        // A two-point table queried far above its band must not silently
        // clamp to the last measured point.
        let model = TabulatedComplexModulus::new(
            vec![
                ModulusPoint {
                    frequency_hz: 50.0,
                    storage_pa: 1.0e6,
                    loss_pa: 0.2e6,
                },
                ModulusPoint {
                    frequency_hz: 500.0,
                    storage_pa: 2.0e6,
                    loss_pa: 0.5e6,
                },
            ],
            0.49,
        )
        .unwrap();
        assert!(model.complex_youngs_modulus_pa(5_000.0).is_err());
    }
}
