//! Independent compact rotating-dipole observer reference.
//!
//! This reference is deliberately narrower than the production source model:
//! it applies the far-field dipole envelope `|sin(theta)|` to a declared
//! equatorial harmonic magnitude and checks the corresponding sphere-integrated
//! power factor. It contains no observer phase, wake, installation, or
//! resolved directivity information.

use std::f64::consts::PI;

use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;

/// Errors constructing or sampling the compact rotating-dipole reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotatingDipoleReferenceError {
    /// A source or derived scalar was outside the reference domain.
    InvalidInput {
        /// Name of the invalid input.
        field: &'static str,
        /// Required domain for that input.
        requirement: &'static str,
    },
    /// The observer polar angle was not in the far-field spherical domain.
    InvalidPolarAngle,
}

impl std::fmt::Display for RotatingDipoleReferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput { field, requirement } => {
                write!(f, "rotating-dipole reference {field} must be {requirement}")
            }
            Self::InvalidPolarAngle => {
                write!(
                    f,
                    "rotating-dipole polar angle must be finite and in [0, pi]"
                )
            }
        }
    }
}

impl std::error::Error for RotatingDipoleReferenceError {}

/// Compact rotating-dipole magnitude reference for one source harmonic.
///
/// `equatorial_peak_pressure_pa` is the positive peak magnitude at the
/// reference radius in the plane normal to the rotation axis. The reference
/// predicts only the canonical compact-dipole envelope:
///
/// `p_peak(theta) = p_peak(pi/2) * |sin(theta)|`.
///
/// The base pressure API is magnitude-only. The checked
/// [`RotatingDipoleReference::complex_pressure_pa`] extension adds the
/// declared compact-source phase convention, but remains an analytic
/// observer reference rather than a wake or resolved propeller directivity
/// field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotatingDipoleReference {
    /// Harmonic number, where one is the blade-passing-frequency component.
    pub harmonic: usize,
    /// Harmonic frequency in hertz.
    pub frequency_hz: f64,
    /// Radius at which the equatorial pressure magnitude is specified.
    pub reference_distance_m: f64,
    /// Equatorial positive peak pressure magnitude in pascals.
    pub equatorial_peak_pressure_pa: f64,
    /// Medium density used for the power conversion.
    pub density_kg_m3: f64,
    /// Medium sound speed used for the power conversion.
    pub speed_of_sound_m_s: f64,
}

impl RotatingDipoleReference {
    /// The exact sphere-average of `sin²(theta)` for this dipole envelope.
    pub const DIRECTIVITY_POWER_FACTOR: f64 = 2.0 / 3.0;

    /// Build a reference from the production source's equatorial harmonic
    /// magnitude at `reference_distance_m`.
    pub fn try_from_source(
        source: &PropellerSource,
        harmonic: usize,
        reference_distance_m: f64,
    ) -> Result<Self, RotatingDipoleReferenceError> {
        validate_source(source)?;
        if harmonic == 0 {
            return Err(RotatingDipoleReferenceError::InvalidInput {
                field: "harmonic",
                requirement: "non-zero",
            });
        }
        validate_positive_finite("reference_distance_m", reference_distance_m)?;

        let frequency_hz = source.harmonic_frequency(harmonic);
        let equatorial_peak_pressure_pa = source
            .harmonic_amplitudes_at_distance(harmonic, reference_distance_m)
            .get(harmonic - 1)
            .copied()
            .ok_or(RotatingDipoleReferenceError::InvalidInput {
                field: "harmonic",
                requirement: "representable by the source harmonic model",
            })?;

        Self::try_from_equatorial_amplitude(
            harmonic,
            frequency_hz,
            reference_distance_m,
            equatorial_peak_pressure_pa,
            source.density,
            source.speed_of_sound,
        )
    }

    /// Build the observer reference from an independently supplied equatorial
    /// harmonic magnitude.
    pub fn try_from_equatorial_amplitude(
        harmonic: usize,
        frequency_hz: f64,
        reference_distance_m: f64,
        equatorial_peak_pressure_pa: f64,
        density_kg_m3: f64,
        speed_of_sound_m_s: f64,
    ) -> Result<Self, RotatingDipoleReferenceError> {
        if harmonic == 0 {
            return Err(RotatingDipoleReferenceError::InvalidInput {
                field: "harmonic",
                requirement: "non-zero",
            });
        }
        validate_positive_finite("frequency_hz", frequency_hz)?;
        validate_positive_finite("reference_distance_m", reference_distance_m)?;
        validate_nonnegative_finite("equatorial_peak_pressure_pa", equatorial_peak_pressure_pa)?;
        validate_positive_finite("density_kg_m3", density_kg_m3)?;
        validate_positive_finite("speed_of_sound_m_s", speed_of_sound_m_s)?;

        Ok(Self {
            harmonic,
            frequency_hz,
            reference_distance_m,
            equatorial_peak_pressure_pa,
            density_kg_m3,
            speed_of_sound_m_s,
        })
    }

    /// Return the positive peak pressure magnitude at polar angle `theta_rad`.
    pub fn peak_pressure_pa(self, theta_rad: f64) -> Result<f64, RotatingDipoleReferenceError> {
        if !theta_rad.is_finite() || !(0.0..=PI).contains(&theta_rad) {
            return Err(RotatingDipoleReferenceError::InvalidPolarAngle);
        }
        Ok(self.equatorial_peak_pressure_pa * theta_rad.sin().abs())
    }

    /// Isotropic-reference power represented by the equatorial pressure
    /// magnitude, using peak-to-RMS conversion.
    pub fn isotropic_reference_power_w(self) -> f64 {
        4.0 * PI * self.reference_distance_m.powi(2) * self.equatorial_peak_pressure_pa.powi(2)
            / (2.0 * self.density_kg_m3 * self.speed_of_sound_m_s)
    }

    /// Dipole power after integrating the `sin(theta)` envelope over a sphere.
    pub fn radiated_sound_power_w(self) -> f64 {
        self.isotropic_reference_power_w() * Self::DIRECTIVITY_POWER_FACTOR
    }

    /// Ratio of dipole power to the isotropic power represented by the same
    /// equatorial peak magnitude.
    pub const fn power_ratio_to_isotropic_reference() -> f64 {
        Self::DIRECTIVITY_POWER_FACTOR
    }
}

/// Complex far-field peak-pressure phasor for one compact rotating-dipole
/// observer location.
///
/// The convention is `exp(i omega t)` with outgoing propagation
/// `exp(-i k r)`. A positive rotating-dipole azimuth contributes the phase
/// factor `exp(-i phi)`, so the returned phase is `-(k r + phi)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotatingDipoleComplexPressure {
    /// Real component of the peak-pressure phasor [Pa].
    pub real_pa: f64,
    /// Imaginary component of the peak-pressure phasor [Pa].
    pub imaginary_pa: f64,
}

impl RotatingDipoleComplexPressure {
    /// Peak-pressure magnitude [Pa].
    pub fn magnitude_pa(self) -> f64 {
        self.real_pa.hypot(self.imaginary_pa)
    }

    /// Principal phasor phase [rad].
    pub fn phase_rad(self) -> f64 {
        self.imaginary_pa.atan2(self.real_pa)
    }
}

impl RotatingDipoleReference {
    /// Minimum `k*r` admitted by the far-field observer evaluation.
    ///
    /// Below this the dipole's reactive near-field terms (decaying as
    /// `1/r^2` and faster) are comparable to the radiating `1/r` term,
    /// so the far-field phasor and the progressive-wave intensity
    /// relation derived from it are not valid.
    pub const MIN_FAR_FIELD_KR: f64 = 1.0;

    /// Returns the dimensionless observer distance `k*r` for this harmonic.
    pub fn far_field_kr(self, observer_distance_m: f64) -> f64 {
        2.0 * PI * self.frequency_hz / self.speed_of_sound_m_s * observer_distance_m
    }

    /// Evaluate a phase-aware compact-dipole observer phasor.
    ///
    /// The source amplitude is scaled as `1/r`, the polar envelope is
    /// `sin(theta)`, and the circularly rotating source contributes the
    /// azimuthal phase `-phi`. This is valid only as a declared far-field
    /// analytic reference; it does not resolve installation, wake, or blade
    /// loading effects.
    pub fn complex_pressure_pa(
        self,
        observer_distance_m: f64,
        polar_angle_rad: f64,
        azimuth_angle_rad: f64,
    ) -> Result<RotatingDipoleComplexPressure, RotatingDipoleReferenceError> {
        validate_positive_finite("observer_distance_m", observer_distance_m)?;
        if self.far_field_kr(observer_distance_m) < Self::MIN_FAR_FIELD_KR {
            return Err(RotatingDipoleReferenceError::InvalidInput {
                field: "observer_distance_m",
                requirement: "in the far field (k*r >= 1) for this 1/r reference",
            });
        }
        if !polar_angle_rad.is_finite() || !(0.0..=PI).contains(&polar_angle_rad) {
            return Err(RotatingDipoleReferenceError::InvalidPolarAngle);
        }
        if !azimuth_angle_rad.is_finite() {
            return Err(RotatingDipoleReferenceError::InvalidInput {
                field: "azimuth_angle_rad",
                requirement: "finite",
            });
        }

        let wavenumber_rad_m = 2.0 * PI * self.frequency_hz / self.speed_of_sound_m_s;
        let phase_rad = -(wavenumber_rad_m * observer_distance_m + azimuth_angle_rad);
        let amplitude_pa = self.equatorial_peak_pressure_pa * self.reference_distance_m
            / observer_distance_m
            * polar_angle_rad.sin();
        if !wavenumber_rad_m.is_finite() || !phase_rad.is_finite() || !amplitude_pa.is_finite() {
            return Err(RotatingDipoleReferenceError::InvalidInput {
                field: "observer_phasor",
                requirement: "finite",
            });
        }

        Ok(RotatingDipoleComplexPressure {
            real_pa: amplitude_pa * phase_rad.cos(),
            imaginary_pa: amplitude_pa * phase_rad.sin(),
        })
    }
}

fn validate_source(source: &PropellerSource) -> Result<(), RotatingDipoleReferenceError> {
    if source.n_blades == 0 {
        return Err(RotatingDipoleReferenceError::InvalidInput {
            field: "source.n_blades",
            requirement: "non-zero",
        });
    }
    for (field, value) in [
        ("source.rpm", source.rpm),
        ("source.thrust", source.thrust),
        ("source.torque", source.torque),
        ("source.tip_radius", source.tip_radius),
        ("source.hub_radius", source.hub_radius),
        ("source.speed_of_sound", source.speed_of_sound),
        ("source.density", source.density),
    ] {
        if !value.is_finite() {
            return Err(RotatingDipoleReferenceError::InvalidInput {
                field,
                requirement: "finite",
            });
        }
    }
    if source.tip_radius <= source.hub_radius || source.hub_radius <= 0.0 {
        return Err(RotatingDipoleReferenceError::InvalidInput {
            field: "source.hub_radius/source.tip_radius",
            requirement: "0 < hub_radius < tip_radius",
        });
    }
    validate_positive_finite("source.speed_of_sound", source.speed_of_sound)?;
    validate_positive_finite("source.density", source.density)?;
    if source.rpm == 0.0 {
        return Err(RotatingDipoleReferenceError::InvalidInput {
            field: "source.rpm",
            requirement: "non-zero",
        });
    }
    Ok(())
}

fn validate_positive_finite(
    field: &'static str,
    value: f64,
) -> Result<(), RotatingDipoleReferenceError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(RotatingDipoleReferenceError::InvalidInput {
            field,
            requirement: "finite and positive",
        });
    }
    Ok(())
}

fn validate_nonnegative_finite(
    field: &'static str,
    value: f64,
) -> Result<(), RotatingDipoleReferenceError> {
    if !value.is_finite() || value < 0.0 {
        return Err(RotatingDipoleReferenceError::InvalidInput {
            field,
            requirement: "finite and non-negative",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dipole_has_axial_null_and_equatorial_peak() {
        let source = PropellerSource::drone_5inch(18_000.0);
        let reference = RotatingDipoleReference::try_from_source(&source, 1, 1.0).unwrap();
        let equatorial = reference.peak_pressure_pa(PI / 2.0).unwrap();
        assert!((equatorial - reference.equatorial_peak_pressure_pa).abs() < 1.0e-15);
        assert!(reference.peak_pressure_pa(0.0).unwrap().abs() < 1.0e-15);
        assert!(reference.peak_pressure_pa(PI).unwrap().abs() < 1.0e-15);
        assert!(
            (reference.peak_pressure_pa(PI / 6.0).unwrap()
                - reference.peak_pressure_pa(5.0 * PI / 6.0).unwrap())
            .abs()
                < 1.0e-15
        );
    }

    #[test]
    fn pressure_and_power_use_declared_far_field_conventions() {
        let source = PropellerSource::drone_5inch(18_000.0);
        let one_metre = RotatingDipoleReference::try_from_source(&source, 1, 1.0).unwrap();
        let two_metres = RotatingDipoleReference::try_from_source(&source, 1, 2.0).unwrap();
        assert!(
            (one_metre.equatorial_peak_pressure_pa / two_metres.equatorial_peak_pressure_pa - 2.0)
                .abs()
                < 1.0e-12
        );
        assert!(
            (one_metre.radiated_sound_power_w() / one_metre.isotropic_reference_power_w()
                - RotatingDipoleReference::DIRECTIVITY_POWER_FACTOR)
                .abs()
                < 1.0e-15
        );
        assert_eq!(
            RotatingDipoleReference::power_ratio_to_isotropic_reference(),
            RotatingDipoleReference::DIRECTIVITY_POWER_FACTOR
        );
        assert!(
            (one_metre.radiated_sound_power_w() - two_metres.radiated_sound_power_w()).abs()
                < 1.0e-12 * one_metre.radiated_sound_power_w().max(1.0)
        );
    }

    #[test]
    fn numerical_sphere_average_matches_analytic_power_factor() {
        let reference = RotatingDipoleReference::try_from_equatorial_amplitude(
            1, 1_000.0, 1.0, 0.25, 1.204, 343.0,
        )
        .unwrap();
        let n_theta = 256;
        let n_phi = 64;
        let d_theta = PI / n_theta as f64;
        let d_phi = 2.0 * PI / n_phi as f64;
        let mut integral = 0.0;
        for i in 0..n_theta {
            let theta = (i as f64 + 0.5) * d_theta;
            let pressure = reference.peak_pressure_pa(theta).unwrap();
            integral += n_phi as f64 * pressure.powi(2) * theta.sin() * d_theta * d_phi;
        }
        let normalized = integral / (4.0 * PI * reference.equatorial_peak_pressure_pa.powi(2));
        assert!((normalized - RotatingDipoleReference::DIRECTIVITY_POWER_FACTOR).abs() < 2.0e-5);
    }

    #[test]
    fn rejects_invalid_source_and_observer_inputs() {
        let source = PropellerSource::drone_5inch(18_000.0);
        assert!(matches!(
            RotatingDipoleReference::try_from_source(&source, 0, 1.0),
            Err(RotatingDipoleReferenceError::InvalidInput {
                field: "harmonic",
                ..
            })
        ));
        assert!(matches!(
            RotatingDipoleReference::try_from_source(&source, 1, 0.0),
            Err(RotatingDipoleReferenceError::InvalidInput {
                field: "reference_distance_m",
                ..
            })
        ));
        let reference = RotatingDipoleReference::try_from_source(&source, 1, 1.0).unwrap();
        assert_eq!(
            reference.peak_pressure_pa(-f64::EPSILON),
            Err(RotatingDipoleReferenceError::InvalidPolarAngle)
        );
        assert_eq!(
            reference.peak_pressure_pa(PI + 1.0e-12),
            Err(RotatingDipoleReferenceError::InvalidPolarAngle)
        );
        assert!(
            RotatingDipoleReferenceError::InvalidPolarAngle
                .to_string()
                .contains("polar angle")
        );
    }
}
#[cfg(test)]
mod phase_tests {
    use super::*;

    #[test]
    fn complex_observer_pressure_preserves_distance_and_phase_conventions() {
        let reference = RotatingDipoleReference::try_from_equatorial_amplitude(
            1, 1_000.0, 1.0, 0.25, 1.204, 343.0,
        )
        .unwrap();
        let at_two_metres = reference.complex_pressure_pa(2.0, PI / 2.0, 0.0).unwrap();
        let at_two_metres_quarter_turn = reference
            .complex_pressure_pa(2.0, PI / 2.0, PI / 2.0)
            .unwrap();

        assert!((at_two_metres.magnitude_pa() - 0.125).abs() < 1.0e-12);
        assert!(
            (at_two_metres.real_pa.powi(2) + at_two_metres.imaginary_pa.powi(2)
                - at_two_metres_quarter_turn.real_pa.powi(2)
                - at_two_metres_quarter_turn.imaginary_pa.powi(2))
            .abs()
                < 1.0e-12
        );
        assert!(
            (at_two_metres.real_pa * at_two_metres_quarter_turn.real_pa
                + at_two_metres.imaginary_pa * at_two_metres_quarter_turn.imaginary_pa)
                .abs()
                < 1.0e-12
        );

        let expected_phase = -(2.0 * PI * 1_000.0 / 343.0 * 2.0);
        assert!(
            (at_two_metres.phase_rad() - expected_phase.rem_euclid(2.0 * PI)).abs() < 1.0e-12
                || (at_two_metres.phase_rad() - (expected_phase.rem_euclid(2.0 * PI) - 2.0 * PI))
                    .abs()
                    < 1.0e-12
        );
    }

    #[test]
    fn complex_observer_directivity_power_is_distance_invariant() {
        let reference = RotatingDipoleReference::try_from_equatorial_amplitude(
            1, 1_000.0, 1.0, 0.25, 1.204, 343.0,
        )
        .unwrap();
        let n_theta = 96;
        let n_phi = 48;
        let d_theta = PI / n_theta as f64;
        let d_phi = 2.0 * PI / n_phi as f64;
        let mut integral = 0.0;
        for i in 0..n_theta {
            let theta = (i as f64 + 0.5) * d_theta;
            for j in 0..n_phi {
                let phi = (j as f64 + 0.5) * d_phi;
                let pressure = reference.complex_pressure_pa(3.0, theta, phi).unwrap();
                integral += pressure.magnitude_pa().powi(2) * theta.sin() * d_theta * d_phi;
            }
        }
        let peak_power = 3.0_f64.powi(2) * integral
            / (2.0 * reference.density_kg_m3 * reference.speed_of_sound_m_s);
        assert!(
            (peak_power - reference.radiated_sound_power_w()).abs()
                < 2.0e-5 * reference.radiated_sound_power_w().max(1.0)
        );
    }

    #[test]
    fn complex_observer_rejects_near_field_distances() {
        // 100 Hz in air: k = 1.83 rad/m, so r = 0.1 m gives kr = 0.18,
        // far outside the 1/r far-field regime of this reference.
        let reference = RotatingDipoleReference::try_from_equatorial_amplitude(
            1, 100.0, 1.0, 0.25, 1.204, 343.0,
        )
        .unwrap();
        assert!(reference.complex_pressure_pa(0.1, PI / 2.0, 0.0).is_err());
        assert!(reference.complex_pressure_pa(1.0, PI / 2.0, 0.0).is_ok());
        assert!((reference.far_field_kr(1.0) - 1.831).abs() < 0.01);
    }

    #[test]
    fn complex_observer_rejects_invalid_distance_and_azimuth() {
        let reference = RotatingDipoleReference::try_from_equatorial_amplitude(
            1, 1_000.0, 1.0, 0.25, 1.204, 343.0,
        )
        .unwrap();
        assert!(matches!(
            reference.complex_pressure_pa(0.0, PI / 2.0, 0.0),
            Err(RotatingDipoleReferenceError::InvalidInput {
                field: "observer_distance_m",
                ..
            })
        ));
        assert!(matches!(
            reference.complex_pressure_pa(1.0, PI / 2.0, f64::NAN),
            Err(RotatingDipoleReferenceError::InvalidInput {
                field: "azimuth_angle_rad",
                ..
            })
        ));
    }
}
