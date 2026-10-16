use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use super::{
    AcousticMedium, ModelingError, baffled_piston_mechanical_radiation_impedance, require_finite,
    require_nonnegative, require_positive,
};

/// Circular bass-reflex port represented as a lumped acoustic mass and resistance.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PortSpec {
    pub radius_m: f64,
    pub length_m: f64,
    /// Total end correction for both terminations.
    pub end_correction_m: f64,
    /// Additional viscous/porous acoustic resistance in Pa·s/m³.
    pub resistance_pa_s_per_m3: f64,
}

impl PortSpec {
    pub fn validate(self) -> Result<(), ModelingError> {
        require_positive("enclosure.port.radius_m", self.radius_m)?;
        require_positive("enclosure.port.length_m", self.length_m)?;
        require_nonnegative("enclosure.port.end_correction_m", self.end_correction_m)?;
        require_nonnegative(
            "enclosure.port.resistance_pa_s_per_m3",
            self.resistance_pa_s_per_m3,
        )
    }

    pub fn area_m2(self) -> Result<f64, ModelingError> {
        self.validate()?;
        let area = std::f64::consts::PI * self.radius_m * self.radius_m;
        require_positive("enclosure.port.area_m2", area)?;
        Ok(area)
    }

    /// Acoustic port impedance `pressure / volume_velocity`.
    pub fn acoustic_impedance(
        self,
        frequency_hz: f64,
        medium: AcousticMedium,
    ) -> Result<Complex64, ModelingError> {
        self.validate()?;
        require_positive("frequency_hz", frequency_hz)?;
        medium.validate()?;
        let area = self.area_m2()?;
        let effective_length = self.length_m + self.end_correction_m;
        require_positive("enclosure.port.effective_length_m", effective_length)?;
        let omega = std::f64::consts::TAU * frequency_hz;
        require_positive("angular_frequency_rad_per_s", omega)?;
        let inertance = medium.density_kg_per_m3 * effective_length / area;
        require_positive("enclosure.port.acoustic_inertance", inertance)?;

        // Only the resistive part of the exterior piston load is added here;
        // reactive termination mass is represented explicitly by end_correction_m.
        let radiation =
            baffled_piston_mechanical_radiation_impedance(self.radius_m, frequency_hz, medium)?;
        let radiation_resistance = radiation.re / (area * area);
        let impedance = Complex64::new(
            self.resistance_pa_s_per_m3 + radiation_resistance,
            omega * inertance,
        );
        require_finite("enclosure.port.impedance.re", impedance.re)?;
        require_finite("enclosure.port.impedance.im", impedance.im)?;
        Ok(impedance)
    }
}

/// Lumped passive-radiator (drone cone) parameters.
///
/// The radiator is a mass-compliance-resistance mechanical branch with
/// no motor. Its acoustic impedance is `Z_m / S^2`, where `Z_m` is the
/// mechanical impedance `force / velocity` and `S` is the radiator area.
/// The lumped form holds only while the radiator and cavity stay
/// acoustically compact; above that regime a resolved model is required.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PassiveRadiatorSpec {
    /// Radiator diaphragm area in m².
    pub area_m2: f64,
    /// Radiator moving mass in kg.
    pub moving_mass_kg: f64,
    /// Mechanical compliance in m/N.
    pub compliance_m_per_n: f64,
    /// Mechanical resistance in kg/s.
    pub resistance_kg_per_s: f64,
}

impl PassiveRadiatorSpec {
    /// Validates the radiator parameters.
    pub fn validate(self) -> Result<(), ModelingError> {
        require_positive("enclosure.passive_radiator.area_m2", self.area_m2)?;
        require_positive("enclosure.passive_radiator.moving_mass_kg", self.moving_mass_kg)?;
        require_positive(
            "enclosure.passive_radiator.compliance_m_per_n",
            self.compliance_m_per_n,
        )?;
        require_nonnegative(
            "enclosure.passive_radiator.resistance_kg_per_s",
            self.resistance_kg_per_s,
        )
    }

    /// Acoustic impedance `pressure / volume_velocity` in Pa·s/m³.
    pub fn acoustic_impedance(
        self,
        frequency_hz: f64,
    ) -> Result<Complex64, ModelingError> {
        self.validate()?;
        require_positive("frequency_hz", frequency_hz)?;
        let omega = std::f64::consts::TAU * frequency_hz;
        let stiffness = 1.0 / self.compliance_m_per_n;
        let mechanical = Complex64::new(
            self.resistance_kg_per_s,
            omega * self.moving_mass_kg - stiffness / omega,
        );
        let impedance = mechanical / (self.area_m2 * self.area_m2);
        require_finite("enclosure.passive_radiator.impedance.re", impedance.re)?;
        require_finite("enclosure.passive_radiator.impedance.im", impedance.im)?;
        Ok(impedance)
    }

    /// Undamped radiator resonance `1/(2π√(M·C))` in Hz.
    ///
    /// This is the undamped natural frequency, not the damped response
    /// peak; the two must never be reported as the same quantity.
    pub fn resonance_hz(self) -> Result<f64, ModelingError> {
        self.validate()?;
        let frequency =
            1.0 / (std::f64::consts::TAU * (self.moving_mass_kg * self.compliance_m_per_n).sqrt());
        require_positive("enclosure.passive_radiator.resonance_hz", frequency)?;
        Ok(frequency)
    }
}

/// Rear acoustic loading attached to the diaphragm.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum EnclosureSpec {
    #[default]
    FreeAir,
    Sealed {
        volume_m3: f64,
        /// Parallel leakage resistance; zero means airtight.
        leakage_resistance_pa_s_per_m3: f64,
        /// Parallel porous/damping resistance; zero disables this loss path.
        absorption_resistance_pa_s_per_m3: f64,
    },
    Vented {
        volume_m3: f64,
        port: PortSpec,
        leakage_resistance_pa_s_per_m3: f64,
        absorption_resistance_pa_s_per_m3: f64,
    },
    /// Sealed box loaded by a passive radiator instead of a port.
    PassiveRadiator {
        volume_m3: f64,
        radiator: PassiveRadiatorSpec,
        leakage_resistance_pa_s_per_m3: f64,
        absorption_resistance_pa_s_per_m3: f64,
    },
    /// Fourth-order bandpass: sealed rear chamber, ported front chamber.
    ///
    /// The rear impedance seen by the diaphragm is the sealed rear
    /// chamber; the front chamber and its port shape the radiated
    /// output and are exposed separately, never folded into the rear
    /// load. Lumped relations hold only while both chambers and the
    /// front neck stay acoustically compact.
    Bandpass {
        rear_volume_m3: f64,
        front_volume_m3: f64,
        front_port: PortSpec,
        leakage_resistance_pa_s_per_m3: f64,
        absorption_resistance_pa_s_per_m3: f64,
    },
}

impl EnclosureSpec {
    pub fn validate(&self) -> Result<(), ModelingError> {
        match self {
            Self::FreeAir => Ok(()),
            Self::Sealed {
                volume_m3,
                leakage_resistance_pa_s_per_m3,
                absorption_resistance_pa_s_per_m3,
            }
            | Self::Vented {
                volume_m3,
                leakage_resistance_pa_s_per_m3,
                absorption_resistance_pa_s_per_m3,
                ..
            }
            | Self::PassiveRadiator {
                volume_m3,
                leakage_resistance_pa_s_per_m3,
                absorption_resistance_pa_s_per_m3,
                ..
            } => {
                require_positive("enclosure.volume_m3", *volume_m3)?;
                require_nonnegative(
                    "enclosure.leakage_resistance_pa_s_per_m3",
                    *leakage_resistance_pa_s_per_m3,
                )?;
                require_nonnegative(
                    "enclosure.absorption_resistance_pa_s_per_m3",
                    *absorption_resistance_pa_s_per_m3,
                )?;
                if let Self::Vented { port, .. } = self {
                    port.validate()?;
                }
                if let Self::PassiveRadiator { radiator, .. } = self {
                    radiator.validate()?;
                }
                Ok(())
            }
            Self::Bandpass {
                rear_volume_m3,
                front_volume_m3,
                front_port,
                leakage_resistance_pa_s_per_m3,
                absorption_resistance_pa_s_per_m3,
            } => {
                require_positive("enclosure.bandpass.rear_volume_m3", *rear_volume_m3)?;
                require_positive("enclosure.bandpass.front_volume_m3", *front_volume_m3)?;
                require_nonnegative(
                    "enclosure.leakage_resistance_pa_s_per_m3",
                    *leakage_resistance_pa_s_per_m3,
                )?;
                require_nonnegative(
                    "enclosure.absorption_resistance_pa_s_per_m3",
                    *absorption_resistance_pa_s_per_m3,
                )?;
                front_port.validate()
            }
        }
    }

    /// Rear acoustic impedance `pressure / diaphragm volume_velocity`.
    pub fn acoustic_impedance(
        &self,
        frequency_hz: f64,
        medium: AcousticMedium,
    ) -> Result<Complex64, ModelingError> {
        self.validate()?;
        require_positive("frequency_hz", frequency_hz)?;
        medium.validate()?;
        let (volume_m3, leakage, absorption, port, radiator) = match self {
            Self::FreeAir => return Ok(Complex64::new(0.0, 0.0)),
            Self::Sealed {
                volume_m3,
                leakage_resistance_pa_s_per_m3,
                absorption_resistance_pa_s_per_m3,
            } => (
                *volume_m3,
                *leakage_resistance_pa_s_per_m3,
                *absorption_resistance_pa_s_per_m3,
                None,
                None,
            ),
            Self::Vented {
                volume_m3,
                port,
                leakage_resistance_pa_s_per_m3,
                absorption_resistance_pa_s_per_m3,
            } => (
                *volume_m3,
                *leakage_resistance_pa_s_per_m3,
                *absorption_resistance_pa_s_per_m3,
                Some(*port),
                None,
            ),
            Self::PassiveRadiator {
                volume_m3,
                radiator,
                leakage_resistance_pa_s_per_m3,
                absorption_resistance_pa_s_per_m3,
            } => (
                *volume_m3,
                *leakage_resistance_pa_s_per_m3,
                *absorption_resistance_pa_s_per_m3,
                None,
                Some(*radiator),
            ),
            Self::Bandpass {
                rear_volume_m3,
                leakage_resistance_pa_s_per_m3,
                absorption_resistance_pa_s_per_m3,
                ..
            } => (
                *rear_volume_m3,
                *leakage_resistance_pa_s_per_m3,
                *absorption_resistance_pa_s_per_m3,
                None,
                None,
            ),
        };
        let omega = std::f64::consts::TAU * frequency_hz;
        let compliance =
            volume_m3 / (medium.density_kg_per_m3 * medium.sound_speed_m_per_s.powi(2));
        require_positive("enclosure.acoustic_compliance", compliance)?;
        let mut admittance = Complex64::new(0.0, omega * compliance);
        for resistance in [leakage, absorption] {
            if resistance > 0.0 {
                admittance += Complex64::new(1.0 / resistance, 0.0);
            }
        }
        if let Some(port) = port {
            admittance +=
                Complex64::new(1.0, 0.0) / port.acoustic_impedance(frequency_hz, medium)?;
        }
        if let Some(radiator) = radiator {
            admittance +=
                Complex64::new(1.0, 0.0) / radiator.acoustic_impedance(frequency_hz)?;
        }
        require_finite("enclosure.acoustic_admittance.re", admittance.re)?;
        require_finite("enclosure.acoustic_admittance.im", admittance.im)?;
        let impedance = Complex64::new(1.0, 0.0) / admittance;
        require_finite("enclosure.acoustic_impedance.re", impedance.re)?;
        require_finite("enclosure.acoustic_impedance.im", impedance.im)?;
        Ok(impedance)
    }

    /// Helmholtz tuning frequency of a vented enclosure.
    pub fn tuning_frequency_hz(
        &self,
        medium: AcousticMedium,
    ) -> Result<Option<f64>, ModelingError> {
        self.validate()?;
        medium.validate()?;
        let Self::Vented {
            volume_m3, port, ..
        } = self
        else {
            return Ok(None);
        };
        let area = port.area_m2()?;
        let effective_length = port.length_m + port.end_correction_m;
        let frequency = medium.sound_speed_m_per_s / std::f64::consts::TAU
            * (area / (volume_m3 * effective_length)).sqrt();
        require_positive("enclosure.tuning_frequency_hz", frequency)?;
        Ok(Some(frequency))
    }

    /// Helmholtz tuning of the bandpass front chamber, if bandpass.
    ///
    /// Uses `f = c/(2π)·√(S/(V_front·L_eff))`, valid only while the
    /// front neck and cavity stay acoustically compact.
    pub fn front_tuning_frequency_hz(
        &self,
        medium: AcousticMedium,
    ) -> Result<Option<f64>, ModelingError> {
        self.validate()?;
        medium.validate()?;
        let Self::Bandpass { front_volume_m3, front_port, .. } = self else {
            return Ok(None);
        };
        let area = front_port.area_m2()?;
        let effective_length = front_port.length_m + front_port.end_correction_m;
        let frequency = medium.sound_speed_m_per_s / std::f64::consts::TAU
            * (area / (front_volume_m3 * effective_length)).sqrt();
        require_positive("enclosure.bandpass.front_tuning_frequency_hz", frequency)?;
        Ok(Some(frequency))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn medium() -> AcousticMedium {
        AcousticMedium {
            density_kg_per_m3: 1.204,
            sound_speed_m_per_s: 343.0,
            dynamic_viscosity_pa_s: 1.81e-5,
            thermal_diffusivity_m2_per_s: 2.5e-5,
        }
    }

    #[test]
    fn passive_radiator_resonance_and_passivity() {
        let radiator = PassiveRadiatorSpec {
            area_m2: 0.02,
            moving_mass_kg: 0.05,
            compliance_m_per_n: 0.001,
            resistance_kg_per_s: 0.5,
        };
        let resonance = radiator.resonance_hz().expect("resonance");
        assert!((resonance - 22.51).abs() < 0.1);
        let impedance = radiator.acoustic_impedance(50.0).expect("impedance");
        assert!(impedance.re >= 0.0, "passive radiator must stay passive");
        let enclosure = EnclosureSpec::PassiveRadiator {
            volume_m3: 0.02,
            radiator,
            leakage_resistance_pa_s_per_m3: 0.0,
            absorption_resistance_pa_s_per_m3: 0.0,
        };
        let load = enclosure.acoustic_impedance(50.0, medium()).expect("load");
        assert!(load.re >= 0.0 && load.is_finite());
    }

    #[test]
    fn bandpass_rear_load_is_sealed_rear_chamber() {
        let port = PortSpec {
            radius_m: 0.03,
            length_m: 0.1,
            end_correction_m: 0.02,
            resistance_pa_s_per_m3: 0.0,
        };
        let bandpass = EnclosureSpec::Bandpass {
            rear_volume_m3: 0.02,
            front_volume_m3: 0.01,
            front_port: port,
            leakage_resistance_pa_s_per_m3: 0.0,
            absorption_resistance_pa_s_per_m3: 0.0,
        };
        let sealed = EnclosureSpec::Sealed {
            volume_m3: 0.02,
            leakage_resistance_pa_s_per_m3: 0.0,
            absorption_resistance_pa_s_per_m3: 0.0,
        };
        let actual = bandpass.acoustic_impedance(100.0, medium()).expect("bandpass");
        let expected = sealed.acoustic_impedance(100.0, medium()).expect("sealed");
        assert!((actual - expected).norm() < 1e-9);
        assert!(bandpass.front_tuning_frequency_hz(medium()).expect("tuning").is_some());
    }
}

/// Infinite-baffle circular-piston pressure directivity, normalized on axis.
pub fn baffled_piston_directivity(
    radius_m: f64,
    frequency_hz: f64,
    angle_degrees: f64,
    medium: AcousticMedium,
) -> Result<f64, ModelingError> {
    use spec_math::Bessel;

    require_positive("radius_m", radius_m)?;
    require_positive("frequency_hz", frequency_hz)?;
    require_finite("angle_degrees", angle_degrees)?;
    if !(0.0..=90.0).contains(&angle_degrees) {
        return Err(ModelingError::OutOfRange {
            field: "angle_degrees",
            constraint: "between zero and 90 degrees",
        });
    }
    medium.validate()?;
    let ka = std::f64::consts::TAU * frequency_hz * radius_m / medium.sound_speed_m_per_s;
    let argument = ka * angle_degrees.to_radians().sin();
    let directivity = if argument.abs() < 1.0e-6 {
        1.0 - argument * argument / 8.0
    } else {
        2.0 * argument.bessel_jv(1.0) / argument
    };
    require_finite("piston_directivity", directivity)?;
    Ok(directivity)
}

/// Far-field RMS pressure phasor of a uniformly moving baffled piston.
pub fn baffled_piston_far_field_pressure(
    radius_m: f64,
    frequency_hz: f64,
    velocity_rms: Complex64,
    distance_m: f64,
    angle_degrees: f64,
    medium: AcousticMedium,
) -> Result<Complex64, ModelingError> {
    require_finite("velocity_rms.re", velocity_rms.re)?;
    require_finite("velocity_rms.im", velocity_rms.im)?;
    require_positive("distance_m", distance_m)?;
    let directivity = baffled_piston_directivity(radius_m, frequency_hz, angle_degrees, medium)?;
    let area = std::f64::consts::PI * radius_m * radius_m;
    let wavenumber = std::f64::consts::TAU * frequency_hz / medium.sound_speed_m_per_s;
    let phase = Complex64::from_polar(1.0, -wavenumber * distance_m);
    let pressure = Complex64::new(0.0, 1.0)
        * medium.density_kg_per_m3
        * medium.sound_speed_m_per_s
        * wavenumber
        * area
        * velocity_rms
        * directivity
        * phase
        / (2.0 * std::f64::consts::PI * distance_m);
    require_finite("far_field_pressure.re", pressure.re)?;
    require_finite("far_field_pressure.im", pressure.im)?;
    Ok(pressure)
}

pub fn sound_pressure_level_db(pressure_rms_pa: f64) -> Result<f64, ModelingError> {
    require_nonnegative("pressure_rms_pa", pressure_rms_pa)?;
    if pressure_rms_pa == 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    let spl = 20.0 * (pressure_rms_pa / 20.0e-6).log10();
    require_finite("sound_pressure_level_db", spl)?;
    Ok(spl)
}
