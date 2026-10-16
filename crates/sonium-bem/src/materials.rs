//! Acoustic material impedance models.
//!
//! These models turn porous absorber parameters into frequency-dependent
//! surface impedance/admittance values that can be fed into BEM admittance
//! boundary conditions.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

/// Typed failure from the porous-material acoustic boundary.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MaterialError {
    /// A medium parameter was not finite or positive.
    #[error("invalid ambient medium {field}: {value}")]
    InvalidMedium {
        /// Name of the rejected medium field.
        field: &'static str,
        /// Supplied, invalid field value.
        value: f64,
    },
    /// A porous-layer parameter failed its physical domain check.
    #[error("invalid porous layer: {0}")]
    InvalidLayer(String),
    /// A thin-shell parameter failed its physical domain check.
    #[error("invalid thin shell: {0}")]
    InvalidThinShell(String),
    /// Frequency is outside the model's supported domain.
    #[error("frequency must be finite and positive, got {0}")]
    InvalidFrequency(f64),
    /// Incidence angle is not a finite angle in the physically distinct
    /// interval `[0, π/2)`.
    #[error("incidence angle must be finite and in [0, π/2), got {0}")]
    InvalidIncidenceAngle(f64),
    /// A model evaluation produced a non-finite complex impedance.
    #[error("porous impedance evaluation produced a non-finite result")]
    NonFiniteResult,
    /// The requested passivity tolerance is not finite and non-negative.
    #[error("passivity tolerance must be finite and non-negative, got {0}")]
    InvalidTolerance(f64),
    /// The evaluated surface violates the requested passivity interval.
    #[error("porous surface is not passive (absorption={0})")]
    NonPassive(f64),
    /// Interface media carry incompatible exchange conventions.
    #[error("incompatible XEM interface convention: {0}")]
    InvalidConvention(&'static str),
}

/// Plane-wave transmission result for a fluid-fluid planar interface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaneWaveTransmission {
    /// Complex pressure reflection coefficient at the interface.
    pub pressure_reflection: Complex64,
    /// Complex pressure transmission coefficient at the interface.
    pub pressure_transmission: Complex64,
    /// Reflected normal power fraction.
    pub reflected_power: f64,
    /// Transmitted normal power fraction. This is zero for total internal
    /// reflection, where the transmitted field is evanescent.
    pub transmitted_power: f64,
    /// Whether the transmitted wave is evanescent.
    pub total_internal_reflection: bool,
}

impl PlaneWaveTransmission {
    /// Return the passive power-conservation residual `1 - R - T`.
    pub fn power_balance_residual(self) -> f64 {
        1.0 - self.reflected_power - self.transmitted_power
    }
}

/// Ambient fluid properties used by acoustic impedance models.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AmbientMedium {
    /// Air density in kg/m^3.
    pub density: f64,
    /// Sound speed in m/s.
    pub speed_of_sound: f64,
}

impl AmbientMedium {
    /// Create a medium from density and sound speed.
    pub fn new(density: f64, speed_of_sound: f64) -> Self {
        Self {
            density,
            speed_of_sound,
        }
    }

    /// Construct a validated ambient medium from exchange quantities.
    pub fn try_from_xem(
        density: sonium_xem::DensityKgM3,
        speed_of_sound: sonium_xem::SpeedOfSoundMPerS,
    ) -> Result<Self, MaterialError> {
        let medium = Self::new(density.get(), speed_of_sound.get());
        medium.validate()?;
        Ok(medium)
    }

    /// Construct a validated ambient medium from raw compatibility scalars.
    pub fn try_new(density: f64, speed_of_sound: f64) -> Result<Self, MaterialError> {
        let medium = Self::new(density, speed_of_sound);
        medium.validate()?;
        Ok(medium)
    }

    /// Validate density and sound speed before using the medium in a model.
    pub fn validate(self) -> Result<(), MaterialError> {
        if !self.density.is_finite() || self.density <= 0.0 {
            return Err(MaterialError::InvalidMedium {
                field: "density",
                value: self.density,
            });
        }
        if !self.speed_of_sound.is_finite() || self.speed_of_sound <= 0.0 {
            return Err(MaterialError::InvalidMedium {
                field: "speed_of_sound",
                value: self.speed_of_sound,
            });
        }
        Ok(())
    }

    /// Characteristic impedance rho*c in Pa*s/m.
    pub fn characteristic_impedance(self) -> f64 {
        self.density * self.speed_of_sound
    }
}

/// Lossless fluid-fluid planar interface with incident and transmitted media.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FluidInterface {
    /// Medium on the incident-wave side of the interface.
    pub incident: AmbientMedium,
    /// Medium on the transmitted-wave side of the interface.
    pub transmitted: AmbientMedium,
}

impl FluidInterface {
    /// Construct a lossless fluid-fluid interface.
    pub const fn new(incident: AmbientMedium, transmitted: AmbientMedium) -> Self {
        Self {
            incident,
            transmitted,
        }
    }

    /// Adapt two typed XEM media to a plane-wave interface.
    ///
    /// A transmission interface is monochromatic and must use one shared
    /// phasor convention. This boundary rejects mixed frequencies or
    /// harmonic signs before adapting the typed media to the material model.
    pub fn try_from_xem(
        incident: sonium_xem::AcousticMedium,
        transmitted: sonium_xem::AcousticMedium,
    ) -> Result<Self, MaterialError> {
        if incident.frequency != transmitted.frequency {
            return Err(MaterialError::InvalidConvention(
                "interface media must use the same frequency",
            ));
        }
        if incident.conventions.harmonic_time != transmitted.conventions.harmonic_time {
            return Err(MaterialError::InvalidConvention(
                "interface media must use the same harmonic convention",
            ));
        }
        Ok(Self::new(
            AmbientMedium::try_from_xem(incident.density, incident.speed_of_sound)?,
            AmbientMedium::try_from_xem(transmitted.density, transmitted.speed_of_sound)?,
        ))
    }

    /// Evaluate pressure and normal-power coefficients for a plane wave.
    ///
    /// `incidence_angle_radians` is measured from the interface normal on the
    /// incident side. Tangential wavenumber is conserved by Snell's law. The
    /// result handles total internal reflection with a complex reflected phase
    /// and zero transmitted normal power.
    pub fn try_plane_wave_transmission(
        self,
        incidence_angle_radians: f64,
    ) -> Result<PlaneWaveTransmission, MaterialError> {
        self.incident.validate()?;
        self.transmitted.validate()?;
        if !incidence_angle_radians.is_finite()
            || !(0.0..std::f64::consts::FRAC_PI_2).contains(&incidence_angle_radians)
        {
            return Err(MaterialError::InvalidIncidenceAngle(
                incidence_angle_radians,
            ));
        }

        let sin_incident = incidence_angle_radians.sin();
        let cos_incident = incidence_angle_radians.cos();
        let sin_transmitted =
            self.transmitted.speed_of_sound / self.incident.speed_of_sound * sin_incident;
        let normal_admittance_incident = cos_incident / self.incident.characteristic_impedance();
        let (normal_admittance_transmitted, total_internal_reflection) =
            if sin_transmitted.abs() <= 1.0 {
                (
                    Complex64::new(
                        (1.0 - sin_transmitted * sin_transmitted).sqrt()
                            / self.transmitted.characteristic_impedance(),
                        0.0,
                    ),
                    false,
                )
            } else {
                (
                    Complex64::new(
                        0.0,
                        (sin_transmitted * sin_transmitted - 1.0).sqrt()
                            / self.transmitted.characteristic_impedance(),
                    ),
                    true,
                )
            };
        let y_incident = Complex64::new(normal_admittance_incident, 0.0);
        let pressure_reflection = (y_incident - normal_admittance_transmitted)
            / (y_incident + normal_admittance_transmitted);
        let pressure_transmission = Complex64::new(1.0, 0.0) + pressure_reflection;
        let reflected_power = pressure_reflection.norm_sqr();
        let transmitted_power = if total_internal_reflection {
            0.0
        } else {
            normal_admittance_transmitted.re / normal_admittance_incident
                * pressure_transmission.norm_sqr()
        };
        if !pressure_reflection.re.is_finite()
            || !pressure_reflection.im.is_finite()
            || !pressure_transmission.re.is_finite()
            || !pressure_transmission.im.is_finite()
            || !reflected_power.is_finite()
            || !transmitted_power.is_finite()
        {
            return Err(MaterialError::NonFiniteResult);
        }
        Ok(PlaneWaveTransmission {
            pressure_reflection,
            pressure_transmission,
            reflected_power,
            transmitted_power,
            total_internal_reflection,
        })
    }
}

impl Default for AmbientMedium {
    fn default() -> Self {
        Self {
            density: 1.21,
            speed_of_sound: 343.0,
        }
    }
}

/// Empirical porous-layer model family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PorousModel {
    /// Delany-Bazley fibrous absorber model.
    DelanyBazley,
    /// Miki correction to Delany-Bazley with improved low-frequency behavior.
    Miki,
}

/// Rigid-backed equivalent-fluid porous absorber.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PorousLayer {
    /// Empirical impedance model.
    pub model: PorousModel,
    /// Layer thickness in meters.
    pub thickness: f64,
    /// Static flow resistivity in Pa*s/m^2.
    pub flow_resistivity: f64,
    /// Optional air gap behind the material in meters.
    pub air_gap: f64,
}

impl PorousLayer {
    /// Construct a validated Miki layer from typed exchange quantities.
    pub fn try_from_xem(
        thickness: sonium_xem::LengthM,
        flow_resistivity: sonium_xem::FlowResistivityPaSPerm2,
    ) -> Result<Self, MaterialError> {
        let layer = Self::miki(thickness.get(), flow_resistivity.get());
        layer.validate().map_err(MaterialError::InvalidLayer)?;
        Ok(layer)
    }

    /// Create a Delany-Bazley rigid-backed porous layer.
    pub fn delany_bazley(thickness: f64, flow_resistivity: f64) -> Self {
        Self {
            model: PorousModel::DelanyBazley,
            thickness,
            flow_resistivity,
            air_gap: 0.0,
        }
    }

    /// Create a Miki rigid-backed porous layer.
    pub fn miki(thickness: f64, flow_resistivity: f64) -> Self {
        Self {
            model: PorousModel::Miki,
            thickness,
            flow_resistivity,
            air_gap: 0.0,
        }
    }

    /// Add a rigid-backed air gap behind the porous layer.
    pub fn with_air_gap(mut self, air_gap: f64) -> Self {
        self.air_gap = air_gap;
        self
    }

    /// Validate model parameters.
    pub fn validate(self) -> Result<(), String> {
        if !self.thickness.is_finite() || self.thickness <= 0.0 {
            return Err(format!(
                "porous layer thickness must be finite and positive, got {}",
                self.thickness
            ));
        }
        if !self.flow_resistivity.is_finite() || self.flow_resistivity <= 0.0 {
            return Err(format!(
                "flow resistivity must be finite and positive, got {}",
                self.flow_resistivity
            ));
        }
        if !self.air_gap.is_finite() || self.air_gap < 0.0 {
            return Err(format!(
                "air gap must be finite and non-negative, got {}",
                self.air_gap
            ));
        }
        Ok(())
    }

    /// Return the equivalent-fluid characteristic impedance and propagation
    /// constant together with their frequency derivatives.
    ///
    /// Keeping the empirical Delany--Bazley/Miki coefficients in one helper
    /// makes the single-layer and multilayer transfer-matrix paths share the
    /// same model and derivative contract.
    ///
    /// Model limit: the dimensionless frequency `x = f/flow_resistivity` is
    /// floored at `1e-12` (with `dx/df = 0` below the floor) to keep the
    /// `x.powf(-0.6…-0.75)` terms finite. The floor creates a derivative kink
    /// at the boundary; frequencies that low are far outside the calibrated
    /// range of the Delany--Bazley/Miki fits, so treat results there as a
    /// regularized extension rather than model predictions.
    fn try_characteristic_data(
        self,
        frequency_hz: f64,
        medium: AmbientMedium,
    ) -> Result<(Complex64, Complex64, Complex64, Complex64), MaterialError> {
        self.validate().map_err(MaterialError::InvalidLayer)?;
        medium.validate()?;
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(MaterialError::InvalidFrequency(frequency_hz));
        }

        let rho_c = medium.characteristic_impedance();
        let raw_x = frequency_hz / self.flow_resistivity;
        let x = raw_x.max(1.0e-12);
        let dx_df = if raw_x >= 1.0e-12 {
            1.0 / self.flow_resistivity
        } else {
            0.0
        };
        let (zc_norm, dzc_norm_df, gamma_over_k, dgamma_over_k_df) = match self.model {
            PorousModel::DelanyBazley => (
                Complex64::new(1.0 + 0.0571 * x.powf(-0.754), -0.087 * x.powf(-0.732)),
                Complex64::new(
                    0.0571 * -0.754 * x.powf(-1.754) * dx_df,
                    -0.087 * -0.732 * x.powf(-1.732) * dx_df,
                ),
                Complex64::new(0.0978 * x.powf(-0.700), 1.0 + 0.189 * x.powf(-0.595)),
                Complex64::new(
                    0.0978 * -0.700 * x.powf(-1.700) * dx_df,
                    0.189 * -0.595 * x.powf(-1.595) * dx_df,
                ),
            ),
            PorousModel::Miki => (
                Complex64::new(1.0 + 0.070 * x.powf(-0.632), -0.107 * x.powf(-0.632)),
                Complex64::new(
                    0.070 * -0.632 * x.powf(-1.632) * dx_df,
                    -0.107 * -0.632 * x.powf(-1.632) * dx_df,
                ),
                Complex64::new(0.160 * x.powf(-0.618), 1.0 + 0.109 * x.powf(-0.618)),
                Complex64::new(
                    0.160 * -0.618 * x.powf(-1.618) * dx_df,
                    0.109 * -0.618 * x.powf(-1.618) * dx_df,
                ),
            ),
        };
        let k0 = 2.0 * std::f64::consts::PI * frequency_hz / medium.speed_of_sound;
        let dk0_df = 2.0 * std::f64::consts::PI / medium.speed_of_sound;
        let zc = zc_norm * rho_c;
        let dzc_df = dzc_norm_df * rho_c;
        let gamma = gamma_over_k * k0;
        let dgamma_df = dgamma_over_k_df * k0 + gamma_over_k * dk0_df;
        let values = [zc, dzc_df, gamma, dgamma_df];
        if values
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
        {
            Ok((zc, dzc_df, gamma, dgamma_df))
        } else {
            Err(MaterialError::NonFiniteResult)
        }
    }

    /// Normal-incidence surface impedance in Pa*s/m.
    ///
    /// # Panics
    ///
    /// Panics on invalid layer/medium/frequency input. Prefer
    /// [`try_surface_impedance`](Self::try_surface_impedance), which returns
    /// a [`MaterialError`] instead, in all library and application code.
    #[deprecated(note = "panics on invalid input; use try_surface_impedance")]
    pub fn surface_impedance(self, frequency_hz: f64, medium: AmbientMedium) -> Complex64 {
        self.try_surface_impedance(frequency_hz, medium)
            .expect("invalid porous material passed to surface_impedance")
    }

    /// Checked normal-incidence surface impedance in Pa*s/m.
    pub fn try_surface_impedance(
        self,
        frequency_hz: f64,
        medium: AmbientMedium,
    ) -> Result<Complex64, MaterialError> {
        self.validate().map_err(MaterialError::InvalidLayer)?;
        medium.validate()?;
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(MaterialError::InvalidFrequency(frequency_hz));
        }

        let rho_c = medium.characteristic_impedance();
        let x = (frequency_hz / self.flow_resistivity).max(1.0e-12);
        let (zc_norm, gamma_over_k) = match self.model {
            PorousModel::DelanyBazley => (
                Complex64::new(1.0 + 0.0571 * x.powf(-0.754), -0.087 * x.powf(-0.732)),
                Complex64::new(0.0978 * x.powf(-0.700), 1.0 + 0.189 * x.powf(-0.595)),
            ),
            PorousModel::Miki => (
                Complex64::new(1.0 + 0.070 * x.powf(-0.632), -0.107 * x.powf(-0.632)),
                Complex64::new(0.160 * x.powf(-0.618), 1.0 + 0.109 * x.powf(-0.618)),
            ),
        };

        let k0 = 2.0 * std::f64::consts::PI * frequency_hz / medium.speed_of_sound;
        let zc = zc_norm * rho_c;
        let gamma = gamma_over_k * k0;
        let propagation = gamma * self.thickness;

        let impedance = if self.air_gap <= 0.0 {
            zc / propagation.tanh()
        } else {
            let backing = Complex64::new(0.0, -rho_c / (k0 * self.air_gap).tan());
            zc * (backing + zc * propagation.tanh()) / (zc + backing * propagation.tanh())
        };
        if impedance.re.is_finite() && impedance.im.is_finite() {
            Ok(impedance)
        } else {
            Err(MaterialError::NonFiniteResult)
        }
    }

    /// Analytic frequency derivative of the normal-incidence surface
    /// impedance in Pa*s/(m*Hz).
    ///
    /// This differentiates the supported Delany--Bazley and Miki empirical
    /// models at fixed layer geometry and medium. It is not a shape or
    /// resistivity derivative.
    pub fn try_surface_impedance_frequency_derivative(
        self,
        frequency_hz: f64,
        medium: AmbientMedium,
    ) -> Result<Complex64, MaterialError> {
        self.validate().map_err(MaterialError::InvalidLayer)?;
        medium.validate()?;
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(MaterialError::InvalidFrequency(frequency_hz));
        }

        let rho_c = medium.characteristic_impedance();
        let raw_x = frequency_hz / self.flow_resistivity;
        let x = raw_x.max(1.0e-12);
        let dx_df = if raw_x >= 1.0e-12 {
            1.0 / self.flow_resistivity
        } else {
            0.0
        };
        let (zc_norm, dzc_norm_df, gamma_over_k, dgamma_over_k_df) = match self.model {
            PorousModel::DelanyBazley => (
                Complex64::new(1.0 + 0.0571 * x.powf(-0.754), -0.087 * x.powf(-0.732)),
                Complex64::new(
                    0.0571 * -0.754 * x.powf(-1.754) * dx_df,
                    -0.087 * -0.732 * x.powf(-1.732) * dx_df,
                ),
                Complex64::new(0.0978 * x.powf(-0.700), 1.0 + 0.189 * x.powf(-0.595)),
                Complex64::new(
                    0.0978 * -0.700 * x.powf(-1.700) * dx_df,
                    0.189 * -0.595 * x.powf(-1.595) * dx_df,
                ),
            ),
            PorousModel::Miki => (
                Complex64::new(1.0 + 0.070 * x.powf(-0.632), -0.107 * x.powf(-0.632)),
                Complex64::new(
                    0.070 * -0.632 * x.powf(-1.632) * dx_df,
                    -0.107 * -0.632 * x.powf(-1.632) * dx_df,
                ),
                Complex64::new(0.160 * x.powf(-0.618), 1.0 + 0.109 * x.powf(-0.618)),
                Complex64::new(
                    0.160 * -0.618 * x.powf(-1.618) * dx_df,
                    0.109 * -0.618 * x.powf(-1.618) * dx_df,
                ),
            ),
        };
        let k0 = 2.0 * std::f64::consts::PI * frequency_hz / medium.speed_of_sound;
        let dk0_df = 2.0 * std::f64::consts::PI / medium.speed_of_sound;
        let zc = zc_norm * rho_c;
        let dzc_df = dzc_norm_df * rho_c;
        let gamma = gamma_over_k * k0;
        let dgamma_df = dgamma_over_k_df * k0 + gamma_over_k * dk0_df;
        let propagation = gamma * self.thickness;
        let dpropagation_df = dgamma_df * self.thickness;
        let tanh_propagation = propagation.tanh();
        let dtanh_df =
            (Complex64::new(1.0, 0.0) - tanh_propagation * tanh_propagation) * dpropagation_df;

        let derivative = if self.air_gap <= 0.0 {
            dzc_df / tanh_propagation - zc * dtanh_df / (tanh_propagation * tanh_propagation)
        } else {
            let angle = k0 * self.air_gap;
            let backing = Complex64::new(0.0, -rho_c / angle.tan());
            let dbacking_df =
                Complex64::new(0.0, rho_c * self.air_gap * dk0_df / angle.sin().powi(2));
            let numerator = zc * (backing + zc * tanh_propagation);
            let denominator = zc + backing * tanh_propagation;
            let dnumerator_df = dzc_df * (backing + zc * tanh_propagation)
                + zc * (dbacking_df + dzc_df * tanh_propagation + zc * dtanh_df);
            let ddenominator_df = dzc_df + dbacking_df * tanh_propagation + backing * dtanh_df;
            (dnumerator_df * denominator - numerator * ddenominator_df)
                / (denominator * denominator)
        };
        if derivative.re.is_finite() && derivative.im.is_finite() {
            Ok(derivative)
        } else {
            Err(MaterialError::NonFiniteResult)
        }
    }

    /// Normalized acoustic admittance rho*c / Zs.
    ///
    /// # Panics
    ///
    /// Panics on invalid layer/medium/frequency input. Prefer
    /// [`try_normalized_admittance`](Self::try_normalized_admittance), which
    /// returns a [`MaterialError`] instead, in all library and application code.
    #[deprecated(note = "panics on invalid input; use try_normalized_admittance")]
    pub fn normalized_admittance(self, frequency_hz: f64, medium: AmbientMedium) -> Complex64 {
        self.try_normalized_admittance(frequency_hz, medium)
            .expect("invalid porous material passed to normalized_admittance")
    }

    /// Checked normalized acoustic admittance `rho*c / Zs`.
    pub fn try_normalized_admittance(
        self,
        frequency_hz: f64,
        medium: AmbientMedium,
    ) -> Result<Complex64, MaterialError> {
        let admittance = Complex64::new(medium.characteristic_impedance(), 0.0)
            / self.try_surface_impedance(frequency_hz, medium)?;
        if admittance.re.is_finite() && admittance.im.is_finite() {
            Ok(admittance)
        } else {
            Err(MaterialError::NonFiniteResult)
        }
    }

    /// Analytic frequency derivative of normalized acoustic admittance at
    /// fixed layer geometry and medium.
    pub fn try_normalized_admittance_frequency_derivative(
        self,
        frequency_hz: f64,
        medium: AmbientMedium,
    ) -> Result<Complex64, MaterialError> {
        let impedance = self.try_surface_impedance(frequency_hz, medium)?;
        let derivative = -Complex64::new(medium.characteristic_impedance(), 0.0)
            * self.try_surface_impedance_frequency_derivative(frequency_hz, medium)?
            / (impedance * impedance);
        if derivative.re.is_finite() && derivative.im.is_finite() {
            Ok(derivative)
        } else {
            Err(MaterialError::NonFiniteResult)
        }
    }

    /// Normal-incidence absorption coefficient.
    ///
    /// # Panics
    ///
    /// Panics on invalid layer/medium/frequency input. Prefer
    /// [`try_absorption_coefficient`](Self::try_absorption_coefficient),
    /// which returns a [`MaterialError`] instead, in all library and
    /// application code.
    #[deprecated(note = "panics on invalid input; use try_absorption_coefficient")]
    pub fn absorption_coefficient(self, frequency_hz: f64, medium: AmbientMedium) -> f64 {
        self.try_absorption_coefficient(frequency_hz, medium)
            .expect("invalid porous material passed to absorption_coefficient")
    }

    /// Checked normal-incidence absorption coefficient.
    pub fn try_absorption_coefficient(
        self,
        frequency_hz: f64,
        medium: AmbientMedium,
    ) -> Result<f64, MaterialError> {
        let z = self.try_surface_impedance(frequency_hz, medium)?;
        let rho_c = Complex64::new(medium.characteristic_impedance(), 0.0);
        let reflection = (z - rho_c) / (z + rho_c);
        let alpha = (1.0 - reflection.norm_sqr()).clamp(0.0, 1.0);
        alpha
            .is_finite()
            .then_some(alpha)
            .ok_or(MaterialError::NonFiniteResult)
    }
}

/// A checked normal-incidence stack of equivalent-fluid porous layers.
///
/// Layers are ordered from the incident fluid toward the backing.  Each
/// layer's `air_gap` is treated as a fluid spacer immediately behind that
/// layer, so the existing single-layer rigid-backed semantics are preserved
/// when a stack contains one layer.  The stack is intentionally a material
/// boundary model; callers can feed its impedance or normalized admittance to
/// the BEM impedance boundary condition without assembling a volume mesh.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayeredPorousSurface {
    /// Porous layers ordered from the incident side to the rigid backing.
    pub layers: Vec<PorousLayer>,
}

impl LayeredPorousSurface {
    /// Construct a validated non-empty porous stack.
    pub fn try_new(layers: Vec<PorousLayer>) -> Result<Self, MaterialError> {
        let surface = Self { layers };
        surface.validate()?;
        Ok(surface)
    }

    /// Validate the stack and every layer before evaluation.
    pub fn validate(&self) -> Result<(), MaterialError> {
        if self.layers.is_empty() {
            return Err(MaterialError::InvalidLayer(
                "layered porous surface requires at least one layer".to_string(),
            ));
        }
        for (index, layer) in self.layers.iter().copied().enumerate() {
            layer
                .validate()
                .map_err(|error| MaterialError::InvalidLayer(format!("layer {index}: {error}")))?;
        }
        Ok(())
    }

    /// Borrow the ordered layer list.
    pub fn layers(&self) -> &[PorousLayer] {
        &self.layers
    }

    /// Checked alias matching the `PorousLayer` naming convention.
    pub fn try_surface_impedance(
        &self,
        frequency_hz: f64,
        medium: AmbientMedium,
    ) -> Result<Complex64, MaterialError> {
        self.surface_impedance(frequency_hz, medium)
    }

    /// Return the physical velocity/pressure admittance (`1 / Z_s`).
    ///
    /// [`normalized_admittance`](Self::normalized_admittance) is convenient
    /// for comparing a surface against the ambient characteristic impedance,
    /// but BEM's `VelocityWithAdmittance` boundary condition consumes the
    /// dimensional admittance in `(m/s)/Pa`.  Keeping this conversion here
    /// prevents callers from accidentally passing the dimensionless value to
    /// the operator assembly.
    pub fn try_admittance(
        &self,
        frequency_hz: f64,
        medium: AmbientMedium,
    ) -> Result<Complex64, MaterialError> {
        let impedance = self.surface_impedance(frequency_hz, medium)?;
        let admittance = Complex64::new(1.0, 0.0) / impedance;
        if complex_is_finite(admittance) {
            Ok(admittance)
        } else {
            Err(MaterialError::NonFiniteResult)
        }
    }

    /// Infallible compatibility wrapper for [`Self::try_admittance`].
    ///
    /// # Panics
    ///
    /// Panics on invalid input. Prefer [`try_admittance`](Self::try_admittance)
    /// in all library and application code.
    #[deprecated(note = "panics on invalid input; use try_admittance")]
    pub fn admittance(&self, frequency_hz: f64, medium: AmbientMedium) -> Complex64 {
        self.try_admittance(frequency_hz, medium)
            .expect("validated layered porous surface admittance")
    }

    /// Analytic frequency derivative of the dimensional admittance.
    pub fn admittance_frequency_derivative(
        &self,
        frequency_hz: f64,
        medium: AmbientMedium,
    ) -> Result<Complex64, MaterialError> {
        let impedance = self.surface_impedance(frequency_hz, medium)?;
        let derivative = self.surface_impedance_frequency_derivative(frequency_hz, medium)?;
        let result = -derivative / (impedance * impedance);
        if complex_is_finite(result) {
            Ok(result)
        } else {
            Err(MaterialError::NonFiniteResult)
        }
    }

    /// Checked alias for [`Self::admittance_frequency_derivative`].
    pub fn try_admittance_frequency_derivative(
        &self,
        frequency_hz: f64,
        medium: AmbientMedium,
    ) -> Result<Complex64, MaterialError> {
        self.admittance_frequency_derivative(frequency_hz, medium)
    }

    /// Normal-incidence stack impedance in Pa*s/m.
    pub fn surface_impedance(
        &self,
        frequency_hz: f64,
        medium: AmbientMedium,
    ) -> Result<Complex64, MaterialError> {
        self.validate()?;
        medium.validate()?;
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(MaterialError::InvalidFrequency(frequency_hz));
        }
        let mut termination = None;
        for &layer in self.layers.iter().rev() {
            if layer.air_gap > 0.0 {
                termination = Some(input_impedance(
                    Complex64::new(medium.characteristic_impedance(), 0.0),
                    Complex64::new(
                        0.0,
                        2.0 * std::f64::consts::PI * frequency_hz / medium.speed_of_sound,
                    ),
                    layer.air_gap,
                    termination,
                )?);
            }
            let (zc, _, gamma, _) = layer.try_characteristic_data(frequency_hz, medium)?;
            termination = Some(input_impedance(zc, gamma, layer.thickness, termination)?);
        }
        termination.ok_or(MaterialError::NonFiniteResult)
    }

    /// Analytic frequency derivative of [`Self::surface_impedance`].
    pub fn surface_impedance_frequency_derivative(
        &self,
        frequency_hz: f64,
        medium: AmbientMedium,
    ) -> Result<Complex64, MaterialError> {
        self.validate()?;
        medium.validate()?;
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(MaterialError::InvalidFrequency(frequency_hz));
        }
        let mut termination: Option<(Complex64, Complex64)> = None;
        for &layer in self.layers.iter().rev() {
            if layer.air_gap > 0.0 {
                let k0 = 2.0 * std::f64::consts::PI * frequency_hz / medium.speed_of_sound;
                let dk0_df = 2.0 * std::f64::consts::PI / medium.speed_of_sound;
                termination = Some(input_impedance_with_derivative(
                    Complex64::new(medium.characteristic_impedance(), 0.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, k0),
                    Complex64::new(0.0, dk0_df),
                    layer.air_gap,
                    termination,
                )?);
            }
            let (zc, dzc_df, gamma, dgamma_df) =
                layer.try_characteristic_data(frequency_hz, medium)?;
            termination = Some(input_impedance_with_derivative(
                zc,
                dzc_df,
                gamma,
                dgamma_df,
                layer.thickness,
                termination,
            )?);
        }
        termination
            .map(|(_, derivative)| derivative)
            .ok_or(MaterialError::NonFiniteResult)
    }

    /// Normalized acoustic admittance `rho*c / Zs`.
    pub fn normalized_admittance(
        &self,
        frequency_hz: f64,
        medium: AmbientMedium,
    ) -> Result<Complex64, MaterialError> {
        let impedance = self.surface_impedance(frequency_hz, medium)?;
        let admittance = Complex64::new(medium.characteristic_impedance(), 0.0) / impedance;
        if complex_is_finite(admittance) {
            Ok(admittance)
        } else {
            Err(MaterialError::NonFiniteResult)
        }
    }

    /// Checked alias for normalized admittance.
    pub fn try_normalized_admittance(
        &self,
        frequency_hz: f64,
        medium: AmbientMedium,
    ) -> Result<Complex64, MaterialError> {
        self.normalized_admittance(frequency_hz, medium)
    }

    /// Analytic frequency derivative of normalized stack admittance.
    pub fn normalized_admittance_frequency_derivative(
        &self,
        frequency_hz: f64,
        medium: AmbientMedium,
    ) -> Result<Complex64, MaterialError> {
        let impedance = self.surface_impedance(frequency_hz, medium)?;
        let impedance_derivative =
            self.surface_impedance_frequency_derivative(frequency_hz, medium)?;
        let derivative = -Complex64::new(medium.characteristic_impedance(), 0.0)
            * impedance_derivative
            / (impedance * impedance);
        if complex_is_finite(derivative) {
            Ok(derivative)
        } else {
            Err(MaterialError::NonFiniteResult)
        }
    }

    /// Checked alias for the normalized-admittance frequency derivative.
    pub fn try_normalized_admittance_frequency_derivative(
        &self,
        frequency_hz: f64,
        medium: AmbientMedium,
    ) -> Result<Complex64, MaterialError> {
        self.normalized_admittance_frequency_derivative(frequency_hz, medium)
    }

    /// Normal-incidence absorption coefficient of the stack.
    pub fn absorption_coefficient(
        &self,
        frequency_hz: f64,
        medium: AmbientMedium,
    ) -> Result<f64, MaterialError> {
        let impedance = self.surface_impedance(frequency_hz, medium)?;
        let reference = Complex64::new(medium.characteristic_impedance(), 0.0);
        let denominator = impedance + reference;
        if denominator.norm() <= f64::MIN_POSITIVE {
            return Err(MaterialError::NonFiniteResult);
        }
        let reflection = (impedance - reference) / denominator;
        let absorption = 1.0 - reflection.norm_sqr();
        if absorption.is_finite() {
            Ok(absorption.clamp(0.0, 1.0))
        } else {
            Err(MaterialError::NonFiniteResult)
        }
    }

    /// Checked alias for the normal-incidence absorption coefficient.
    pub fn try_absorption_coefficient(
        &self,
        frequency_hz: f64,
        medium: AmbientMedium,
    ) -> Result<f64, MaterialError> {
        self.absorption_coefficient(frequency_hz, medium)
    }

    /// Evaluate absorption without clamping and enforce a passivity budget.
    ///
    /// [`Self::absorption_coefficient`] retains its historical clamped return
    /// value for compatibility.  Release/QA gates should use this method so
    /// an active or numerically invalid surface cannot be hidden by that
    /// clamp.
    pub fn try_passive_absorption(
        &self,
        frequency_hz: f64,
        medium: AmbientMedium,
        tolerance: f64,
    ) -> Result<f64, MaterialError> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(MaterialError::InvalidTolerance(tolerance));
        }
        let impedance = self.surface_impedance(frequency_hz, medium)?;
        let reference = Complex64::new(medium.characteristic_impedance(), 0.0);
        let denominator = impedance + reference;
        if denominator.norm() <= f64::MIN_POSITIVE {
            return Err(MaterialError::NonFiniteResult);
        }
        let reflection = (impedance - reference) / denominator;
        let absorption = 1.0 - reflection.norm_sqr();
        if !absorption.is_finite() {
            return Err(MaterialError::NonFiniteResult);
        }
        if absorption < -tolerance || absorption > 1.0 + tolerance {
            return Err(MaterialError::NonPassive(absorption));
        }
        Ok(absorption.clamp(0.0, 1.0))
    }
}

fn complex_is_finite(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

fn input_impedance(
    characteristic_impedance: Complex64,
    propagation_constant: Complex64,
    thickness: f64,
    termination: Option<Complex64>,
) -> Result<Complex64, MaterialError> {
    let propagation = propagation_constant * thickness;
    let tangent = propagation.tanh();
    if !complex_is_finite(tangent) {
        return Err(MaterialError::NonFiniteResult);
    }
    let value = match termination {
        Some(load) => {
            let denominator = characteristic_impedance + load * tangent;
            if !complex_is_finite(denominator) || denominator.norm() <= f64::MIN_POSITIVE {
                return Err(MaterialError::NonFiniteResult);
            }
            characteristic_impedance * (load + characteristic_impedance * tangent) / denominator
        }
        None => {
            if tangent.norm() <= f64::MIN_POSITIVE {
                return Err(MaterialError::NonFiniteResult);
            }
            characteristic_impedance / tangent
        }
    };
    if complex_is_finite(value) {
        Ok(value)
    } else {
        Err(MaterialError::NonFiniteResult)
    }
}

fn input_impedance_with_derivative(
    characteristic_impedance: Complex64,
    characteristic_derivative: Complex64,
    propagation_constant: Complex64,
    propagation_derivative: Complex64,
    thickness: f64,
    termination: Option<(Complex64, Complex64)>,
) -> Result<(Complex64, Complex64), MaterialError> {
    let propagation = propagation_constant * thickness;
    let propagation_derivative = propagation_derivative * thickness;
    let tangent = propagation.tanh();
    let tangent_derivative =
        (Complex64::new(1.0, 0.0) - tangent * tangent) * propagation_derivative;
    if !complex_is_finite(tangent) || !complex_is_finite(tangent_derivative) {
        return Err(MaterialError::NonFiniteResult);
    }
    let (value, derivative) = match termination {
        Some((load, load_derivative)) => {
            let numerator = characteristic_impedance * (load + characteristic_impedance * tangent);
            let denominator = characteristic_impedance + load * tangent;
            let numerator_derivative = characteristic_derivative
                * (load + characteristic_impedance * tangent)
                + characteristic_impedance
                    * (load_derivative
                        + characteristic_derivative * tangent
                        + characteristic_impedance * tangent_derivative);
            let denominator_derivative =
                characteristic_derivative + load_derivative * tangent + load * tangent_derivative;
            if !complex_is_finite(denominator) || denominator.norm() <= f64::MIN_POSITIVE {
                return Err(MaterialError::NonFiniteResult);
            }
            (
                numerator / denominator,
                (numerator_derivative * denominator - numerator * denominator_derivative)
                    / (denominator * denominator),
            )
        }
        None => {
            if tangent.norm() <= f64::MIN_POSITIVE {
                return Err(MaterialError::NonFiniteResult);
            }
            (
                characteristic_impedance / tangent,
                characteristic_derivative / tangent
                    - characteristic_impedance * tangent_derivative / (tangent * tangent),
            )
        }
    };
    if complex_is_finite(value) && complex_is_finite(derivative) {
        Ok((value, derivative))
    } else {
        Err(MaterialError::NonFiniteResult)
    }
}

/// Thin, locally reacting shell represented by a mass--spring--damper
/// surface impedance.
///
/// The parameters are specific (per unit area): areal mass `m` in kg/m²,
/// normal stiffness `k` in Pa/m, and viscous damping `c` in Pa·s/m.  With the
/// repository's canonical `exp(-iωt)` convention the pressure/velocity
/// relation is
/// `Z(f) = c + i (k / ω - ω m)`, and the BEM boundary consumes `1 / Z`.
/// This is a local thin-shell model; it intentionally does not claim bending
/// or in-plane modal coupling.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThinShellSurface {
    /// Shell mass per unit area in kg/m².
    pub mass_per_area_kg_m2: f64,
    /// Normal spring stiffness per unit area in Pa/m.
    pub stiffness_per_area_pa_per_m: f64,
    /// Viscous normal damping per unit area in Pa·s/m.
    pub damping_per_area_pa_s_per_m: f64,
}

impl ThinShellSurface {
    /// Construct a validated local thin-shell model.
    pub fn try_new(
        mass_per_area_kg_m2: f64,
        stiffness_per_area_pa_per_m: f64,
        damping_per_area_pa_s_per_m: f64,
    ) -> Result<Self, MaterialError> {
        let shell = Self {
            mass_per_area_kg_m2,
            stiffness_per_area_pa_per_m,
            damping_per_area_pa_s_per_m,
        };
        shell.validate()?;
        Ok(shell)
    }

    /// Validate shell parameters before any frequency evaluation.
    pub fn validate(self) -> Result<(), MaterialError> {
        if !self.mass_per_area_kg_m2.is_finite() || self.mass_per_area_kg_m2 <= 0.0 {
            return Err(MaterialError::InvalidThinShell(
                "mass_per_area_kg_m2 must be finite and positive".to_string(),
            ));
        }
        if !self.stiffness_per_area_pa_per_m.is_finite() || self.stiffness_per_area_pa_per_m < 0.0 {
            return Err(MaterialError::InvalidThinShell(
                "stiffness_per_area_pa_per_m must be finite and non-negative".to_string(),
            ));
        }
        if !self.damping_per_area_pa_s_per_m.is_finite() || self.damping_per_area_pa_s_per_m < 0.0 {
            return Err(MaterialError::InvalidThinShell(
                "damping_per_area_pa_s_per_m must be finite and non-negative".to_string(),
            ));
        }
        Ok(())
    }

    /// Checked normal-incidence surface impedance in Pa·s/m.
    pub fn try_surface_impedance(self, frequency_hz: f64) -> Result<Complex64, MaterialError> {
        self.validate()?;
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(MaterialError::InvalidFrequency(frequency_hz));
        }
        let omega = 2.0 * std::f64::consts::PI * frequency_hz;
        let reactance = self.stiffness_per_area_pa_per_m / omega - omega * self.mass_per_area_kg_m2;
        let impedance = Complex64::new(self.damping_per_area_pa_s_per_m, reactance);
        if complex_is_finite(impedance) {
            Ok(impedance)
        } else {
            Err(MaterialError::NonFiniteResult)
        }
    }

    /// Checked dimensional admittance in m/(Pa·s).
    pub fn try_admittance(self, frequency_hz: f64) -> Result<Complex64, MaterialError> {
        let impedance = self.try_surface_impedance(frequency_hz)?;
        if impedance.norm_sqr() <= f64::MIN_POSITIVE {
            return Err(MaterialError::NonFiniteResult);
        }
        let admittance = Complex64::new(1.0, 0.0) / impedance;
        if complex_is_finite(admittance) {
            Ok(admittance)
        } else {
            Err(MaterialError::NonFiniteResult)
        }
    }

    /// Analytic frequency derivative of the shell impedance.
    pub fn try_surface_impedance_frequency_derivative(
        self,
        frequency_hz: f64,
    ) -> Result<Complex64, MaterialError> {
        self.validate()?;
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(MaterialError::InvalidFrequency(frequency_hz));
        }
        let omega = 2.0 * std::f64::consts::PI * frequency_hz;
        let derivative = Complex64::new(
            0.0,
            -2.0 * std::f64::consts::PI
                * (self.mass_per_area_kg_m2 + self.stiffness_per_area_pa_per_m / (omega * omega)),
        );
        if complex_is_finite(derivative) {
            Ok(derivative)
        } else {
            Err(MaterialError::NonFiniteResult)
        }
    }

    /// Analytic frequency derivative of the shell admittance.
    pub fn try_admittance_frequency_derivative(
        self,
        frequency_hz: f64,
    ) -> Result<Complex64, MaterialError> {
        let impedance = self.try_surface_impedance(frequency_hz)?;
        let derivative = self.try_surface_impedance_frequency_derivative(frequency_hz)?;
        let result = -derivative / (impedance * impedance);
        if complex_is_finite(result) {
            Ok(result)
        } else {
            Err(MaterialError::NonFiniteResult)
        }
    }

    /// Checked normal-incidence absorption coefficient.
    pub fn try_absorption_coefficient(
        self,
        frequency_hz: f64,
        medium: AmbientMedium,
        tolerance: f64,
    ) -> Result<f64, MaterialError> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(MaterialError::InvalidTolerance(tolerance));
        }
        medium.validate()?;
        let impedance = self.try_surface_impedance(frequency_hz)?;
        let characteristic = Complex64::new(medium.characteristic_impedance(), 0.0);
        let reflection = (impedance - characteristic) / (impedance + characteristic);
        let absorption = 1.0 - reflection.norm_sqr();
        if !absorption.is_finite() || absorption < -tolerance || absorption > 1.0 + tolerance {
            return Err(MaterialError::NonPassive(absorption));
        }
        Ok(absorption.clamp(0.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layered_surface_matches_single_layer_and_manual_transfer_matrix() {
        let medium = AmbientMedium::default();
        let frequency = 800.0;
        let first = PorousLayer::miki(0.035, 12_000.0);
        let second = PorousLayer::delany_bazley(0.055, 18_000.0);

        let one_layer = LayeredPorousSurface::try_new(vec![first]).expect("valid stack");
        let single = first
            .try_surface_impedance(frequency, medium)
            .expect("single-layer impedance");
        let stacked_single = one_layer
            .surface_impedance(frequency, medium)
            .expect("stacked single-layer impedance");
        assert!((single - stacked_single).norm() < 1.0e-12);

        let stack = LayeredPorousSurface::try_new(vec![first, second]).expect("valid stack");
        let actual = stack
            .surface_impedance(frequency, medium)
            .expect("stack impedance");
        let (second_zc, _, second_gamma, _) = second
            .try_characteristic_data(frequency, medium)
            .expect("second data");
        let second_load = second_zc / (second_gamma * second.thickness).tanh();
        let (first_zc, _, first_gamma, _) = first
            .try_characteristic_data(frequency, medium)
            .expect("first data");
        let first_tangent = (first_gamma * first.thickness).tanh();
        let expected = first_zc * (second_load + first_zc * first_tangent)
            / (first_zc + second_load * first_tangent);
        assert!((actual - expected).norm() / expected.norm().max(1.0) < 1.0e-12);
        let alpha = stack
            .absorption_coefficient(frequency, medium)
            .expect("stack absorption");
        assert!((0.0..=1.0).contains(&alpha));
        assert!(
            stack
                .normalized_admittance(frequency, medium)
                .expect("stack admittance")
                .re
                .is_finite()
        );
    }

    #[test]
    fn layered_surface_frequency_derivative_matches_centered_difference() {
        let medium = AmbientMedium::default();
        let stack = LayeredPorousSurface::try_new(vec![
            PorousLayer::miki(0.03, 10_000.0).with_air_gap(0.02),
            PorousLayer::delany_bazley(0.045, 20_000.0),
        ])
        .expect("valid stack");
        let frequency = 1_200.0;
        let step = 0.25;
        let derivative = stack
            .surface_impedance_frequency_derivative(frequency, medium)
            .expect("analytic stack derivative");
        let plus = stack
            .surface_impedance(frequency + step, medium)
            .expect("plus impedance");
        let minus = stack
            .surface_impedance(frequency - step, medium)
            .expect("minus impedance");
        let centered = (plus - minus) / (2.0 * step);
        let scale = derivative.norm().max(centered.norm()).max(1.0);
        assert!((derivative - centered).norm() / scale < 2.0e-5);

        let admittance_derivative = stack
            .normalized_admittance_frequency_derivative(frequency, medium)
            .expect("analytic admittance derivative");
        let admittance_plus = stack
            .normalized_admittance(frequency + step, medium)
            .expect("plus admittance");
        let admittance_minus = stack
            .normalized_admittance(frequency - step, medium)
            .expect("minus admittance");
        let admittance_centered = (admittance_plus - admittance_minus) / (2.0 * step);
        let admittance_scale = admittance_derivative
            .norm()
            .max(admittance_centered.norm())
            .max(1.0);
        assert!((admittance_derivative - admittance_centered).norm() / admittance_scale < 2.0e-5);

        let dimensional_derivative = stack
            .admittance_frequency_derivative(frequency, medium)
            .expect("analytic dimensional admittance derivative");
        let dimensional_plus = stack
            .try_admittance(frequency + step, medium)
            .expect("plus dimensional admittance");
        let dimensional_minus = stack
            .try_admittance(frequency - step, medium)
            .expect("minus dimensional admittance");
        let dimensional_centered = (dimensional_plus - dimensional_minus) / (2.0 * step);
        let dimensional_scale = dimensional_derivative
            .norm()
            .max(dimensional_centered.norm())
            .max(1.0);
        assert!(
            (dimensional_derivative - dimensional_centered).norm() / dimensional_scale < 2.0e-5
        );
    }

    #[test]
    fn layered_surface_admittance_is_dimensional_reciprocal_of_impedance() {
        let medium = AmbientMedium::default();
        let stack = LayeredPorousSurface::try_new(vec![
            PorousLayer::miki(0.04, 14_000.0).with_air_gap(0.01),
            PorousLayer::delany_bazley(0.025, 9_000.0),
        ])
        .expect("valid stack");
        let impedance = stack
            .surface_impedance(900.0, medium)
            .expect("finite impedance");
        let expected = Complex64::new(1.0, 0.0) / impedance;
        let actual = stack
            .try_admittance(900.0, medium)
            .expect("finite dimensional admittance");
        assert!((actual - expected).norm() < 1.0e-12);
    }

    #[test]
    fn layered_surface_passivity_gate_is_unclamped_and_typed() {
        let medium = AmbientMedium::default();
        let stack = LayeredPorousSurface::try_new(vec![
            PorousLayer::miki(0.04, 14_000.0),
            PorousLayer::delany_bazley(0.03, 18_000.0).with_air_gap(0.005),
        ])
        .expect("valid stack");
        let absorption = stack
            .try_passive_absorption(1_000.0, medium, 1.0e-12)
            .expect("passive stack");
        assert!((0.0..=1.0).contains(&absorption));
        assert!(matches!(
            stack.try_passive_absorption(1_000.0, medium, -1.0),
            Err(MaterialError::InvalidTolerance(-1.0))
        ));
        assert!(matches!(
            stack.try_passive_absorption(1_000.0, medium, f64::NAN),
            Err(MaterialError::InvalidTolerance(value)) if value.is_nan()
        ));
    }

    #[test]
    fn layered_surface_rejects_empty_invalid_and_nonfinite_requests() {
        assert!(matches!(
            LayeredPorousSurface::try_new(Vec::new()),
            Err(MaterialError::InvalidLayer(_))
        ));
        let invalid = LayeredPorousSurface {
            layers: vec![PorousLayer::miki(-0.01, 12_000.0)],
        };
        assert!(matches!(
            invalid.surface_impedance(800.0, AmbientMedium::default()),
            Err(MaterialError::InvalidLayer(_))
        ));
        let valid = LayeredPorousSurface::try_new(vec![PorousLayer::miki(0.03, 12_000.0)])
            .expect("valid stack");
        assert!(matches!(
            valid.surface_impedance(f64::NAN, AmbientMedium::default()),
            Err(MaterialError::InvalidFrequency(_))
        ));
        assert!(matches!(
            valid.surface_impedance(800.0, AmbientMedium::new(0.0, 343.0)),
            Err(MaterialError::InvalidMedium { .. })
        ));
    }

    #[test]
    fn porous_layer_is_passive() {
        let layer = PorousLayer::miki(0.05, 12_000.0);
        let medium = AmbientMedium::default();

        for frequency in [125.0, 500.0, 2_000.0] {
            let z = layer
                .try_surface_impedance(frequency, medium)
                .expect("valid fixture input");
            let alpha = layer
                .try_absorption_coefficient(frequency, medium)
                .expect("valid fixture input");
            assert!(z.re > 0.0);
            assert!((0.0..=1.0).contains(&alpha));
        }
    }

    #[test]
    fn planar_fluid_interface_conserves_power_and_is_reciprocal_at_normal_incidence() {
        let air = AmbientMedium::default();
        let water = AmbientMedium::new(1_000.0, 1_480.0);
        let air_to_water = FluidInterface::new(air, water)
            .try_plane_wave_transmission(0.0)
            .expect("valid interface");
        let water_to_air = FluidInterface::new(water, air)
            .try_plane_wave_transmission(0.0)
            .expect("valid interface");
        assert!(air_to_water.power_balance_residual().abs() < 1.0e-12);
        assert!(water_to_air.power_balance_residual().abs() < 1.0e-12);
        assert!((air_to_water.transmitted_power - water_to_air.transmitted_power).abs() < 1.0e-12);
        assert!(
            (air_to_water.pressure_reflection + water_to_air.pressure_reflection).norm() < 1.0e-12
        );
    }

    #[test]
    fn planar_fluid_interface_stays_finite_at_grazing_incidence() {
        // Steep water -> air incidence (fast -> slow, no critical angle in
        // this direction): coefficients must stay finite and power-conserving
        // from near-normal up to grazing. The critical angle asin(343/1480)
        // ≈ 0.234 rad belongs to the reverse (air -> water) direction.
        let air = AmbientMedium::default();
        let water = AmbientMedium::new(1_000.0, 1_480.0);
        for angle in [0.23, 0.233, 0.24, 1.3, 1.5, 1.57] {
            let t = FluidInterface::new(water, air)
                .try_plane_wave_transmission(angle)
                .expect("valid grazing angle");
            assert!(t.pressure_reflection.re.is_finite());
            assert!(t.pressure_reflection.im.is_finite());
            assert!(t.transmitted_power.is_finite());
            assert!(t.power_balance_residual().abs() < 1.0e-9);
        }
        // Water -> air has no critical angle (fast -> slow always transmits):
        // at 0.5 rad most power reflects (|R| ~ 1) but a small transmitted
        // fraction remains and power is conserved. Total internal reflection
        // only exists in the air -> water direction (covered by
        // `planar_fluid_interface_handles_oblique_and_total_internal_reflection`).
        let fast_to_slow = FluidInterface::new(water, air)
            .try_plane_wave_transmission(0.5)
            .expect("valid fast-to-slow angle");
        assert!(!fast_to_slow.total_internal_reflection);
        assert!(fast_to_slow.transmitted_power > 0.0);
        assert!(fast_to_slow.transmitted_power < 1.0e-2);
        assert!(fast_to_slow.power_balance_residual().abs() < 1.0e-12);
    }

    #[test]
    fn planar_fluid_interface_handles_oblique_and_total_internal_reflection() {
        let water = AmbientMedium::new(1_000.0, 1_480.0);
        let air = AmbientMedium::default();
        let oblique = FluidInterface::new(air, water)
            .try_plane_wave_transmission(0.1)
            .expect("propagating oblique interface");
        assert!(!oblique.total_internal_reflection);
        assert!(oblique.power_balance_residual().abs() < 1.0e-12);

        let total = FluidInterface::new(air, water)
            .try_plane_wave_transmission(1.0)
            .expect("total internal reflection is physical");
        assert!(total.total_internal_reflection);
        assert_eq!(total.transmitted_power, 0.0);
        assert!((total.reflected_power - 1.0).abs() < 1.0e-12);
        assert!(matches!(
            FluidInterface::new(air, water)
                .try_plane_wave_transmission(std::f64::consts::FRAC_PI_2),
            Err(MaterialError::InvalidIncidenceAngle(_))
        ));
    }

    #[test]
    fn porous_absorption_rises_from_low_to_mid_band() {
        let layer = PorousLayer::delany_bazley(0.08, 10_000.0);
        let medium = AmbientMedium::default();

        let low = layer
            .try_absorption_coefficient(125.0, medium)
            .expect("valid fixture input");
        let mid = layer
            .try_absorption_coefficient(1_000.0, medium)
            .expect("valid fixture input");
        assert!(
            mid > low,
            "expected mid-band absorption {mid} > low-band {low}"
        );
    }

    #[test]
    fn air_gap_changes_low_frequency_absorption() {
        let flush = PorousLayer::miki(0.05, 12_000.0);
        let spaced = flush.with_air_gap(0.05);
        let medium = AmbientMedium::default();

        let flush_alpha = flush
            .try_absorption_coefficient(250.0, medium)
            .expect("valid fixture input");
        let spaced_alpha = spaced
            .try_absorption_coefficient(250.0, medium)
            .expect("valid fixture input");
        assert!((spaced_alpha - flush_alpha).abs() > 0.01);
    }

    #[test]
    fn checked_material_boundary_rejects_invalid_units_and_medium() {
        let layer = PorousLayer::miki(0.05, 12_000.0);
        let medium = AmbientMedium::default();
        assert!(matches!(
            layer.try_surface_impedance(0.0, medium),
            Err(MaterialError::InvalidFrequency(0.0))
        ));
        assert!(matches!(
            layer.try_surface_impedance(1_000.0, AmbientMedium::new(0.0, 343.0)),
            Err(MaterialError::InvalidMedium {
                field: "density",
                ..
            })
        ));
        assert!(matches!(
            PorousLayer::miki(-0.01, 12_000.0).try_surface_impedance(1_000.0, medium),
            Err(MaterialError::InvalidLayer(_))
        ));
    }

    #[test]
    fn checked_material_boundary_accepts_xem_quantities() {
        let medium = AmbientMedium::try_from_xem(
            sonium_xem::DensityKgM3::try_new(1.21).unwrap(),
            sonium_xem::SpeedOfSoundMPerS::try_new(343.0).unwrap(),
        )
        .unwrap();
        let impedance = PorousLayer::miki(0.05, 12_000.0)
            .try_surface_impedance(500.0, medium)
            .unwrap();
        assert!(impedance.re.is_finite() && impedance.im.is_finite());

        let typed_layer = PorousLayer::try_from_xem(
            sonium_xem::LengthM::try_new(0.05).unwrap(),
            sonium_xem::FlowResistivityPaSPerm2::try_new(12_000.0).unwrap(),
        )
        .unwrap();
        assert_eq!(typed_layer.thickness, 0.05);
        assert_eq!(typed_layer.flow_resistivity, 12_000.0);
    }

    #[test]
    fn porous_frequency_derivatives_match_central_difference_and_taylor_rates() {
        let medium = AmbientMedium::default();
        for layer in [
            PorousLayer::delany_bazley(0.08, 10_000.0),
            PorousLayer::miki(0.05, 12_000.0).with_air_gap(0.03),
        ] {
            let frequency = 800.0;
            let analytic = layer
                .try_surface_impedance_frequency_derivative(frequency, medium)
                .expect("valid porous derivative");
            let admittance_analytic = layer
                .try_normalized_admittance_frequency_derivative(frequency, medium)
                .expect("valid admittance derivative");
            let mut remainders = Vec::new();
            for step in [1.0, 0.5, 0.25] {
                let plus = layer
                    .try_surface_impedance(frequency + step, medium)
                    .expect("positive frequency");
                let minus = layer
                    .try_surface_impedance(frequency - step, medium)
                    .expect("positive frequency");
                let central = (plus - minus) / (2.0 * step);
                let scale = analytic.norm().max(central.norm()).max(1.0);
                assert!(
                    (analytic - central).norm() / scale < 2.0e-5,
                    "impedance derivative central-difference mismatch for {layer:?}"
                );

                let admittance_plus = layer
                    .try_normalized_admittance(frequency + step, medium)
                    .expect("positive frequency");
                let admittance_minus = layer
                    .try_normalized_admittance(frequency - step, medium)
                    .expect("positive frequency");
                let admittance_central = (admittance_plus - admittance_minus) / (2.0 * step);
                let admittance_scale = admittance_analytic
                    .norm()
                    .max(admittance_central.norm())
                    .max(1.0);
                assert!(
                    (admittance_analytic - admittance_central).norm() / admittance_scale < 2.0e-5,
                    "admittance derivative central-difference mismatch for {layer:?}"
                );

                let base = layer
                    .try_surface_impedance(frequency, medium)
                    .expect("positive frequency");
                remainders.push((plus - base - analytic * step).norm());
            }
            assert!(
                remainders[2] < remainders[1] * 0.3 && remainders[1] < remainders[0] * 0.3,
                "first-order Taylor remainder should be second-order: {remainders:?}"
            );
        }
    }

    #[test]
    fn fluid_interface_xem_adapter_rejects_mixed_exchange_metadata() {
        let incident = sonium_xem::AcousticMedium::try_new(500.0, 343.0, 1.21).unwrap();
        let transmitted = sonium_xem::AcousticMedium::try_new(500.0, 1_480.0, 1_000.0).unwrap();
        let interface = FluidInterface::try_from_xem(incident, transmitted).unwrap();
        let result = interface.try_plane_wave_transmission(0.0).unwrap();
        assert!(result.power_balance_residual().abs() < 1.0e-12);

        let different_frequency =
            sonium_xem::AcousticMedium::try_new(600.0, 1_480.0, 1_000.0).unwrap();
        assert!(matches!(
            FluidInterface::try_from_xem(incident, different_frequency),
            Err(MaterialError::InvalidConvention(_))
        ));

        let mut plus_convention = sonium_xem::ConventionRegistry::default();
        plus_convention.harmonic_time = sonium_xem::HarmonicConvention::ExpPlusIwt;
        let different_harmonic = sonium_xem::AcousticMedium::try_new_with_conventions(
            sonium_xem::FrequencyHz::new(500.0),
            sonium_xem::SpeedOfSoundMPerS::new(1_480.0),
            sonium_xem::DensityKgM3::new(1_000.0),
            plus_convention,
        )
        .unwrap();
        assert!(matches!(
            FluidInterface::try_from_xem(incident, different_harmonic),
            Err(MaterialError::InvalidConvention(_))
        ));
    }
}
