//! Shared numerical and physical conventions for speaker FEM studies.
//!
//! Every field solver, coupling operator, and exporter must use this contract.
//! Keeping the conventions executable avoids local sign, RMS/peak, and coordinate
//! assumptions drifting between the magnetic, structural, and acoustic models.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use sonium_fem::physics::electromagnetics::magnetostatics::MU_0;
use std::f64::consts::SQRT_2;

/// Version of the conventions and study-metadata schema.
pub const CONVENTIONS_SCHEMA_VERSION: u32 = 1;

/// Schema version for provenance attached to exported fields and reduced
/// model reports.
pub const ARTIFACT_PROVENANCE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactBackend {
    Cpu,
    Cuda,
    Metal,
    Wasm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactExecutionProfile {
    Preview,
    Golden,
    ProjectedReanalysis,
    Fixture,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactPrecision {
    F32,
    F64,
    Mixed,
}

/// Portable provenance shared by field-export sidecars and modeling reports.
///
/// The speaker-product crate has a parallel provenance type because it cannot
/// depend back on this modeling crate. Keeping the wire-level fields aligned
/// lets artifact consumers distinguish a native golden result from a reduced
/// WASM preview without inspecting filenames or build targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactProvenance {
    pub schema_version: u32,
    pub backend: ArtifactBackend,
    pub execution_profile: ArtifactExecutionProfile,
    pub precision: ArtifactPrecision,
    pub deterministic: bool,
}

impl ArtifactProvenance {
    pub const fn native_golden() -> Self {
        Self {
            schema_version: ARTIFACT_PROVENANCE_SCHEMA_VERSION,
            backend: ArtifactBackend::Cpu,
            execution_profile: ArtifactExecutionProfile::Golden,
            precision: ArtifactPrecision::F64,
            deterministic: true,
        }
    }

    pub const fn wasm_preview() -> Self {
        Self {
            schema_version: ARTIFACT_PROVENANCE_SCHEMA_VERSION,
            backend: ArtifactBackend::Wasm,
            execution_profile: ArtifactExecutionProfile::Preview,
            precision: ArtifactPrecision::F32,
            deterministic: false,
        }
    }

    pub const fn native_preview() -> Self {
        Self {
            schema_version: ARTIFACT_PROVENANCE_SCHEMA_VERSION,
            backend: ArtifactBackend::Cpu,
            execution_profile: ArtifactExecutionProfile::Preview,
            precision: ArtifactPrecision::F64,
            deterministic: true,
        }
    }
}

impl Default for ArtifactProvenance {
    fn default() -> Self {
        Self::native_golden()
    }
}

/// Peak-to-RMS conversion used at the product BEM/modeling boundary.
pub const PEAK_TO_RMS: f64 = 1.0 / SQRT_2;

/// Time dependence suppressed by a frequency-domain phasor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HarmonicTimeConvention {
    /// Physical fields are `Re{x exp(+i omega t)}`.
    ExpPositiveIomegaT,
}

impl HarmonicTimeConvention {
    /// Multiplier representing one time derivative at angular frequency `omega`.
    pub fn derivative_factor(self, omega_rad_per_s: f64) -> Complex64 {
        match self {
            Self::ExpPositiveIomegaT => Complex64::new(0.0, omega_rad_per_s),
        }
    }
}

/// Amplitude represented by a complex phasor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhasorAmplitude {
    /// Magnitudes are root-mean-square values.
    Rms,
    /// Magnitudes are sinusoidal peak values.
    Peak,
}

/// Reversible adaptation from an XEM acoustic phasor to this crate's
/// canonical `exp(+iωt)` RMS representation.
///
/// XEM documents make the time and amplitude conventions explicit, while the
/// speaker modeling kernels use one internal convention. Keeping this
/// conversion at the boundary prevents individual acoustic, mechanical, and
/// electrical consumers from independently deciding whether a phasor must be
/// conjugated or scaled by `sqrt(2)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExchangePhasorBridge {
    source: sonium_xem::ConventionRegistry,
}

impl ExchangePhasorBridge {
    /// The canonical exterior-BEM boundary: outward world-frame
    /// `exp(-iωt)` peak phasors.
    pub const BEM_PEAK: Self = Self {
        source: sonium_xem::ConventionRegistry {
            harmonic_time: sonium_xem::HarmonicConvention::ExpMinusIwt,
            coordinate_frame: sonium_xem::CoordinateFrame::WorldRightHanded,
            outward_normals: true,
            rms_amplitudes: false,
        },
    };

    /// Construct a bridge after checking the mesh-orientation and coordinate
    /// frame contracts shared by the speaker and XEM boundaries.
    pub fn from_xem(source: sonium_xem::ConventionRegistry) -> Result<Self, ConventionError> {
        if !source.outward_normals {
            return Err(ConventionError::InwardExchangeNormals);
        }
        if !matches!(
            source.coordinate_frame,
            sonium_xem::CoordinateFrame::WorldRightHanded
        ) {
            return Err(ConventionError::UnsupportedExchangeCoordinateFrame);
        }
        Ok(Self { source })
    }

    /// Return the source convention accepted by this bridge.
    pub const fn source(self) -> sonium_xem::ConventionRegistry {
        self.source
    }

    /// Convert an XEM acoustic phasor into the modeling RMS phasor.
    pub fn to_modeling_rms(self, value: Complex64) -> Complex64 {
        let rms = if self.source.rms_amplitudes {
            value
        } else {
            PhasorAmplitude::Peak.to_rms(value)
        };
        match self.source.harmonic_time {
            sonium_xem::HarmonicConvention::ExpPlusIwt => rms,
            sonium_xem::HarmonicConvention::ExpMinusIwt => rms.conj(),
        }
    }

    /// Convert a modeling RMS phasor back into the declared XEM convention.
    pub fn from_modeling_rms(self, value: Complex64) -> Complex64 {
        let source_time = match self.source.harmonic_time {
            sonium_xem::HarmonicConvention::ExpPlusIwt => value,
            sonium_xem::HarmonicConvention::ExpMinusIwt => value.conj(),
        };
        if self.source.rms_amplitudes {
            source_time
        } else {
            PhasorAmplitude::Peak.from_rms(source_time)
        }
    }
}

impl PhasorAmplitude {
    /// Convert a phasor from this amplitude convention to RMS.
    pub fn to_rms(self, value: Complex64) -> Complex64 {
        match self {
            Self::Rms => value,
            Self::Peak => value / SQRT_2,
        }
    }

    /// Convert an RMS phasor to this amplitude convention.
    pub fn from_rms(self, value: Complex64) -> Complex64 {
        match self {
            Self::Rms => value,
            Self::Peak => value * SQRT_2,
        }
    }

    /// Time-average active power for conjugate effort and flow phasors.
    pub fn active_power(self, effort: Complex64, flow: Complex64) -> f64 {
        let effort_rms = self.to_rms(effort);
        let flow_rms = self.to_rms(flow);
        (effort_rms * flow_rms.conj()).re
    }
}

/// Convert a BEM `e^{-iωt}` peak phasor into the modeling crate's canonical
/// `e^{+iωt}` RMS phasor.  The conjugation is part of the physical boundary
/// contract; it must not be left to individual driver or enclosure callers.
pub fn bem_peak_to_modeling_rms(value: Complex64) -> Complex64 {
    ExchangePhasorBridge::BEM_PEAK.to_modeling_rms(value)
}

/// Convert a modeling `e^{+iωt}` RMS phasor into the BEM `e^{-iωt}` peak
/// convention used by the exterior solver.
pub fn modeling_rms_to_bem_peak(value: Complex64) -> Complex64 {
    ExchangePhasorBridge::BEM_PEAK.from_modeling_rms(value)
}

/// Convert a mechanical impedance ratio from the BEM time convention to the
/// modeling time convention. Peak/RMS scaling cancels in an impedance ratio;
/// the time-sign change still requires conjugation.
pub fn bem_impedance_to_modeling(value: Complex64) -> Complex64 {
    value.conj()
}

/// Convert a modeling mechanical impedance to the BEM time convention.
pub fn modeling_impedance_to_bem(value: Complex64) -> Complex64 {
    value.conj()
}

/// Return the bending frequency of a thin, isotropic, clamped circular plate.
///
/// `mode_coefficient` is the dimensionless clamped-plate eigenvalue `λ` for
/// the selected radial mode.  The reference uses the independent closed form
/// `f = λ²/(2πa²) · sqrt(D/(ρh))`, with
/// `D = E h³ / (12(1-ν²))`.  It is intentionally a reference contract for
/// reduced structural projections, not a replacement for a resolved cone or
/// cabinet structural eigenproblem.
pub fn circular_plate_bending_frequency_hz(
    radius_m: f64,
    thickness_m: f64,
    density_kg_per_m3: f64,
    youngs_modulus_pa: f64,
    poisson_ratio: f64,
    mode_coefficient: f64,
) -> Result<f64, ConventionError> {
    validate_positive("radius_m", radius_m)?;
    validate_positive("thickness_m", thickness_m)?;
    validate_positive("density_kg_per_m3", density_kg_per_m3)?;
    validate_positive("youngs_modulus_pa", youngs_modulus_pa)?;
    validate_positive("mode_coefficient", mode_coefficient)?;
    if !poisson_ratio.is_finite() || !(0.0..0.5).contains(&poisson_ratio) {
        return Err(ConventionError::InvalidValue {
            field: "poisson_ratio",
        });
    }
    let flexural_rigidity =
        youngs_modulus_pa * thickness_m.powi(3) / (12.0 * (1.0 - poisson_ratio.powi(2)));
    let frequency_hz = mode_coefficient.powi(2)
        * (flexural_rigidity / (density_kg_per_m3 * thickness_m)).sqrt()
        / radius_m.powi(2)
        / std::f64::consts::TAU;
    if frequency_hz.is_finite() && frequency_hz > 0.0 {
        Ok(frequency_hz)
    } else {
        Err(ConventionError::InvalidValue {
            field: "circular_plate_frequency_hz",
        })
    }
}

/// Unit system used by public study inputs and exported fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnitSystem {
    /// SI base and derived units: metre, kilogram, second, ampere, pascal, tesla, etc.
    Si,
}

/// Orientation of an acoustic interface normal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcousticNormalConvention {
    /// Interface normals point outward from the acoustic fluid domain.
    OutwardFromFluid,
}

/// Ordering and meaning of mechanical unknowns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MechanicalDofConvention {
    /// Cartesian displacement components are ordered `(u_x, u_y, u_z)`.
    CartesianDisplacementXyz,
    /// Axisymmetric displacement components are ordered `(u_r, u_z)`.
    AxisymmetricDisplacementRz,
}

/// Coordinate and sign conventions used by electromagnetic models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ElectromagneticOrientation {
    /// Right-handed cylindrical `(r, theta, z)`, with positive `A_theta` and `J_theta`.
    RightHandedCylindricalRThetaZ,
}

/// Complete convention contract attached to every speaker study.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelingConventions {
    pub schema_version: u32,
    pub units: UnitSystem,
    pub harmonic_time: HarmonicTimeConvention,
    pub phasor_amplitude: PhasorAmplitude,
    pub acoustic_normal: AcousticNormalConvention,
    pub mechanical_cartesian_dofs: MechanicalDofConvention,
    pub mechanical_axisymmetric_dofs: MechanicalDofConvention,
    pub electromagnetic_orientation: ElectromagneticOrientation,
}

impl ModelingConventions {
    /// Canonical conventions for all Sonium speaker-modeling studies.
    pub const SONIUM: Self = Self {
        schema_version: CONVENTIONS_SCHEMA_VERSION,
        units: UnitSystem::Si,
        harmonic_time: HarmonicTimeConvention::ExpPositiveIomegaT,
        phasor_amplitude: PhasorAmplitude::Rms,
        acoustic_normal: AcousticNormalConvention::OutwardFromFluid,
        mechanical_cartesian_dofs: MechanicalDofConvention::CartesianDisplacementXyz,
        mechanical_axisymmetric_dofs: MechanicalDofConvention::AxisymmetricDisplacementRz,
        electromagnetic_orientation: ElectromagneticOrientation::RightHandedCylindricalRThetaZ,
    };
}

impl Default for ModelingConventions {
    fn default() -> Self {
        Self::SONIUM
    }
}

/// Metadata stored with a solved field or frequency-sweep sample.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StudyMetadata {
    pub study_name: String,
    pub conventions: ModelingConventions,
    pub frequency_hz: Option<f64>,
    pub drive_voltage_rms_v: Option<f64>,
    pub sector_angle_degrees: Option<f64>,
    pub circumferential_mode: i32,
    pub reference_case: Option<String>,
}

impl StudyMetadata {
    pub fn new(study_name: impl Into<String>) -> Self {
        Self {
            study_name: study_name.into(),
            conventions: ModelingConventions::SONIUM,
            frequency_hz: None,
            drive_voltage_rms_v: None,
            sector_angle_degrees: None,
            circumferential_mode: 0,
            reference_case: None,
        }
    }

    /// Validate values before attaching the metadata to exported results.
    pub fn validate(&self) -> Result<(), ConventionError> {
        if self.study_name.trim().is_empty() {
            return Err(ConventionError::BlankStudyName);
        }
        if self.conventions != ModelingConventions::SONIUM {
            return Err(ConventionError::UnsupportedConventions);
        }
        validate_optional_positive("frequency_hz", self.frequency_hz)?;
        validate_optional_nonnegative("drive_voltage_rms_v", self.drive_voltage_rms_v)?;
        if let Some(angle) = self.sector_angle_degrees
            && (!angle.is_finite() || !(0.0..=360.0).contains(&angle) || angle == 0.0)
        {
            return Err(ConventionError::InvalidValue {
                field: "sector_angle_degrees",
            });
        }
        if self.sector_angle_degrees.is_none() && self.circumferential_mode != 0 {
            return Err(ConventionError::ModeRequiresSector);
        }
        if self
            .reference_case
            .as_ref()
            .is_some_and(|name| name.trim().is_empty())
        {
            return Err(ConventionError::BlankReferenceCase);
        }
        Ok(())
    }
}

/// Analytical configurations used as mesh, sign, and convergence references.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReferenceModel {
    BaffledPiston {
        radius_m: f64,
        velocity_rms_m_per_s: f64,
        observation_distance_m: f64,
    },
    PulsatingSphere {
        radius_m: f64,
        radial_velocity_rms_m_per_s: f64,
        observation_distance_m: f64,
    },
    RigidRectangularCavity {
        dimensions_m: [f64; 3],
    },
    CircularRigidDuct {
        radius_m: f64,
        length_m: f64,
    },
    UniformMagneticCircuit {
        path_length_m: f64,
        cross_section_m2: f64,
        relative_permeability: f64,
        turns: u32,
        current_rms_a: f64,
    },
}

impl ReferenceModel {
    pub fn name(&self) -> &'static str {
        match self {
            Self::BaffledPiston { .. } => "baffled_piston",
            Self::PulsatingSphere { .. } => "pulsating_sphere",
            Self::RigidRectangularCavity { .. } => "rigid_rectangular_cavity",
            Self::CircularRigidDuct { .. } => "circular_rigid_duct",
            Self::UniformMagneticCircuit { .. } => "uniform_magnetic_circuit",
        }
    }

    pub fn validate(&self) -> Result<(), ConventionError> {
        match *self {
            Self::BaffledPiston {
                radius_m,
                velocity_rms_m_per_s,
                observation_distance_m,
            } => {
                validate_positive("radius_m", radius_m)?;
                validate_nonnegative("velocity_rms_m_per_s", velocity_rms_m_per_s)?;
                validate_positive("observation_distance_m", observation_distance_m)
            }
            Self::PulsatingSphere {
                radius_m,
                radial_velocity_rms_m_per_s,
                observation_distance_m,
            } => {
                validate_positive("radius_m", radius_m)?;
                validate_nonnegative("radial_velocity_rms_m_per_s", radial_velocity_rms_m_per_s)?;
                validate_positive("observation_distance_m", observation_distance_m)?;
                if observation_distance_m < radius_m {
                    return Err(ConventionError::InvalidValue {
                        field: "observation_distance_m",
                    });
                }
                Ok(())
            }
            Self::RigidRectangularCavity { dimensions_m } => {
                for dimension in dimensions_m {
                    validate_positive("dimensions_m", dimension)?;
                }
                Ok(())
            }
            Self::CircularRigidDuct { radius_m, length_m } => {
                validate_positive("radius_m", radius_m)?;
                validate_positive("length_m", length_m)
            }
            Self::UniformMagneticCircuit {
                path_length_m,
                cross_section_m2,
                relative_permeability,
                turns,
                current_rms_a,
            } => {
                validate_positive("path_length_m", path_length_m)?;
                validate_positive("cross_section_m2", cross_section_m2)?;
                validate_positive("relative_permeability", relative_permeability)?;
                if turns == 0 {
                    return Err(ConventionError::InvalidValue { field: "turns" });
                }
                validate_nonnegative("current_rms_a", current_rms_a)
            }
        }
    }

    /// First non-plane-mode cutoff of a hard-walled circular duct.
    pub fn first_higher_mode_cutoff_hz(
        &self,
        sound_speed_m_per_s: f64,
    ) -> Option<Result<f64, ConventionError>> {
        let Self::CircularRigidDuct { radius_m, .. } = *self else {
            return None;
        };
        Some((|| {
            self.validate()?;
            validate_positive("sound_speed_m_per_s", sound_speed_m_per_s)?;
            Ok(1.8412 * sound_speed_m_per_s / (2.0 * std::f64::consts::PI * radius_m))
        })())
    }

    /// Return the uniform magnetic-circuit coenergy and closing force used by
    /// the speaker reference ladder.
    ///
    /// The circuit is intentionally the small-signal constant-μ reference:
    /// `L = μ₀μᵣ A N² / l`, `W' = L I² / 2`, and
    /// `F = μ₀μᵣ A (N I / l)² / 2`.  Nonlinear FEM bias results
    /// remain the production source of motor force; this helper is an
    /// independently evaluable admission/reference contract.
    pub fn uniform_magnetic_circuit_energy_force(
        &self,
    ) -> Option<Result<(f64, f64), ConventionError>> {
        let Self::UniformMagneticCircuit {
            path_length_m,
            cross_section_m2,
            relative_permeability,
            turns,
            current_rms_a,
        } = *self
        else {
            return None;
        };
        Some((|| {
            self.validate()?;
            let turns = f64::from(turns);
            let inductance_h =
                MU_0 * relative_permeability * cross_section_m2 * turns * turns / path_length_m;
            let energy_j = 0.5 * inductance_h * current_rms_a * current_rms_a;
            let force_n = 0.5
                * MU_0
                * relative_permeability
                * cross_section_m2
                * (turns * current_rms_a / path_length_m).powi(2);
            if !energy_j.is_finite() || !force_n.is_finite() {
                return Err(ConventionError::InvalidValue {
                    field: "uniform_magnetic_circuit_result",
                });
            }
            Ok((energy_j, force_n))
        })())
    }

    /// Analytical rigid-wall rectangular-cavity mode used for enclosure
    /// resonance checks. The zero mode is rejected because it is the static
    /// pressure mode, not an acoustic resonance.
    pub fn rigid_rectangular_cavity_mode_hz(
        &self,
        sound_speed_m_per_s: f64,
        mode: [u32; 3],
    ) -> Option<Result<f64, ConventionError>> {
        let Self::RigidRectangularCavity { dimensions_m } = *self else {
            return None;
        };
        Some((|| {
            self.validate()?;
            validate_positive("sound_speed_m_per_s", sound_speed_m_per_s)?;
            if mode == [0, 0, 0] {
                return Err(ConventionError::InvalidValue { field: "mode" });
            }
            let squared_sum = mode
                .into_iter()
                .zip(dimensions_m)
                .map(|(index, length)| f64::from(index) / length)
                .map(|value| value * value)
                .sum::<f64>();
            Ok(0.5 * sound_speed_m_per_s * squared_sum.sqrt())
        })())
    }
}

/// Baseline analytical cases required by the speaker FEM validation ladder.
pub fn baseline_reference_models() -> Vec<ReferenceModel> {
    vec![
        ReferenceModel::BaffledPiston {
            radius_m: 0.05,
            velocity_rms_m_per_s: 0.01,
            observation_distance_m: 1.0,
        },
        ReferenceModel::PulsatingSphere {
            radius_m: 0.05,
            radial_velocity_rms_m_per_s: 0.01,
            observation_distance_m: 1.0,
        },
        ReferenceModel::RigidRectangularCavity {
            dimensions_m: [0.4, 0.3, 0.25],
        },
        ReferenceModel::CircularRigidDuct {
            radius_m: 0.035,
            length_m: 0.15,
        },
        ReferenceModel::UniformMagneticCircuit {
            path_length_m: 0.1,
            cross_section_m2: 1.0e-4,
            relative_permeability: 1_000.0,
            turns: 100,
            current_rms_a: 1.0,
        },
    ]
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ConventionError {
    #[error("study name must not be blank")]
    BlankStudyName,
    #[error("only the canonical Sonium convention set is supported")]
    UnsupportedConventions,
    #[error("{field} is not finite or outside its physical range")]
    InvalidValue { field: &'static str },
    #[error("a nonzero circumferential mode requires a periodic sector")]
    ModeRequiresSector,
    #[error("reference case name must not be blank")]
    BlankReferenceCase,
    #[error("XEM acoustic boundary requires outward physical-domain normals")]
    InwardExchangeNormals,
    #[error("XEM acoustic boundary requires the global right-handed world frame")]
    UnsupportedExchangeCoordinateFrame,
}

fn validate_positive(field: &'static str, value: f64) -> Result<(), ConventionError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(ConventionError::InvalidValue { field })
    }
}

fn validate_nonnegative(field: &'static str, value: f64) -> Result<(), ConventionError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(ConventionError::InvalidValue { field })
    }
}

fn validate_optional_positive(
    field: &'static str,
    value: Option<f64>,
) -> Result<(), ConventionError> {
    value.map_or(Ok(()), |value| validate_positive(field, value))
}

fn validate_optional_nonnegative(
    field: &'static str,
    value: Option<f64>,
) -> Result<(), ConventionError> {
    value.map_or(Ok(()), |value| validate_nonnegative(field, value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_iomega_time_derivative_maps_displacement_to_velocity() {
        let displacement = Complex64::new(0.25, -0.5);
        let omega = 20.0;
        let velocity =
            HarmonicTimeConvention::ExpPositiveIomegaT.derivative_factor(omega) * displacement;

        assert_eq!(velocity, Complex64::new(10.0, 5.0));
    }

    #[test]
    fn peak_and_rms_phasors_report_the_same_active_power() {
        let voltage_rms = Complex64::new(2.0, 1.0);
        let current_rms = Complex64::new(0.5, -0.25);
        let rms_power = PhasorAmplitude::Rms.active_power(voltage_rms, current_rms);
        let peak_power = PhasorAmplitude::Peak.active_power(
            PhasorAmplitude::Peak.from_rms(voltage_rms),
            PhasorAmplitude::Peak.from_rms(current_rms),
        );

        assert!((rms_power - peak_power).abs() < 1.0e-14);
    }

    #[test]
    fn bem_modeling_bridge_preserves_impedance_and_power_conventions() {
        let bem_velocity = Complex64::new(0.4, -0.2);
        let modeling_velocity = bem_peak_to_modeling_rms(bem_velocity);
        let round_trip = modeling_rms_to_bem_peak(modeling_velocity);
        assert!((round_trip - bem_velocity).norm() < 1.0e-14);

        let bem_impedance = Complex64::new(2.0, -3.0);
        assert_eq!(
            bem_impedance_to_modeling(bem_impedance),
            bem_impedance.conj()
        );
        assert_eq!(
            modeling_impedance_to_bem(bem_impedance_to_modeling(bem_impedance)),
            bem_impedance
        );
    }

    #[test]
    fn exchange_phasor_bridge_handles_every_xem_time_and_amplitude_pair() {
        let modeling_rms = Complex64::new(0.4, -0.2);
        for harmonic_time in [
            sonium_xem::HarmonicConvention::ExpMinusIwt,
            sonium_xem::HarmonicConvention::ExpPlusIwt,
        ] {
            for rms_amplitudes in [false, true] {
                let source = sonium_xem::ConventionRegistry {
                    harmonic_time,
                    rms_amplitudes,
                    ..Default::default()
                };
                let bridge = ExchangePhasorBridge::from_xem(source).expect("valid XEM boundary");
                let encoded = bridge.from_modeling_rms(modeling_rms);
                assert!(
                    (bridge.to_modeling_rms(encoded) - modeling_rms).norm() < 1.0e-14,
                    "round trip failed for {source:?}"
                );
                assert_eq!(bridge.source(), source);
            }
        }
    }

    #[test]
    fn exchange_phasor_bridge_rejects_incompatible_geometry_conventions() {
        let inward = sonium_xem::ConventionRegistry {
            outward_normals: false,
            ..Default::default()
        };
        assert_eq!(
            ExchangePhasorBridge::from_xem(inward),
            Err(ConventionError::InwardExchangeNormals)
        );

        let local_frame = sonium_xem::ConventionRegistry {
            coordinate_frame: sonium_xem::CoordinateFrame::SourceLocalRightHanded,
            ..Default::default()
        };
        assert_eq!(
            ExchangePhasorBridge::from_xem(local_frame),
            Err(ConventionError::UnsupportedExchangeCoordinateFrame)
        );
    }

    #[test]
    fn metadata_round_trips_with_explicit_canonical_conventions() {
        let mut metadata = StudyMetadata::new("axisymmetric golden model");
        metadata.frequency_hz = Some(1_000.0);
        metadata.drive_voltage_rms_v = Some(2.83);
        metadata.sector_angle_degrees = Some(30.0);
        metadata.circumferential_mode = 1;
        metadata.reference_case = Some("baffled_piston".to_owned());
        metadata.validate().expect("valid metadata");

        let json = serde_json::to_string(&metadata).expect("serialize metadata");
        let decoded: StudyMetadata = serde_json::from_str(&json).expect("deserialize metadata");
        assert_eq!(decoded, metadata);
        assert!(json.contains("ExpPositiveIomegaT"));
        assert!(json.contains("Rms"));
    }

    #[test]
    fn nonzero_circumferential_mode_requires_a_sector() {
        let mut metadata = StudyMetadata::new("invalid full model");
        metadata.circumferential_mode = 2;
        assert_eq!(
            metadata.validate(),
            Err(ConventionError::ModeRequiresSector)
        );
    }

    #[test]
    fn baseline_reference_suite_is_valid_and_named_uniquely() {
        let references = baseline_reference_models();
        assert_eq!(references.len(), 5);
        for reference in &references {
            reference.validate().expect("valid baseline reference");
        }
        let mut names: Vec<_> = references.iter().map(ReferenceModel::name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), references.len());
    }

    #[test]
    fn circular_duct_cutoff_uses_first_hard_wall_root() {
        let duct = ReferenceModel::CircularRigidDuct {
            radius_m: 0.05,
            length_m: 0.5,
        };
        let cutoff = duct
            .first_higher_mode_cutoff_hz(343.0)
            .expect("duct reference")
            .expect("valid duct and medium");
        let expected = 1.8412 * 343.0 / (2.0 * std::f64::consts::PI * 0.05);
        assert!((cutoff - expected).abs() < 1.0e-12);
    }

    #[test]
    fn reference_models_reject_nonphysical_dimensions() {
        let reference = ReferenceModel::PulsatingSphere {
            radius_m: 1.0,
            radial_velocity_rms_m_per_s: 0.1,
            observation_distance_m: 0.5,
        };
        assert!(reference.validate().is_err());
    }

    #[test]
    fn uniform_magnetic_reference_matches_closed_form_energy_and_force() {
        let reference = ReferenceModel::UniformMagneticCircuit {
            path_length_m: 0.1,
            cross_section_m2: 1.0e-4,
            relative_permeability: 1_000.0,
            turns: 100,
            current_rms_a: 1.0,
        };
        let (energy_j, force_n) = reference
            .uniform_magnetic_circuit_energy_force()
            .expect("magnetic reference should expose energy and force")
            .expect("physical magnetic reference should validate");
        let inductance_h = MU_0 * 1_000.0 * 1.0e-4 * 100.0_f64.powi(2) / 0.1;
        assert!((energy_j - 0.5 * inductance_h).abs() < 1.0e-15);
        let expected_force = 0.5 * MU_0 * 1_000.0 * 1.0e-4 * (100.0_f64 / 0.1_f64).powi(2);
        assert!((force_n - expected_force).abs() / expected_force < 1.0e-14);
        assert!(
            ReferenceModel::BaffledPiston {
                radius_m: 0.05,
                velocity_rms_m_per_s: 0.01,
                observation_distance_m: 1.0,
            }
            .uniform_magnetic_circuit_energy_force()
            .is_none()
        );
    }

    #[test]
    fn rectangular_cavity_reference_matches_axial_mode_and_rejects_static_mode() {
        let reference = ReferenceModel::RigidRectangularCavity {
            dimensions_m: [0.4, 0.3, 0.25],
        };
        let mode = reference
            .rigid_rectangular_cavity_mode_hz(343.0, [1, 0, 0])
            .expect("cavity reference should expose modes")
            .expect("physical cavity reference should validate");
        assert!((mode - 343.0 / (2.0 * 0.4)).abs() < 1.0e-12);
        assert!(
            reference
                .rigid_rectangular_cavity_mode_hz(343.0, [0, 0, 0])
                .expect("cavity reference should expose modes")
                .is_err()
        );
    }

    #[test]
    fn circular_plate_reference_matches_closed_form_and_rejects_invalid_materials() {
        let frequency =
            circular_plate_bending_frequency_hz(0.1, 0.001, 1_200.0, 2.5e9, 0.35, 10.215)
                .expect("valid circular-plate reference");
        let flexural_rigidity = 2.5e9 * 0.001_f64.powi(3) / (12.0 * (1.0 - 0.35_f64.powi(2)));
        let expected = 10.215_f64.powi(2) * (flexural_rigidity / (1_200.0 * 0.001)).sqrt()
            / 0.1_f64.powi(2)
            / std::f64::consts::TAU;
        assert!((frequency - expected).abs() / expected < 1.0e-14);
        assert!(
            circular_plate_bending_frequency_hz(0.1, 0.001, 1_200.0, 2.5e9, 0.5, 10.215).is_err()
        );
        assert!(
            circular_plate_bending_frequency_hz(0.1, 0.001, 1_200.0, 2.5e9, 0.35, 0.0).is_err()
        );
    }
}
