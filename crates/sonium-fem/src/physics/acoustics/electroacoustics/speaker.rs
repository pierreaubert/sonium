//! FEM-side contracts for coupled loudspeaker multiphysics workflows.

use num_complex::Complex64;

use crate::assembly::TripletMatrix;

use super::coupling::{CoupledSystem, CoupledSystemError, HarmonicCoupledPoint};
use super::transducer::ElectrodynamicCoupling;

/// Magnetic operating point used by perturbation and Lorentz coupling solves.
#[derive(Debug, Clone, PartialEq)]
pub struct MagneticBiasState {
    pub vector_potential_dofs: Vec<f64>,
    pub b_at_quadrature: Vec<[f64; 3]>,
}

impl MagneticBiasState {
    pub fn average_flux_density(&self) -> [f64; 3] {
        if self.b_at_quadrature.is_empty() {
            return [0.0; 3];
        }
        let mut sum = [0.0; 3];
        for b in &self.b_at_quadrature {
            for d in 0..3 {
                sum[d] += b[d];
            }
        }
        let inv = 1.0 / self.b_at_quadrature.len() as f64;
        [sum[0] * inv, sum[1] * inv, sum[2] * inv]
    }
}

/// FEM-owned description of a bias-plus-small-signal magnetic workflow.
#[derive(Debug, Clone, PartialEq)]
pub struct MagneticPerturbationWorkflow {
    pub bias_solve_name: String,
    pub perturbation_solve_name: String,
    pub reuse_bias_tangent: bool,
}

impl Default for MagneticPerturbationWorkflow {
    fn default() -> Self {
        Self {
            bias_solve_name: "stationary_magnetic_bias".to_string(),
            perturbation_solve_name: "harmonic_magnetic_perturbation".to_string(),
            reuse_bias_tangent: true,
        }
    }
}

/// One distributed Lorentz integration contribution.
#[derive(Debug, Clone, PartialEq)]
pub struct LorentzIntegrationPoint {
    pub dof_weights: Vec<(usize, [f64; 3])>,
    pub magnetic_flux_density: [f64; 3],
    pub conductor_direction: [f64; 3],
    pub jacobian_weight: f64,
}

impl LorentzIntegrationPoint {
    pub fn force_density_per_ampere(&self) -> [f64; 3] {
        cross(self.conductor_direction, self.magnetic_flux_density)
    }
}

/// Distributed Lorentz force assembly for voice coils with spatial field variation.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DistributedLorentzCoupling {
    pub points: Vec<LorentzIntegrationPoint>,
}

impl DistributedLorentzCoupling {
    pub fn force_vector_per_ampere(&self, n_solid_dofs: usize) -> Vec<f64> {
        let mut rhs = vec![0.0; n_solid_dofs];
        for point in &self.points {
            let f = point.force_density_per_ampere();
            for &(dof, shape) in &point.dof_weights {
                rhs[dof] += dot(f, shape) * point.jacobian_weight;
            }
        }
        rhs
    }

    pub fn bl_from_velocity_shape(&self, velocity_shape: &[f64]) -> f64 {
        self.force_vector_per_ampere(velocity_shape.len())
            .iter()
            .zip(velocity_shape.iter())
            .map(|(force, velocity)| force * velocity)
            .sum()
    }
}

/// Penalty tie between a solid voice-coil body DOF and a shell structure DOF.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolidShellTie {
    pub solid_dof: usize,
    pub shell_dof: usize,
    pub weight: f64,
}

/// Assemble `penalty * (u_solid - weight*u_shell)^2` into a monolithic block.
pub fn assemble_solid_shell_tie_penalty(
    n_solid_dofs: usize,
    n_shell_dofs: usize,
    ties: &[SolidShellTie],
    penalty: f64,
) -> TripletMatrix<f64> {
    let mut matrix = TripletMatrix::new(n_solid_dofs + n_shell_dofs, n_solid_dofs + n_shell_dofs);
    let shell_offset = n_solid_dofs;
    for tie in ties {
        let s = tie.solid_dof;
        let h = shell_offset + tie.shell_dof;
        let w = tie.weight;
        matrix.push(s, s, penalty);
        matrix.push(s, h, -penalty * w);
        matrix.push(h, s, -penalty * w);
        matrix.push(h, h, penalty * w * w);
    }
    matrix
}

/// Physical coefficients for acoustic-structure boundary coupling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcousticStructureBoundaryCoupling {
    pub fluid_to_structure: Complex64,
    pub structure_to_fluid: Complex64,
}

impl AcousticStructureBoundaryCoupling {
    pub fn pressure_displacement(omega: f64, fluid_density: f64) -> Self {
        Self {
            fluid_to_structure: Complex64::new(1.0, 0.0),
            structure_to_fluid: Complex64::new(-omega * omega * fluid_density, 0.0),
        }
    }
}

/// Speaker-gap model hierarchy selected from local boundary-layer thickness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeakerGapModel {
    FullThermoviscousRegion,
    BoundaryLayerImpedance,
    LumpedNarrowGapImpedance,
}

/// Pick a gap model from the gap size relative to viscous/thermal layers.
pub fn recommend_speaker_gap_model(
    gap: f64,
    viscous_delta: f64,
    thermal_delta: f64,
) -> SpeakerGapModel {
    let layer = viscous_delta.max(thermal_delta);
    if gap <= 6.0 * layer {
        SpeakerGapModel::FullThermoviscousRegion
    } else if gap <= 30.0 * layer {
        SpeakerGapModel::BoundaryLayerImpedance
    } else {
        SpeakerGapModel::LumpedNarrowGapImpedance
    }
}

/// Errors raised while resolving a thermoviscous speaker-gap model.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ThermoviscousGapError {
    /// Gap, frequency, or wall state was non-finite or non-positive.
    #[error("invalid thermoviscous gap input {field}: {detail}")]
    InvalidInput {
        /// Offending field name.
        field: &'static str,
        /// Human-readable detail.
        detail: String,
    },
}

/// Fully resolved thermoviscous gap state at one frequency.
///
/// Exactly one payload is populated, matching [`SpeakerGapModel`]:
/// the LRF equivalent-fluid coefficients (resolved or lumped narrow gap),
/// or the thermal boundary-layer admittance. The moving-wall state is
/// explicit: `moving_wall_velocity_rms` is `Some` only when the caller
/// supplied a wall velocity phasor (RMS, `exp(+iωt)`), and the resolved
/// region option means the caller must assemble a boundary-layer-resolving
/// mesh rather than an equivalent-fluid block.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedThermoviscousGap {
    /// Selected model tier.
    pub model: SpeakerGapModel,
    /// Viscous boundary-layer thickness `δ_v` in metres.
    pub viscous_delta_m: f64,
    /// Thermal boundary-layer thickness `δ_t` in metres.
    pub thermal_delta_m: f64,
    /// Ratio `gap / max(δ_v, δ_t)`.
    pub gap_to_layer_ratio: f64,
    /// LRF complex effective density in kg/m³ (equivalent-fluid tiers).
    pub lrf_effective_density: Option<Complex64>,
    /// LRF complex effective bulk modulus in Pa (equivalent-fluid tiers).
    pub lrf_effective_bulk_modulus_pa: Option<Complex64>,
    /// Thermal BLI wall admittance (boundary-layer-impedance tier).
    pub boundary_layer_admittance: Option<Complex64>,
    /// Moving-wall RMS normal velocity phasor, when explicitly supplied.
    pub moving_wall_velocity_rms: Option<Complex64>,
}

impl ResolvedThermoviscousGap {
    /// Whether the gap wall is an explicitly moving (driven) wall.
    pub fn is_moving_wall(&self) -> bool {
        self.moving_wall_velocity_rms.is_some()
    }

    /// Whether this tier requires an assembled, boundary-layer-resolved
    /// region instead of an equivalent-fluid or impedance closure.
    pub fn requires_resolved_region(&self) -> bool {
        matches!(self.model, SpeakerGapModel::FullThermoviscousRegion)
    }
}

/// Resolve the thermoviscous gap model at one frequency.
///
/// Uses [`recommend_speaker_gap_model`] for tier selection, then fills the
/// tier payload from the LRF equivalent-fluid model on a slit of half-width
/// `gap_m / 2` (the voice-coil gap) or the thermal BLI admittance. The wall
/// state is never assumed: pass `Some(velocity)` for a moving wall or
/// `None` for a rigid wall.
pub fn resolve_thermoviscous_gap(
    fluid: &crate::physics::acoustics::pressure::FluidProperties,
    frequency_hz: f64,
    gap_m: f64,
    moving_wall_velocity_rms: Option<Complex64>,
) -> Result<ResolvedThermoviscousGap, ThermoviscousGapError> {
    if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        return Err(ThermoviscousGapError::InvalidInput {
            field: "frequency_hz",
            detail: "must be finite and positive".to_string(),
        });
    }
    if !gap_m.is_finite() || gap_m <= 0.0 {
        return Err(ThermoviscousGapError::InvalidInput {
            field: "gap_m",
            detail: "must be finite and positive".to_string(),
        });
    }
    if let Some(velocity) = moving_wall_velocity_rms
        && !velocity.is_finite()
    {
        return Err(ThermoviscousGapError::InvalidInput {
            field: "moving_wall_velocity_rms",
            detail: "must be finite".to_string(),
        });
    }
    let omega = 2.0 * std::f64::consts::PI * frequency_hz;
    let viscous_delta_m = fluid.viscous_bl_thickness(omega);
    let thermal_delta_m = fluid.thermal_bl_thickness(omega);
    let model = recommend_speaker_gap_model(gap_m, viscous_delta_m, thermal_delta_m);
    let layer = viscous_delta_m.max(thermal_delta_m);
    let (lrf_density, lrf_bulk, bli) = match model {
        SpeakerGapModel::BoundaryLayerImpedance => (
            None,
            None,
            Some(crate::physics::acoustics::pressure::thermal_wall_admittance(fluid, omega)),
        ),
        SpeakerGapModel::FullThermoviscousRegion | SpeakerGapModel::LumpedNarrowGapImpedance => {
            let equivalent = crate::physics::acoustics::pressure::EquivalentFluid::new(
                fluid.clone(),
                crate::physics::acoustics::pressure::CrossSection::Slit {
                    half_width: 0.5 * gap_m,
                },
            );
            (
                Some(equivalent.effective_density(omega)),
                Some(equivalent.effective_bulk_modulus(omega)),
                None,
            )
        }
    };
    Ok(ResolvedThermoviscousGap {
        model,
        viscous_delta_m,
        thermal_delta_m,
        gap_to_layer_ratio: gap_m / layer,
        lrf_effective_density: lrf_density,
        lrf_effective_bulk_modulus_pa: lrf_bulk,
        boundary_layer_admittance: bli,
        moving_wall_velocity_rms,
    })
}

/// Frequency point for coupled electroacoustic sweep assembly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoupledSweepPoint {
    pub frequency_hz: f64,
    pub sound_speed: f64,
    pub fluid_density: f64,
}

impl CoupledSweepPoint {
    pub fn omega(self) -> f64 {
        2.0 * std::f64::consts::PI * self.frequency_hz
    }

    pub fn k_squared(self) -> Complex64 {
        let k = self.omega() / self.sound_speed;
        Complex64::new(k * k, 0.0)
    }

    pub fn coupling(self) -> AcousticStructureBoundaryCoupling {
        AcousticStructureBoundaryCoupling::pressure_displacement(self.omega(), self.fluid_density)
    }

    pub fn harmonic_point(self) -> HarmonicCoupledPoint {
        HarmonicCoupledPoint {
            omega_rad_per_s: self.omega(),
            sound_speed_m_per_s: self.sound_speed,
            fluid_density_kg_per_m3: self.fluid_density,
        }
    }
}

/// Reusable coupled system wrapper for frequency sweeps.
pub struct CoupledFrequencySweep<'a> {
    pub system: &'a CoupledSystem,
}

impl<'a> CoupledFrequencySweep<'a> {
    /// Assemble a physical harmonic point, panicking only for legacy callers
    /// that provide an invalid model or frequency point.
    pub fn assemble(&self, point: CoupledSweepPoint) -> TripletMatrix<Complex64> {
        self.try_assemble(point)
            .expect("invalid coupled electroacoustic frequency point")
    }

    pub fn try_assemble(
        &self,
        point: CoupledSweepPoint,
    ) -> Result<TripletMatrix<Complex64>, CoupledSystemError> {
        self.system.assemble_harmonic(point.harmonic_point())
    }

    /// Structural signature for detecting whether a sweep may reuse its
    /// sparsity pattern and symbolic factorization.
    pub fn reuse_signature(&self) -> Vec<usize> {
        let mut signature = vec![
            self.system.n_fluid,
            self.system.n_solid,
            self.system.fluid_stiffness.nnz(),
            self.system.fluid_mass.nnz(),
            self.system.fluid_damping.nnz(),
            self.system.solid_stiffness.nnz(),
            self.system.solid_mass.nnz(),
            self.system.solid_damping.nnz(),
            self.system.coupling.nnz(),
        ];
        if let Some(electrical) = &self.system.electrical {
            signature.extend([
                electrical.n_dofs,
                electrical.resistance.nnz(),
                electrical.inductance.nnz(),
                electrical.coupling.nnz(),
            ]);
        } else {
            signature.push(0);
        }
        signature
    }
}

/// Stable speaker result extraction contract for modeling crates and examples.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerResponse {
    pub frequency_hz: f64,
    pub voltage: Complex64,
    pub current: Complex64,
    pub impedance: Complex64,
    pub bl_force: Complex64,
    pub coil_velocity: Complex64,
    pub diaphragm_displacement: Complex64,
}

impl SpeakerResponse {
    pub fn from_lumped_voice_coil(
        frequency_hz: f64,
        voltage: Complex64,
        current: Complex64,
        coil_velocity: Complex64,
        diaphragm_displacement: Complex64,
        coupling: &ElectrodynamicCoupling,
    ) -> Self {
        Self {
            frequency_hz,
            voltage,
            current,
            impedance: voltage / current,
            bl_force: Complex64::new(coupling.bl, 0.0) * current,
            coil_velocity,
            diaphragm_displacement,
        }
    }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magnetic_bias_state_averages_quadrature_flux() {
        let state = MagneticBiasState {
            vector_potential_dofs: vec![0.0, 1.0],
            b_at_quadrature: vec![[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
        };
        assert_eq!(state.average_flux_density(), [2.0, 3.0, 4.0]);
    }

    #[test]
    fn distributed_lorentz_assembles_force_per_ampere() {
        let coupling = DistributedLorentzCoupling {
            points: vec![LorentzIntegrationPoint {
                dof_weights: vec![(0, [1.0, 0.0, 0.0]), (1, [0.0, 0.5, 0.0])],
                magnetic_flux_density: [0.0, 0.0, 2.0],
                conductor_direction: [0.0, 1.0, 0.0],
                jacobian_weight: 3.0,
            }],
        };
        let force = coupling.force_vector_per_ampere(2);
        assert!((force[0] - 6.0).abs() < 1.0e-12);
        assert_eq!(force[1], 0.0);
        assert!((coupling.bl_from_velocity_shape(&[0.25, 0.0]) - 1.5).abs() < 1.0e-12);
    }

    #[test]
    fn tie_penalty_is_symmetric_and_weighted() {
        let matrix = assemble_solid_shell_tie_penalty(
            2,
            1,
            &[SolidShellTie {
                solid_dof: 1,
                shell_dof: 0,
                weight: 0.5,
            }],
            10.0,
        )
        .to_csr();
        let a12: f64 = matrix
            .row_entries(1)
            .filter(|(j, _)| *j == 2)
            .map(|(_, v)| v)
            .sum();
        let a21: f64 = matrix
            .row_entries(2)
            .filter(|(j, _)| *j == 1)
            .map(|(_, v)| v)
            .sum();
        let a22: f64 = matrix
            .row_entries(2)
            .filter(|(j, _)| *j == 2)
            .map(|(_, v)| v)
            .sum();
        assert_eq!(a12, -5.0);
        assert_eq!(a21, -5.0);
        assert_eq!(a22, 2.5);
    }

    #[test]
    fn acoustic_structure_coupling_has_expected_sign() {
        let coupling = AcousticStructureBoundaryCoupling::pressure_displacement(10.0, 1.2);
        assert_eq!(coupling.fluid_to_structure, Complex64::new(1.0, 0.0));
        assert_eq!(coupling.structure_to_fluid, Complex64::new(-120.0, 0.0));
    }

    #[test]
    fn gap_model_hierarchy_selects_resolution_level() {
        assert_eq!(
            recommend_speaker_gap_model(1.0e-4, 2.0e-5, 1.0e-5),
            SpeakerGapModel::FullThermoviscousRegion
        );
        assert_eq!(
            recommend_speaker_gap_model(3.0e-4, 2.0e-5, 1.0e-5),
            SpeakerGapModel::BoundaryLayerImpedance
        );
        assert_eq!(
            recommend_speaker_gap_model(1.0e-3, 2.0e-5, 1.0e-5),
            SpeakerGapModel::LumpedNarrowGapImpedance
        );
    }

    #[test]
    fn sweep_point_builds_frequency_dependent_coefficients() {
        let point = CoupledSweepPoint {
            frequency_hz: 100.0,
            sound_speed: 340.0,
            fluid_density: 1.2,
        };
        assert!(point.omega() > 600.0);
        assert!(point.k_squared().re > 0.0);
        assert!(point.coupling().structure_to_fluid.re < 0.0);
    }

    #[test]
    fn thermoviscous_gap_tiers_carry_matching_payloads() {
        let fluid = crate::physics::acoustics::pressure::FluidProperties::air_20c();
        // Narrow gap at audio frequency: LRF equivalent-fluid payload.
        let narrow = resolve_thermoviscous_gap(&fluid, 1_000.0, 30.0e-6, None).unwrap();
        assert!(narrow.lrf_effective_density.is_some());
        assert!(narrow.boundary_layer_admittance.is_none());
        assert!(!narrow.is_moving_wall());
        // Wide gap: boundary-layer-impedance payload, explicit moving wall.
        let wide =
            resolve_thermoviscous_gap(&fluid, 1_000.0, 5.0e-3, Some(Complex64::new(0.01, 0.0)))
                .unwrap();
        assert!(wide.is_moving_wall());
        assert!(resolve_thermoviscous_gap(&fluid, 0.0, 1.0e-3, None).is_err());
        assert!(resolve_thermoviscous_gap(&fluid, 1_000.0, -1.0, None).is_err());
    }

    #[test]
    fn speaker_response_extracts_impedance_and_bl_force() {
        let coupling = ElectrodynamicCoupling {
            resistance: 6.0,
            inductance: 1.0e-3,
            bl: 5.0,
            coil_dofs: vec![(0, 1.0)],
        };
        let response = SpeakerResponse::from_lumped_voice_coil(
            100.0,
            Complex64::new(2.0, 0.0),
            Complex64::new(0.5, 0.0),
            Complex64::new(0.1, 0.0),
            Complex64::new(1.0e-4, 0.0),
            &coupling,
        );
        assert_eq!(response.impedance, Complex64::new(4.0, 0.0));
        assert_eq!(response.bl_force, Complex64::new(2.5, 0.0));
    }
}
