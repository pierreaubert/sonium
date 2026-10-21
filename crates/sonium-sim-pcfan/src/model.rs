use serde::{Deserialize, Serialize};
use sonium_solvers_optimization::OptimStatus;

use crate::resolved_adjoint::PcFanResolvedFlowPicardLinearization;

pub const DEFAULT_BLADE_COUNT: usize = 8;
pub const FRAME_MOUNT_COUNT: usize = 4;
pub const SUPPORT_STRUT_COUNT: usize = 4;
pub const BLADE_PATH_CONTROL_COUNT: usize = 5;
pub const BLADE_SECTION_STATION_COUNT: usize = 5;
pub const SHROUD_CONTROL_COUNT: usize = 5;
/// Number of scalar radial-envelope controls: hub radius and blade-tip radius.
pub const RADIAL_ENVELOPE_CONTROL_COUNT: usize = 2;
/// Maximum absolute radial-envelope offset exposed to the reduced-order
/// optimizer. The geometry relation checks impose the tighter bound when a
/// configuration has a smaller available gap.
pub const RADIAL_ENVELOPE_OFFSET_LIMIT_M: f64 = 0.004;
/// Keep a finite positive gap between hub, blade tip, and shroud when an
/// independently bounded radial control is active.
pub const RADIAL_ENVELOPE_MIN_GAP_M: f64 = 5.0e-4;
pub const ROOT_MERGE_SPAN_FRACTION: f64 = 0.18;
/// Reference speed for the empirical 120 mm fan airflow priors.
pub const AIRFLOW_PRIOR_REFERENCE_RPM: f64 = 600.0;
/// Free-air target at [`AIRFLOW_PRIOR_REFERENCE_RPM`].
pub const BENCH_TARGET_AIRFLOW_CFM_AT_REFERENCE_RPM: f64 = 20.0;
/// Chassis-installed target at [`AIRFLOW_PRIOR_REFERENCE_RPM`].
pub const CHASSIS_TARGET_AIRFLOW_CFM_AT_REFERENCE_RPM: f64 = 15.0;
/// Free-air sensitivity interval at [`AIRFLOW_PRIOR_REFERENCE_RPM`].
pub const BENCH_AIRFLOW_PRIOR_CFM_AT_REFERENCE_RPM: [f64; 2] = [14.0, 24.0];
/// Chassis-installed sensitivity interval at [`AIRFLOW_PRIOR_REFERENCE_RPM`].
pub const CHASSIS_AIRFLOW_PRIOR_CFM_AT_REFERENCE_RPM: [f64; 2] = [10.0, 20.0];
/// Central rectangular-cavity loss-factor prior (`Q = 1 / eta`).
pub const CHASSIS_MODAL_LOSS_FACTOR_DEFAULT: f64 = 0.08;
/// Sensitivity interval for the uncalibrated chassis modal loss factor.
pub const CHASSIS_MODAL_LOSS_FACTOR_RANGE: [f64; 2] = [0.03, 0.20];
pub const DESIGN_VARIABLE_COUNT: usize = BLADE_PATH_CONTROL_COUNT * 2
    + BLADE_SECTION_STATION_COUNT * 4
    + RADIAL_ENVELOPE_CONTROL_COUNT
    + SHROUD_CONTROL_COUNT;
pub type PcFanDesignVector = [f64; DESIGN_VARIABLE_COUNT];

/// Reduced-order motor thermal envelope used for design admission.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PcFanThermalConfig {
    pub ambient_temperature_c: f64,
    pub motor_thermal_resistance_k_per_w: f64,
    pub motor_efficiency: f64,
    pub max_winding_temperature_c: f64,
    pub max_electrical_power_w: f64,
}

impl Default for PcFanThermalConfig {
    fn default() -> Self {
        Self {
            ambient_temperature_c: 25.0,
            motor_thermal_resistance_k_per_w: 8.0,
            motor_efficiency: 0.82,
            max_winding_temperature_c: 105.0,
            max_electrical_power_w: 25.0,
        }
    }
}

/// Conservative blade stress and modal-separation envelope.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PcFanStructuralConfig {
    pub material_density_kg_m3: f64,
    pub youngs_modulus_pa: f64,
    pub allowable_stress_pa: f64,
    pub safety_factor: f64,
    pub min_modal_separation_hz: f64,
    pub max_imbalance_g_mm: f64,
    /// Effective blade thermal-expansion coefficient used by the
    /// reduced-order thermo-structural coupling.
    #[serde(default = "default_thermal_expansion_coefficient_per_k")]
    pub thermal_expansion_coefficient_per_k: f64,
    /// Fraction of free thermal strain converted into constrained stress.
    #[serde(default = "default_thermal_constraint_factor")]
    pub thermal_constraint_factor: f64,
    /// Alternating-stress reference for the reduced-order Basquin fatigue
    /// screen, in pascals.
    #[serde(default = "default_fatigue_reference_stress_pa")]
    pub fatigue_reference_stress_pa: f64,
    /// Cycle count at which `fatigue_reference_stress_pa` is defined.
    #[serde(default = "default_fatigue_reference_cycles")]
    pub fatigue_reference_cycles: f64,
    /// Required design life for the reduced-order fatigue screen.
    #[serde(default = "default_fatigue_design_cycles")]
    pub fatigue_design_cycles: f64,
    /// Negative Basquin exponent used to scale allowable alternating stress
    /// from the reference cycle count to the required design life.
    #[serde(default = "default_fatigue_exponent")]
    pub fatigue_exponent: f64,
}

impl Default for PcFanStructuralConfig {
    fn default() -> Self {
        Self {
            material_density_kg_m3: 1_050.0,
            youngs_modulus_pa: 2.4e9,
            allowable_stress_pa: 35.0e6,
            safety_factor: 1.5,
            min_modal_separation_hz: 12.0,
            max_imbalance_g_mm: 12.0,
            thermal_expansion_coefficient_per_k: default_thermal_expansion_coefficient_per_k(),
            thermal_constraint_factor: default_thermal_constraint_factor(),
            fatigue_reference_stress_pa: default_fatigue_reference_stress_pa(),
            fatigue_reference_cycles: default_fatigue_reference_cycles(),
            fatigue_design_cycles: default_fatigue_design_cycles(),
            fatigue_exponent: default_fatigue_exponent(),
        }
    }
}

fn default_thermal_expansion_coefficient_per_k() -> f64 {
    70.0e-6
}

fn default_thermal_constraint_factor() -> f64 {
    0.25
}

fn default_fatigue_reference_stress_pa() -> f64 {
    18.0e6
}

fn default_fatigue_reference_cycles() -> f64 {
    1.0e6
}

fn default_fatigue_design_cycles() -> f64 {
    1.0e8
}

fn default_fatigue_exponent() -> f64 {
    -0.10
}

/// Manufacturing admission limits for the blade and shroud geometry.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PcFanManufacturingConfig {
    pub min_blade_thickness_m: f64,
    pub min_blade_chord_m: f64,
    pub max_twist_gradient_deg: f64,
    pub max_shroud_offset_m: f64,
    pub min_root_thickness_ratio: f64,
    /// Symmetric manufacturing half-widths used by the explicit robust
    /// reduced-order admission pass.  These are not random samples: the
    /// evaluator checks the nominal point and deterministic tolerance
    /// corners.
    #[serde(default)]
    pub tolerance: PcFanManufacturingTolerance,
}

/// Symmetric manufacturing tolerance half-widths for robust admission.
///
/// Thickness and chord values are relative fractions.  Twist is in degrees
/// and shroud offset is in metres.  A zero value disables that axis while
/// retaining the nominal sample in the ensemble.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PcFanManufacturingTolerance {
    pub blade_thickness_fraction: f64,
    pub blade_chord_fraction: f64,
    pub twist_deg: f64,
    pub shroud_offset_m: f64,
}

impl Default for PcFanManufacturingTolerance {
    fn default() -> Self {
        Self {
            blade_thickness_fraction: 0.0,
            blade_chord_fraction: 0.0,
            twist_deg: 0.0,
            shroud_offset_m: 0.0,
        }
    }
}

/// One deterministic manufacturing realization of a design.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PcFanManufacturingVariation {
    pub blade_thickness_fraction: f64,
    pub blade_chord_fraction: f64,
    pub twist_deg: f64,
    pub shroud_offset_m: f64,
}

impl PcFanManufacturingVariation {
    pub const fn nominal() -> Self {
        Self {
            blade_thickness_fraction: 0.0,
            blade_chord_fraction: 0.0,
            twist_deg: 0.0,
            shroud_offset_m: 0.0,
        }
    }

    /// Apply this realization without modifying the nominal design.
    pub fn apply(self, design: &PcFanDesignVariables) -> PcFanDesignVariables {
        let mut realization = design.clone();
        let thickness_scale = 1.0 + self.blade_thickness_fraction;
        let chord_scale = 1.0 + self.blade_chord_fraction;
        for section in &mut realization.blade_sections {
            section.thickness_m *= thickness_scale;
            section.chord_m *= chord_scale;
            section.twist_deg += self.twist_deg;
        }
        for offset in &mut realization.shroud_radius_offsets_m {
            *offset += self.shroud_offset_m;
        }
        realization
    }
}

impl Default for PcFanManufacturingConfig {
    fn default() -> Self {
        Self {
            min_blade_thickness_m: 0.001,
            min_blade_chord_m: 0.006,
            max_twist_gradient_deg: 35.0,
            max_shroud_offset_m: 0.004,
            min_root_thickness_ratio: 0.06,
            tolerance: PcFanManufacturingTolerance::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3 {
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub(crate) fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    pub(crate) fn cross(self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    pub(crate) fn norm(self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanConfig {
    pub blade_count: usize,
    pub dimensions: PcFanDimensions,
    pub operating_point: PcFanOperatingPoint,
    pub mesh: PcFanMeshResolution,
    pub cfd: PcFanCfdConfig,
    pub weights: PcFanOptimizationWeights,
    pub optimizer: PcFanOptimizerConfig,
    /// Far-field observer grid used to compute SPL directivity spread.
    pub observers: PcFanObserverArrayConfig,
    /// Toggles for the broadband noise contributions (tip-leakage and
    /// strut/motor wakes). The analytic shape-gradient adjoint covers only
    /// the tonal blade-passing source, so callers exercising that path
    /// should disable broadband to keep adjoint and finite-difference
    /// gradients in agreement. Default: all sources enabled.
    pub broadband: PcFanBroadbandConfig,
    /// Off-design RPMs used when `operating_spread_weight > 0`.
    pub operating_point_spread: PcFanOperatingPointSpread,
    /// Acoustic propagation environment for the per-harmonic source.
    /// `FreeField`/`DuctCutoff` reproduce the legacy bench-style models;
    /// `RigidCavity` applies coherent rigid-rectangular-cavity modal
    /// coloration so the optimiser sees chassis resonances/antiresonances.
    /// Default: `DuctCutoff` (preserves prior behaviour).
    #[serde(default)]
    pub enclosure: PcFanEnclosureConfig,
    #[serde(default)]
    pub thermal: PcFanThermalConfig,
    #[serde(default)]
    pub structural: PcFanStructuralConfig,
    #[serde(default)]
    pub manufacturing: PcFanManufacturingConfig,
}

impl Default for PcFanConfig {
    fn default() -> Self {
        Self {
            blade_count: DEFAULT_BLADE_COUNT,
            dimensions: PcFanDimensions::default(),
            operating_point: PcFanOperatingPoint::default(),
            mesh: PcFanMeshResolution::default(),
            cfd: PcFanCfdConfig::default(),
            weights: PcFanOptimizationWeights::default(),
            optimizer: PcFanOptimizerConfig::default(),
            observers: PcFanObserverArrayConfig::default(),
            broadband: PcFanBroadbandConfig::default(),
            operating_point_spread: PcFanOperatingPointSpread::default(),
            enclosure: PcFanEnclosureConfig::default(),
            thermal: PcFanThermalConfig::default(),
            structural: PcFanStructuralConfig::default(),
            manufacturing: PcFanManufacturingConfig::default(),
        }
    }
}

/// Switches for the broadband-noise contributions and propagation
/// filters in `PcFanAcousticSolution`. Despite the name, this also hosts
/// the duct cut-off propagation filter — a noise-model knob that fits
/// alongside the broadband sources rather than the geometry or operating
/// point.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PcFanBroadbandConfig {
    pub tip_leakage_enabled: bool,
    pub strut_motor_enabled: bool,
    /// Brooks–Pope–Marcolini turbulent-boundary-layer trailing-edge
    /// self-noise. The dominant broadband mechanism for low-tip-Mach
    /// shrouded axial fans at design point.
    pub trailing_edge_enabled: bool,
    /// Hard-walled-cylinder duct cut-off propagation filter. Attenuates
    /// the (1,0) modal content of every per-harmonic pressure below the
    /// cut-on frequency `f_10 = 1.8412·c / (2π·R_shroud)` by
    /// `exp(−|k_z|·L_shroud)`. Plane-wave content propagates at all
    /// frequencies. First-order model — assumes a fixed plane-wave / (1,0)
    /// modal split.
    #[serde(default = "default_duct_cutoff_enabled")]
    pub duct_cutoff_enabled: bool,
    /// Amiet leading-edge inflow-turbulence broadband. Models the
    /// blade-leading-edge response to ingested turbulent eddies. Only
    /// non-zero when `operating_point.inflow_turbulence_intensity > 0`,
    /// so the default operating point stays silent unless the user
    /// opts into a chassis-like inflow condition.
    #[serde(default = "default_leading_edge_inflow_enabled")]
    pub leading_edge_inflow_enabled: bool,
}

fn default_duct_cutoff_enabled() -> bool {
    true
}

fn default_leading_edge_inflow_enabled() -> bool {
    true
}

impl Default for PcFanBroadbandConfig {
    fn default() -> Self {
        Self {
            tip_leakage_enabled: true,
            strut_motor_enabled: true,
            trailing_edge_enabled: true,
            duct_cutoff_enabled: true,
            leading_edge_inflow_enabled: true,
        }
    }
}

impl PcFanBroadbandConfig {
    /// All broadband sources off and the duct cut-off filter disabled —
    /// useful for tests that only exercise the tonal blade-passing path
    /// not covered by the current finite-difference objective-gradient path;
    /// no propagation filtering
    /// either.
    pub fn off() -> Self {
        Self {
            tip_leakage_enabled: false,
            strut_motor_enabled: false,
            trailing_edge_enabled: false,
            duct_cutoff_enabled: false,
            leading_edge_inflow_enabled: false,
        }
    }
}

impl PcFanConfig {
    pub fn default_120mm() -> Self {
        Self::default()
    }

    /// 120 mm fan in a typical mid-tower chassis (`mid_tower` enclosure,
    /// 10 % inflow turbulence, rigid-cavity propagation). The canonical
    /// installed-deployment preset: Amiet ingestion noise active and
    /// chassis modes shape the spectrum.
    pub fn chassis_installed_120mm() -> Self {
        let mut config = Self::default_120mm();
        config.operating_point = PcFanOperatingPoint::chassis_installed(config.operating_point.rpm);
        config.enclosure = PcFanEnclosureConfig::mid_tower();
        config.enclosure.mode = PcFanPropagationMode::RigidCavity;
        config
    }

    /// 120 mm fan in anechoic / bench conditions. Free-field propagation
    /// and TI = 0 — Amiet LE noise contributes nothing. Use this to
    /// reproduce published self-noise data and validate the optimised
    /// design before committing to a chassis study.
    pub fn bench_120mm() -> Self {
        let mut config = Self::default_120mm();
        config.operating_point = PcFanOperatingPoint::bench(config.operating_point.rpm);
        config.enclosure = PcFanEnclosureConfig::free_field();
        config
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PcFanDimensions {
    pub frame_side_m: f64,
    pub depth_m: f64,
    pub hub_radius_m: f64,
    pub blade_tip_radius_m: f64,
    pub shroud_inner_radius_m: f64,
    pub shroud_wall_thickness_m: f64,
    pub frame_wall_thickness_m: f64,
    pub support_strut_width_m: f64,
    pub support_strut_thickness_m: f64,
    pub motor_mount_radius_m: f64,
    pub motor_mount_depth_m: f64,
    pub frame_mount_inset_m: f64,
    pub mount_pad_size_m: f64,
    /// Axial gap between the rotor's trailing edge and the X-support
    /// strut leading edge (m). Drives Sears rotor–stator interaction:
    /// the strut-wake velocity deficit decays roughly as
    /// `1/√(1 + gap/c_strut)` between rotor and strut, so increasing
    /// the gap is the cheapest way to suppress BPF tones.
    #[serde(default = "default_rotor_strut_axial_gap_m")]
    pub rotor_strut_axial_gap_m: f64,
    /// Centerline velocity deficit of the X-support strut wake at the
    /// strut trailing edge, normalised by the local axial mean flow.
    /// Typical thin-strut values: 0.25–0.40. Higher → stronger BPF tones.
    #[serde(default = "default_strut_wake_centerline_deficit")]
    pub strut_wake_centerline_deficit: f64,
    /// Half-width of the (Gaussian) strut wake at the strut trailing
    /// edge (m). Roughly equal to the strut chord for a compact bluff
    /// body. Wider wakes carry more low-order Fourier content and so
    /// concentrate the Sears tones at lower BPF harmonics.
    #[serde(default = "default_strut_wake_half_width_m")]
    pub strut_wake_half_width_m: f64,
}

fn default_rotor_strut_axial_gap_m() -> f64 {
    0.005
}

fn default_strut_wake_centerline_deficit() -> f64 {
    0.30
}

fn default_strut_wake_half_width_m() -> f64 {
    0.008
}

impl Default for PcFanDimensions {
    fn default() -> Self {
        Self {
            frame_side_m: 0.120,
            depth_m: 0.025,
            hub_radius_m: 0.018,
            // Tip radius 54 mm — a compromise: close enough to the
            // 57 mm shroud (3 mm gap) that the blade actually fills the
            // available radius, but with enough breathing room that the
            // ±30 mm tangential-sweep bound still gives the optimiser
            // ~`√(57² − 54²) ≈ 18 mm` of sweep before
            // `shroud_clearance_penalty` activates.
            blade_tip_radius_m: 0.054,
            shroud_inner_radius_m: 0.057,
            shroud_wall_thickness_m: 0.001,
            frame_wall_thickness_m: 0.002,
            support_strut_width_m: 0.006,
            support_strut_thickness_m: 0.004,
            // The motor sits inside the hub, so there is no separate
            // motor-mount cylinder protruding into the air path. We keep
            // the field for blockage/wake-noise terms that already clamp
            // any motor-vs-hub overshoot to zero, and pin both dims to
            // the hub so the data is internally consistent.
            motor_mount_radius_m: 0.018,
            motor_mount_depth_m: 0.0,
            frame_mount_inset_m: 0.008,
            mount_pad_size_m: 0.008,
            rotor_strut_axial_gap_m: default_rotor_strut_axial_gap_m(),
            strut_wake_centerline_deficit: default_strut_wake_centerline_deficit(),
            strut_wake_half_width_m: default_strut_wake_half_width_m(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PcFanOperatingPoint {
    pub rpm: f64,
    pub target_airflow_cfm: f64,
    pub air_density_kg_m3: f64,
    pub dynamic_viscosity_pa_s: f64,
    pub speed_of_sound_m_s: f64,
    /// Inlet turbulence intensity (RMS upwash velocity normalised by
    /// mean axial velocity). Drives Amiet leading-edge inflow noise:
    /// `p²_LE ∝ TI²`. Default 0 → benchtop / clean-inflow operation,
    /// LE term contributes nothing. Typical chassis-installed values
    /// are 0.05–0.15 (5–15 %).
    #[serde(default = "default_inflow_turbulence_intensity")]
    pub inflow_turbulence_intensity: f64,
    /// Streamwise integral length scale of the inlet turbulence (m).
    /// Sets the von Kármán cutoff frequency `f_c = U_local / (2π·Λ)`
    /// — small Λ shifts LE-noise spectral weight to higher BPF
    /// harmonics; large Λ concentrates it at the fundamentals.
    #[serde(default = "default_inflow_integral_length_scale_m")]
    pub inflow_integral_length_scale_m: f64,
}

fn default_inflow_turbulence_intensity() -> f64 {
    0.0
}

fn default_inflow_integral_length_scale_m() -> f64 {
    0.005
}

impl Default for PcFanOperatingPoint {
    fn default() -> Self {
        Self {
            rpm: AIRFLOW_PRIOR_REFERENCE_RPM,
            target_airflow_cfm: BENCH_TARGET_AIRFLOW_CFM_AT_REFERENCE_RPM,
            air_density_kg_m3: 1.204,
            dynamic_viscosity_pa_s: 1.825e-5,
            speed_of_sound_m_s: 343.0,
            inflow_turbulence_intensity: default_inflow_turbulence_intensity(),
            inflow_integral_length_scale_m: default_inflow_integral_length_scale_m(),
        }
    }
}

impl PcFanOperatingPoint {
    fn scale_airflow_from_reference(rpm: f64, airflow_cfm_at_reference: f64) -> f64 {
        airflow_cfm_at_reference * rpm / AIRFLOW_PRIOR_REFERENCE_RPM
    }

    /// RPM-scaled empirical free-air interval for a conventional 120 mm fan.
    pub fn bench_airflow_prior_cfm(rpm: f64) -> [f64; 2] {
        BENCH_AIRFLOW_PRIOR_CFM_AT_REFERENCE_RPM
            .map(|flow| Self::scale_airflow_from_reference(rpm, flow))
    }

    /// RPM-scaled empirical chassis-installed interval for a 120 mm fan.
    pub fn chassis_airflow_prior_cfm(rpm: f64) -> [f64; 2] {
        CHASSIS_AIRFLOW_PRIOR_CFM_AT_REFERENCE_RPM
            .map(|flow| Self::scale_airflow_from_reference(rpm, flow))
    }

    /// Anechoic / bench-validation preset. Clean inflow, TI = 0, so the
    /// Amiet leading-edge ingestion-turbulence term is silent. Use this
    /// for self-noise studies (BPM TE + tip-leakage + tonal) and for
    /// reproducing published anechoic-chamber measurements.
    pub fn bench(rpm: f64) -> Self {
        Self {
            rpm,
            target_airflow_cfm: Self::scale_airflow_from_reference(
                rpm,
                BENCH_TARGET_AIRFLOW_CFM_AT_REFERENCE_RPM,
            ),
            inflow_turbulence_intensity: 0.0,
            ..Self::default()
        }
    }

    /// Chassis-installed preset. Models the inflow disturbance from
    /// upstream chassis components (heatsink fins, CPU coolers, other
    /// fans). TI = 10 %, integral length scale ≈ 3 mm — a reasonable
    /// proxy for fan-after-heatsink ingestion. Activates the Amiet
    /// leading-edge inflow noise term, which often dominates self-noise
    /// at low rpm. Pair with `PcFanEnclosureConfig::mid_tower()` for
    /// the canonical chassis-installed study.
    pub fn chassis_installed(rpm: f64) -> Self {
        Self {
            rpm,
            target_airflow_cfm: Self::scale_airflow_from_reference(
                rpm,
                CHASSIS_TARGET_AIRFLOW_CFM_AT_REFERENCE_RPM,
            ),
            inflow_turbulence_intensity: 0.10,
            inflow_integral_length_scale_m: 0.003,
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PcFanMeshResolution {
    pub radial_segments: usize,
    pub chord_segments: usize,
    pub circumferential_segments: usize,
    pub harmonic_count: usize,
}

impl Default for PcFanMeshResolution {
    fn default() -> Self {
        Self {
            radial_segments: 16,
            chord_segments: 6,
            circumferential_segments: 64,
            harmonic_count: 6,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PcFanFlowModel {
    /// Closed-form mean-flow + harmonic acoustic model. Fastest, used for
    /// the adjoint shape-gradient path.
    ReducedOrder,
    /// Incompressible Navier-Stokes with the blade represented as a
    /// distributed actuator-disk body force (Gaussian-weighted forcing).
    /// The blade is NOT a wall — flow can pass through it; the body force
    /// imparts axial+tangential momentum proportional to the local
    /// loading polynomial.
    IncompressibleCfd,
    /// Incompressible Navier-Stokes with the blade represented as a real
    /// **rotating wall** via Brinkman penalty. Each cell whose centroid
    /// lies inside the swept blade volume is forced toward `u = ω × r`
    /// with a stiff penalty. The actuator-disk body force is disabled in
    /// this mode (otherwise the blade is double-counted).
    ///
    /// Costs more per Picard iteration (penalty forces are stiffer and
    /// the geometry is sampled per quadrature point), but captures the
    /// real blade geometry — including separation, wake, tip leakage.
    /// Pair with a generous `cfd.side_clearance_m` and large inlet/outlet
    /// extensions so the outer no-slip walls don't constrain the flow.
    IncompressibleCfdFull,
}

impl PcFanFlowModel {
    /// True for any flow model that runs the FEM Navier-Stokes solver.
    pub fn is_cfd(self) -> bool {
        matches!(self, Self::IncompressibleCfd | Self::IncompressibleCfdFull)
    }

    /// True when the blade is treated as a rotating wall (penalty),
    /// false when it is a distributed actuator-disk body force.
    pub fn blade_as_wall(self) -> bool {
        matches!(self, Self::IncompressibleCfdFull)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PcFanCfdConfig {
    pub flow_model: PcFanFlowModel,
    pub inlet_extension_m: f64,
    pub outlet_extension_m: f64,
    pub side_clearance_m: f64,
    pub cells_x: usize,
    pub cells_y: usize,
    pub cells_z: usize,
    pub max_picard_iterations: usize,
    pub nonlinear_tolerance: f64,
    /// Maximum relative inlet/outlet mass-flow imbalance admitted before a
    /// CFD result may be used by the objective.  The denominator is the
    /// larger boundary mass flow, so this remains meaningful for either
    /// flow direction.  The default is deliberately loose enough for the
    /// coarse production smoke mesh; refined validation must set a tighter
    /// bound explicitly.
    #[serde(default = "default_max_relative_mass_imbalance")]
    pub max_relative_mass_imbalance: f64,
    /// Maximum relative axial momentum-ledger residual admitted before a
    /// CFD field is used by the objective.
    #[serde(default = "default_max_relative_momentum_imbalance")]
    pub max_relative_momentum_imbalance: f64,
    /// Maximum relative pressure-force closure error.
    #[serde(default = "default_max_relative_pressure_closure")]
    pub max_relative_pressure_closure: f64,
    /// Maximum relative mechanical-power closure error.
    #[serde(default = "default_max_relative_energy_imbalance")]
    pub max_relative_energy_imbalance: f64,
    /// Maximum GMRES iterations for each Picard linear solve.
    #[serde(default = "default_linear_solver_max_iterations")]
    pub linear_solver_max_iterations: usize,
    /// Relative residual tolerance for each Picard linear solve.
    #[serde(default = "default_linear_solver_tolerance")]
    pub linear_solver_tolerance: f64,
    /// CPU preconditioner used when the CFD linear solve runs or falls
    /// back on CPU.
    #[serde(default)]
    pub cpu_preconditioner: PcFanCpuPreconditionerKind,
    /// Turbulence closure used in the Picard solve (`Laminar`,
    /// `KOmegaSstAlgebraic`, or segregated `KOmegaSstTransported`).
    pub turbulence: PcFanTurbulenceModel,
    /// Tetrahedralisation pattern used to build the CFD volumetric mesh.
    /// `Symmetric` (default) preserves cube octahedral symmetry — the
    /// pressure field reflects the physical 4-fold rotational symmetry of
    /// the fan. `Kuhn` is the legacy 6-tet decomposition with one fixed
    /// diagonal per cube; faster/lighter but biases the field along
    /// the diagonal of every cell.
    pub mesh_pattern: PcFanCfdMeshPattern,
    /// Compute backend for the per-Picard linear solve. `Cpu` keeps the
    /// legacy ILU(0) + GMRES path; `Cuda` and `Metal` route through
    /// `sonium_compute::Backend` (polynomial Jacobi + backend-aware GMRES
    /// from `sonium_solvers`). The non-CPU backends fall back to CPU
    /// with a warning when their feature isn't built.
    pub backend: PcFanCfdBackend,
    /// Permit an unavailable or failed accelerator to fall back to the CPU
    /// solver. The fallback remains visible in the returned provenance. Set
    /// this to `false` for production runs that must fail closed instead of
    /// accepting a substituted backend.
    #[serde(default = "default_allow_host_fallback")]
    pub allow_host_fallback: bool,
    /// Require the complete CFD trajectory to remain device-resident. The
    /// current FEM bridge only offloads the linear solve, so enabling this
    /// flag intentionally rejects that host-assembled path until resident
    /// assembly/state kernels are available.
    #[serde(default)]
    pub require_device_resident: bool,
    /// Polynomial Jacobi preconditioner degree on the GPU/backend path.
    /// `1` (default) is plain point-Jacobi; `2..4` gives a stronger
    /// preconditioner at the cost of `K-1` extra SpMVs per GMRES iter.
    /// Used when `precond_kind == PolynomialJacobi`. Ignored on the CPU backend.
    pub precond_polynomial_degree: usize,
    /// Damping factor `ω` for the polynomial Jacobi preconditioner.
    /// `1.0` (default) is plain Jacobi-Richardson; `2/3` is the classical
    /// weighted-Jacobi smoother. `0.5..0.8` for indefinite Stokes blocks.
    /// Used when `precond_kind == PolynomialJacobi`. Ignored on the CPU backend.
    pub precond_polynomial_damping: f64,
    /// Polynomial degree for the pressure Schur solve inside backend
    /// Jacobi/Schur. `1` preserves diagonal Schur behavior.
    #[serde(default = "default_schur_polynomial_degree")]
    pub precond_schur_degree: usize,
    /// Damping for pressure Schur polynomial correction steps.
    #[serde(default = "default_schur_polynomial_damping")]
    pub precond_schur_damping: f64,
    /// Pressure-block solve used inside backend Jacobi/Schur.
    #[serde(default)]
    pub precond_schur_kind: PcFanBackendPressureSchurPreconditionerKind,
    /// Chebyshev degree for the pressure Schur solve inside backend
    /// Jacobi/Schur. Used when `precond_schur_kind == Chebyshev`.
    #[serde(default = "default_chebyshev_degree")]
    pub precond_schur_chebyshev_degree: usize,
    /// Pressure Schur Chebyshev eigenvalue interval ratio.
    #[serde(default = "default_chebyshev_eigenvalue_ratio")]
    pub precond_schur_chebyshev_eigenvalue_ratio: f64,
    /// Which GPU-friendly preconditioner the backend GMRES path uses.
    /// `PolynomialJacobi` (default) is robust for indefinite Stokes
    /// blocks; `Chebyshev` is stronger on SPD/near-SPD problems.
    #[serde(default)]
    pub precond_kind: PcFanBackendPreconditionerKind,
    /// Chebyshev preconditioner degree (number of 3-term-recurrence
    /// steps). Default `5`. Used when `precond_kind == Chebyshev`.
    #[serde(default = "default_chebyshev_degree")]
    pub precond_chebyshev_degree: usize,
    /// Chebyshev eigenvalue-interval ratio `λ_max / λ_min`. Default
    /// `30.0` (PETSc/HYPRE). Used when `precond_kind == Chebyshev`.
    #[serde(default = "default_chebyshev_eigenvalue_ratio")]
    pub precond_chebyshev_eigenvalue_ratio: f64,
    /// Working precision for the per-Picard linear solve. `F64`
    /// (default) is the legacy path. `MixedF32F64` routes the inner
    /// Krylov through f32 (FGMRES-IR, Carson-Higham 2018) — required
    /// to get any Metal acceleration on Apple Silicon since Metal has
    /// no `double` support.
    #[serde(default)]
    pub solver_kind: PcFanLinearSolverKind,
    /// Maximum inner f32 GMRES iterations per outer FGMRES-IR step
    /// (only used when `solver_kind == MixedF32F64`).
    #[serde(default = "default_ir_inner_max_iterations")]
    pub solver_ir_inner_max_iterations: usize,
    /// Inner f32 GMRES relative-residual tolerance.
    #[serde(default = "default_ir_inner_tolerance")]
    pub solver_ir_inner_tolerance: f32,
    /// Solve each Picard/Oseen linear system with sparse-direct LU instead
    /// of the configured iterative solver.
    ///
    /// This isolates mesh-convergence assessment from iterative-solver luck:
    /// an odd-cell coarse level whose Oseen system stalls GMRES+ILU would
    /// otherwise read as a discretization failure. Only suitable for small
    /// reference meshes; production paths keep the iterative solver.
    #[serde(default)]
    pub use_direct_solver: bool,
}

fn default_ir_inner_max_iterations() -> usize {
    50
}

fn default_ir_inner_tolerance() -> f32 {
    1.0e-3
}

fn default_allow_host_fallback() -> bool {
    true
}

fn default_linear_solver_max_iterations() -> usize {
    5000
}

fn default_linear_solver_tolerance() -> f64 {
    1.0e-4
}

fn default_max_relative_mass_imbalance() -> f64 {
    5.0e-2
}
fn default_max_relative_momentum_imbalance() -> f64 {
    2.0
}
fn default_max_relative_pressure_closure() -> f64 {
    2.0
}
fn default_max_relative_energy_imbalance() -> f64 {
    2.0
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PcFanCpuPreconditionerKind {
    #[default]
    Ilu,
    JacobiSchur,
    /// Block saddle-point preconditioner with ILUT on the velocity block.
    ///
    /// The FEM-recommended choice for these saddle systems: monolithic
    /// ILU(0) is fragile on the velocity/pressure scale separation once
    /// the immersed penalty terms stiffen the momentum block.
    IlutSchur,
}

impl PcFanCpuPreconditionerKind {
    pub fn into_fem(
        self,
    ) -> sonium_fem::physics::fluid::stokes::navier_stokes::LinearSolverCpuPreconditionerKind {
        use sonium_fem::physics::fluid::stokes::navier_stokes::LinearSolverCpuPreconditionerKind as Kind;
        match self {
            Self::Ilu => Kind::Ilu,
            Self::JacobiSchur => Kind::JacobiSchur,
            Self::IlutSchur => Kind::IlutSchur,
        }
    }
}

/// Working precision for the iterative linear solve.
#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PcFanLinearSolverKind {
    /// Standard f64 GMRES end-to-end.
    #[default]
    F64,
    /// Mixed-precision FGMRES-IR — outer f64, inner f32 Krylov.
    MixedF32F64,
}

impl PcFanLinearSolverKind {
    pub fn into_fem(self) -> sonium_fem::physics::fluid::stokes::navier_stokes::LinearSolverKind {
        use sonium_fem::physics::fluid::stokes::navier_stokes::LinearSolverKind as Kind;
        match self {
            Self::F64 => Kind::F64,
            Self::MixedF32F64 => Kind::MixedF32F64,
        }
    }
}

fn default_chebyshev_degree() -> usize {
    5
}

fn default_schur_polynomial_degree() -> usize {
    1
}

fn default_schur_polynomial_damping() -> f64 {
    1.0
}

fn default_chebyshev_eigenvalue_ratio() -> f64 {
    30.0
}

/// Which GPU-friendly preconditioner the backend GMRES path uses.
#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PcFanBackendPreconditionerKind {
    /// Truncated Neumann series of `D⁻¹A` (point-Jacobi at degree 1).
    /// Robust on indefinite Stokes blocks with appropriate damping.
    #[default]
    PolynomialJacobi,
    /// Chebyshev polynomial of `D⁻¹A`. Optimal min-max damping for SPD
    /// or near-SPD systems.
    Chebyshev,
    /// Upper-triangular saddle-point preconditioner with diagonal
    /// velocity Jacobi and pressure-mass Schur diagonal inverse.
    JacobiSchur,
}

impl PcFanBackendPreconditionerKind {
    pub fn into_fem(
        self,
    ) -> sonium_fem::physics::fluid::stokes::navier_stokes::BackendPreconditionerKind {
        use sonium_fem::physics::fluid::stokes::navier_stokes::BackendPreconditionerKind as Kind;
        match self {
            Self::PolynomialJacobi => Kind::PolynomialJacobi,
            Self::Chebyshev => Kind::Chebyshev,
            Self::JacobiSchur => Kind::JacobiSchur,
        }
    }
}

/// Pressure-block solve used inside backend Jacobi/Schur.
#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PcFanBackendPressureSchurPreconditionerKind {
    #[default]
    Polynomial,
    Chebyshev,
    Amg,
}

impl PcFanBackendPressureSchurPreconditionerKind {
    pub fn into_fem(
        self,
    ) -> sonium_fem::physics::fluid::stokes::navier_stokes::BackendPressureSchurPreconditionerKind
    {
        use sonium_fem::physics::fluid::stokes::navier_stokes::BackendPressureSchurPreconditionerKind as Kind;
        match self {
            Self::Polynomial => Kind::Polynomial,
            Self::Chebyshev => Kind::Chebyshev,
            Self::Amg => Kind::Amg,
        }
    }
}

/// Compute backend used by the CFD per-iteration linear solve.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PcFanCfdBackend {
    /// Pick the best available backend at runtime: Metal on macOS (with
    /// the `metal` feature), CUDA on Linux/Windows (with the `cuda`
    /// feature), CPU otherwise. This is the default so an unflagged
    /// `cargo run --features metal` on macOS lands on Metal automatically.
    #[default]
    Auto,
    Cpu,
    /// NVIDIA CUDA via `sonium-compute`'s cudarc-backed CUDA backend.
    /// Requires the `cuda` feature and a CUDA driver at runtime.
    Cuda,
    /// Apple Metal via `sonium-compute`'s Metal backend. Requires the `metal` feature.
    Metal,
}

impl PcFanCfdBackend {
    /// Lower this enum to the equivalent
    /// `sonium_fem::physics::fluid::stokes::navier_stokes::LinearSolverBackend`
    /// expected by the FEM Picard solver. `Auto` round-trips as `Auto`
    /// — the FEM dispatcher resolves it on first solve.
    pub fn into_fem(
        self,
    ) -> sonium_fem::physics::fluid::stokes::navier_stokes::LinearSolverBackend {
        use sonium_fem::physics::fluid::stokes::navier_stokes::LinearSolverBackend as Lsb;
        match self {
            Self::Auto => Lsb::Auto,
            Self::Cpu => Lsb::Cpu,
            Self::Cuda => Lsb::Cuda,
            Self::Metal => Lsb::Metal,
        }
    }

    /// Convert the concrete backend reported by the FEM dispatcher.
    pub(crate) fn from_fem(
        backend: sonium_fem::physics::fluid::stokes::navier_stokes::LinearSolverBackend,
    ) -> Self {
        use sonium_fem::physics::fluid::stokes::navier_stokes::LinearSolverBackend as Lsb;

        match backend {
            Lsb::Auto => Self::Auto,
            Lsb::Cpu => Self::Cpu,
            Lsb::Cuda => Self::Cuda,
            Lsb::Metal => Self::Metal,
        }
    }
}

/// Execution residency and fallback provenance for one accepted CFD result.
///
/// `AcceleratorLinearSolve` is deliberately distinct from
/// `DeviceResident`: the current FEM bridge assembles fields and transient
/// state on the host, then offloads only the sparse linear solve. This type
/// prevents that partial path from being reported as a complete accelerator
/// trajectory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PcFanCfdExecutionMode {
    Cpu,
    AcceleratorLinearSolve,
    DeviceResident,
}

impl PcFanCfdExecutionMode {
    pub const fn is_device_resident(self) -> bool {
        matches!(self, Self::DeviceResident)
    }
}

/// Explicit reuse contract for a published CFD execution environment.
///
/// The FEM bridge currently rebuilds host-side assembly and stages each
/// accelerator linear solve.  Keeping these flags next to backend provenance
/// prevents a CUDA linear-solve result from being mistaken for a retained
/// device trajectory.  A future resident implementation must opt into the
/// `DeviceResident` mode before any of the component reuse flags become true.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PcFanCfdReuseValidity {
    /// Mesh coordinates/topology can be retained across accepted steps.
    pub mesh_reused: bool,
    /// The FEM sparsity pattern can be retained across accepted steps.
    pub sparsity_reused: bool,
    /// The backend preconditioner can be retained across accepted steps.
    pub preconditioner_reused: bool,
    /// Mean-flow state remains on the execution device between steps.
    pub flow_state_reused: bool,
    /// Transported k--omega state remains on the execution device between
    /// BDF updates.
    pub transported_state_reused: bool,
    /// Coupled aeroacoustic state remains on the execution device.
    pub aeroacoustic_state_reused: bool,
}

impl Default for PcFanCfdReuseValidity {
    fn default() -> Self {
        Self::host_staged()
    }
}

impl PcFanCfdReuseValidity {
    /// Conservative contract for the current host-assembled FEM bridge.
    pub const fn host_staged() -> Self {
        Self {
            mesh_reused: false,
            sparsity_reused: false,
            preconditioner_reused: false,
            flow_state_reused: false,
            transported_state_reused: false,
            aeroacoustic_state_reused: false,
        }
    }

    /// Contract reserved for a complete retained-device implementation.
    pub const fn device_resident() -> Self {
        Self {
            mesh_reused: true,
            sparsity_reused: true,
            preconditioner_reused: true,
            flow_state_reused: true,
            transported_state_reused: true,
            aeroacoustic_state_reused: true,
        }
    }

    pub const fn all_components_reused(self) -> bool {
        self.mesh_reused
            && self.sparsity_reused
            && self.preconditioner_reused
            && self.flow_state_reused
            && self.transported_state_reused
            && self.aeroacoustic_state_reused
    }

    pub const fn for_execution_mode(mode: PcFanCfdExecutionMode) -> Self {
        if mode.is_device_resident() {
            Self::device_resident()
        } else {
            Self::host_staged()
        }
    }
}

/// Why a requested backend was not the backend that produced the result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PcFanCfdFallbackReason {
    RequestedAcceleratorUnavailable,
    RequestedBackendSubstituted,
}

/// Backend and residency provenance attached to every production CFD result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PcFanCfdBackendProvenance {
    pub requested_backend: PcFanCfdBackend,
    pub actual_backend: PcFanCfdBackend,
    pub execution_mode: PcFanCfdExecutionMode,
    pub fallback_reason: Option<PcFanCfdFallbackReason>,
}

impl PcFanCfdBackendProvenance {
    pub fn from_dispatch(
        requested_backend: PcFanCfdBackend,
        actual_backend: PcFanCfdBackend,
    ) -> Self {
        let fallback_reason = match (requested_backend, actual_backend) {
            (PcFanCfdBackend::Auto, _) | (PcFanCfdBackend::Cpu, PcFanCfdBackend::Cpu) => None,
            (requested, actual) if requested == actual => None,
            (PcFanCfdBackend::Cuda | PcFanCfdBackend::Metal, PcFanCfdBackend::Cpu) => {
                Some(PcFanCfdFallbackReason::RequestedAcceleratorUnavailable)
            }
            _ => Some(PcFanCfdFallbackReason::RequestedBackendSubstituted),
        };
        let execution_mode = match actual_backend {
            PcFanCfdBackend::Cpu => PcFanCfdExecutionMode::Cpu,
            PcFanCfdBackend::Cuda | PcFanCfdBackend::Metal => {
                PcFanCfdExecutionMode::AcceleratorLinearSolve
            }
            PcFanCfdBackend::Auto => PcFanCfdExecutionMode::Cpu,
        };
        Self {
            requested_backend,
            actual_backend,
            execution_mode,
            fallback_reason,
        }
    }

    pub const fn is_fallback(self) -> bool {
        self.fallback_reason.is_some()
    }

    pub const fn is_admitted(
        self,
        allow_host_fallback: bool,
        require_device_resident: bool,
    ) -> bool {
        (!self.is_fallback() || allow_host_fallback)
            && (!require_device_resident || self.execution_mode.is_device_resident())
    }
}

/// Build/runtime details needed to reproduce or compare a CFD result across
/// architectures. This is metadata only; it does not claim that a requested
/// accelerator was actually available or that the entire trajectory was
/// device-resident.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanCfdExecutionEnvironment {
    pub target_arch: String,
    pub target_os: String,
    pub compiler: String,
    pub cpu_features: String,
    /// SIMD implementation selected by the CPU kernels for this process.
    pub simd_dispatch: String,
    pub thread_count: usize,
    pub requested_backend: PcFanCfdBackend,
    pub actual_backend: PcFanCfdBackend,
    pub execution_mode: PcFanCfdExecutionMode,
    /// Per-component reuse claims.  For the current CPU and accelerator
    /// linear-solve modes this is explicitly all false.
    #[serde(default)]
    pub reuse_validity: PcFanCfdReuseValidity,
    pub linear_solver: PcFanLinearSolverKind,
    pub cpu_preconditioner: PcFanCpuPreconditionerKind,
    pub backend_preconditioner: PcFanBackendPreconditionerKind,
    pub linear_solver_max_iterations: usize,
    pub linear_solver_tolerance: f64,
}

impl PcFanCfdExecutionEnvironment {
    pub fn capture(config: &PcFanCfdConfig, provenance: PcFanCfdBackendProvenance) -> Self {
        Self {
            target_arch: std::env::consts::ARCH.into(),
            target_os: std::env::consts::OS.into(),
            compiler: option_env!("RUSTC_VERSION").unwrap_or("unknown").into(),
            cpu_features: detected_cpu_features(),
            simd_dispatch: detected_simd_dispatch(),
            thread_count: std::thread::available_parallelism().map_or(1, std::num::NonZero::get),
            requested_backend: provenance.requested_backend,
            actual_backend: provenance.actual_backend,
            execution_mode: provenance.execution_mode,
            reuse_validity: PcFanCfdReuseValidity::for_execution_mode(provenance.execution_mode),
            linear_solver: config.solver_kind,
            cpu_preconditioner: config.cpu_preconditioner,
            backend_preconditioner: config.precond_kind,
            linear_solver_max_iterations: config.linear_solver_max_iterations,
            linear_solver_tolerance: config.linear_solver_tolerance,
        }
    }

    pub fn matches_provenance(&self, provenance: PcFanCfdBackendProvenance) -> bool {
        self.requested_backend == provenance.requested_backend
            && self.actual_backend == provenance.actual_backend
            && self.execution_mode == provenance.execution_mode
    }
}

#[cfg(target_arch = "x86_64")]
fn detected_cpu_features() -> String {
    [
        ("avx2", std::is_x86_feature_detected!("avx2")),
        ("fma", std::is_x86_feature_detected!("fma")),
        ("avx", std::is_x86_feature_detected!("avx")),
        ("sse4.2", std::is_x86_feature_detected!("sse4.2")),
    ]
    .into_iter()
    .filter_map(|(feature, enabled)| enabled.then_some(feature))
    .collect::<Vec<_>>()
    .join(",")
}

#[cfg(target_arch = "x86_64")]
fn detected_simd_dispatch() -> String {
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        "avx2-fma".into()
    } else {
        "scalar".into()
    }
}

#[cfg(target_arch = "aarch64")]
fn detected_simd_dispatch() -> String {
    "neon".into()
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn detected_simd_dispatch() -> String {
    "scalar".into()
}

#[cfg(target_arch = "aarch64")]
fn detected_cpu_features() -> String {
    [
        ("neon", std::arch::is_aarch64_feature_detected!("neon")),
        ("fp16", std::arch::is_aarch64_feature_detected!("fp16")),
    ]
    .into_iter()
    .filter_map(|(feature, enabled)| enabled.then_some(feature))
    .collect::<Vec<_>>()
    .join(",")
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn detected_cpu_features() -> String {
    "unknown".into()
}

/// CFD mesh decomposition pattern.
#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PcFanCfdMeshPattern {
    /// 24-tet barycentric subdivision per cube. Octahedral-symmetric, so
    /// a symmetric flow problem produces a visually symmetric pressure
    /// field. 4× more elements than Kuhn.
    #[default]
    Symmetric,
    /// 6-tet Kuhn triangulation per cube. Cheaper but breaks 4-fold
    /// rotational symmetry by the fixed `(0,0,0)→(1,1,1)` diagonal.
    Kuhn,
}

/// Turbulence selection wrapping `sonium_fem::physics::fluid::stokes::turbulence::TurbulenceModel`
/// with a serialisable interface. Defaults to `Laminar` so existing tests
/// see no behavioural change.
#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PcFanTurbulenceModel {
    #[default]
    Laminar,
    KOmegaSstAlgebraic,
    /// Segregated transported k--omega SST state with lagged momentum coupling.
    KOmegaSstTransported,
}

impl PcFanTurbulenceModel {
    pub fn into_fem(self) -> sonium_fem::physics::fluid::stokes::turbulence::TurbulenceModel {
        use sonium_fem::physics::fluid::stokes::turbulence::TurbulenceModel;
        match self {
            Self::Laminar => TurbulenceModel::Laminar,
            Self::KOmegaSstAlgebraic | Self::KOmegaSstTransported => {
                TurbulenceModel::pcfan_default()
            }
        }
    }
}

impl Default for PcFanCfdConfig {
    fn default() -> Self {
        Self {
            flow_model: PcFanFlowModel::ReducedOrder,
            inlet_extension_m: 0.035,
            outlet_extension_m: 0.045,
            side_clearance_m: 0.0,
            cells_x: 8,
            cells_y: 8,
            cells_z: 12,
            max_picard_iterations: 40,
            nonlinear_tolerance: 1.0e-6,
            max_relative_mass_imbalance: default_max_relative_mass_imbalance(),
            max_relative_momentum_imbalance: default_max_relative_momentum_imbalance(),
            max_relative_pressure_closure: default_max_relative_pressure_closure(),
            max_relative_energy_imbalance: default_max_relative_energy_imbalance(),
            linear_solver_max_iterations: default_linear_solver_max_iterations(),
            linear_solver_tolerance: default_linear_solver_tolerance(),
            // Block ILUT-Schur, not monolithic ILU: the implicit immersed
            // penalty block is orders of magnitude stiffer than the viscous
            // block, and ILU(0) stalls on that scale separation.
            cpu_preconditioner: PcFanCpuPreconditionerKind::IlutSchur,
            turbulence: PcFanTurbulenceModel::Laminar,
            mesh_pattern: PcFanCfdMeshPattern::Symmetric,
            backend: PcFanCfdBackend::Auto,
            allow_host_fallback: default_allow_host_fallback(),
            require_device_resident: false,
            precond_polynomial_degree: 1,
            precond_polynomial_damping: 1.0,
            precond_schur_degree: default_schur_polynomial_degree(),
            precond_schur_damping: default_schur_polynomial_damping(),
            precond_schur_kind: PcFanBackendPressureSchurPreconditionerKind::Polynomial,
            precond_schur_chebyshev_degree: default_chebyshev_degree(),
            precond_schur_chebyshev_eigenvalue_ratio: default_chebyshev_eigenvalue_ratio(),
            precond_kind: PcFanBackendPreconditionerKind::PolynomialJacobi,
            precond_chebyshev_degree: 5,
            precond_chebyshev_eigenvalue_ratio: 30.0,
            solver_kind: PcFanLinearSolverKind::F64,
            solver_ir_inner_max_iterations: 50,
            solver_ir_inner_tolerance: 1.0e-3,
            use_direct_solver: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PcFanOptimizationWeights {
    pub acoustic_weight: f64,
    pub airflow_penalty_weight: f64,
    pub collision_penalty_weight: f64,
    /// Fractional half-width of the airflow tolerance band. The squared
    /// airflow penalty is zero whenever `|airflow_cfm − target_airflow_cfm|`
    /// stays inside this fraction of the target; outside it grows
    /// quadratically with `airflow_penalty_weight`. Default 0.10 (±10%).
    pub airflow_band_fraction: f64,
    /// Weight on the spectral-spread penalty `(1 − spectral_flatness)`.
    /// `spectral_flatness ∈ (0,1]` is the geometric/arithmetic-mean ratio
    /// of harmonic acoustic powers; a tonal spectrum (one big peak at
    /// blade-passing frequency) gives flatness ≈ 0, a flat broadband
    /// spectrum gives flatness ≈ 1. The penalty `1 − flatness` is added
    /// to the objective scaled by this weight. Default 0.0 (off).
    pub spectral_spread_weight: f64,
    /// Weight on the directivity-spread penalty
    /// `std-dev of per-observer SPL across the observer grid (in dB)`.
    /// Lower std-dev means the fan radiates more uniformly in space,
    /// which is preferred for desktop placement. Default 0.0 (off).
    pub directivity_spread_weight: f64,
    /// Weight on the operating-point spread penalty (std-dev in dB of the
    /// overall SPL across the configured `operating_rpms`). Higher values
    /// push toward designs whose noise level is robust to RPM changes.
    /// Default 0.0 (off — only the centre RPM is solved, fast path).
    pub operating_spread_weight: f64,
    /// Minimum required radial gap between blade tip and shroud, in
    /// metres. Designs whose tip extends within this distance of the
    /// shroud incur a quadratic penalty.
    pub shroud_clearance_min_m: f64,
    /// Quadratic-penalty weight for tip-shroud clearance violations.
    /// Default `1.0e6` — high enough to dominate the new physics-based
    /// noise gradients (which can be several dB per millimetre of
    /// shroud-piercing sweep), so the optimiser treats it as a hard
    /// constraint. Old default `1.0e3` was tuned against the empirical
    /// `0.45/√h` strut-wake source and was not strong enough versus
    /// the Sears RSI / Amiet LE / BPM TE terms.
    pub shroud_clearance_penalty_weight: f64,
    /// Quadratic-penalty weight for blade vertices that exit the
    /// inner-frame cavity. Default `1.0e6`. See note on
    /// `shroud_clearance_penalty_weight` above for the rationale on
    /// the 1000× bump from the legacy `1.0e3`.
    pub frame_containment_penalty_weight: f64,
    /// Quadratic-penalty weight for the blade root deviating from the
    /// hub radius. Walks panels with `span_fraction < 0.05` and
    /// penalises any vertex whose distance from the rotation axis
    /// differs from `hub_radius_m`. Belt-and-braces alongside the
    /// `blade_path[0] = 0` bound pin: even if a future config bypasses
    /// the bound, this penalty keeps the blade root anchored to the
    /// rotor.
    pub hub_attachment_penalty_weight: f64,
    /// Weight for normalized thermal-envelope violations.
    pub thermal_penalty_weight: f64,
    /// Weight for normalized stress/modal/imbalance violations.
    pub structural_penalty_weight: f64,
    /// Weight for normalized manufacturing-envelope violations.
    pub manufacturing_penalty_weight: f64,
    /// When true, the acoustic objective term uses the A-weighted
    /// overall SPL (IEC 61672-1) rather than the unweighted level.
    /// At BPF ≈ 80 Hz the A-weighting is roughly −22 dB versus 1 kHz, so
    /// disabling this makes the optimiser chase low-frequency tonal
    /// content that is barely audible. Default: `true`.
    #[serde(default = "default_a_weighting_enabled")]
    pub a_weighting_enabled: bool,
    /// Weight on a tonality penalty applied to the per-harmonic
    /// (A-weighted) acoustic powers. Uses
    /// `sound_metrics::tonality_penalty` — bins whose power exceeds the
    /// average of their neighbours by more than `tonality_threshold_db`
    /// contribute. Pure broadband → 0; a strong tone → tens of dB.
    /// Default: `0.0` (off). Adjoint coverage is FD-only.
    #[serde(default)]
    pub tonality_penalty_weight: f64,
    /// Threshold (dB above the local broadband floor) above which a
    /// per-harmonic bin counts as a tone for the tonality penalty.
    /// Default: 6 dB (one doubling of power).
    #[serde(default = "default_tonality_threshold_db")]
    pub tonality_threshold_db: f64,
}

fn default_a_weighting_enabled() -> bool {
    true
}

fn default_tonality_threshold_db() -> f64 {
    6.0
}

impl Default for PcFanOptimizationWeights {
    fn default() -> Self {
        Self {
            acoustic_weight: 1.0,
            airflow_penalty_weight: 0.05,
            collision_penalty_weight: 1.0e6,
            airflow_band_fraction: 0.10,
            spectral_spread_weight: 0.0,
            directivity_spread_weight: 0.0,
            operating_spread_weight: 0.0,
            shroud_clearance_min_m: 0.001, // 1 mm minimum gap
            shroud_clearance_penalty_weight: 1.0e6,
            frame_containment_penalty_weight: 1.0e6,
            hub_attachment_penalty_weight: 1.0e6,
            thermal_penalty_weight: 0.0,
            structural_penalty_weight: 0.0,
            manufacturing_penalty_weight: 0.0,
            a_weighting_enabled: default_a_weighting_enabled(),
            tonality_penalty_weight: 0.0,
            tonality_threshold_db: default_tonality_threshold_db(),
        }
    }
}

/// Acoustic propagation environment dispatched per harmonic in
/// `evaluate_acoustics`. Each variant defines how the per-harmonic source
/// pressure is mapped to the level reported on `PcFanAcousticSolution`:
///
/// - `FreeField`: no correction. Pressure passes through unchanged.
/// - `DuctCutoff`: legacy first-order hard-walled circular-duct (1,0)-
///   mode cut-off filter — the only propagation model the crate had
///   before Phase 3. Preserves existing test behaviour.
/// - `RigidCavity`: rigid-rectangular-cavity modal Green's-function
///   gain. The dominant model for "fan installed in a PC chassis":
///   per-harmonic gain peaks at cavity resonances and dips between
///   them, producing the 5–15 dB coloration that free-field models miss.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PcFanPropagationMode {
    FreeField,
    DuctCutoff,
    RigidCavity,
}

impl PcFanPropagationMode {
    pub const fn is_free_field(self) -> bool {
        matches!(self, PcFanPropagationMode::FreeField)
    }

    pub const fn is_duct_cutoff(self) -> bool {
        matches!(self, PcFanPropagationMode::DuctCutoff)
    }

    pub const fn is_rigid_cavity(self) -> bool {
        matches!(self, PcFanPropagationMode::RigidCavity)
    }
}

/// Chassis enclosure parameters used by `PcFanPropagationMode::RigidCavity`.
///
/// Models the cavity as a rigid-walled rectangular box of internal
/// dimensions `chassis_dimensions_m`. Source and listener positions are
/// inside the cavity (relative to the `(0, 0, 0)` corner). The modal sum
/// runs over `(m, n, p) ∈ [0, max_mode_index]³` excluding `(0,0,0)`.
/// `modal_damping` is the loss factor `η` applied uniformly to all
/// modes (covers absorber padding and panel losses).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PcFanEnclosureConfig {
    /// Propagation environment (see `PcFanPropagationMode`). Default:
    /// `DuctCutoff` (preserves the bench-style legacy behaviour). Switch
    /// to `RigidCavity` for installed-in-chassis design studies.
    #[serde(default = "default_propagation_mode")]
    pub mode: PcFanPropagationMode,
    /// Internal chassis dimensions [m]: `[width_x, height_y, depth_z]`.
    /// Default: a typical mid-tower internal cavity (0.40 × 0.40 × 0.20).
    #[serde(default = "default_chassis_dimensions")]
    pub chassis_dimensions_m: [f64; 3],
    /// Source position relative to the `(0,0,0)` corner [m]. Default:
    /// chassis centre (where the fan typically sits on a side panel).
    #[serde(default = "default_chassis_centre")]
    pub source_position_m: [f64; 3],
    /// Listener position relative to the `(0,0,0)` corner [m]. Default:
    /// chassis front face centre, i.e. one face away from the source.
    #[serde(default = "default_chassis_listener")]
    pub listener_position_m: [f64; 3],
    /// Modal loss factor η (Q^{-1}). The uncalibrated prior is 0.08
    /// (`Q = 12.5`); use 0.03–0.20 for sensitivity analysis.
    #[serde(default = "default_modal_damping")]
    pub modal_damping: f64,
    /// Maximum mode index per axis. The total mode count is
    /// `(max+1)^3 − 1` (DC mode is excluded). Default 8 covers the
    /// audible band up to ~3.4 kHz on a 0.40 m dimension.
    #[serde(default = "default_max_mode_index")]
    pub max_mode_index: usize,
    /// Reference distance [m] for all source pressures and free-field sound
    /// power. Observer pressures scale from this radius as `r_ref/r`.
    #[serde(default = "default_reference_distance")]
    pub reference_distance_m: f64,
}

fn default_propagation_mode() -> PcFanPropagationMode {
    PcFanPropagationMode::DuctCutoff
}

fn default_chassis_dimensions() -> [f64; 3] {
    [0.40, 0.40, 0.20]
}

fn default_chassis_centre() -> [f64; 3] {
    [0.20, 0.20, 0.10]
}

fn default_chassis_listener() -> [f64; 3] {
    [0.20, 0.20, 0.18]
}

fn default_modal_damping() -> f64 {
    CHASSIS_MODAL_LOSS_FACTOR_DEFAULT
}

fn default_max_mode_index() -> usize {
    8
}

fn default_reference_distance() -> f64 {
    1.0
}

impl Default for PcFanEnclosureConfig {
    fn default() -> Self {
        Self {
            mode: default_propagation_mode(),
            chassis_dimensions_m: default_chassis_dimensions(),
            source_position_m: default_chassis_centre(),
            listener_position_m: default_chassis_listener(),
            modal_damping: default_modal_damping(),
            max_mode_index: default_max_mode_index(),
            reference_distance_m: default_reference_distance(),
        }
    }
}

impl PcFanEnclosureConfig {
    /// Anechoic / bench-validation preset. No propagation correction.
    pub const fn free_field() -> Self {
        Self {
            mode: PcFanPropagationMode::FreeField,
            chassis_dimensions_m: [0.0; 3],
            source_position_m: [0.0; 3],
            listener_position_m: [0.0; 3],
            modal_damping: 0.0,
            max_mode_index: 0,
            reference_distance_m: 1.0,
        }
    }

    /// Mid-tower chassis preset (0.40 × 0.40 × 0.20 m, 5% modal damping).
    pub fn mid_tower() -> Self {
        Self::default()
    }
}

/// Off-design RPMs evaluated when `operating_spread_weight > 0`. The
/// optimiser additionally solves the design at each entry and reports the
/// std-dev of overall SPL across them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanOperatingPointSpread {
    /// RPMs to evaluate. Must include at least two distinct values for the
    /// std-dev to be non-zero. Empty → fast path (centre RPM only).
    pub rpms: Vec<f64>,
}

/// One reduced-order operating-envelope sample for a fixed design.
#[derive(Debug, Clone, PartialEq)]
pub struct PcFanOperatingEnvelopePoint {
    pub rpm: f64,
    pub airflow_cfm: f64,
    pub pressure_rise_pa: f64,
    pub acoustic_level_db: f64,
    pub sound_power_level_db: f64,
    pub temperature_margin_c: f64,
    pub stress_margin_pa: f64,
    pub modal_margin_hz: f64,
    pub feasible: bool,
}

/// Deterministic operating-envelope report for a fixed design.
#[derive(Debug, Clone, PartialEq)]
pub struct PcFanOperatingEnvelope {
    pub points: Vec<PcFanOperatingEnvelopePoint>,
    pub finite: bool,
    pub airflow_strictly_increases_with_rpm: bool,
    pub pressure_strictly_increases_with_rpm: bool,
    pub all_points_feasible: bool,
}

/// One point on the reduced-order static fan curve at a fixed RPM.
///
/// `flow_fraction` is normalized by the model's free-airflow prediction. The
/// fan pressure is the reduced-order pressure-rise curve, while
/// `system_pressure_rise_pa` is the installed enclosure loss at the same
/// flow. Keeping both values makes the curve useful for both product-style
/// fan-curve plots and installed operating-point diagnostics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcFanFanCurvePoint {
    pub rpm: f64,
    pub flow_fraction: f64,
    pub airflow_cfm: f64,
    pub fan_pressure_rise_pa: f64,
    pub system_pressure_rise_pa: f64,
    pub acoustic_level_db: f64,
    pub sound_power_level_db: f64,
    pub fluid_power_w: f64,
    pub electrical_power_w: f64,
    pub efficiency: f64,
    pub feasible: bool,
}

/// Intersection of the reduced fan curve and installed system-resistance
/// curve at one RPM.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcFanFanCurveOperatingPoint {
    pub flow_fraction: f64,
    pub airflow_cfm: f64,
    pub fan_pressure_rise_pa: f64,
    pub system_pressure_rise_pa: f64,
    pub residual_pa: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Checked reduced-order fan-curve report.
#[derive(Debug, Clone, PartialEq)]
pub struct PcFanFanCurve {
    pub points: Vec<PcFanFanCurvePoint>,
    pub free_airflow_cfm: f64,
    pub shutoff_pressure_pa: f64,
    pub operating_point: Option<PcFanFanCurveOperatingPoint>,
    pub finite: bool,
    pub airflow_strictly_increases: bool,
    pub fan_pressure_strictly_decreases: bool,
    pub efficiencies_bounded: bool,
    pub all_points_feasible: bool,
}

impl PcFanFanCurve {
    /// Whether the curve has a physically ordered, finite, feasible sample
    /// set. The reduced-order model is intentionally explicit about its
    /// limits; this is not a resolved-flow or measured-product admission.
    pub fn is_valid(&self) -> bool {
        self.points.len() >= 2
            && self.finite
            && self
                .operating_point
                .is_some_and(|point| point.converged && point.residual_pa.is_finite())
            && self.airflow_strictly_increases
            && self.fan_pressure_strictly_decreases
            && self.efficiencies_bounded
            && self.all_points_feasible
    }
}

/// Targets and weights for reduced-order multi-point fan-curve optimization.
///
/// All target vectors follow `flow_fractions` point-for-point. Acoustic and
/// efficiency targets are optional so a caller can optimize only the
/// aerodynamic pressure/flow curve when no measured acoustic data is
/// available.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanFanCurveObjectiveConfig {
    pub rpm: f64,
    pub flow_fractions: Vec<f64>,
    pub target_airflow_cfm: Vec<f64>,
    pub target_pressure_rise_pa: Vec<f64>,
    #[serde(default)]
    pub target_acoustic_level_db: Option<Vec<f64>>,
    #[serde(default)]
    pub target_efficiency: Option<Vec<f64>>,
    pub airflow_weight: f64,
    pub pressure_weight: f64,
    #[serde(default)]
    pub acoustic_weight: f64,
    #[serde(default)]
    pub efficiency_weight: f64,
}

/// Result of scoring a design against a multi-point fan-curve target.
#[derive(Debug, Clone, PartialEq)]
pub struct PcFanFanCurveObjective {
    pub score: f64,
    pub airflow_rmse_cfm: f64,
    pub pressure_rmse_pa: f64,
    pub acoustic_rmse_db: Option<f64>,
    pub efficiency_rmse: Option<f64>,
    pub curve: PcFanFanCurve,
    pub feasible: bool,
}

impl PcFanOperatingEnvelope {
    /// Whether the envelope is internally consistent and every sample meets
    /// the hard reduced-order feasibility contract.
    pub fn is_valid(&self) -> bool {
        self.finite
            && self.airflow_strictly_increases_with_rpm
            && self.pressure_strictly_increases_with_rpm
            && self.all_points_feasible
    }
}

impl Default for PcFanOperatingPointSpread {
    fn default() -> Self {
        // 70% / 100% / 130% of the centre RPM is a typical robustness sweep.
        Self {
            rpms: vec![420.0, 600.0, 780.0],
        }
    }
}

/// Configuration of the far-field observer grid used to compute SPL
/// directivity. Observers are placed on a sphere of radius `distance_m`
/// centred on the fan, sampled on a regular `azimuth_count × elevation_count`
/// grid in spherical coordinates `(θ, φ)`. `θ` is the polar angle measured
/// from the fan's axis of rotation (`+Z`), `φ` is the azimuth.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PcFanObserverArrayConfig {
    pub azimuth_count: usize,
    pub elevation_count: usize,
    pub distance_m: f64,
}

impl Default for PcFanObserverArrayConfig {
    fn default() -> Self {
        Self {
            azimuth_count: 8,
            elevation_count: 3,
            distance_m: 1.0,
        }
    }
}

impl PcFanObserverArrayConfig {
    pub fn observer_count(&self) -> usize {
        self.azimuth_count.max(1) * self.elevation_count.max(1)
    }
}

/// Derivative policy for the PC-fan optimization/evaluation surface.
///
/// Finite difference covers the full declared objective; resolved-CFD
/// perturbations must pass checked objective admission, while the reduced
/// adjoint covers only the reduced-order scalar-airflow state contract.
/// `DerivativeFree` is explicit so callers can route objectives containing
/// unsupported or nonsmooth terms to DE without presenting a zero-valued
/// placeholder as an adjoint gradient.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PcFanDerivativePolicy {
    #[default]
    FiniteDifferenceFullObjective,
    /// Discrete adjoint of the reduced-order scalar airflow state. The
    /// acoustic algebraic state is eliminated analytically through the
    /// reduced acoustic source model; resolved CFD rejects this policy.
    ReducedOrderDiscreteAdjoint,
    DerivativeFree,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PcFanOptimizerConfig {
    pub max_iterations: usize,
    pub finite_difference_step: f64,
    pub gradient_tolerance: f64,
    pub objective_tolerance: f64,
    pub memory: usize,
    /// If true (default), the shroud-radius design variables are held at
    /// their baseline values during optimisation. The frame, shroud and
    /// hub geometry are physical fixed parts of a PC fan; only the blade
    /// shape DOFs (chord/camber/thickness/twist + path offsets per
    /// station) should be tunable. Set to false to also optimise the
    /// shroud profile.
    pub freeze_shroud: bool,
    /// If true (default), camber is constrained to the non-negative
    /// half-line. Negative camber represents an inverted blade that
    /// would push air the opposite direction at fixed RPM; for a forward-
    /// blowing fan the optimiser should not be allowed to converge there.
    pub require_positive_camber: bool,
}

impl Default for PcFanOptimizerConfig {
    fn default() -> Self {
        Self {
            max_iterations: 8,
            finite_difference_step: 1.0e-4,
            gradient_tolerance: 1.0e-5,
            objective_tolerance: 1.0e-8,
            memory: 6,
            freeze_shroud: true,
            require_positive_camber: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanDesignVariables {
    pub blade_path: [BladePathControl; BLADE_PATH_CONTROL_COUNT],
    pub blade_sections: [BladeSectionControl; BLADE_SECTION_STATION_COUNT],
    /// Hub-radius offset from `PcFanDimensions::hub_radius_m`.
    #[serde(default)]
    pub hub_radius_offset_m: f64,
    /// Blade-tip-radius offset from `PcFanDimensions::blade_tip_radius_m`.
    #[serde(default)]
    pub tip_radius_offset_m: f64,
    pub shroud_radius_offsets_m: [f64; SHROUD_CONTROL_COUNT],
}

impl Default for PcFanDesignVariables {
    fn default() -> Self {
        Self::baseline_for(&PcFanConfig::default())
    }
}

impl PcFanDesignVariables {
    pub fn to_vector(&self) -> PcFanDesignVector {
        let mut values = [0.0; DESIGN_VARIABLE_COUNT];
        let mut i = 0;

        for control in self.blade_path {
            values[i] = control.tangential_offset_m;
            values[i + 1] = control.axial_offset_m;
            i += 2;
        }
        for section in self.blade_sections {
            values[i] = section.chord_m;
            values[i + 1] = section.camber_m;
            values[i + 2] = section.thickness_m;
            values[i + 3] = section.twist_deg;
            i += 4;
        }
        values[i] = self.hub_radius_offset_m;
        values[i + 1] = self.tip_radius_offset_m;
        i += RADIAL_ENVELOPE_CONTROL_COUNT;
        for offset in self.shroud_radius_offsets_m {
            values[i] = offset;
            i += 1;
        }

        values
    }

    pub fn from_vector(values: PcFanDesignVector) -> Self {
        let mut i = 0;
        let mut blade_path = [BladePathControl::default(); BLADE_PATH_CONTROL_COUNT];
        for control in &mut blade_path {
            control.tangential_offset_m = values[i];
            control.axial_offset_m = values[i + 1];
            i += 2;
        }

        let mut blade_sections = [BladeSectionControl::default(); BLADE_SECTION_STATION_COUNT];
        for section in &mut blade_sections {
            section.chord_m = values[i];
            section.camber_m = values[i + 1];
            section.thickness_m = values[i + 2];
            section.twist_deg = values[i + 3];
            i += 4;
        }

        let hub_radius_offset_m = values[i];
        let tip_radius_offset_m = values[i + 1];
        i += RADIAL_ENVELOPE_CONTROL_COUNT;
        let mut shroud_radius_offsets_m = [0.0; SHROUD_CONTROL_COUNT];
        for offset in &mut shroud_radius_offsets_m {
            *offset = values[i];
            i += 1;
        }

        Self {
            blade_path,
            blade_sections,
            hub_radius_offset_m,
            tip_radius_offset_m,
            shroud_radius_offsets_m,
        }
    }

    /// Effective hub radius represented by this design.
    pub fn effective_hub_radius_m(&self, config: &PcFanConfig) -> f64 {
        config.dimensions.hub_radius_m + self.hub_radius_offset_m
    }

    /// Effective blade-tip radius represented by this design.
    pub fn effective_tip_radius_m(&self, config: &PcFanConfig) -> f64 {
        config.dimensions.blade_tip_radius_m + self.tip_radius_offset_m
    }

    /// Whether this design requests radial-envelope motion whose mesh-motion
    /// derivative is outside the fixed-topology production CFD contract.
    pub fn has_nonzero_radial_envelope_offsets(&self) -> bool {
        !self.hub_radius_offset_m.is_finite()
            || !self.tip_radius_offset_m.is_finite()
            || self.hub_radius_offset_m.abs() > f64::EPSILON
            || self.tip_radius_offset_m.abs() > f64::EPSILON
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BladePathControl {
    pub tangential_offset_m: f64,
    pub axial_offset_m: f64,
}

impl Default for BladePathControl {
    fn default() -> Self {
        Self {
            tangential_offset_m: 0.0,
            axial_offset_m: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BladeSectionControl {
    pub chord_m: f64,
    pub camber_m: f64,
    pub thickness_m: f64,
    pub twist_deg: f64,
}

impl Default for BladeSectionControl {
    fn default() -> Self {
        Self {
            chord_m: 0.018,
            camber_m: 0.0,
            thickness_m: 0.0012,
            twist_deg: 0.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PcFanGeometry {
    pub hub: PcFanHub,
    /// Effective blade-tip radius after applying the radial-envelope design
    /// control. Kept beside the generated mesh so reduced-order consumers do
    /// not silently fall back to the fixed configuration dimension.
    pub blade_tip_radius_m: f64,
    pub blades: Vec<PcFanBlade>,
    pub shroud: PcFanShroud,
    pub frame_box: PcFanFrameBox,
    pub frame_mounts: [FrameMountPoint; FRAME_MOUNT_COUNT],
    pub support_struts: [PcFanSupportStrut; SUPPORT_STRUT_COUNT],
    pub motor_mount: PcFanMotorMount,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcFanHub {
    pub radius_m: f64,
    pub depth_m: f64,
    pub center_z_m: f64,
    pub circumferential_segments: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PcFanBlade {
    pub index: usize,
    pub azimuth_rad: f64,
    pub panels: Vec<PcFanBladePanel>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcFanBladePanel {
    pub vertices: [Point3; 4],
    pub blade_index: usize,
    pub span_fraction: f64,
    pub chord_fraction: f64,
    pub thickness_m: f64,
    pub area: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcFanShroud {
    pub inner_radius_m: f64,
    pub outer_radius_m: f64,
    pub wall_thickness_m: f64,
    pub radius_controls_m: [f64; SHROUD_CONTROL_COUNT],
    pub depth_m: f64,
    pub axial_segments: usize,
    pub circumferential_segments: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcFanFrameBox {
    pub outer_side_m: f64,
    pub inner_side_m: f64,
    pub depth_m: f64,
    pub wall_thickness_m: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameMountPoint {
    pub position: Point3,
    pub pad_size_m: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcFanSupportStrut {
    pub start: Point3,
    pub end: Point3,
    pub width_m: f64,
    pub thickness_m: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcFanMotorMount {
    pub center: Point3,
    pub radius_m: f64,
    pub depth_m: f64,
    pub circumferential_segments: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcFanSurfaceKind {
    Hub,
    Blade,
    Shroud,
    FrameBox,
    FrameMount,
    SupportStrut,
    MotorMount,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcFanSurfacePanel {
    pub vertices: [Point3; 4],
    pub kind: PcFanSurfaceKind,
    pub blade_index: Option<usize>,
    pub span_fraction: f64,
    pub area: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcFanCfdCellRegion {
    InletExtension,
    FanBox,
    OutletExtension,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcFanCfdBoundaryKind {
    Inlet,
    Outlet,
    OuterWall,
    FrameWall,
    ShroudWall,
    SupportWall,
    MotorWall,
    RotatingHub,
    RotatingBlade,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcFanCfdCell {
    pub node_indices: [usize; 8],
    pub region: PcFanCfdCellRegion,
    pub volume_m3: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcFanCfdBoundaryFace {
    pub vertices: [Point3; 4],
    pub kind: PcFanCfdBoundaryKind,
    pub source_surface: Option<PcFanSurfaceKind>,
    pub blade_index: Option<usize>,
    pub area: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PcFanCfdDomain {
    pub nodes: Vec<Point3>,
    pub cells: Vec<PcFanCfdCell>,
    pub boundary_faces: Vec<PcFanCfdBoundaryFace>,
    pub min_corner: Point3,
    pub max_corner: Point3,
    pub inlet_area_m2: f64,
    pub outlet_area_m2: f64,
    pub volume_m3: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PcFanCfdFlowSolution {
    pub domain: PcFanCfdDomain,
    /// Concrete FEM linear-solve backend used for this solution.
    ///
    /// This is intentionally separate from `PcFanCfdConfig::backend`, which
    /// records only the requested backend and may fall back to CPU when an
    /// accelerator is unavailable.
    pub actual_backend: PcFanCfdBackend,
    /// Requested/actual backend, fallback, and residency contract.
    pub backend_provenance: PcFanCfdBackendProvenance,
    /// Build/runtime execution metadata for reproducible cross-architecture
    /// comparisons.
    pub execution_environment: PcFanCfdExecutionEnvironment,
    pub nodes: Vec<Point3>,
    pub velocity_m_s: Vec<[f64; 3]>,
    pub pressure_pa: Vec<f64>,
    pub airflow_cfm: f64,
    pub inlet_airflow_cfm: f64,
    pub mass_balance_error_kg_s: f64,
    pub pressure_drop_pa: f64,
    pub max_velocity_m_s: f64,
    pub reynolds_number: f64,
    pub fem_node_count: usize,
    pub fem_element_count: usize,
    pub boundary_dof_count: usize,
    /// Integrated actuator-disk loading, when the CFD model applies a blade
    /// body force.  Rotating-wall CFD publishes its recovered loading separately.
    pub actuator_loading: Option<PcFanActuatorLoading>,
    /// Recovered immersed-wall loading for the rotating-wall CFD path.
    pub rotating_wall_loading: Option<PcFanActuatorLoading>,
    /// Control-volume conservation diagnostics for this CFD evaluation.
    pub conservation: PcFanCfdConservation,
    /// Last production FEM Picard/Oseen linearization, when available. This
    /// is not an adjoint-ready full Newton tangent.
    pub picard_linearization: Option<PcFanResolvedFlowPicardLinearization>,
    /// Exact analytic Navier--Stokes convection tangent at the prescribed
    /// state, when the FEM production map was evaluated with an initial state.
    pub newton_linearization: Option<crate::resolved_adjoint::PcFanResolvedFlowNewtonLinearization>,
    /// Exact frozen-mesh derivative of the body-force residual with respect to
    /// the declared blade design variables, when the active CFD path provides
    /// that analytic contract. Geometry/topology-dependent cases are `None`.
    pub residual_design_jacobian: Option<Vec<Vec<f64>>>,
    pub iterations: usize,
    pub residual: f64,
    pub converged: bool,
}

/// Control-volume conservation ledger for one steady PC-fan CFD evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcFanCfdConservation {
    pub inlet_mass_flow_kg_s: f64,
    pub outlet_mass_flow_kg_s: f64,
    pub relative_mass_imbalance: f64,
    pub inlet_axial_momentum_flux_n: f64,
    pub outlet_axial_momentum_flux_n: f64,
    /// Net control-volume pressure traction along +z (`p_in A_in - p_out A_out`).
    pub pressure_force_n: f64,
    /// Recovered actuator/wall force on the fluid along +z.
    pub body_force_n: f64,
    pub axial_momentum_residual_n: f64,
    pub relative_momentum_imbalance: f64,
    /// Axial recovered actuator/wall force used for pressure-traction closure.
    pub pressure_force_from_loading_n: f64,
    pub relative_pressure_closure: f64,
    /// Pressure-rise power delivered to the fluid: `-pressure_force_n * Q`.
    pub pressure_power_w: f64,
    pub shaft_power_to_fluid_w: f64,
    pub relative_energy_imbalance: f64,
}

/// Volume-integrated blade actuator loading published by actuator-force CFD.
///
/// All quantities are forces exerted on the fluid.  The shaft-power value is
/// the corresponding `torque_on_fluid_nm × angular_speed_rad_s`; it is not a
/// motor electrical-input estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcFanActuatorLoading {
    /// Net actuator force on the fluid in product Cartesian coordinates.
    pub force_on_fluid_n: [f64; 3],
    /// Axial torque on the fluid about the rotor axis.
    pub torque_on_fluid_nm: f64,
    /// Angular speed used when integrating the actuator force.
    pub angular_speed_rad_s: f64,
    /// Mechanical shaft power transferred from the actuator to the fluid.
    pub shaft_power_to_fluid_w: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcFanCfdMeshConvergenceConfig {
    pub start_cells: usize,
    pub max_cells: usize,
    pub relative_tolerance: f64,
    pub min_levels: usize,
    /// Absolute nonlinear residual budget required at every admitted level.
    pub max_final_residual: f64,
}

/// Configuration errors that must be resolved before a mesh-convergence
/// sweep is allowed to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcFanCfdMeshConvergenceConfigError {
    /// A mesh level must contain at least one cell in each direction.
    StartCellsMustBePositive,
    /// Refinement bounds must be ordered from coarse to fine.
    MaxCellsBeforeStart {
        start_cells: usize,
        max_cells: usize,
    },
    /// Relative tolerances must be finite and non-negative.
    InvalidRelativeTolerance,
    /// Final nonlinear residual budget must be finite non-negative.
    InvalidFinalResidual,
    /// A CFD conservation tolerance must be finite non-negative.
    InvalidConservationTolerance { metric: &'static str },
    /// At least two levels are required for a change estimate.
    MinLevelsTooSmall,
    /// The requested bounds cannot provide the requested number of levels.
    InsufficientLevels { available: usize, required: usize },
}

impl std::fmt::Display for PcFanCfdMeshConvergenceConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StartCellsMustBePositive => {
                write!(
                    f,
                    "PC-fan CFD mesh convergence start_cells must be positive"
                )
            }
            Self::MaxCellsBeforeStart {
                start_cells,
                max_cells,
            } => write!(
                f,
                "PC-fan CFD mesh convergence max_cells ({max_cells}) must be at least start_cells ({start_cells})"
            ),
            Self::InvalidRelativeTolerance => write!(
                f,
                "PC-fan CFD mesh convergence relative_tolerance must be finite and non-negative"
            ),
            Self::InvalidFinalResidual => write!(
                f,
                "PC-fan CFD mesh convergence max_final_residual must be finite non-negative"
            ),
            Self::InvalidConservationTolerance { metric } => write!(
                f,
                "PC-fan CFD mesh convergence {metric} tolerance must be finite non-negative"
            ),
            Self::MinLevelsTooSmall => write!(
                f,
                "PC-fan CFD mesh convergence min_levels must be at least two"
            ),
            Self::InsufficientLevels {
                available,
                required,
            } => write!(
                f,
                "PC-fan CFD mesh convergence provides {available} levels but requires {required}"
            ),
        }
    }
}

impl std::error::Error for PcFanCfdMeshConvergenceConfigError {}

impl PcFanCfdMeshConvergenceConfig {
    /// Validate the bounded, unit-cell refinement sweep before any CFD solve
    /// is started.
    pub fn validate(&self) -> Result<(), PcFanCfdMeshConvergenceConfigError> {
        if self.start_cells == 0 {
            return Err(PcFanCfdMeshConvergenceConfigError::StartCellsMustBePositive);
        }
        if self.max_cells < self.start_cells {
            return Err(PcFanCfdMeshConvergenceConfigError::MaxCellsBeforeStart {
                start_cells: self.start_cells,
                max_cells: self.max_cells,
            });
        }
        if !self.relative_tolerance.is_finite() || self.relative_tolerance < 0.0 {
            return Err(PcFanCfdMeshConvergenceConfigError::InvalidRelativeTolerance);
        }
        if !self.max_final_residual.is_finite() || self.max_final_residual < 0.0 {
            return Err(PcFanCfdMeshConvergenceConfigError::InvalidFinalResidual);
        }
        if self.min_levels < 2 {
            return Err(PcFanCfdMeshConvergenceConfigError::MinLevelsTooSmall);
        }

        let available = self.max_cells - self.start_cells + 1;
        if available < self.min_levels {
            return Err(PcFanCfdMeshConvergenceConfigError::InsufficientLevels {
                available,
                required: self.min_levels,
            });
        }

        Ok(())
    }
}

impl Default for PcFanCfdMeshConvergenceConfig {
    fn default() -> Self {
        Self {
            start_cells: 6,
            max_cells: 8,
            relative_tolerance: 0.08,
            min_levels: 3,
            max_final_residual: 1.0e-3,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PcFanCfdMeshLevel {
    pub cells_x: usize,
    pub cells_y: usize,
    pub cells_z: usize,
    pub solution: PcFanCfdFlowSolution,
    pub airflow_relative_change: Option<f64>,
    pub pressure_relative_change: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PcFanCfdMeshConvergence {
    pub levels: Vec<PcFanCfdMeshLevel>,
    pub selected_solution: PcFanCfdFlowSolution,
    pub converged: bool,
    /// Every level used for the final comparison reached nonlinear
    /// convergence, satisfied the residual budget, and reported finite
    /// diagnostics.
    pub all_levels_converged_and_finite: bool,
    pub airflow_relative_change: Option<f64>,
    pub pressure_relative_change: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcFanCollisionStatus {
    pub is_valid: bool,
    pub root_merge_count: usize,
    pub invalid_intersection_count: usize,
    pub minimum_blade_gap_m: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcFanConstraintMargins {
    /// Signed distance to the allowed airflow band in CFM. Non-negative
    /// values satisfy the target-flow constraint.
    pub airflow_cfm: f64,
    /// Smallest radial clearance from any blade vertex to the shroud inner
    /// surface, in metres.
    pub minimum_shroud_clearance_m: f64,
    /// Smallest signed clearance from any blade vertex to the inner frame
    /// cavity, in metres. Negative values are frame-containment violations.
    pub minimum_frame_clearance_m: f64,
    /// Absolute tangential offset of the hub-attached blade root, in metres.
    /// A non-zero value would detach the root from the rotor.
    pub root_tangential_attachment_error_m: f64,
    pub invalid_intersection_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcFanFlowSolution {
    pub airflow_cfm: f64,
    pub axial_velocity_m_s: f64,
    pub pressure_rise_pa: f64,
    pub reynolds_number: f64,
    pub mach_number: f64,
    pub disk_solidity: f64,
    pub fixed_hardware: PcFanHardwareFlowEffects,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcFanHardwareFlowEffects {
    pub inlet_open_area_m2: f64,
    pub blockage_area_m2: f64,
    pub blockage_ratio: f64,
    pub motor_blockage_ratio: f64,
    pub strut_blockage_ratio: f64,
    pub frame_lip_loss_factor: f64,
    pub support_wake_loss_factor: f64,
    pub total_loss_factor: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcFanAcousticHarmonic {
    pub harmonic: usize,
    pub frequency_hz: f64,
    pub pressure_pa: f64,
    pub sound_pressure_level_db: f64,
    pub sound_power_level_db: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PcFanAcousticSolution {
    pub blade_passing_frequency_hz: f64,
    pub overall_spl_db: f64,
    /// A-weighted overall SPL (dB(A), IEC 61672-1) — the per-harmonic
    /// pressures are scaled by `a_weighting(f_h)` before energy summing.
    /// At BPF 80 Hz this is ~22 dB below `overall_spl_db`; the gap closes
    /// as harmonic content moves toward 1 kHz. Used by the optimiser when
    /// `weights.a_weighting_enabled = true` (the default).
    pub a_weighted_overall_spl_db: f64,
    pub overall_swl_db: f64,
    pub fixed_hardware_noise: PcFanHardwareNoise,
    pub harmonics: Vec<PcFanAcousticHarmonic>,
    /// SPL contribution (dB, energy-summed across harmonics) of the
    /// tip-leakage vortex broadband source. Reported separately for
    /// diagnostics; already folded into per-harmonic pressures above.
    pub tip_leakage_spl_db: f64,
    /// SPL contribution (dB, energy-summed across harmonics) of the
    /// Brooks–Pope–Marcolini turbulent-boundary-layer / trailing-edge
    /// broadband source, integrated across the blade span. Reported
    /// separately for diagnostics; already folded into per-harmonic
    /// pressures above.
    pub trailing_edge_spl_db: f64,
    /// SPL contribution (dB, energy-summed across harmonics) of the
    /// Amiet leading-edge inflow-turbulence broadband source. Zero
    /// when `inflow_turbulence_intensity == 0` (the default). Reported
    /// separately for diagnostics; already folded into per-harmonic
    /// pressures above.
    pub leading_edge_inflow_spl_db: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcFanHardwareNoise {
    pub support_wake_spl_db: f64,
    pub motor_mount_spl_db: f64,
    pub combined_spl_db: f64,
    /// Broadband contribution from rotor-strut interaction (Goldstein-style
    /// wake-deficit spectrum). Reported separately for diagnostics; folded
    /// into the per-harmonic pressures and `combined_spl_db`.
    pub support_wake_broadband_spl_db: f64,
    /// Broadband contribution from rotor-motor-mount wake/buffeting.
    /// Reported separately for diagnostics; folded into the per-harmonic
    /// pressures and `combined_spl_db`.
    pub motor_mount_broadband_spl_db: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcFanThermalSolution {
    pub fluid_power_w: f64,
    pub shaft_power_w: f64,
    pub electrical_power_w: f64,
    pub motor_loss_w: f64,
    pub winding_temperature_c: f64,
    pub temperature_margin_c: f64,
    pub power_margin_w: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcFanStructuralSolution {
    pub maximum_stress_pa: f64,
    pub thermal_stress_pa: f64,
    pub thermal_expansion_m: f64,
    pub allowable_stress_pa: f64,
    pub first_mode_hz: f64,
    pub nearest_bpf_separation_hz: f64,
    pub stress_margin_pa: f64,
    pub modal_margin_hz: f64,
    pub imbalance_g_mm: f64,
    pub imbalance_margin_g_mm: f64,
    pub fatigue_alternating_stress_pa: f64,
    pub fatigue_allowable_stress_pa: f64,
    pub fatigue_margin_pa: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcFanManufacturingStatus {
    pub valid: bool,
    pub minimum_thickness_m: f64,
    pub minimum_chord_m: f64,
    pub maximum_twist_gradient_deg: f64,
    pub maximum_shroud_offset_m: f64,
    pub minimum_root_thickness_ratio: f64,
    pub minimum_margin: f64,
}

/// One point in the deterministic manufacturing-tolerance ensemble.
#[derive(Debug, Clone, PartialEq)]
pub struct PcFanManufacturingEnsemblePoint {
    pub variation: PcFanManufacturingVariation,
    pub design_variables: PcFanDesignVariables,
    pub evaluation: PcFanEvaluation,
}

/// Reduced-order robustness report for manufacturing tolerances.
///
/// The nominal point is always first.  Remaining points are the Cartesian
/// corners of all non-zero tolerance axes, so `is_robust()` is an explicit
/// indication that tolerance coverage was requested rather than an implicit
/// claim based on a single nominal evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct PcFanManufacturingEnsemble {
    pub points: Vec<PcFanManufacturingEnsemblePoint>,
    pub nominal_index: usize,
    pub finite: bool,
    pub all_points_feasible: bool,
    pub worst_objective: f64,
    pub mean_objective: f64,
    pub worst_feasibility_violation: f64,
}

impl PcFanManufacturingEnsemble {
    pub fn is_valid(&self) -> bool {
        !self.points.is_empty()
            && self.nominal_index < self.points.len()
            && self.finite
            && self.all_points_feasible
            && self.worst_objective.is_finite()
            && self.mean_objective.is_finite()
            && self.worst_feasibility_violation.is_finite()
    }

    pub fn has_tolerance_coverage(&self) -> bool {
        self.points
            .iter()
            .any(|point| point.variation != PcFanManufacturingVariation::nominal())
    }

    /// Whether the design passes every checked tolerance corner and actually
    /// requested a non-zero manufacturing ensemble.
    pub fn is_robust(&self) -> bool {
        self.is_valid() && self.has_tolerance_coverage()
    }
}

/// One reduced-order operating-envelope point for one manufacturing
/// realization.
#[derive(Debug, Clone, PartialEq)]
pub struct PcFanManufacturingOperatingEnvelopePoint {
    pub variation: PcFanManufacturingVariation,
    pub rpm: f64,
    pub design_variables: PcFanDesignVariables,
    pub point: PcFanOperatingEnvelopePoint,
    pub objective: f64,
}

/// Cross-product report for manufacturing tolerances and operating RPMs.
///
/// Points are ordered by manufacturing realization, then ascending RPM.  The
/// nominal realization is first and each realization has the same RPM list.
/// This is deliberately a reduced-order contract: a resolved-flow fan-curve
/// sweep must use its own checked pressure/load boundary and convergence
/// evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct PcFanManufacturingOperatingEnvelope {
    pub points: Vec<PcFanManufacturingOperatingEnvelopePoint>,
    pub variation_count: usize,
    pub rpm_count: usize,
    pub finite: bool,
    pub all_points_feasible: bool,
    pub all_variations_monotonic: bool,
    pub worst_objective: f64,
    pub mean_objective: f64,
}

impl PcFanManufacturingOperatingEnvelope {
    pub fn is_valid(&self) -> bool {
        !self.points.is_empty()
            && self.variation_count > 0
            && self.rpm_count > 0
            && self.points.len() == self.variation_count * self.rpm_count
            && self.finite
            && self.all_points_feasible
            && self.all_variations_monotonic
            && self.worst_objective.is_finite()
            && self.mean_objective.is_finite()
    }

    pub fn has_tolerance_coverage(&self) -> bool {
        !self.points.is_empty()
            && self
                .points
                .iter()
                .any(|candidate| candidate.variation != PcFanManufacturingVariation::nominal())
    }

    /// Whether both a non-zero tolerance ensemble and at least two operating
    /// points were checked successfully.
    pub fn is_robust(&self) -> bool {
        self.is_valid() && self.has_tolerance_coverage() && self.rpm_count >= 2
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PcFanEvaluation {
    pub airflow_cfm: f64,
    pub acoustic_level_db: f64,
    /// A-weighted overall SPL (dB(A)). This is the level the optimiser
    /// minimises when `weights.a_weighting_enabled = true` (the default).
    /// `acoustic_level_db` is retained for diagnostics.
    pub a_weighted_acoustic_level_db: f64,
    /// Tonality penalty in dB, computed via
    /// `sound_metrics::tonality_penalty` over the (A-weighted, when
    /// enabled) per-harmonic powers. Multiplied by
    /// `weights.tonality_penalty_weight` to form the
    /// `tonality_penalty` contribution to the objective.
    pub tonality_penalty_db: f64,
    /// Tonality contribution to the objective:
    /// `weights.tonality_penalty_weight · tonality_penalty_db`.
    pub tonality_penalty: f64,
    pub sound_power_level_db: f64,
    pub flow: PcFanFlowSolution,
    pub acoustics: PcFanAcousticSolution,
    pub airflow_penalty: f64,
    pub collision_penalty: f64,
    pub collision_status: PcFanCollisionStatus,
    pub objective: f64,
    pub constraint_margins: PcFanConstraintMargins,
    /// Spectral flatness of the per-harmonic acoustic powers, in `(0, 1]`.
    /// 1.0 means perfectly flat (broadband); values near 0 mean tonal-heavy.
    pub spectral_flatness: f64,
    /// Spectral-spread contribution to the objective, equal to
    /// `weights.spectral_spread_weight · (1 − spectral_flatness)`.
    pub spectral_spread_penalty: f64,
    /// Per-observer SPL (dB) on the configured far-field grid. Length =
    /// `config.observers.observer_count()`. Order: outer loop over
    /// elevations (polar angle), inner loop over azimuths.
    pub observer_spls_db: Vec<f64>,
    /// Standard deviation of `observer_spls_db` in dB. Used as the
    /// directivity-spread metric.
    pub directivity_spread_db: f64,
    /// Directivity-spread contribution to the objective, equal to
    /// `weights.directivity_spread_weight · directivity_spread_db`.
    pub directivity_spread_penalty: f64,
    /// Overall SPL (dB) at each off-design RPM in
    /// `config.operating_point_spread.rpms`. Empty when the
    /// `operating_spread_weight` is zero (fast path).
    pub operating_point_spls_db: Vec<f64>,
    /// Standard deviation across `operating_point_spls_db` in dB.
    pub operating_point_spread_db: f64,
    /// Operating-point-spread contribution to the objective, equal to
    /// `weights.operating_spread_weight · operating_point_spread_db`.
    pub operating_point_spread_penalty: f64,
    /// Quadratic penalty for tip-shroud clearance violations. Zero when
    /// the largest blade-tip radial extent leaves at least
    /// `weights.shroud_clearance_min_m` of gap to the shroud's inner
    /// surface.
    pub shroud_clearance_penalty: f64,
    /// Quadratic penalty for any blade vertex that crosses the
    /// inner-frame cavity wall. Zero when every blade vertex stays
    /// inside `|x|, |y| ≤ 0.5·frame_inner_side` and `|z| ≤ 0.5·depth`.
    pub frame_containment_penalty: f64,
    /// Quadratic penalty for blade-root deviation from the hub radius.
    /// Walks panels with `span_fraction < 0.05` and penalises any
    /// vertex whose distance from the rotation axis differs from
    /// `hub_radius_m`. Zero on the baseline geometry (root anchored
    /// at the hub by `blade_path[0] = (0, 0)`).
    pub hub_attachment_penalty: f64,
    pub thermal: PcFanThermalSolution,
    pub structural: PcFanStructuralSolution,
    pub manufacturing: PcFanManufacturingStatus,
    pub thermal_penalty: f64,
    pub structural_penalty: f64,
    pub manufacturing_penalty: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PcFanAdjointState {
    pub airflow_multiplier: f64,
    pub acoustic_pressure_multipliers: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PcFanPrimalResidual {
    pub airflow_cfm: f64,
    pub acoustic_pressure_pa: Vec<f64>,
    pub max_abs: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PcFanShapeGradient {
    pub objective_gradient: PcFanDesignVector,
    pub adjoint: PcFanAdjointState,
    pub residual: PcFanPrimalResidual,
    /// True when the gradient came from finite differences and `adjoint`
    /// holds placeholder zeros rather than a genuine discrete adjoint.
    /// Callers must not take adjoint-driven steps from such a gradient.
    pub adjoint_is_finite_difference: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OptimizedPcFan {
    pub config: PcFanConfig,
    pub design_variables: PcFanDesignVariables,
    pub geometry: PcFanGeometry,
    pub evaluation: PcFanEvaluation,
    pub optimization: PcFanOptimizationSummary,
}

impl OptimizedPcFan {
    /// Whether the returned design may be promoted beyond optimization.
    ///
    /// An optimizer can still return its least-violating design for
    /// diagnostics when no feasible iterate was found, but that result must
    /// never be treated as an accepted design.
    pub fn is_feasible(&self) -> bool {
        self.optimization.feasible
            && self.optimization.fan_curve_feasible
            && self.evaluation.is_feasible(&self.config)
    }

    /// Evaluate the selected design at the configured manufacturing
    /// tolerance corners. This is an explicit promotion check and is not
    /// folded into the hot-loop nominal optimizer feasibility predicate.
    pub fn manufacturing_ensemble(
        &self,
    ) -> Result<PcFanManufacturingEnsemble, crate::evaluation::PcFanManufacturingEnsembleError>
    {
        crate::evaluation::evaluate_pc_fan_manufacturing_ensemble(
            &self.config,
            &self.design_variables,
        )
    }

    /// Whether the selected design passes nominal and non-zero manufacturing
    /// tolerance coverage.
    pub fn is_manufacturing_robustly_feasible(
        &self,
    ) -> Result<bool, crate::evaluation::PcFanManufacturingEnsembleError> {
        self.manufacturing_ensemble()
            .map(|ensemble| ensemble.is_robust())
    }

    /// Evaluate the selected design across manufacturing corners and the
    /// configured reduced-order RPM envelope.
    pub fn manufacturing_operating_envelope(
        &self,
    ) -> Result<
        PcFanManufacturingOperatingEnvelope,
        crate::evaluation::PcFanManufacturingOperatingEnvelopeError,
    > {
        crate::evaluation::evaluate_pc_fan_manufacturing_operating_envelope(
            &self.config,
            &self.design_variables,
        )
    }

    /// Whether the selected design passes tolerance corners at every
    /// configured operating RPM, including trend and feasibility gates.
    pub fn is_manufacturing_envelope_robustly_feasible(
        &self,
    ) -> Result<bool, crate::evaluation::PcFanManufacturingOperatingEnvelopeError> {
        self.manufacturing_operating_envelope()
            .map(|envelope| envelope.is_robust())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PcFanOptimizationSummary {
    pub baseline_objective: f64,
    pub best_objective: f64,
    pub iterations: usize,
    pub function_evaluations: usize,
    pub gradient_evaluations: usize,
    /// Explicit, weight-independent feasibility result for the selected
    /// design. Callers must gate product promotion on this value.
    pub feasible: bool,
    /// Optional reduced-order multi-point fan-curve objective used for
    /// promotion. `None` means the ordinary single-point objective was used.
    pub fan_curve_objective: Option<f64>,
    /// Independent admission result for the optional fan-curve objective.
    /// Ordinary optimizers set this to true.
    pub fan_curve_feasible: bool,
    pub converged: bool,
    pub status: OptimStatus,
}

/// Result of the explicit reduced-order feasibility-restoration pass.
///
/// Restoration is deliberately separate from objective penalties: an
/// optimizer may use penalties to search, but callers must be able to see
/// whether a candidate was brought back into the hard geometry, airflow,
/// thermal, structural, and manufacturing constraints before promotion.
#[derive(Debug, Clone, PartialEq)]
pub struct PcFanFeasibilityRestoration {
    pub design_variables: PcFanDesignVariables,
    pub evaluation: PcFanEvaluation,
    pub iterations: usize,
    pub function_evaluations: usize,
    pub restored: bool,
}

#[cfg(test)]
mod backend_provenance_tests {
    use super::*;

    #[test]
    fn accelerator_fallback_is_explicitly_admitted_or_rejected() {
        let fallback =
            PcFanCfdBackendProvenance::from_dispatch(PcFanCfdBackend::Cuda, PcFanCfdBackend::Cpu);
        assert_eq!(
            fallback.fallback_reason,
            Some(PcFanCfdFallbackReason::RequestedAcceleratorUnavailable)
        );
        assert!(!fallback.is_admitted(false, false));
        assert!(fallback.is_admitted(true, false));
    }

    #[test]
    fn accelerator_linear_solve_is_not_device_resident() {
        let provenance =
            PcFanCfdBackendProvenance::from_dispatch(PcFanCfdBackend::Cuda, PcFanCfdBackend::Cuda);
        assert_eq!(
            provenance.execution_mode,
            PcFanCfdExecutionMode::AcceleratorLinearSolve
        );
        assert!(provenance.is_admitted(true, false));
        assert!(!provenance.is_admitted(true, true));
        assert!(
            !PcFanCfdReuseValidity::for_execution_mode(provenance.execution_mode)
                .all_components_reused()
        );
    }

    #[test]
    fn reuse_contract_is_explicit_for_each_execution_mode() {
        let host = PcFanCfdReuseValidity::for_execution_mode(PcFanCfdExecutionMode::Cpu);
        assert_eq!(host, PcFanCfdReuseValidity::host_staged());
        assert!(!host.all_components_reused());

        let accelerator = PcFanCfdReuseValidity::for_execution_mode(
            PcFanCfdExecutionMode::AcceleratorLinearSolve,
        );
        assert_eq!(accelerator, PcFanCfdReuseValidity::host_staged());

        let resident =
            PcFanCfdReuseValidity::for_execution_mode(PcFanCfdExecutionMode::DeviceResident);
        assert_eq!(resident, PcFanCfdReuseValidity::device_resident());
        assert!(resident.all_components_reused());
    }

    #[test]
    fn execution_environment_matches_backend_provenance() {
        let config = PcFanCfdConfig::default();
        let provenance =
            PcFanCfdBackendProvenance::from_dispatch(PcFanCfdBackend::Cpu, PcFanCfdBackend::Cpu);
        let environment = PcFanCfdExecutionEnvironment::capture(&config, provenance);
        assert!(environment.matches_provenance(provenance));
        assert!(!environment.target_arch.is_empty());
        assert!(!environment.simd_dispatch.is_empty());
        assert!(environment.thread_count > 0);
        assert_eq!(
            environment.reuse_validity,
            PcFanCfdReuseValidity::host_staged()
        );
    }
}
