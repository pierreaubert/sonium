//! Rotating-machine electromagnetics on a 2D cross-section.
//!
//! Planar `A_z` magnetoquasistatics with nonlinear iron, permanent
//! magnets on a recoil line, imposed winding currents, an optional
//! backward-Euler eddy term, flux-linkage extraction, and torque
//! evaluators. This is the field layer a BLDC/PMSM design workflow
//! (geometry, windings, sliding-band motion, losses, thermal coupling)
//! is built on; those layers live beside it, not inside the field solve.

// Rust guideline compliant 2026-02-21

pub mod az;
pub mod demag;
pub mod losses;
pub mod materials;
pub mod mechanical;
pub mod motor_mesh;
pub mod planar_mesh;
pub mod thermal;
pub mod torque;
pub mod transient;
pub mod winding;

pub use az::{
    AxialCurrentForm, AzEnergyReport, AzError, AzLinearBackend, AzProblem, AzQuadraturePoint,
    AzSolution, AzSolveConfig, ConductorBlock, CurrentDensityFn, MagnetCoercivityForm,
    ReluctivityForm, TransientStep, solve_az,
};
pub use demag::{DemagError, MagnetDemagCurve};
pub use losses::{
    BertottiCoefficients, CopperLoss, CopperLossInput, IronLossReport, LossError,
    MeasuredLossSurface, SteelLoss, copper_loss_dc, end_winding_factor, harmonic_amplitudes,
    iron_loss,
};
pub use materials::{AzMaterial, AzMaterialError, MagnetLaw};
pub use mechanical::{
    MechError, MechMaterial, RotorStressProblem, RotorStressResult, RotorStressSolver,
};
pub use motor_mesh::{
    MARKER_AIR, MARKER_COIL_BASE, MARKER_MAGNET_BASE, MARKER_ROTOR, MARKER_SHAFT, MARKER_STATOR,
    MotorMesh, MotorMeshOptions, MotorPolygons, build_motor_mesh,
};
pub use planar_mesh::{PlanarMeshError, PlanarMeshOptions, RegionPolygon, triangulate_regions};
pub use thermal::{
    RegionHeat, RobinSurface, SinkReport, SurfaceReport, ThermalError, ThermalProblem,
    ThermalResult, VolumeSink, element_areas_m2, element_centroids_m, heat_sources_from_regions,
    marker_mean_temperature_c, solve_steady_thermal,
};
pub use torque::{
    TorqueError, TorqueMetrics, annulus_point_count, arkkio_torque_nm, flux_linkage_mean_torque_nm,
    space_vector_mean_nm, terminal_work_mean_nm, torque_metrics,
};
pub use transient::{
    DemagConfig, DrivePoint, EddyResult, EddyTransientConfig, MotorMaterials, TransientConfig,
    TransientError, TransientFrame, TransientResult, VoltageDrivePoint, VoltageTransientConfig,
    VoltageWaveform, laminated_bh_curve, run_current_drive_transient, run_eddy_current_drive,
    run_voltage_drive_transient, step_mean_voltages_v,
};
pub use winding::{
    Phase, SlotWinding, Winding, WindingError, calibrate_daxis_deg, inverse_park, park,
    region_mean_a_wb_per_m, step_voltage_series,
};
