//! JLR case definition, CPU runner, and time-averaged observables.
//!
//! Contains the three published operating points (paper Section 3.1.4),
//! the physical-to-lattice conversion (ν = 1e-6 m²/s water at 20 °C,
//! ρ = 1000 kg/m³), the run loop with the paper's 1 s inlet ramp and
//! t = 5–20 s averaging window, and the extraction of the annular-gap
//! radial profile, the annular volume flow (Eq. 3.3), the circulation
//! number (Eq. 3.2), and the characteristic Reynolds numbers (Table 4.1).

// Rust guideline compliant 2026-02-21

use serde::{Deserialize, Serialize};

use crate::geometry::JlrGeometry;
use crate::lbm::{Cell, Lattice, LatticeConfig, tau_for_viscosity};

/// One published operating point of the experimental campaign.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OperatingPoint {
    /// Specific power input φ in kW/m³.
    pub specific_power_kw_m3: f64,
    /// External recycle (= nozzle) volume flow in m³/s.
    pub external_flow_m3_s: f64,
}

/// The three operating points of paper Section 3.1.4 / Table 4.1.
pub const OPERATING_POINTS: [OperatingPoint; 3] = [
    OperatingPoint { specific_power_kw_m3: 2.243, external_flow_m3_s: 1.175e-3 },
    OperatingPoint { specific_power_kw_m3: 4.038, external_flow_m3_s: 1.429e-3 },
    OperatingPoint { specific_power_kw_m3: 5.833, external_flow_m3_s: 1.616e-3 },
];

/// Feed-pipe inlet velocity profile shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InletProfile {
    /// Uniform plug flow, as stated in the paper for M-Star.
    Plug,
    /// Parabolic profile with the same volume flow; removes the
    /// one-cell shear layer the plug creates at the pipe wall, which
    /// is numerically unstable at tau close to 0.5 (measured).
    Parabolic,
}

/// Full case configuration; serde so cases are checked-in JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseConfig {
    /// Lattice spacing in mm (paper level I: 1.5).
    pub dx_mm: f64,
    /// Physical time step in s (paper level I: 4e-6).
    pub dt_s: f64,
    /// Kinematic viscosity in m²/s (water: 1e-6).
    pub nu_m2_s: f64,
    /// Simulated physical time in s (paper: 20).
    pub total_time_s: f64,
    /// Start of the time-averaging window in s (paper: 5).
    pub averaging_start_s: f64,
    /// Inlet ramp time in s (paper: 1).
    pub ramp_time_s: f64,
    /// Smagorinsky constant, `None` for laminar (paper: 0.1).
    pub smagorinsky: Option<f64>,
    /// Inlet velocity profile in the feed pipe.
    pub inlet_profile: InletProfile,
    /// Central-moment (cascaded) collision; the production collision
    /// model. Takes precedence over `regularized`.
    pub central_moment: bool,
    /// Regularised collision; required for stability at the paper's
    /// tau = 0.500005 (SRT-BGK ghost modes otherwise grow from
    /// round-off, verified on CPU and CUDA probes). Documented
    /// deviation: M-Star's collision model is not published.
    pub regularized: bool,
    /// Azimuth in degrees at which the radial profile is extracted
    /// (paper measurement azimuth: 45, between support struts).
    pub profile_azimuth_deg: f64,
    /// Geometry rotation in degrees (paper's rotated-grid study: 45).
    pub rotation_deg: f64,
    /// Log progress every this many steps (0 = silent).
    pub log_every_steps: u64,
}

impl Default for CaseConfig {
    fn default() -> Self {
        Self {
            dx_mm: 1.5,
            dt_s: 4.0e-6,
            nu_m2_s: 1.0e-6,
            total_time_s: 20.0,
            averaging_start_s: 5.0,
            ramp_time_s: 1.0,
            smagorinsky: Some(0.1),
            regularized: true,
            central_moment: true,
            inlet_profile: InletProfile::Plug,
            profile_azimuth_deg: 45.0,
            rotation_deg: 0.0,
            log_every_steps: 0,
        }
    }
}

/// Time-averaged observables of one finished run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseResult {
    /// Specific power input in kW/m³.
    pub specific_power_kw_m3: f64,
    /// External recycle flow in m³/s.
    pub external_flow_m3_s: f64,
    /// Radial profile at the measuring height: (r in mm, mean axial
    /// velocity in m/s), innermost first.
    pub annular_profile: Vec<(f64, f64)>,
    /// Annular-gap volume flow in m³/s (Eq. 3.3, plane integral).
    pub annular_flow_m3_s: f64,
    /// Circulation number Ncirc (Eq. 3.2).
    pub circulation_number: f64,
    /// Nozzle Reynolds number (Table 4.1 definition).
    pub re_nozzle: f64,
    /// Annular-gap Reynolds number, hydraulic diameter dR − dDT,outer.
    pub re_annular_gap: f64,
    /// Draft-tube Reynolds number.
    pub re_draft_tube: f64,
    /// Lattice steps executed.
    pub steps: u64,
    /// Simulated physical time in s.
    pub simulated_time_s: f64,
    /// Wall-clock seconds of the run loop.
    pub wall_time_s: f64,
    /// Million lattice updates per second over fluid+boundary cells.
    pub mlups: f64,
    /// Grid dimensions.
    pub grid: [usize; 3],
    /// Total lattice cells.
    pub lattice_cells: usize,
}

/// Runs one operating point on the CPU solver.
///
/// Returns the averaged observables; the caller decides on persistence.
///
/// # Panics
/// Panics if the resulting lattice relaxation time is not above ½, which
/// is a case-configuration error (Δt too small for the chosen Δx and ν).
pub fn run_case_cpu(cfg: &CaseConfig, op: OperatingPoint, geom: &JlrGeometry) -> CaseResult {
    let dx_m = cfg.dx_mm * 1.0e-3;
    let nu_lb = cfg.nu_m2_s * cfg.dt_s / (dx_m * dx_m);
    let tau = tau_for_viscosity(nu_lb);
    let mut geom = geom.clone();
    geom.rotation_deg = cfg.rotation_deg;
    let (cells, nx, ny, nz) = geom.voxelise(cfg.dx_mm);
    let n_cells = nx * ny * nz;
    // Feed-pipe inlet velocity (plug flow in the Ø36 mm feed, downward).
    let a_feed = std::f64::consts::PI * (geom.nozzle_inlet_diameter_mm * 1.0e-3 / 2.0).powi(2);
    let v_feed_lb = (op.external_flow_m3_s / a_feed) * cfg.dt_s / dx_m;
    let mut lcfg = LatticeConfig::new(nx, ny, nz, tau);
    lcfg.smagorinsky = cfg.smagorinsky;
    lcfg.regularized = cfg.regularized;
    lcfg.central_moment = cfg.central_moment;
    lcfg.inlet_velocity = [0.0, 0.0, -v_feed_lb];
    {
        let a_patch = std::f64::consts::PI * (geom.baffle_diameter_mm * 1.0e-3 / 2.0).powi(2);
        let v_out_lb = (op.external_flow_m3_s / a_patch) * cfg.dt_s / dx_m;
        lcfg.outlet_velocity = [0.0, 0.0, -v_out_lb];
    }
    lcfg.outlet_normal = [0, 0, 1];
    let mut lat = Lattice::new(lcfg, cells);
    if cfg.inlet_profile == InletProfile::Parabolic {
        let r_feed_mm = geom.nozzle_inlet_diameter_mm / 2.0;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let idx = lat.index(x, y, z);
                    if lat.cell(idx) != Cell::Inlet {
                        continue;
                    }
                    let xm = (x as f64 + 0.5) * cfg.dx_mm - nx as f64 * cfg.dx_mm / 2.0;
                    let ym = (y as f64 + 0.5) * cfg.dx_mm - ny as f64 * cfg.dx_mm / 2.0;
                    let r = (xm * xm + ym * ym).sqrt();
                    let shape = 2.0 * (1.0 - (r / r_feed_mm).powi(2)).max(0.0);
                    lat.set_inlet_velocity(idx, [0.0, 0.0, -v_feed_lb * shape]);
                }
            }
        }
    }
    let total_steps = (cfg.total_time_s / cfg.dt_s).round() as u64;
    let avg_start_step = (cfg.averaging_start_s / cfg.dt_s).round() as u64;
    let ramp_steps = (cfg.ramp_time_s / cfg.dt_s).max(1.0);
    let mut avg_uz = vec![0.0f64; n_cells];
    let mut avg_count: u64 = 0;
    let vel_scale = dx_m / cfg.dt_s; // lattice velocity -> m/s
    let t0 = std::time::Instant::now();
    for step in 0..total_steps {
        let ramp = ((step as f64 + 1.0) / ramp_steps).min(1.0);
        lat.step(ramp);
        if step >= avg_start_step {
            avg_count += 1;
            lat.accumulate_uz(&mut avg_uz, vel_scale);
        }
        if cfg.log_every_steps > 0 && step % cfg.log_every_steps == 0 {
            eprintln!("jlr step {step}/{total_steps} t={:.3}s", step as f64 * cfg.dt_s);
        }
    }
    let wall = t0.elapsed().as_secs_f64();
    if avg_count > 0 {
        for v in &mut avg_uz {
            *v /= avg_count as f64;
        }
    }
    extract_result(cfg, op, &geom, &avg_uz, nx, ny, nz, total_steps, wall)
}

pub(crate) fn extract_result(
    cfg: &CaseConfig,
    op: OperatingPoint,
    geom: &JlrGeometry,
    avg_uz: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
    steps: u64,
    wall: f64,
) -> CaseResult {
    let dx_mm = cfg.dx_mm;
    let z_layer = ((geom.measuring_height_mm / dx_mm).round() as usize).min(nz - 1);
    let az = cfg.profile_azimuth_deg.to_radians();
    let mut profile = Vec::new();
    let mut r = 55.0;
    while r <= 82.0 {
        let xm = r * az.cos();
        let ym = r * az.sin();
        let xi = ((xm + nx as f64 * dx_mm / 2.0) / dx_mm).floor() as usize;
        let yi = ((ym + ny as f64 * dx_mm / 2.0) / dx_mm).floor() as usize;
        if xi < nx && yi < ny {
            profile.push((r, avg_uz[xi + nx * (yi + ny * z_layer)]));
        }
        r += 1.25;
    }
    // Annular volume flow: plane integral of mean axial velocity.
    let dt_out = (geom.draft_tube_diameter_mm + 2.0 * geom.draft_tube_wall_mm) / 2.0;
    let r_reactor = geom.reactor_diameter_mm / 2.0;
    let mut v_ag = 0.0;
    for y in 0..ny {
        for x in 0..nx {
            let xm = (x as f64 + 0.5) * dx_mm - nx as f64 * dx_mm / 2.0;
            let ym = (y as f64 + 0.5) * dx_mm - ny as f64 * dx_mm / 2.0;
            let rr = (xm * xm + ym * ym).sqrt();
            if rr > dt_out && rr <= r_reactor {
                v_ag += avg_uz[x + nx * (y + ny * z_layer)] * (dx_mm * 1.0e-3).powi(2);
            }
        }
    }
    let nu = cfg.nu_m2_s;
    let a_nozzle = std::f64::consts::PI * (geom.nozzle_outlet_diameter_mm * 1.0e-3 / 2.0).powi(2);
    let v_jet = op.external_flow_m3_s / a_nozzle;
    let a_ann = geom.annulus_area_mm2() * 1.0e-6;
    let v_ann = v_ag / a_ann;
    let d_hyd = (geom.reactor_diameter_mm
        - (geom.draft_tube_diameter_mm + 2.0 * geom.draft_tube_wall_mm))
        * 1.0e-3;
    let a_dt = std::f64::consts::PI * (geom.draft_tube_diameter_mm * 1.0e-3 / 2.0).powi(2);
    let v_dt = (v_ag + op.external_flow_m3_s) / a_dt;
    CaseResult {
        specific_power_kw_m3: op.specific_power_kw_m3,
        external_flow_m3_s: op.external_flow_m3_s,
        annular_profile: profile,
        annular_flow_m3_s: v_ag,
        circulation_number: v_ag / op.external_flow_m3_s,
        re_nozzle: v_jet * geom.nozzle_outlet_diameter_mm * 1.0e-3 / nu,
        re_annular_gap: v_ann * d_hyd / nu,
        re_draft_tube: v_dt * geom.draft_tube_diameter_mm * 1.0e-3 / nu,
        steps,
        simulated_time_s: steps as f64 * cfg.dt_s,
        wall_time_s: wall,
        mlups: (nx * ny * nz) as f64 * steps as f64 / wall / 1.0e6,
        grid: [nx, ny, nz],
        lattice_cells: nx * ny * nz,
    }
}

/// Cell-kind histogram of a voxelised geometry, for area artefacts.
///
/// # Examples
/// ```rust
/// use sonium_sim_jlr::geometry::JlrGeometry;
/// use sonium_sim_jlr::case::fluid_fraction;
/// let g = JlrGeometry::default();
/// let (cells, nx, ny, _nz) = g.voxelise(3.0);
/// assert!(fluid_fraction(&cells) > 0.5);
/// ```
#[must_use]
pub fn fluid_fraction(cells: &[Cell]) -> f64 {
    let fluid = cells.iter().filter(|c| **c != Cell::Solid).count();
    fluid as f64 / cells.len() as f64
}
