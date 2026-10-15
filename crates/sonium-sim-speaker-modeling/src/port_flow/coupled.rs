//! Coupling loop between the resolved unsteady Navier–Stokes port solve and
//! the lumped box + driver ODE — the heart of M1 (resolved nonlinear port
//! flow).
//!
//! Per time step, the electrical/mechanical driver advances once and the
//! scalar box-pressure/port-flow interface is closed at the new time level:
//!
//! 1. advance the driver and diaphragm contribution to `p_box`,
//! 2. solve the affine BDF response at two box-cap pressures,
//! 3. close `p_box^{n+1} = p_base − ρc²Δt/V · q_port^{n+1}` exactly,
//! 4. interpolate the affine fluid state to that pressure, with tractions
//!    `[MARKER_BOX_CAP → p_applied, MARKER_EXTERIOR_CAP → exterior_pressure_pa]`
//!    and no-slip Dirichlet on every [`MARKER_WALL`] edge.
//!
//! Sign conventions (the traps this module exists to pin down):
//!
//! - `PressureTraction` adds `∫(−p·n)·φ dΓ` to the momentum RHS with `n` the
//!   OUTWARD normal, so a positive `p_applied` at the box cap pushes fluid in
//!   `+x` — out of the box, through the duct. Physical.
//! - `q_port` is positive box → exterior. The outward-normal flux at the box
//!   cap is NEGATIVE when fluid leaves the box, so measuring there would flip
//!   the feedback sign (p_box > 0 → q < 0 → dp/dt > 0 → runaway). The flux
//!   is therefore integrated at [`MARKER_EXTERIOR_CAP`], whose outward normal
//!   is `+x`; discrete incompressibility makes it identical to the negated
//!   box-cap flux up to the linear-solver residual.
//! - `BoxDriverState::p_box_pa` is gauge pressure, positive when the box is
//!   compressed; the exterior cap traction anchors the gauge at
//!   `exterior_pressure_pa`.
//!
//! Solver quirks honoured here (carried forward from Tasks 2–3): all-Dirichlet
//! inlet+walls sets are LU-singular, so the caps must be traction-driven and
//! only the walls carry Dirichlet DOFs. Velocity DOFs are interleaved
//! `scalar_dof * dim + comp`, where scalar DOFs are the mesh vertices followed
//! by the P2 edge midpoints. The saddle system runs on Taylor–Hood P2–P1
//! spaces (inf-sup stable), so no residual stabilisation is used:
//! `NavierStokesConfig::use_residual_stabilisation` must stay off. Equal-order
//! P1–P1 on this geometry is rank-deficient without stabilisation (isolated
//! corner pressure nodes and duct-band spurious pressure modes — see the
//! Task 9 report) and is not supported here.
//!
//! Two-dimensional slot fixtures retain the per-unit-depth convention: their
//! flux is m²/s, interpreted as m³/s per metre of depth.  Three-dimensional
//! round-port meshes use the same coupling equations with physical m³/s flux
//! and physical box volume.  The lumped reference in the 2-D tests uses the
//! former convention; the 3-D regression exercises the latter path directly.

use sonium_fem::mesh::Mesh;
use sonium_fem::physics::fluid::stokes::navier_stokes::{
    BdfStepAudit, NavierStokesConfig, audit_bdf_step, solve_bdf_step_cached,
};
use sonium_fem::physics::fluid::stokes::time_stepping::{BdfConfig, BdfState};
use sonium_fem::physics::fluid::stokes::traction::PressureTraction;
use sonium_fem::physics::fluid::stokes::workspace::{BdfLinearSolveTelemetry, BdfWorkspace};
use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};

use super::box_driver::{BoxDriverEnergyStep, BoxDriverModel, BoxDriverState};
use super::error::PortFlowError;
use super::geometry::{MARKER_BOX_CAP, MARKER_EXTERIOR_CAP, MARKER_WALL};
use super::measure::integrate_face_flux_for_space;

/// Everything needed to run the coupled port/box/driver lockstep.
pub struct CoupledPortConfig {
    /// Lumped box-compliance + (nonlinear) driver model (Task 8).
    pub driver: BoxDriverModel,
    /// Navier–Stokes solver configuration (viscosity, density, linear solver).
    pub ns_config: NavierStokesConfig,
    /// BDF time-stepping configuration (`dt`, order, `num_steps`).
    pub bdf: BdfConfig,
    /// Terminal voltage (V) as a function of time (s).
    pub voltage_fn: Box<dyn Fn(f64) -> f64 + Send + Sync>,
    /// Legacy pressure under-relaxation setting. The affine interface closure
    /// is direct rather than iterative, so spec-faithful runs require `1.0`.
    pub under_relaxation: f64,
    /// Exterior gauge reference pressure (Pa), usually 0.0.
    pub exterior_pressure_pa: f64,
}

/// Time series recorded by [`run_coupled_port`], one entry per BDF step.
pub struct CoupledPortResult {
    /// Time at the end of each step (s).
    pub time_s: Vec<f64>,
    /// Port volume velocity, box → exterior, positive out (m³/s per unit depth).
    pub port_flow_m3s: Vec<f64>,
    /// Exterior-cap volume velocity, box → exterior, positive out (m³/s per
    /// unit depth). This is identical to `port_flow_m3s`, which remains as the
    /// compatibility field.
    pub exterior_cap_flow_m3s: Vec<f64>,
    /// Outward-normal volume flux at the box cap (m³/s per unit depth).
    ///
    /// For an incompressible port this is the negative of `port_flow_m3s`;
    /// their sum is the discrete cap-continuity diagnostic.
    pub box_cap_outward_flow_m3s: Vec<f64>,
    /// Diaphragm volume velocity (`Sd * v`), positive compressing the box (m³/s).
    pub diaphragm_volume_velocity_m3s: Vec<f64>,
    /// Box gauge pressure (Pa).
    pub box_pressure_pa: Vec<f64>,
    /// Diaphragm displacement (m), positive compressing the box.
    pub displacement_m: Vec<f64>,
    /// Voice-coil current (A).
    pub current_a: Vec<f64>,
    /// Per-step affine box conservation residual (Pa). For a vented box this
    /// includes the diaphragm pressure-feedback term in addition to
    /// `p_box - p_base + ρc²Δt/V·q_port`, and should be at roundoff scale.
    pub box_conservation_residual_pa: Vec<f64>,
    /// Quantitative signed energy ledger, aligned one-to-one with `time_s`.
    pub energy_steps: Vec<CoupledEnergyStep>,
    /// One entry per affine fluid solve, including setup-reuse and true
    /// residual diagnostics. A vented step normally contributes two entries.
    pub linear_solves: Vec<BdfLinearSolveTelemetry>,
    /// Independently recomputed true residual of each accepted affine fluid
    /// state against the correspondingly interpolated right-hand side.
    pub accepted_linear_relative_residual: Vec<f64>,
    /// Accepted unsteady fields retained at `BdfConfig::output_interval` for
    /// one-way M4 aeroacoustic post-processing. Empty when output is disabled.
    pub field_snapshots: Vec<PortFlowFieldSnapshot>,
    /// Final accepted BDF fluid state, retained for direct/iterative parity
    /// and restart diagnostics.
    pub final_flow_state: Option<BdfState>,
    /// Number of BDF steps completed (== `config.bdf.num_steps` on success).
    pub steps_completed: usize,
}

/// One accepted M1 field snapshot. DOF layouts are those of the velocity and
/// pressure spaces supplied to `run_coupled_port`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PortFlowFieldSnapshot {
    pub time_s: f64,
    pub velocity_dofs_m_per_s: Vec<f64>,
    pub pressure_dofs_pa: Vec<f64>,
}

/// Signed per-step coupled energy ledger. Positive sources enter the complete
/// system; positive dissipations/exports leave physical stored energy.
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct CoupledEnergyStep {
    /// Electrical, mechanical, suspension, and box channels (J).
    pub driver: BoxDriverEnergyStep,
    /// Accepted-state FEM fluid channels (J).
    pub fluid: BdfStepAudit,
    /// Work delivered by the box to the fluid at the box cap (J).
    pub box_to_fluid_work_j: f64,
    /// Endpoint-pressure box compliance residual (J).
    pub box_endpoint_numerical_j: f64,
    /// Trapezoidal box-compliance identity residual (J). This should close at
    /// roundoff because the affine pressure update uses the same cap flow.
    pub box_trapezoidal_residual_j: f64,
    /// Box-cap versus exterior-cap continuity work mismatch (J).
    pub cap_continuity_work_j: f64,
}

impl CoupledEnergyStep {
    /// Return true only when every recorded scalar channel is finite.
    pub fn is_finite(self) -> bool {
        let d = self.driver;
        let f = self.fluid;
        [
            d.source_j,
            d.copper_loss_j,
            d.inductive_storage_old_j,
            d.inductive_storage_new_j,
            d.electrical_numerical_j,
            d.electromagnetic_staggering_j,
            d.mechanical_kinetic_old_j,
            d.mechanical_kinetic_new_j,
            d.suspension_storage_old_j,
            d.suspension_storage_new_j,
            d.suspension_numerical_j,
            d.mechanical_damping_j,
            d.mechanical_damping_midpoint_j,
            d.mechanical_residual_j,
            d.box_storage_old_j,
            d.box_storage_new_j,
            d.driver_to_box_work_j,
            d.box_on_driver_work_j,
            f.fluid_kinetic_old_j,
            f.fluid_kinetic_new_j,
            f.bdf_mass_work_j,
            f.bdf_numerical_j,
            f.viscous_dissipation_j,
            f.molecular_viscous_dissipation_j,
            f.eddy_viscous_dissipation_j,
            f.convection_export_j,
            f.pressure_work_j,
            f.traction_work_j,
            f.box_traction_work_j,
            f.exterior_traction_work_j,
            f.solve_residual_j,
            f.effective_viscosity_pa_s,
            self.box_to_fluid_work_j,
            self.box_endpoint_numerical_j,
            self.box_trapezoidal_residual_j,
            self.cap_continuity_work_j,
        ]
        .iter()
        .all(|value| value.is_finite())
    }

    /// Check the physical loss channels against a caller-supplied roundoff
    /// tolerance. Signed transport and numerical residuals are intentionally
    /// excluded.
    pub fn physical_dissipations_nonnegative(self, tolerance_j: f64) -> bool {
        self.driver.copper_loss_j >= -tolerance_j
            && self.driver.mechanical_damping_midpoint_j >= -tolerance_j
            && self.fluid.molecular_viscous_dissipation_j >= -tolerance_j
            && self.fluid.eddy_viscous_dissipation_j >= -tolerance_j
            && self.fluid.viscous_dissipation_j >= -tolerance_j
    }

    /// Total physical stored energy at the old endpoint (J).
    pub fn stored_old_j(self) -> f64 {
        self.driver.inductive_storage_old_j
            + self.driver.mechanical_kinetic_old_j
            + self.driver.suspension_storage_old_j
            + self.driver.box_storage_old_j
            + self.fluid.fluid_kinetic_old_j
    }

    /// Total physical stored energy at the new endpoint (J).
    pub fn stored_new_j(self) -> f64 {
        self.driver.inductive_storage_new_j
            + self.driver.mechanical_kinetic_new_j
            + self.driver.suspension_storage_new_j
            + self.driver.box_storage_new_j
            + self.fluid.fluid_kinetic_new_j
    }

    /// Total physical storage change over this step (J).
    pub fn stored_delta_j(self) -> f64 {
        self.stored_new_j() - self.stored_old_j()
    }

    /// Sum of copper, midpoint mechanical, and modeled viscous losses (J).
    pub fn physical_dissipation_j(self) -> f64 {
        self.driver.copper_loss_j
            + self.driver.mechanical_damping_midpoint_j
            + self.fluid.viscous_dissipation_j
    }

    /// Physical-only energy residual for this step (J).
    pub fn physical_residual_j(self) -> f64 {
        self.driver.source_j
            - self.stored_delta_j()
            - self.physical_dissipation_j()
            - self.fluid.convection_export_j
            + self.fluid.exterior_traction_work_j
    }

    /// Sum of explicitly classified numerical channels (J).
    pub fn numerical_residual_j(self) -> f64 {
        self.driver.electrical_numerical_j
            + self.driver.electromagnetic_staggering_j
            + self.driver.suspension_numerical_j
            + self.driver.mechanical_residual_j
            + self.driver.mechanical_damping_j
            - self.driver.mechanical_damping_midpoint_j
            + self.box_endpoint_numerical_j
            + self.fluid.bdf_numerical_j
            + self.fluid.pressure_work_j
            + self.fluid.solve_residual_j
    }

    /// Residual after physical and numerical channels are included (J).
    pub fn algebraic_residual_j(self) -> f64 {
        self.physical_residual_j() - self.numerical_residual_j()
    }
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct CoupledEnergySummary {
    /// Signed electrical source work over the window (J).
    pub source_j: f64,
    /// Final minus initial total stored energy (J).
    pub stored_delta_j: f64,
    /// Nonnegative copper loss (J).
    pub copper_loss_j: f64,
    /// Nonnegative midpoint mechanical damping loss (J).
    pub mechanical_damping_j: f64,
    /// Nonnegative modeled viscous loss (J).
    pub viscous_dissipation_j: f64,
    /// Sum of copper, mechanical, and viscous losses (J).
    pub physical_dissipation_j: f64,
    /// Signed advective export minus exterior pressure work (J).
    pub exterior_export_j: f64,
    /// Sum of explicitly recorded numerical residual channels (J).
    pub numerical_residual_j: f64,
    /// Normalization energy used for relative closure metrics (J).
    pub energy_scale_j: f64,
    /// Absolute numerical residual divided by `energy_scale_j`.
    pub numerical_relative_residual: f64,
    /// Residual after all physical and numerical channels, normalized.
    pub algebraic_relative_closure: f64,
    /// Physical-only energy residual, normalized.
    pub physical_relative_closure: f64,
    /// Cumulative physical power-transfer mismatch at the driver/box and
    /// box/fluid interfaces, normalized. The per-step cap-continuity
    /// discretisation residual remains available as
    /// [`CoupledEnergyStep::cap_continuity_work_j`] and is not folded into
    /// this physical interface metric.
    pub interface_relative_residual: f64,
}

impl CoupledEnergySummary {
    /// Return true only when every summary field is finite and the
    /// normalization scale is strictly positive.
    pub fn is_finite(self) -> bool {
        [
            self.source_j,
            self.stored_delta_j,
            self.copper_loss_j,
            self.mechanical_damping_j,
            self.viscous_dissipation_j,
            self.physical_dissipation_j,
            self.exterior_export_j,
            self.numerical_residual_j,
            self.energy_scale_j,
            self.numerical_relative_residual,
            self.algebraic_relative_closure,
            self.physical_relative_closure,
            self.interface_relative_residual,
        ]
        .iter()
        .all(|value| value.is_finite())
            && self.energy_scale_j > 0.0
    }
}

/// Current serialized schema version for [`CoupledEnergyBlock`].
pub const COUPLED_ENERGY_SCHEMA_VERSION: u32 = 1;

/// Versioned serialization envelope for coupled energy-ledger consumers.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CoupledEnergyBlock {
    /// Schema version; currently [`COUPLED_ENERGY_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Signed per-step energy records in chronological order.
    pub steps: Vec<CoupledEnergyStep>,
}

impl CoupledPortResult {
    /// Copy the ledger into its stable, versioned serialization envelope.
    pub fn energy_block(&self) -> CoupledEnergyBlock {
        CoupledEnergyBlock {
            schema_version: COUPLED_ENERGY_SCHEMA_VERSION,
            steps: self.energy_steps.clone(),
        }
    }
    fn with_capacity(steps: usize) -> Self {
        Self {
            time_s: Vec::with_capacity(steps),
            port_flow_m3s: Vec::with_capacity(steps),
            exterior_cap_flow_m3s: Vec::with_capacity(steps),
            box_cap_outward_flow_m3s: Vec::with_capacity(steps),
            diaphragm_volume_velocity_m3s: Vec::with_capacity(steps),
            box_pressure_pa: Vec::with_capacity(steps),
            displacement_m: Vec::with_capacity(steps),
            current_a: Vec::with_capacity(steps),
            box_conservation_residual_pa: Vec::with_capacity(steps),
            energy_steps: Vec::with_capacity(steps),
            linear_solves: Vec::with_capacity(2 * steps),
            accepted_linear_relative_residual: Vec::with_capacity(steps),
            field_snapshots: Vec::new(),
            final_flow_state: None,
            steps_completed: 0,
        }
    }

    fn push(
        &mut self,
        time_s: f64,
        q_port: f64,
        q_box_outward: f64,
        driver_model: &BoxDriverModel,
        driver_state: &BoxDriverState,
        conservation_residual_pa: f64,
        energy: CoupledEnergyStep,
    ) {
        self.time_s.push(time_s);
        self.port_flow_m3s.push(q_port);
        self.exterior_cap_flow_m3s.push(q_port);
        self.box_cap_outward_flow_m3s.push(q_box_outward);
        self.diaphragm_volume_velocity_m3s
            .push(driver_model.sd_m2 * driver_state.v_m_per_s);
        self.box_pressure_pa.push(driver_state.p_box_pa);
        self.displacement_m.push(driver_state.x_m);
        self.current_a.push(driver_state.circuit.current);
        self.box_conservation_residual_pa
            .push(conservation_residual_pa);
        self.energy_steps.push(energy);
    }

    /// Summarize an exact half-open step window. Storage uses the first old and
    /// last new state rather than summing rounded per-step deltas.
    pub fn energy_summary(&self, range: std::ops::Range<usize>) -> CoupledEnergySummary {
        let steps = &self.energy_steps[range];
        if steps.is_empty() {
            return CoupledEnergySummary::default();
        }
        let first = steps[0];
        let last = steps[steps.len() - 1];
        let stored_old = first.driver.inductive_storage_old_j
            + first.driver.mechanical_kinetic_old_j
            + first.driver.suspension_storage_old_j
            + first.driver.box_storage_old_j
            + first.fluid.fluid_kinetic_old_j;
        let stored_new = last.driver.inductive_storage_new_j
            + last.driver.mechanical_kinetic_new_j
            + last.driver.suspension_storage_new_j
            + last.driver.box_storage_new_j
            + last.fluid.fluid_kinetic_new_j;
        let source = steps.iter().map(|step| step.driver.source_j).sum::<f64>();
        let absolute_source = steps
            .iter()
            .map(|step| step.driver.source_j.abs())
            .sum::<f64>();
        let copper = steps
            .iter()
            .map(|step| step.driver.copper_loss_j)
            .sum::<f64>();
        let mechanical = steps
            .iter()
            .map(|step| step.driver.mechanical_damping_midpoint_j)
            .sum::<f64>();
        let viscous = steps
            .iter()
            .map(|step| step.fluid.viscous_dissipation_j)
            .sum::<f64>();
        let dissipation = copper + mechanical + viscous;
        let exterior = steps
            .iter()
            .map(|step| step.fluid.convection_export_j - step.fluid.exterior_traction_work_j)
            .sum::<f64>();
        let physical = source - (stored_new - stored_old) - dissipation - exterior;
        let numerical = steps
            .iter()
            .map(|step| step.numerical_residual_j())
            .sum::<f64>();
        let algebraic = physical - numerical;
        // Cap continuity is a fluid discretisation/linear-solve residual and
        // is reported separately per step.  The interface-work audit here
        // covers only the two physical power-transfer interfaces; including
        // the cap residual would mix a PDE solve diagnostic into this ledger.
        let interface = steps
            .iter()
            .map(|step| {
                step.driver.driver_to_box_work_j
                    + step.driver.box_on_driver_work_j
                    + step.box_to_fluid_work_j
                    - step.fluid.box_traction_work_j
            })
            .sum::<f64>()
            .abs();
        let stored_energy = |step: &CoupledEnergyStep, new: bool| {
            if new {
                step.driver.inductive_storage_new_j
                    + step.driver.mechanical_kinetic_new_j
                    + step.driver.suspension_storage_new_j
                    + step.driver.box_storage_new_j
                    + step.fluid.fluid_kinetic_new_j
            } else {
                step.driver.inductive_storage_old_j
                    + step.driver.mechanical_kinetic_old_j
                    + step.driver.suspension_storage_old_j
                    + step.driver.box_storage_old_j
                    + step.fluid.fluid_kinetic_old_j
            }
        };
        let mut stored_min = stored_energy(&steps[0], false);
        let mut stored_max = stored_min;
        for step in steps {
            let stored = stored_energy(step, true);
            stored_min = stored_min.min(stored);
            stored_max = stored_max.max(stored);
        }
        let scale = absolute_source
            .max(dissipation + exterior.abs())
            .max(stored_max - stored_min)
            .max(1.0e-18);
        CoupledEnergySummary {
            source_j: source,
            stored_delta_j: stored_new - stored_old,
            copper_loss_j: copper,
            mechanical_damping_j: mechanical,
            viscous_dissipation_j: viscous,
            physical_dissipation_j: dissipation,
            exterior_export_j: exterior,
            numerical_residual_j: numerical,
            energy_scale_j: scale,
            numerical_relative_residual: numerical.abs() / scale,
            algebraic_relative_closure: algebraic.abs() / scale,
            physical_relative_closure: physical.abs() / scale,
            interface_relative_residual: interface / scale,
        }
    }
}

/// Run the coupled N-S port + lumped box/driver lockstep.
///
/// Returns [`PortFlowError::SolverDiverged`] when a BDF step's linear solve
/// fails or any coupled state goes non-finite. The absurd-`dt` misuse path is
/// covered because the affine pressure response can become singular when
/// `dt·ω_b ≫ 1`. Never panics on solver failure.
pub fn run_coupled_port(
    mesh: &Mesh,
    v_space: &VectorH1Space,
    p_space: &H1Space,
    config: &CoupledPortConfig,
) -> Result<CoupledPortResult, PortFlowError> {
    run_coupled_port_initialized(mesh, v_space, p_space, config, None, None)
}

fn run_coupled_port_initialized(
    mesh: &Mesh,
    v_space: &VectorH1Space,
    p_space: &H1Space,
    config: &CoupledPortConfig,
    initial_flow: Option<BdfState>,
    initial_driver: Option<BoxDriverState>,
) -> Result<CoupledPortResult, PortFlowError> {
    if config.bdf.dt <= 0.0 || config.under_relaxation != 1.0 {
        return Err(PortFlowError::InvalidConfig(format!(
            "dt {} / under_relaxation {} (direct affine closure requires 1.0)",
            config.bdf.dt, config.under_relaxation
        )));
    }
    let dim = mesh.dimension;
    if v_space.polynomial_degree() != 2 || p_space.polynomial_degree() != 1 {
        return Err(PortFlowError::InvalidConfig(format!(
            "run_coupled_port requires Taylor-Hood P2-P1 spaces, got P{}-P{}",
            v_space.polynomial_degree(),
            p_space.polynomial_degree()
        )));
    }
    if v_space.num_components() != dim {
        return Err(PortFlowError::InvalidConfig(format!(
            "velocity components {} must match mesh dimension {dim}",
            v_space.num_components()
        )));
    }
    if config.ns_config.use_residual_stabilisation {
        return Err(PortFlowError::InvalidConfig(
            "residual stabilization must be disabled for Taylor-Hood P2-P1".to_owned(),
        ));
    }
    if config.driver.beta != 0.25 || config.driver.gamma != 0.5 {
        return Err(PortFlowError::InvalidConfig(format!(
            "audited coupled energy requires average-acceleration Newmark beta=0.25, gamma=0.5; got beta={}, gamma={}",
            config.driver.beta, config.driver.gamma
        )));
    }
    if config.driver.compliance_force_fn.is_some()
        && config.driver.compliance_potential_fn.is_none()
    {
        return Err(PortFlowError::InvalidConfig(
            "audited nonlinear suspension requires both force and potential-energy functions"
                .to_owned(),
        ));
    }
    let n_v = v_space.num_global_dofs(mesh);
    let n_p = p_space.num_global_dofs(mesh);
    // Walls no-slip; caps driven by traction (no Dirichlet there — the
    // all-Dirichlet saddle system is LU-singular). `boundary_dofs` includes
    // the P2 edge-midpoint DOFs on wall edges.
    let bc_dofs = v_space.boundary_dofs(mesh, MARKER_WALL);
    let bc_fn = |_x: f64, _y: f64, _z: f64| vec![0.0; dim];
    let mut workspace = BdfWorkspace::new(mesh, v_space, p_space);
    let mesh_centroid = mesh.nodes.iter().fold([0.0_f64; 3], |mut sum, point| {
        sum[0] += point.x;
        sum[1] += point.y;
        sum[2] += point.z;
        sum
    });
    let mesh_centroid = mesh_centroid.map(|sum| sum / mesh.num_nodes().max(1) as f64);
    let pressure_pin = p_space
        .boundary_dofs(mesh, MARKER_EXTERIOR_CAP)
        .into_iter()
        .min_by(|&a, &b| {
            let point_a = p_space.dof_coordinate(mesh, a).unwrap_or([0.0; 3]);
            let point_b = p_space.dof_coordinate(mesh, b).unwrap_or([0.0; 3]);
            let distance_a = (0..dim)
                .map(|axis| (point_a[axis] - mesh_centroid[axis]).powi(2))
                .sum::<f64>();
            let distance_b = (0..dim)
                .map(|axis| (point_b[axis] - mesh_centroid[axis]).powi(2))
                .sum::<f64>();
            distance_a.total_cmp(&distance_b).then(a.cmp(&b))
        })
        .ok_or_else(|| {
            PortFlowError::InvalidConfig(
                "exterior cap exposes no pressure DOF for the prescribed-pressure gauge".into(),
            )
        })?;
    workspace.set_pressure_gauge(pressure_pin, config.exterior_pressure_pa);
    let mut flow = initial_flow.unwrap_or_else(|| BdfState::from_rest(n_v, n_p));
    if flow.velocity.len() != n_v || flow.velocity_prev.len() != n_v || flow.pressure.len() != n_p {
        return Err(PortFlowError::InvalidConfig(
            "initial BDF state dimensions do not match the spaces".to_owned(),
        ));
    }
    let mut driver = initial_driver.unwrap_or_else(|| config.driver.new_state());
    let mut out = CoupledPortResult::with_capacity(config.bdf.num_steps);
    for step in 0..config.bdf.num_steps {
        let t_next = (step + 1) as f64 * config.bdf.dt;
        let p_old = driver.p_box_pa;
        let driver_audit = config.driver.step_pressure_base_audited(
            &mut driver,
            (config.voltage_fn)(t_next),
            config.bdf.dt,
        );
        let p_base = driver.p_box_pa;
        let alpha = config.driver.port_pressure_coefficient(config.bdf.dt);
        let diaphragm_feedback = config.driver.diaphragm_pressure_feedback(config.bdf.dt);

        let solve_at_pressure = |pressure_pa: f64, trial: &mut BdfState| {
            let tractions = [
                PressureTraction {
                    marker: MARKER_BOX_CAP,
                    pressure_pa,
                },
                PressureTraction {
                    marker: MARKER_EXTERIOR_CAP,
                    pressure_pa: config.exterior_pressure_pa,
                },
            ];
            solve_bdf_step_cached(
                mesh,
                v_space,
                p_space,
                &config.ns_config,
                &config.bdf,
                trial,
                &bc_fn,
                &bc_dofs,
                &tractions,
                &workspace,
            )
        };

        let previous_flow = flow.clone();
        let mut flow_base = previous_flow.clone();
        if !solve_at_pressure(p_base, &mut flow_base) {
            return Err(PortFlowError::SolverDiverged {
                step,
                time_s: flow.time,
            });
        }
        out.linear_solves.push(
            workspace
                .last_linear_solve_telemetry()
                .expect("successful cached BDF solve records telemetry"),
        );
        let q_base =
            integrate_face_flux_for_space(mesh, v_space, MARKER_EXTERIOR_CAP, &flow_base.velocity);

        let (q_port, conservation_residual_pa, accepted_linear_residual) = if alpha == 0.0 {
            flow = flow_base;
            let residual = out
                .linear_solves
                .last()
                .expect("base solve telemetry was recorded")
                .true_relative_residual;
            (q_base, 0.0, residual)
        } else {
            const PRESSURE_PROBE_PA: f64 = 1.0;
            let mut flow_probe = previous_flow.clone();
            if !solve_at_pressure(p_base + PRESSURE_PROBE_PA, &mut flow_probe) {
                return Err(PortFlowError::SolverDiverged {
                    step,
                    time_s: flow.time,
                });
            }
            out.linear_solves.push(
                workspace
                    .last_linear_solve_telemetry()
                    .expect("successful cached BDF solve records telemetry"),
            );
            let q_probe = integrate_face_flux_for_space(
                mesh,
                v_space,
                MARKER_EXTERIOR_CAP,
                &flow_probe.velocity,
            );
            let dq_dp = (q_probe - q_base) / PRESSURE_PROBE_PA;
            let denominator = 1.0 + diaphragm_feedback + alpha * dq_dp;
            if !denominator.is_finite() || denominator.abs() <= f64::EPSILON {
                return Err(PortFlowError::SolverDiverged {
                    step,
                    time_s: flow.time,
                });
            }
            let pressure_delta =
                (-alpha * q_base - diaphragm_feedback * (p_base - p_old)) / denominator;
            driver.p_box_pa = p_base + pressure_delta;
            let pressure_delta_from_old = driver.p_box_pa - p_old;
            config.driver.apply_pressure_correction(
                &mut driver,
                pressure_delta_from_old,
                config.bdf.dt,
            );

            flow = flow_base;
            for (value, probe) in flow.velocity.iter_mut().zip(&flow_probe.velocity) {
                *value += pressure_delta * (*probe - *value) / PRESSURE_PROBE_PA;
            }
            for (value, probe) in flow.pressure.iter_mut().zip(&flow_probe.pressure) {
                *value += pressure_delta * (*probe - *value) / PRESSURE_PROBE_PA;
            }
            let q_port =
                integrate_face_flux_for_space(mesh, v_space, MARKER_EXTERIOR_CAP, &flow.velocity);
            let residual = driver.p_box_pa - p_base
                + diaphragm_feedback * (driver.p_box_pa - p_old)
                + alpha * q_port;
            let mut accepted_state = Vec::with_capacity(flow.velocity.len() + flow.pressure.len());
            accepted_state.extend_from_slice(&flow.velocity);
            accepted_state.extend_from_slice(&flow.pressure);
            let accepted_linear_residual = workspace
                .affine_true_relative_residual(&accepted_state, pressure_delta)
                .ok_or(PortFlowError::SolverDiverged {
                    step,
                    time_s: flow.time,
                })?;
            (q_port, residual, accepted_linear_residual)
        };
        if !flow.velocity.iter().all(|v| v.is_finite())
            || !flow.pressure.iter().all(|p| p.is_finite())
            || !driver.p_box_pa.is_finite()
            || !driver.x_m.is_finite()
            || !q_port.is_finite()
            || !conservation_residual_pa.is_finite()
        {
            return Err(PortFlowError::SolverDiverged {
                step,
                time_s: flow.time,
            });
        }
        out.accepted_linear_relative_residual
            .push(accepted_linear_residual);
        let q_box_outward =
            integrate_face_flux_for_space(mesh, v_space, MARKER_BOX_CAP, &flow.velocity);
        if !q_box_outward.is_finite() {
            return Err(PortFlowError::SolverDiverged {
                step,
                time_s: flow.time,
            });
        }
        let final_tractions = [
            PressureTraction {
                marker: MARKER_BOX_CAP,
                pressure_pa: driver.p_box_pa,
            },
            PressureTraction {
                marker: MARKER_EXTERIOR_CAP,
                pressure_pa: config.exterior_pressure_pa,
            },
        ];
        let fluid_energy = audit_bdf_step(
            mesh,
            v_space,
            &config.ns_config,
            &config.bdf,
            &previous_flow,
            &flow,
            &final_tractions,
            MARKER_BOX_CAP,
            MARKER_EXTERIOR_CAP,
            &workspace,
        );
        let driver_energy =
            config
                .driver
                .finalize_energy_step(driver_audit, &driver, config.bdf.dt);
        let box_to_fluid_work = -config.bdf.dt * driver.p_box_pa * q_box_outward;
        let box_endpoint_numerical = driver_energy.driver_to_box_work_j
            - box_to_fluid_work
            - (driver_energy.box_storage_new_j - driver_energy.box_storage_old_j);
        let box_trapezoidal_residual = 0.5
            * (p_old + driver.p_box_pa)
            * (config.driver.sd_m2 * (driver.x_m - driver_audit.x_old_m) - config.bdf.dt * q_port)
            - (driver_energy.box_storage_new_j - driver_energy.box_storage_old_j);
        let energy = CoupledEnergyStep {
            driver: driver_energy,
            fluid: fluid_energy,
            box_to_fluid_work_j: box_to_fluid_work,
            box_endpoint_numerical_j: box_endpoint_numerical,
            box_trapezoidal_residual_j: box_trapezoidal_residual,
            cap_continuity_work_j: config.bdf.dt * driver.p_box_pa * (q_box_outward + q_port),
        };
        out.push(
            flow.time,
            q_port,
            q_box_outward,
            &config.driver,
            &driver,
            conservation_residual_pa,
            energy,
        );
        if config.bdf.output_interval > 0 && (step + 1) % config.bdf.output_interval == 0 {
            out.field_snapshots.push(PortFlowFieldSnapshot {
                time_s: flow.time,
                velocity_dofs_m_per_s: flow.velocity.clone(),
                pressure_dofs_pa: flow.pressure.clone(),
            });
        }
    }
    out.steps_completed = config.bdf.num_steps;
    out.final_flow_state = Some(flow);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::port_flow::analysis::dft_amplitude;
    use crate::{AcousticMedium, EnclosureSpec, PortSpec};
    use num_complex::Complex64;
    use sonium_fem::basis::PolynomialDegree;
    use sonium_fem::physics::coupled::lumped_circuit::VoiceCoilParams;
    use std::f64::consts::{LN_2, PI, TAU};

    use super::super::geometry::{SlotPortGeometry, build_slot_port_mesh};
    use super::super::geometry3d::{RoundPortGeometry, build_round_port_mesh};

    const V_BOX: f64 = 0.005;
    const DUCT_L: f64 = 0.1;
    const DUCT_H: f64 = 0.02;
    const SAMPLES_PER_DRIVE_CYCLE: usize = 126;
    /// Total end correction for both duct terminations: (2·ln2/π)·h per end,
    /// the flanged 2D-slot value. The exterior chamber is 4× the duct height
    /// and 2.5×h long, so the true correction differs from the infinite-baffle
    /// estimate; that residual is what the loose 10% gate absorbs.
    const END_CORRECTION_M: f64 = 4.0 * LN_2 / PI * DUCT_H;

    fn fixture_medium() -> AcousticMedium {
        AcousticMedium {
            density_kg_per_m3: 1.2,
            sound_speed_m_per_s: 343.0,
            dynamic_viscosity_pa_s: 1.85e-5,
            thermal_diffusivity_m2_per_s: 2.2e-5,
        }
    }

    fn fixture_port() -> PortSpec {
        PortSpec {
            // Unit-depth slot: area = h × 1 m, same volume-velocity convention
            // as the 2D flux integral.
            radius_m: (DUCT_H / PI).sqrt(),
            length_m: DUCT_L,
            end_correction_m: END_CORRECTION_M,
            resistance_pa_s_per_m3: 0.0,
        }
    }

    fn fixture_enclosure() -> EnclosureSpec {
        EnclosureSpec::Vented {
            volume_m3: V_BOX,
            port: fixture_port(),
            leakage_resistance_pa_s_per_m3: 0.0,
            absorption_resistance_pa_s_per_m3: 0.0,
        }
    }

    /// Drive frequency: HALF the lumped tuning frequency, not the tuning
    /// frequency itself. Deviation from the task brief, forced by physics:
    /// the M1 N-S model is incompressible, so the box–port Helmholtz resonance
    /// is damped only by boundary-layer viscous drag (Q ≈ 190 at f_b ≈ 318 Hz)
    /// while the lumped port carries baffled-piston radiation resistance
    /// (Q ≈ 7). An amplitude comparison AT f_b compares 1/R values and fails
    /// by orders of magnitude; even against a radiation-free reference it
    /// would need the two resonance frequencies to agree to ≲ 1/(2Q) ≈ 0.3%,
    /// impossible with end-correction uncertainty. At f_b/2 the port flow is
    /// compliance/inertance controlled: the resistive parts (radiation,
    /// boundary layer) perturb the result by < 0.1%, and the inertance
    /// sensitivity is (ω/ω_b)²/(1 − (ω/ω_b)²) = 1/3, so the 10% tolerance
    /// measures exactly the "2D slot vs lumped end corrections" residual the
    /// brief names.
    fn fixture_drive_frequency_hz() -> f64 {
        fixture_enclosure()
            .tuning_frequency_hz(fixture_medium())
            .unwrap()
            .unwrap()
            / 2.0
    }

    fn fixture_driver() -> BoxDriverModel {
        BoxDriverModel {
            mms_kg: 0.02,
            rms_kg_per_s: 1.0,
            kms_n_per_m: 4000.0,
            voice_coil: VoiceCoilParams::linear(6.0, 0.5e-3, 10.0),
            compliance_force_fn: None,
            compliance_potential_fn: None,
            sd_m2: 0.01,
            box_volume_m3: Some(V_BOX),
            medium_density_kg_per_m3: 1.2,
            medium_sound_speed_m_per_s: 343.0,
            beta: 0.25,
            gamma: 0.5,
        }
    }

    fn slot_fixture(mesh_size_m: f64) -> (Mesh, VectorH1Space, H1Space) {
        let geo = SlotPortGeometry {
            duct_length_m: DUCT_L,
            duct_height_m: DUCT_H,
            chamber_length_m: 0.05,
            chamber_height_m: 0.08,
            mesh_size_m,
        };
        let mesh = build_slot_port_mesh(&geo).unwrap();
        // Taylor–Hood: P2 velocity / P1 pressure, inf-sup stable.
        let v_space = VectorH1Space::new(PolynomialDegree::P2, 2);
        let p_space = H1Space::new(PolynomialDegree::P1);
        (mesh, v_space, p_space)
    }

    fn slot_fixture_coarse() -> (Mesh, VectorH1Space, H1Space) {
        slot_fixture(0.01)
    }

    fn round_port_fixture_3d() -> (Mesh, VectorH1Space, H1Space) {
        let geometry = RoundPortGeometry {
            duct_length_m: 0.04,
            duct_radius_m: 0.01,
            chamber_length_m: 0.02,
            chamber_radius_m: 0.01,
            flare_radius_m: 0.0,
            mesh_size_m: 0.012,
        };
        let mesh = build_round_port_mesh(&geometry).expect("small 3-D round-port mesh");
        let v_space = VectorH1Space::new(PolynomialDegree::P2, 3);
        let p_space = H1Space::new(PolynomialDegree::P1);
        (mesh, v_space, p_space)
    }

    fn coupled_fixture(voltage_amplitude: f64) -> CoupledPortConfig {
        let f_drive = fixture_drive_frequency_hz();
        // The coupled BDF2/Newmark transfer is not converged at 1e-4 s: that
        // step over-predicts the tiny-drive flow by 11.4%. Halving the step
        // brings the unchanged lumped-limit gate below 10%.
        let dt = 1.0 / (f_drive * SAMPLES_PER_DRIVE_CYCLE as f64);
        // 40 drive periods: the f_b free transient (τ = 2Q/ω_b ≈ 30 drive
        // periods) decays to ≲30% and sits near the 2nd harmonic, which the
        // Hann-windowed DFT at the fundamental rejects by ≳45 dB.
        let num_steps = 40 * SAMPLES_PER_DRIVE_CYCLE;
        CoupledPortConfig {
            driver: fixture_driver(),
            ns_config: NavierStokesConfig {
                mu: 1.85e-5,
                rho: 1.2,
                // Taylor–Hood P2–P1 is inf-sup stable: no stabilisation.
                use_residual_stabilisation: false,
                // Zero body force through `solve_bdf_step_cached`: the
                // force quadrature is identical every step, so caching
                // skips two full assemblies per step after the first.
                cache_body_force_assembly: true,
                ..Default::default()
            },
            bdf: BdfConfig::bdf2(dt, num_steps),
            voltage_fn: Box::new(move |t| voltage_amplitude * (TAU * f_drive * t).sin()),
            under_relaxation: 1.0,
            exterior_pressure_pa: 0.0,
        }
    }

    #[test]
    fn three_dimensional_round_port_taylor_hood_step_closes_finite_interfaces() {
        let (mesh, v_space, p_space) = round_port_fixture_3d();
        assert_eq!(mesh.dimension, 3);
        let mut config = coupled_fixture(1.0e-4);
        config.driver.box_volume_m3 = Some(2.0e-5);
        config.bdf = BdfConfig::bdf2(1.0e-5, 1);
        config.bdf.output_interval = 1;

        let result = run_coupled_port(&mesh, &v_space, &p_space, &config)
            .expect("one 3-D Taylor-Hood coupled step");

        assert_eq!(result.steps_completed, 1);
        assert_eq!(result.time_s.len(), 1);
        assert_eq!(result.field_snapshots.len(), 1);
        assert!(result.port_flow_m3s[0].is_finite());
        assert!(result.exterior_cap_flow_m3s[0].is_finite());
        assert!(result.box_cap_outward_flow_m3s[0].is_finite());
        assert!(result.box_conservation_residual_pa[0].is_finite());
        assert!(result.accepted_linear_relative_residual[0].is_finite());
        assert!(result.energy_steps[0].is_finite());
        assert!(result.energy_steps[0].physical_dissipations_nonnegative(1.0e-12));
        assert!(
            (result.exterior_cap_flow_m3s[0] + result.box_cap_outward_flow_m3s[0]).abs() < 1.0e-8
        );
    }

    #[test]
    fn three_dimensional_round_port_reaches_poiseuille_profile() {
        let (mesh, v_space, p_space) = round_port_fixture_3d();
        let n_v = v_space.num_global_dofs(&mesh);
        let n_p = p_space.num_global_dofs(&mesh);
        let mut state = BdfState::from_rest(n_v, n_p);
        let mut workspace = BdfWorkspace::new(&mesh, &v_space, &p_space);
        let pressure_pin = p_space
            .boundary_dofs(&mesh, MARKER_EXTERIOR_CAP)
            .into_iter()
            .min_by(|&a, &b| {
                let point_a = p_space.dof_coordinate(&mesh, a).unwrap_or([0.0; 3]);
                let point_b = p_space.dof_coordinate(&mesh, b).unwrap_or([0.0; 3]);
                point_a[1]
                    .hypot(point_a[2])
                    .total_cmp(&point_b[1].hypot(point_b[2]))
                    .then(a.cmp(&b))
            })
            .expect("round-port exterior cap has a pressure gauge DOF");
        workspace.set_pressure_gauge(pressure_pin, 0.0);

        let mu = 1.0e-3;
        let pressure_drop_pa = 0.01;
        let config = NavierStokesConfig {
            mu,
            rho: 1.0,
            use_residual_stabilisation: false,
            // Zero body force through `solve_bdf_step_cached`: the
            // force quadrature is identical every step, so caching
            // skips two full assemblies per step after the first.
            cache_body_force_assembly: true,
            ..Default::default()
        };
        let bdf = BdfConfig::bdf2(0.125, 4);
        let wall_dofs = v_space.boundary_dofs(&mesh, MARKER_WALL);
        let zero_velocity = |_: f64, _: f64, _: f64| vec![0.0, 0.0, 0.0];
        let tractions = [
            PressureTraction {
                marker: MARKER_BOX_CAP,
                pressure_pa: pressure_drop_pa,
            },
            PressureTraction {
                marker: MARKER_EXTERIOR_CAP,
                pressure_pa: 0.0,
            },
        ];

        for _ in 0..bdf.num_steps {
            assert!(solve_bdf_step_cached(
                &mesh,
                &v_space,
                &p_space,
                &config,
                &bdf,
                &mut state,
                &zero_velocity,
                &wall_dofs,
                &tractions,
                &workspace,
            ));
        }

        let pressure_length = 2.0 * 0.02 + 0.04;
        let expected_centerline_velocity =
            pressure_drop_pa * 0.01_f64.powi(2) / (4.0 * mu * pressure_length);
        let dof_coordinates = v_space
            .dof_coordinates(&mesh)
            .expect("VectorH1Space exposes P2 velocity coordinates");
        let centerline_velocity = dof_coordinates
            .iter()
            .enumerate()
            .filter(|(dof, point)| {
                dof % 3 == 0
                    && point[1].hypot(point[2]) < 1.0e-10
                    && point[0] > 0.25 * pressure_length
                    && point[0] < 0.75 * pressure_length
            })
            .map(|(dof, _)| state.velocity[dof])
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(centerline_velocity.is_finite());
        let relative_error = (centerline_velocity - expected_centerline_velocity).abs()
            / expected_centerline_velocity;
        assert!(
            relative_error < 0.08,
            "3-D round-port Poiseuille centerline velocity expected={expected_centerline_velocity:e}, got={centerline_velocity:e}, relative_error={relative_error:e}"
        );
    }

    /// Linear steady-state port-flow amplitude from `EnclosureSpec::Vented`
    /// acoustics and the driver's linear T/S circuit. Mirrors the admittance
    /// assembly of `enclosure.rs::acoustic_impedance` (box compliance in
    /// parallel with the port branch; zero leakage/absorption), then closes
    /// the electro-mechano-acoustic loop:
    ///
    /// ```text
    /// Z_mt = Z_m + Sd²·Z_encl          (enclosure reflected to mechanical)
    /// v    = BL·V / (Z_e·Z_mt + BL²)   (voltage drive + back-EMF)
    /// U_port = Sd·v · Z_encl/Z_port
    /// ```
    fn lumped_linear_port_flow_amplitude(
        driver: &BoxDriverModel,
        frequency_hz: f64,
        voltage_amplitude: f64,
    ) -> f64 {
        let medium = fixture_medium();
        let z_encl = fixture_enclosure()
            .acoustic_impedance(frequency_hz, medium)
            .unwrap();
        let z_port = fixture_port()
            .acoustic_impedance(frequency_hz, medium)
            .unwrap();
        let w = TAU * frequency_hz;
        let re = driver.voice_coil.resistance;
        let le = (driver.voice_coil.inductance)(0.0);
        let bl = (driver.voice_coil.bl)(0.0);
        let z_e = Complex64::new(re, w * le);
        let z_m = Complex64::new(
            driver.rms_kg_per_s,
            w * driver.mms_kg - driver.kms_n_per_m / w,
        );
        let z_mt = z_m + z_encl * driver.sd_m2 * driver.sd_m2;
        let v = bl * voltage_amplitude / (z_e * z_mt + bl * bl);
        let u_diaphragm = v * driver.sd_m2;
        (u_diaphragm * z_encl / z_port).norm()
    }

    #[test]
    #[ignore = "resolved-to-lumped validation; run via `just qa-nightly-regressions`"]
    fn tiny_drive_matches_linear_lumped_model() {
        // Vented box V = 0.005 m³, slot duct L = 0.1 m, h = 0.02 m, unit width
        // → A = 0.02 m². 0.01 V drive (duct Re ≈ 0.6: deep linear regime) at
        // half the lumped tuning frequency (see fixture_drive_frequency_hz).
        // Steady-state port-flow amplitude from the tail of the record must
        // match the linear lumped prediction within 10% (coarse mesh, 2D slot
        // vs lumped end corrections).
        let (mesh, v_space, p_space) = slot_fixture_coarse();
        let mut config = coupled_fixture(0.01);
        config.bdf.output_interval = config.bdf.num_steps / 4;
        let f_drive = fixture_drive_frequency_hz();
        let result = run_coupled_port(&mesh, &v_space, &p_space, &config).unwrap();
        assert_eq!(result.steps_completed, config.bdf.num_steps);
        assert_eq!(result.field_snapshots.len(), 4);
        assert!(
            result
                .field_snapshots
                .windows(2)
                .all(|pair| pair[0].time_s < pair[1].time_s)
        );
        let n_tail = result.port_flow_m3s.len() / 5; // last 20% of samples
        let q_amp = dft_amplitude(
            &result.port_flow_m3s[result.port_flow_m3s.len() - n_tail..],
            config.bdf.dt,
            f_drive,
        );
        let q_lumped = lumped_linear_port_flow_amplitude(&config.driver, f_drive, 0.01);
        let rel = (q_amp - q_lumped).abs() / q_lumped;
        let tail_start = result.energy_steps.len() - 8 * SAMPLES_PER_DRIVE_CYCLE;
        let summary = result.energy_summary(tail_start..result.energy_steps.len());
        eprintln!("whole-cycle energy summary {summary:?}");
        assert!(summary.algebraic_relative_closure <= 1.0e-8);
        assert!(summary.interface_relative_residual <= 1.0e-8);
        assert!(
            summary.physical_relative_closure <= 0.05,
            "last-complete-cycle physical closure is {}",
            summary.physical_relative_closure
        );
        let omitted_copper_relative = (summary.source_j
            - summary.stored_delta_j
            - (summary.physical_dissipation_j - summary.copper_loss_j)
            - summary.exterior_export_j)
            .abs()
            / summary.energy_scale_j;
        assert!(
            omitted_copper_relative > 0.05,
            "the 5% gate was insensitive to an omitted copper-loss channel: {omitted_copper_relative}"
        );
        assert!(rel < 0.10, "coupled {q_amp}, lumped {q_lumped} (rel {rel})");
    }

    #[test]
    fn solver_failure_returns_err_not_panic() {
        let (mesh, v_space, p_space) = slot_fixture_coarse();
        let mut config = coupled_fixture(1.0);
        config.ns_config.use_direct_solver = false;
        config.bdf.dt = 1.0; // absurd dt makes the affine response singular
        let result = run_coupled_port(&mesh, &v_space, &p_space, &config);
        assert!(matches!(result, Err(PortFlowError::SolverDiverged { .. })));
    }

    #[test]
    fn affine_interface_closure_satisfies_box_conservation_per_step() {
        let (mesh, v_space, p_space) = slot_fixture_coarse();
        let mut config = coupled_fixture(0.01);
        config.bdf.num_steps = 200;
        let result = run_coupled_port(&mesh, &v_space, &p_space, &config).unwrap();
        let max_pressure = result
            .box_pressure_pa
            .iter()
            .map(|value| value.abs())
            .fold(1.0, f64::max);
        let max_residual = result
            .box_conservation_residual_pa
            .iter()
            .map(|value| value.abs())
            .fold(0.0, f64::max);
        assert!(
            max_residual <= 1.0e-11 * max_pressure,
            "box closure residual {max_residual:e} Pa at pressure scale {max_pressure:e} Pa"
        );
    }

    #[test]
    #[ignore = "resolved coupled energy audit; run via `just qa-nightly-regressions`"]
    fn coupled_energy_ledger_is_aligned_finite_and_closes_interfaces() {
        let (mesh, v_space, p_space) = slot_fixture_coarse();
        let mut config = coupled_fixture(0.01);
        config.bdf.num_steps = 20;
        let mut result = run_coupled_port(&mesh, &v_space, &p_space, &config).unwrap();
        assert_eq!(result.energy_steps.len(), result.time_s.len());
        assert!(result.energy_steps.iter().all(|step| {
            [
                step.driver.source_j,
                step.driver.copper_loss_j,
                step.fluid.viscous_dissipation_j,
                step.physical_residual_j(),
                step.algebraic_residual_j(),
            ]
            .iter()
            .all(|value| value.is_finite())
        }));
        assert!(
            result
                .energy_steps
                .iter()
                .all(|step| step.driver.copper_loss_j >= 0.0
                    && step.driver.mechanical_damping_midpoint_j >= 0.0
                    && step.fluid.viscous_dissipation_j >= -1.0e-18)
        );
        let summary = result.energy_summary(0..result.energy_steps.len());
        eprintln!("energy summary {summary:?}");
        let trapezoidal_box_residual = result
            .energy_steps
            .iter()
            .map(|step| step.box_trapezoidal_residual_j.abs())
            .sum::<f64>()
            / summary.energy_scale_j;
        assert!(trapezoidal_box_residual < 1.0e-8);
        assert!(summary.interface_relative_residual < 1.0e-8);
        assert!(summary.algebraic_relative_closure < 1.0e-8);

        let copper: Vec<_> = result
            .energy_steps
            .iter()
            .map(|step| step.driver.copper_loss_j)
            .collect();
        for step in &mut result.energy_steps {
            step.driver.copper_loss_j = 0.0;
        }
        let omitted_copper = result.energy_summary(0..result.energy_steps.len());
        assert!(
            omitted_copper.algebraic_relative_closure > 0.05,
            "omitted copper did not corrupt algebraic closure: {omitted_copper:?}"
        );
        for (step, copper_loss) in result.energy_steps.iter_mut().zip(copper) {
            step.driver.copper_loss_j = copper_loss;
        }
        result.energy_steps[0].fluid.box_traction_work_j += summary.energy_scale_j;
        let corrupted_interface = result.energy_summary(0..result.energy_steps.len());
        assert!(
            corrupted_interface.interface_relative_residual > 0.5,
            "corrupted box traction did not trip the interface audit: {corrupted_interface:?}"
        );
    }

    #[test]
    fn passive_nonzero_initial_state_does_not_gain_total_energy() {
        let (mesh, v_space, p_space) = slot_fixture_coarse();
        let mut config = coupled_fixture(0.0);
        config.bdf.num_steps = 200;
        config.voltage_fn = Box::new(|_| 0.0);
        let n_v = v_space.num_global_dofs(&mesh);
        let n_p = p_space.num_global_dofs(&mesh);
        let flow = BdfState::from_rest(n_v, n_p);
        let mut driver = config.driver.new_state();
        driver.x_m = 1.0e-4;
        driver.a_m_per_s2 = -config.driver.kms_n_per_m * driver.x_m / config.driver.mms_kg;
        let result = run_coupled_port_initialized(
            &mesh,
            &v_space,
            &p_space,
            &config,
            Some(flow),
            Some(driver),
        )
        .unwrap();
        let first = result.energy_steps[0];
        let last = *result.energy_steps.last().unwrap();
        let initial_energy = first.stored_old_j();
        let final_energy = last.stored_new_j();
        let signed_export = result
            .energy_steps
            .iter()
            .map(|step| step.fluid.convection_export_j - step.fluid.exterior_traction_work_j)
            .sum::<f64>();
        let boundary_inflow = (-signed_export).max(0.0);
        assert!(initial_energy > 0.0);
        assert!(
            final_energy <= initial_energy + boundary_inflow + 1.0e-8 * initial_energy,
            "passive coupled state gained energy: initial={initial_energy:e}, final={final_energy:e}, boundary_inflow={boundary_inflow:e}"
        );
        assert!(result.energy_steps.iter().all(|step| step.is_finite()));
    }

    #[test]
    fn coupled_timestep_refines_toward_the_small_step_limit() {
        let (mesh, v_space, p_space) = slot_fixture_coarse();
        let duration_s = 0.01;
        let run_at = |dt: f64| {
            let mut config = coupled_fixture(0.01);
            config.bdf.dt = dt;
            config.bdf.num_steps = (duration_s / dt).round() as usize;
            let result = run_coupled_port(&mesh, &v_space, &p_space, &config).unwrap();
            let final_flow = result.port_flow_m3s.last().copied().unwrap();
            let summary = result.energy_summary(0..result.energy_steps.len());
            (final_flow, summary)
        };
        let (coarse, coarse_energy) = run_at(1.0e-4);
        let (medium, medium_energy) = run_at(5.0e-5);
        let (fine, _) = run_at(2.5e-5);
        let coarse_change = (coarse - medium).abs();
        let fine_change = (medium - fine).abs();
        assert!(
            fine_change < coarse_change,
            "time refinement did not converge: q(1e-4)={coarse:e}, q(5e-5)={medium:e}, q(2.5e-5)={fine:e}"
        );
        assert!(
            medium_energy.physical_relative_closure <= coarse_energy.physical_relative_closure,
            "dt/2 increased physical closure: coarse={coarse_energy:?}, fine={medium_energy:?}"
        );
        assert!(
            (coarse_energy.physical_relative_closure < 0.005
                && medium_energy.physical_relative_closure < 0.005)
                || coarse_energy.physical_relative_closure
                    >= 1.5 * medium_energy.physical_relative_closure,
            "dt/2 closure improvement was below 1.5x: coarse={coarse_energy:?}, fine={medium_energy:?}"
        );
    }

    #[test]
    #[ignore = "resolved spatial-convergence study; run via `just qa-nightly-regressions`"]
    fn coupled_energy_closure_is_stable_under_spatial_refinement() {
        let frequency = fixture_drive_frequency_hz();
        let samples_per_cycle = 32;
        let dt = 1.0 / (frequency * samples_per_cycle as f64);
        let run_on = |mesh_size_m: f64| {
            let (mesh, v_space, p_space) = slot_fixture(mesh_size_m);
            let mut config = coupled_fixture(0.01);
            config.bdf.dt = dt;
            config.bdf.num_steps = 2 * samples_per_cycle;
            config.voltage_fn = Box::new(move |time| 0.01 * (TAU * frequency * time).sin());
            let result = run_coupled_port(&mesh, &v_space, &p_space, &config).unwrap();
            let summary = result.energy_summary(0..result.energy_steps.len());
            let equation_residual = result
                .energy_steps
                .iter()
                .map(|step| {
                    step.fluid.pressure_work_j.abs()
                        + step.cap_continuity_work_j.abs()
                        + step.fluid.solve_residual_j.abs()
                })
                .sum::<f64>()
                / summary.energy_scale_j;
            (summary, equation_residual)
        };
        let (coarse, coarse_equation_residual) = run_on(0.01);
        let (fine, fine_equation_residual) = run_on(0.0075);
        eprintln!(
            "mesh energy convergence coarse={coarse:?} fine={fine:?} equation_residual={coarse_equation_residual:e}->{fine_equation_residual:e}"
        );
        assert!(
            fine.physical_relative_closure <= 1.10 * coarse.physical_relative_closure,
            "mesh refinement worsened closure by more than 10%: coarse={coarse:?}, fine={fine:?}"
        );
        assert!(
            fine_equation_residual < coarse_equation_residual,
            "pressure/continuity/solve residual did not decrease: coarse={coarse_equation_residual:e}, fine={fine_equation_residual:e}"
        );
    }

    #[test]
    #[ignore = "expensive 40-cycle settled spatial energy evidence"]
    fn settled_final_eight_cycle_energy_closure_refines_in_space() {
        // Below this conservative floor, strict ordering is dominated by the
        // accumulated binary64/direct-solve roundoff rather than a resolved
        // spatial-discretisation trend. Approved M1 interpretation: require a
        // literal decrease above the floor, or require both meshes below it.
        const SETTLED_EQUATION_RESIDUAL_FLOOR: f64 = 1.0e-12;
        let run_on = |mesh_size_m: f64| {
            let (mesh, v_space, p_space) = slot_fixture(mesh_size_m);
            let config = coupled_fixture(0.01);
            let result = run_coupled_port(&mesh, &v_space, &p_space, &config).unwrap();
            let tail_start = result.energy_steps.len() - 8 * SAMPLES_PER_DRIVE_CYCLE;
            let summary = result.energy_summary(tail_start..result.energy_steps.len());
            let equation_residual = result.energy_steps[tail_start..]
                .iter()
                .map(|step| {
                    step.fluid.pressure_work_j.abs()
                        + step.cap_continuity_work_j.abs()
                        + step.fluid.solve_residual_j.abs()
                })
                .sum::<f64>()
                / summary.energy_scale_j;
            (summary, equation_residual)
        };
        let (coarse, coarse_equation_residual) = run_on(0.01);
        let (fine, fine_equation_residual) = run_on(0.0075);
        eprintln!(
            "settled mesh energy convergence coarse={coarse:?} fine={fine:?} equation_residual={coarse_equation_residual:e}->{fine_equation_residual:e}"
        );
        assert!(coarse.physical_relative_closure <= 0.05);
        assert!(fine.physical_relative_closure <= 1.10 * coarse.physical_relative_closure);
        assert!(
            fine_equation_residual < coarse_equation_residual
                || (coarse_equation_residual <= SETTLED_EQUATION_RESIDUAL_FLOOR
                    && fine_equation_residual <= SETTLED_EQUATION_RESIDUAL_FLOOR),
            "settled equation residual neither decreased nor reached the approved floor: coarse={coarse_equation_residual:e}, fine={fine_equation_residual:e}, floor={SETTLED_EQUATION_RESIDUAL_FLOOR:e}"
        );
    }

    #[test]
    fn complete_cycle_energy_closure_converges_in_time() {
        let (mesh, v_space, p_space) = slot_fixture_coarse();
        let frequency = fixture_drive_frequency_hz();
        let run_at = |samples_per_cycle: usize| {
            let dt = 1.0 / (frequency * samples_per_cycle as f64);
            let cycles = 4;
            let mut config = coupled_fixture(0.01);
            config.bdf.dt = dt;
            config.bdf.num_steps = cycles * samples_per_cycle;
            config.voltage_fn = Box::new(move |time| 0.01 * (TAU * frequency * time).sin());
            let result = run_coupled_port(&mesh, &v_space, &p_space, &config).unwrap();
            result.energy_summary(0..result.energy_steps.len())
        };
        let coarse = run_at(32);
        let fine = run_at(64);
        eprintln!("whole-cycle dt convergence coarse={coarse:?} fine={fine:?}");
        assert!(
            fine.physical_relative_closure <= coarse.physical_relative_closure,
            "dt/2 increased whole-cycle physical closure: coarse={coarse:?}, fine={fine:?}"
        );
        assert!(
            (coarse.physical_relative_closure < 0.005 && fine.physical_relative_closure < 0.005)
                || coarse.physical_relative_closure >= 1.5 * fine.physical_relative_closure,
            "dt/2 whole-cycle closure improvement was below 1.5x: coarse={coarse:?}, fine={fine:?}"
        );
    }

    #[test]
    fn undriven_coupled_ringdown_does_not_grow() {
        let (mesh, v_space, p_space) = slot_fixture_coarse();
        let mut config = coupled_fixture(0.01);
        let dt = config.bdf.dt;
        config.bdf.num_steps = 600;
        config.voltage_fn = Box::new(move |time| if time <= 20.0 * dt { 0.01 } else { 0.0 });
        let result = run_coupled_port(&mesh, &v_space, &p_space, &config).unwrap();
        let peak = |slice: &[f64]| slice.iter().map(|value| value.abs()).fold(0.0, f64::max);
        let early_peak = peak(&result.port_flow_m3s[100..300]);
        let late_peak = peak(&result.port_flow_m3s[400..600]);
        assert!(
            late_peak <= early_peak,
            "undriven ringdown grew: early {early_peak:e}, late {late_peak:e} m3/s"
        );
    }

    #[test]
    fn result_vectors_align_and_diaphragm_flow_is_signed_sd_times_velocity() {
        let driver_model = fixture_driver();
        let mut driver_state = driver_model.new_state();
        let mut result = CoupledPortResult::with_capacity(2);

        driver_state.v_m_per_s = 0.25;
        driver_state.p_box_pa = 3.0;
        driver_state.x_m = 4.0e-4;
        driver_state.circuit.current = 0.5;
        result.push(
            1.0e-4,
            6.0e-3,
            -6.0e-3,
            &driver_model,
            &driver_state,
            1.0e-14,
            CoupledEnergyStep::default(),
        );

        driver_state.v_m_per_s = -0.125;
        driver_state.p_box_pa = -7.0;
        driver_state.x_m = -8.0e-4;
        driver_state.circuit.current = -0.75;
        result.push(
            2.0e-4,
            -9.0e-3,
            9.0e-3,
            &driver_model,
            &driver_state,
            -2.0e-14,
            CoupledEnergyStep::default(),
        );

        let expected_len = 2;
        assert_eq!(result.time_s.len(), expected_len);
        assert_eq!(result.port_flow_m3s.len(), expected_len);
        assert_eq!(result.exterior_cap_flow_m3s, result.port_flow_m3s);
        assert_eq!(result.energy_steps.len(), expected_len);
        assert_eq!(result.box_cap_outward_flow_m3s.len(), expected_len);
        assert_eq!(result.box_cap_outward_flow_m3s, vec![-6.0e-3, 9.0e-3]);
        assert_eq!(result.diaphragm_volume_velocity_m3s.len(), expected_len);
        assert_eq!(result.box_pressure_pa.len(), expected_len);
        assert_eq!(result.displacement_m.len(), expected_len);
        assert_eq!(result.current_a.len(), expected_len);
        assert_eq!(result.box_conservation_residual_pa.len(), expected_len);
        assert_eq!(result.box_conservation_residual_pa, vec![1.0e-14, -2.0e-14]);
        assert_eq!(
            result.diaphragm_volume_velocity_m3s,
            vec![driver_model.sd_m2 * 0.25, driver_model.sd_m2 * -0.125]
        );
        assert!(result.diaphragm_volume_velocity_m3s[0].is_sign_positive());
        assert!(result.diaphragm_volume_velocity_m3s[1].is_sign_negative());
    }

    fn invalid_config_message(result: Result<CoupledPortResult, PortFlowError>) -> String {
        match result {
            Err(PortFlowError::InvalidConfig(message)) => message,
            Err(other) => panic!("expected InvalidConfig, got {other}"),
            Ok(_) => panic!("expected InvalidConfig, got success"),
        }
    }

    #[test]
    fn rejects_non_taylor_hood_spaces() {
        let (mesh, _, p_space) = slot_fixture_coarse();
        let p1_velocity = VectorH1Space::new(PolynomialDegree::P1, 2);
        let config = coupled_fixture(0.0);
        let message =
            invalid_config_message(run_coupled_port(&mesh, &p1_velocity, &p_space, &config));
        assert!(message.contains("requires Taylor-Hood P2-P1"), "{message}");

        let p2_velocity = VectorH1Space::new(PolynomialDegree::P2, 2);
        let p2_pressure = H1Space::new(PolynomialDegree::P2);
        let config = coupled_fixture(0.0);
        let message =
            invalid_config_message(run_coupled_port(&mesh, &p2_velocity, &p2_pressure, &config));
        assert!(message.contains("requires Taylor-Hood P2-P1"), "{message}");
    }

    #[test]
    fn rejects_nonlinear_suspension_without_potential() {
        let (mesh, v_space, p_space) = slot_fixture_coarse();
        let mut config = coupled_fixture(0.0);
        config.driver.compliance_force_fn = Some(Box::new(|x| 4000.0 * x + 1.0e8 * x.powi(3)));
        config.driver.compliance_potential_fn = None;
        let message = invalid_config_message(run_coupled_port(&mesh, &v_space, &p_space, &config));
        assert!(message.contains("force and potential-energy"));
    }

    #[test]
    fn rejects_newmark_parameters_outside_audited_invariant() {
        let (mesh, v_space, p_space) = slot_fixture_coarse();
        let mut config = coupled_fixture(0.0);
        config.driver.beta = 0.3;
        let message = invalid_config_message(run_coupled_port(&mesh, &v_space, &p_space, &config));
        assert!(message.contains("beta=0.25, gamma=0.5"));
    }

    #[test]
    fn versioned_energy_block_round_trips_json() {
        let mut result = CoupledPortResult::with_capacity(1);
        result.energy_steps.push(CoupledEnergyStep::default());
        let block = result.energy_block();
        assert_eq!(block.schema_version, COUPLED_ENERGY_SCHEMA_VERSION);
        let json = serde_json::to_string(&block).unwrap();
        let decoded: CoupledEnergyBlock = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.schema_version, COUPLED_ENERGY_SCHEMA_VERSION);
        assert_eq!(decoded.steps.len(), 1);
    }

    #[test]
    fn rejects_velocity_component_dimension_mismatch() {
        let (mesh, _, p_space) = slot_fixture_coarse();
        let three_component_velocity = VectorH1Space::new(PolynomialDegree::P2, 3);
        let config = coupled_fixture(0.0);
        let message = invalid_config_message(run_coupled_port(
            &mesh,
            &three_component_velocity,
            &p_space,
            &config,
        ));
        assert!(message.contains("must match mesh dimension"), "{message}");
    }

    #[test]
    fn rejects_residual_stabilization_for_taylor_hood() {
        let (mesh, v_space, p_space) = slot_fixture_coarse();
        let mut config = coupled_fixture(0.0);
        config.ns_config.use_residual_stabilisation = true;
        let message = invalid_config_message(run_coupled_port(&mesh, &v_space, &p_space, &config));
        assert!(message.contains("must be disabled"), "{message}");
    }
}
