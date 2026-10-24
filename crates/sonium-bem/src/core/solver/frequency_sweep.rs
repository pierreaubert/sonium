//! Frequency sweep solver with solution recycling
//!
//! For HRTF computation, the BEM system must be solved at 200-500 frequencies.
//! This module provides an optimized frequency sweep that:
//!
//! 1. Assembles the BEM system at each frequency (the mesh stays fixed)
//! 2. Uses the converged solution at frequency f_{i-1} as the initial guess
//!    for the iterative solver at f_i (Krylov recycling)
//!
//! For closely spaced frequencies, the solution changes slowly, so the solver
//! converges in 5-15 iterations instead of 50-100, dramatically reducing total
//! computation time.

use num_complex::Complex64;
use std::time::Instant;

use crate::core::assembly::hmatrix::{
    HMatrix, apply_tbem_stabilized_exact, build_hmatrix_stabilized,
};
use crate::core::assembly::tbem::build_tbem_system_stabilized;
use crate::core::assembly::{
    build_cluster_tree, build_directional_mlfmm_system_stabilized_checked,
    build_mlfmm_system_stabilized_checked, build_slfmm_system_stabilized,
};
use crate::core::bem_solver::{AssemblyMethod, BemError};
use crate::core::contracts::{OperatorAccuracyReport, require_operator_accuracy};
use crate::core::incident::IncidentField;
use crate::core::mesh::octree::Octree;
use crate::core::mesh::relative_frequency_delta;
use crate::core::solver::fmm_interface::{MlfmmOperator, SlfmmOperator};
use crate::core::types::{BoundaryCondition, Cluster, Element, PhysicsParams};
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::iterative::{
    GcroDrConfig, GcroDrSubspace, GmresConfig, GmresSolution, gcro_dr, gmres_with_guess,
};
use sonium_solvers::traits::LinearOperator;

#[cfg(feature = "cuda")]
#[path = "frequency_sweep_cuda.rs"]
mod cuda;
#[cfg(feature = "cuda")]
#[doc(inline)]
pub use cuda::{CudaFrequencySweepPoint, CudaFrequencySweepResult, frequency_sweep_cuda_tbem_jacobi, frequency_sweep_cuda_tbem_jacobi_device};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for a frequency sweep
pub struct FrequencySweepConfig {
    /// Frequencies to solve at (Hz), should be sorted in ascending order
    pub frequencies: Vec<f64>,
    /// Solver tolerance for iterative methods
    pub tolerance: f64,
    /// Maximum iterations per frequency
    pub max_iterations: usize,
    /// Whether to use previous solution as initial guess (warm-start only).
    /// Independent of `recycle_subspace_size` — both can be enabled together,
    /// in which case the recycled Krylov subspace acts on the residual of the
    /// warm-started initial guess.
    pub recycle_solution: bool,
    /// GMRES restart parameter
    pub gmres_restart: usize,
    /// Burton-Miller beta scale factor
    pub beta_scale: f64,
    /// Number of harmonic Ritz vectors recycled across consecutive
    /// frequencies via GCRO-DR (Parks–de Sturler 2006). `0` keeps the legacy
    /// warm-start-only path. Typical values: 10–25. Cost: `k` extra
    /// matrix-vector products per frequency to rebuild `C := orthonormalize(A·U)`.
    pub recycle_subspace_size: usize,
    /// Largest accepted relative jump between adjacent frequencies when
    /// retaining a GCRO-DR subspace. The normalized distance is
    /// `|f_new - f_old| / max(f_new, f_old)`. A larger jump starts a fresh
    /// subspace; it never changes the solved operator or acceptance checks.
    pub max_recycle_relative_frequency_delta: f64,
}

impl Default for FrequencySweepConfig {
    fn default() -> Self {
        Self {
            frequencies: Vec::new(),
            tolerance: 1e-6,
            max_iterations: 200,
            recycle_solution: true,
            gmres_restart: 30,
            beta_scale: 4.0,
            recycle_subspace_size: 0,
            max_recycle_relative_frequency_delta: 0.1,
        }
    }
}

impl FrequencySweepConfig {
    /// Validate the sweep grid and iterative controls before any assembly.
    pub fn validate(&self) -> Result<(), BemError> {
        if !self.tolerance.is_finite() || self.tolerance <= 0.0 {
            return Err(BemError::InvalidParameters(
                "frequency-sweep tolerance must be finite and positive".into(),
            ));
        }
        if self.max_iterations == 0 || self.gmres_restart == 0 {
            return Err(BemError::InvalidParameters(
                "frequency-sweep iteration and restart limits must be non-zero".into(),
            ));
        }
        if !self.beta_scale.is_finite() || self.beta_scale <= 0.0 {
            return Err(BemError::InvalidParameters(
                "Burton-Miller beta scale must be finite and positive".into(),
            ));
        }
        if !self.max_recycle_relative_frequency_delta.is_finite()
            || self.max_recycle_relative_frequency_delta < 0.0
        {
            return Err(BemError::InvalidParameters(
                "frequency-sweep recycle frequency bound must be finite and non-negative".into(),
            ));
        }
        let _ = self.frequency_quantities()?;
        Ok(())
    }

    /// Return frequencies as validated XEM quantities for typed consumers.
    pub fn frequency_quantities(&self) -> Result<Vec<sonium_xem::FrequencyHz>, BemError> {
        let mut frequencies: Vec<sonium_xem::FrequencyHz> =
            Vec::with_capacity(self.frequencies.len());
        for (index, &frequency) in self.frequencies.iter().enumerate() {
            let quantity = sonium_xem::FrequencyHz::try_new(frequency).map_err(|reason| {
                BemError::InvalidParameters(format!("frequency[{index}] {reason}"))
            })?;
            if let Some(previous) = frequencies.last()
                && quantity.get() <= previous.get()
            {
                return Err(BemError::InvalidParameters(
                    "frequency sweep must be strictly increasing".into(),
                ));
            }
            frequencies.push(quantity);
        }
        Ok(frequencies)
    }

    /// Create a config with linearly spaced frequencies
    pub fn linear(start_hz: f64, end_hz: f64, num_points: usize) -> Self {
        let step = if num_points > 1 {
            (end_hz - start_hz) / (num_points - 1) as f64
        } else {
            0.0
        };
        let frequencies = (0..num_points)
            .map(|i| start_hz + i as f64 * step)
            .collect();
        Self {
            frequencies,
            ..Default::default()
        }
    }

    /// Create a config with logarithmically spaced frequencies
    pub fn logarithmic(start_hz: f64, end_hz: f64, num_points: usize) -> Self {
        let log_start = start_hz.ln();
        let log_end = end_hz.ln();
        let step = if num_points > 1 {
            (log_end - log_start) / (num_points - 1) as f64
        } else {
            0.0
        };
        let frequencies = (0..num_points)
            .map(|i| (log_start + i as f64 * step).exp())
            .collect();
        Self {
            frequencies,
            ..Default::default()
        }
    }
}

// ============================================================================
// Result
// ============================================================================

/// Result of a frequency sweep
pub struct FrequencySweepResult {
    /// Solutions at each frequency
    pub solutions: Vec<Vec<Complex64>>,
    /// Frequencies solved (Hz)
    pub frequencies: Vec<f64>,
    /// Iteration counts at each frequency
    pub iteration_counts: Vec<usize>,
    /// Whether each frequency converged
    pub converged: Vec<bool>,
    /// Exact unpreconditioned relative residual of the assembled operator at
    /// each frequency. This is the post-solve acceptance evidence, not the
    /// Krylov residual estimate.
    pub exact_relative_residuals: Vec<f64>,
    /// Maximum compressed-vs-dense relative action error at each frequency.
    /// Dense sweeps retain `0.0`; H-matrix sweeps certify both the physical RHS
    /// and the solved density before accepting the frequency point.
    pub operator_accuracy_errors: Vec<f64>,
    /// Whether the preceding frequency passed the configured quantitative
    /// bound for retaining a recycled GCRO-DR subspace. The first point is
    /// always `false`; this records eligibility, not a claim that the solver
    /// retained a non-empty subspace.
    pub recycled_subspace_eligible: Vec<bool>,
    /// Total solve time in seconds
    pub total_time_seconds: f64,
    /// Assembly time per frequency in seconds
    pub assembly_times: Vec<f64>,
    /// Solve time per frequency in seconds
    pub solve_times: Vec<f64>,
}

impl FrequencySweepResult {
    /// Total number of iterations across all frequencies
    pub fn total_iterations(&self) -> usize {
        self.iteration_counts.iter().sum()
    }

    /// Average iterations per frequency
    pub fn average_iterations(&self) -> f64 {
        if self.iteration_counts.is_empty() {
            return 0.0;
        }
        self.total_iterations() as f64 / self.iteration_counts.len() as f64
    }

    /// Number of frequencies that converged
    pub fn num_converged(&self) -> usize {
        self.converged.iter().filter(|&&c| c).count()
    }

    /// Validate the complete sweep artifact before an application exports or
    /// compares it. This guards against stale per-frequency evidence being
    /// paired with a different solution grid.
    pub fn validate(&self) -> Result<(), BemError> {
        let count = self.frequencies.len();
        for (field, actual) in [
            ("solutions", self.solutions.len()),
            ("iteration_counts", self.iteration_counts.len()),
            ("converged", self.converged.len()),
            (
                "exact_relative_residuals",
                self.exact_relative_residuals.len(),
            ),
            (
                "operator_accuracy_errors",
                self.operator_accuracy_errors.len(),
            ),
            (
                "recycled_subspace_eligible",
                self.recycled_subspace_eligible.len(),
            ),
            ("assembly_times", self.assembly_times.len()),
            ("solve_times", self.solve_times.len()),
        ] {
            if actual != count {
                return Err(BemError::InvalidParameters(format!(
                    "frequency sweep {field}/frequency length mismatch: {actual} entries for {count} frequencies"
                )));
            }
        }
        if self.frequencies.is_empty()
            || self
                .frequencies
                .windows(2)
                .any(|window| window[1] <= window[0])
            || self
                .frequencies
                .iter()
                .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
        {
            return Err(BemError::InvalidParameters(
                "frequency sweep frequencies must be finite, positive, and strictly increasing"
                    .to_string(),
            ));
        }
        if !self.total_time_seconds.is_finite()
            || self.total_time_seconds < 0.0
            || self
                .assembly_times
                .iter()
                .chain(&self.solve_times)
                .any(|time| !time.is_finite() || *time < 0.0)
            || self
                .exact_relative_residuals
                .iter()
                .chain(&self.operator_accuracy_errors)
                .any(|residual| !residual.is_finite() || *residual < 0.0)
            || self.solutions.iter().any(|solution| {
                solution
                    .iter()
                    .any(|value| !value.re.is_finite() || !value.im.is_finite())
            })
        {
            return Err(BemError::InvalidParameters(
                "frequency sweep result contains non-finite or negative evidence".to_string(),
            ));
        }
        Ok(())
    }
}

// ============================================================================
// Shared sweep-grid semantics for BEM-side sweep callers
// ============================================================================

/// Check that a sweep grid is finite, positive, and strictly increasing.
///
/// This is the grid half of the [`FrequencySweepResult::validate`] invariant,
/// factored out so BEM-side sweep dispatchers outside this module (coupled
/// product sweeps, speaker sweeps) accept exactly the grids the lifecycle
/// itself accepts. Empty grids are rejected here; callers whose edge
/// contract admits an empty sweep keep their own early return before calling.
pub fn validate_frequency_grid(frequencies: &[f64]) -> Result<(), BemError> {
    if frequencies.is_empty()
        || frequencies.windows(2).any(|window| window[1] <= window[0])
        || frequencies
            .iter()
            .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
    {
        return Err(BemError::InvalidParameters(
            "frequency sweep frequencies must be finite, positive, and strictly increasing"
                .to_string(),
        ));
    }
    Ok(())
}

/// One dispatched sweep point with its frequency-tagged solution.
pub struct FrequencySweepPoint<T> {
    /// Position of the point in the dispatched grid.
    pub index: usize,
    /// Frequency of the point in Hz.
    pub frequency_hz: f64,
    /// Caller-computed per-point value (tagged result, evidence, timings).
    pub value: T,
    /// Solution handed to the next point as the warm-start guess when
    /// `recycle_solution` is set. The driver retains ownership so a fast
    /// caller cannot alias the recycled cache.
    pub solution: Vec<Complex64>,
}

/// Dispatch a sequential frequency sweep, threading the warm-start cache.
///
/// The driver owns the previous solution and hands each point an optional
/// borrow of it; with `recycle_solution` disabled every point sees `None`.
/// Per-point assembly, solve, and post-processing stay in `solve_point` —
/// this only owns iteration order and the recycled-solution cache, i.e. the
/// dispatch half of the sweep lifecycle. The incident-field sweep above keeps
/// its own loop (GCRO-DR subspace retention and operator-accuracy gates are
/// not warm-start-only concerns); BEM-side callers with their own operators
/// dispatch through here so grid order and cache ownership match.
///
/// An empty grid dispatches nothing and returns an empty vector.
pub fn dispatch_sequential_sweep<T>(
    frequencies: &[f64],
    recycle_solution: bool,
    mut solve_point: impl FnMut(usize, f64, Option<&[Complex64]>) -> (Vec<Complex64>, T),
) -> Vec<FrequencySweepPoint<T>> {
    let mut previous: Option<Vec<Complex64>> = None;
    frequencies
        .iter()
        .copied()
        .enumerate()
        .map(|(index, frequency_hz)| {
            let guess = if recycle_solution {
                previous.as_deref()
            } else {
                None
            };
            let (solution, value) = solve_point(index, frequency_hz, guess);
            previous = Some(solution.clone());
            FrequencySweepPoint {
                index,
                frequency_hz,
                value,
                solution,
            }
        })
        .collect()
}

// ============================================================================
// Dense matrix operator (for GMRES with initial guess)
// ============================================================================

/// Wrapper around a dense matrix for use as a LinearOperator
struct DenseSweepOperator<'a> {
    matrix: &'a DenseMatrix<Complex64>,
}

impl<'a> LinearOperator<Complex64> for DenseSweepOperator<'a> {
    fn num_rows(&self) -> usize {
        self.matrix.nrows()
    }

    fn num_cols(&self) -> usize {
        self.matrix.ncols()
    }

    fn apply(&self, x: &[Complex64]) -> Vec<Complex64> {
        self.matrix.matvec(x)
    }

    fn apply_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        self.matrix.matvec_into(x, y);
    }

    fn apply_transpose(&self, x: &[Complex64]) -> Vec<Complex64> {
        let mut y = vec![Complex64::new(0.0, 0.0); self.matrix.ncols()];
        self.matrix.matvec_transpose_into(x, &mut y);
        y
    }

    fn apply_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        self.matrix.matvec_transpose_into(x, y);
    }

    fn transpose_capability(&self) -> sonium_solvers::traits::AdjointCapability {
        sonium_solvers::traits::AdjointCapability::Exact
    }

    fn hermitian_capability(&self) -> sonium_solvers::traits::AdjointCapability {
        sonium_solvers::traits::AdjointCapability::Exact
    }
}

/// Matrix-free exact stabilized-TBEM action used to certify a compressed
/// sweep without retaining a dense O(N²) reference matrix.
struct ExactTbemSweepOperator<'a> {
    elements: &'a [Element],
    nodes: &'a [[f64; 3]],
    physics: PhysicsParams,
    sign: f64,
    beta_scale: f64,
    num_dofs: usize,
}

impl LinearOperator<Complex64> for ExactTbemSweepOperator<'_> {
    fn num_rows(&self) -> usize {
        self.num_dofs
    }

    fn num_cols(&self) -> usize {
        self.num_dofs
    }

    fn apply(&self, x: &[Complex64]) -> Vec<Complex64> {
        // Construction validates the geometry and physics. The exact action
        // can therefore fail only if this private operator is misused with an
        // invalid vector shape, which the LinearOperator contract forbids.
        apply_tbem_stabilized_exact(
            self.elements,
            self.nodes,
            &self.physics,
            self.sign,
            self.beta_scale,
            x,
        )
        .expect("validated exact TBEM sweep reference must accept its input")
    }

    fn apply_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        assert_eq!(
            y.len(),
            self.num_rows(),
            "exact TBEM sweep reference output length mismatch"
        );
        y.copy_from_slice(&self.apply(x));
    }

    fn apply_transpose_into(&self, _x: &[Complex64], _y: &mut [Complex64]) {
        panic!("exact TBEM sweep reference is forward-only")
    }
}

enum SweepSystem<'a> {
    Dense(DenseMatrix<Complex64>, Vec<Complex64>),
    Hmatrix(Box<HMatrix>, ExactTbemSweepOperator<'a>, Vec<Complex64>),
    Slfmm(
        Box<SlfmmOperator>,
        ExactTbemSweepOperator<'a>,
        Vec<Complex64>,
    ),
    Mlfmm(
        Box<MlfmmOperator>,
        ExactTbemSweepOperator<'a>,
        Vec<Complex64>,
    ),
}

impl SweepSystem<'_> {
    fn rhs(&self) -> &[Complex64] {
        match self {
            SweepSystem::Dense(_, rhs)
            | SweepSystem::Hmatrix(_, _, rhs)
            | SweepSystem::Slfmm(_, _, rhs)
            | SweepSystem::Mlfmm(_, _, rhs) => rhs,
        }
    }
}

fn solve_sweep_operator<O: LinearOperator<Complex64>>(
    operator: &O,
    rhs: &[Complex64],
    initial_guess: Option<&[Complex64]>,
    gcro_dr_subspace: Option<&mut GcroDrSubspace<Complex64>>,
    gmres_config: &GmresConfig<f64>,
) -> GmresSolution<Complex64> {
    if let Some(sub) = gcro_dr_subspace {
        let cfg = GcroDrConfig::from_gmres(gmres_config);
        gcro_dr(operator, rhs, initial_guess, sub, &cfg)
    } else {
        gmres_with_guess(operator, rhs, initial_guess, gmres_config)
    }
}

// ============================================================================
// Frequency sweep solver
// ============================================================================

/// Solve the BEM system across multiple frequencies with solution recycling.
///
/// At each frequency:
/// 1. Create physics parameters for the current frequency
/// 2. Assemble the BEM system (matrix + RHS)
/// 3. Add incident field contribution to RHS
/// 4. Solve using GMRES with the previous solution as initial guess
///
/// The key optimization is **solution recycling**: for closely spaced frequencies,
/// the BEM solution changes slowly, so using x_{i-1} as the initial guess for x_i
/// causes the iterative solver to converge in very few iterations.
///
/// # Arguments
/// * `elements` - Prepared mesh elements with boundary conditions and DOF addresses
/// * `nodes` - Node coordinates
/// * `incident_field` - Incident field definition
/// * `config` - Frequency sweep configuration
/// * `assembly_method` - Which assembly method to use
/// * `speed_of_sound` - Speed of sound (m/s)
/// * `density` - Medium density (kg/m^3)
///
/// # Returns
/// `FrequencySweepResult` containing solutions at each frequency
///
/// This compatibility entry point uses the canonical exterior BEM convention.
/// Call [`frequency_sweep_with_physics`] when the source problem carries an
/// explicit harmonic convention or interior formulation.
pub fn frequency_sweep(
    elements: &[Element],
    nodes: &[[f64; 3]],
    incident_field: &IncidentField,
    config: &FrequencySweepConfig,
    assembly_method: AssemblyMethod,
    speed_of_sound: f64,
    density: f64,
) -> Result<FrequencySweepResult, BemError> {
    let physics_template = PhysicsParams::try_new(1.0, speed_of_sound, density, false)
        .map_err(|error| BemError::InvalidParameters(error.to_string()))?;
    frequency_sweep_with_physics(
        elements,
        nodes,
        incident_field,
        config,
        assembly_method,
        &physics_template,
    )
}

/// Solve a frequency sweep while preserving the validated medium, phasor, and
/// interior/exterior convention from `physics_template`.
pub fn frequency_sweep_with_physics(
    elements: &[Element],
    nodes: &[[f64; 3]],
    incident_field: &IncidentField,
    config: &FrequencySweepConfig,
    assembly_method: AssemblyMethod,
    physics_template: &PhysicsParams,
) -> Result<FrequencySweepResult, BemError> {
    frequency_sweep_with_physics_and_admittances(
        elements,
        nodes,
        incident_field,
        config,
        assembly_method,
        physics_template,
        None,
    )
}

/// Solve a frequency sweep while replacing impedance-boundary admittances
/// with one checked value per frequency.
///
/// The ordinary sweep API keeps the historical element boundary conditions
/// unchanged.  Frequency-dependent surface models (for example layered
/// porous or thin-shell impedances) use this variant so that each assembled
/// operator receives the model value at its own frequency rather than the
/// value prepared for the first frequency in the sweep.  The slice must be
/// either absent or have exactly one finite value per configured frequency.
pub fn frequency_sweep_with_physics_and_admittances(
    elements: &[Element],
    nodes: &[[f64; 3]],
    incident_field: &IncidentField,
    config: &FrequencySweepConfig,
    assembly_method: AssemblyMethod,
    physics_template: &PhysicsParams,
    surface_admittances: Option<&[Complex64]>,
) -> Result<FrequencySweepResult, BemError> {
    config.validate()?;
    physics_template
        .validate()
        .map_err(|error| BemError::InvalidParameters(error.to_string()))?;
    if let Some(admittances) = surface_admittances {
        if admittances.len() != config.frequencies.len() {
            return Err(BemError::InvalidParameters(format!(
                "frequency-sweep admittance count {} does not match frequency count {}",
                admittances.len(),
                config.frequencies.len()
            )));
        }
        if admittances
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(BemError::InvalidParameters(
                "frequency-sweep admittances must be finite".to_string(),
            ));
        }
    }
    let total_start = Instant::now();
    let num_freqs = config.frequencies.len();

    let mut solutions = Vec::with_capacity(num_freqs);
    let mut iteration_counts = Vec::with_capacity(num_freqs);
    let mut converged_flags = Vec::with_capacity(num_freqs);
    let mut exact_relative_residuals = Vec::with_capacity(num_freqs);
    let mut operator_accuracy_errors = Vec::with_capacity(num_freqs);
    let mut recycled_subspace_eligible = Vec::with_capacity(num_freqs);
    let mut assembly_times = Vec::with_capacity(num_freqs);
    let mut solve_times = Vec::with_capacity(num_freqs);

    let mut previous_solution: Option<Vec<Complex64>> = None;
    let mut previous_frequency: Option<f64> = None;
    let mut gcro_dr_subspace = (config.recycle_subspace_size > 0)
        .then(|| GcroDrSubspace::<Complex64>::new(config.recycle_subspace_size));

    for (freq_idx, &frequency) in config.frequencies.iter().enumerate() {
        let can_recycle_subspace = previous_frequency.is_some_and(|previous| {
            relative_frequency_delta(previous, frequency)
                <= config.max_recycle_relative_frequency_delta
        });
        if !can_recycle_subspace {
            gcro_dr_subspace = (config.recycle_subspace_size > 0)
                .then(|| GcroDrSubspace::<Complex64>::new(config.recycle_subspace_size));
        }
        // Create physics at this frequency
        let mut physics = PhysicsParams::try_new(
            frequency,
            physics_template.speed_of_sound,
            physics_template.density,
            physics_template.tau < 0.0,
        )
        .map_err(|error| BemError::InvalidParameters(error.to_string()))?;
        physics.harmonic_factor = physics_template.harmonic_factor;
        physics.pressure_factor = physics.density * physics.omega * physics.harmonic_factor;
        physics
            .validate()
            .map_err(|error| BemError::InvalidParameters(error.to_string()))?;

        // Assemble the BEM system
        let assembly_start = Instant::now();
        // A frequency-dependent impedance must be visible to every assembly
        // backend.  Keep the legacy borrow-only path allocation-free when no
        // override is requested; otherwise clone only the small element
        // metadata and replace the local admittance before assembling.
        let frequency_elements;
        let elements_at_frequency = if let Some(admittances) = surface_admittances {
            frequency_elements = elements_with_surface_admittance(elements, admittances[freq_idx])?;
            frequency_elements.as_slice()
        } else {
            elements
        };
        let system = assemble_at_frequency(
            elements_at_frequency,
            nodes,
            &physics,
            assembly_method,
            config.beta_scale,
        )?;
        let assembly_elapsed = assembly_start.elapsed().as_secs_f64();
        assembly_times.push(assembly_elapsed);

        // Add incident field RHS contribution
        let beta = physics.burton_miller_beta_scaled(config.beta_scale);
        let gamma = Complex64::new(physics.gamma(), 0.0);
        let tau = Complex64::new(physics.tau, 0.0);
        let rhs = add_incident_field_rhs(
            system.rhs().to_vec(),
            elements,
            nodes,
            incident_field,
            &physics,
            gamma,
            tau,
            beta,
        );

        // Solve using GMRES with initial guess
        let solve_start = Instant::now();

        let gmres_config = GmresConfig {
            max_iterations: config.max_iterations,
            restart: config.gmres_restart,
            tolerance: config.tolerance,
            print_interval: 0,
            convergence_check_interval: 1,
            // Frequency sweeps are a production solve path, so estimated
            // Krylov residuals are not sufficient to certify convergence.
            // Check the assembled operator on every iteration and repeat the
            // same check after the solve below.
            exact_residual_check_interval: 1,
            min_restart: None,
            max_restart: None,
        };

        let initial_guess = if config.recycle_solution {
            previous_solution.as_deref()
        } else {
            None
        };

        let (result, exact_residual, operator_accuracy_error) = match &system {
            SweepSystem::Dense(matrix, _) => {
                let operator = DenseSweepOperator { matrix };
                let result = solve_sweep_operator(
                    &operator,
                    &rhs,
                    initial_guess,
                    gcro_dr_subspace.as_mut(),
                    &gmres_config,
                );
                let exact_residual = exact_relative_residual(&operator, &result.x, &rhs);
                (result, exact_residual, 0.0)
            }
            SweepSystem::Hmatrix(hmatrix, reference, _) => {
                let result = solve_sweep_operator(
                    hmatrix.as_ref(),
                    &rhs,
                    initial_guess,
                    gcro_dr_subspace.as_mut(),
                    &gmres_config,
                );
                let report = require_sweep_operator_accuracy(
                    format!("H-matrix frequency sweep at {frequency:.3} Hz"),
                    hmatrix.as_ref(),
                    reference,
                    &[rhs.clone(), result.x.clone()],
                    hmatrix_action_tolerance(config.tolerance),
                )?;
                let exact_residual = exact_relative_residual(reference, &result.x, &rhs);
                (result, exact_residual, report.max_relative_error)
            }
            SweepSystem::Slfmm(operator, reference, _) => {
                let result = solve_sweep_operator(
                    operator.as_ref(),
                    &rhs,
                    initial_guess,
                    gcro_dr_subspace.as_mut(),
                    &gmres_config,
                );
                let report = require_sweep_operator_accuracy(
                    format!("SLFMM frequency sweep at {frequency:.3} Hz"),
                    operator.as_ref(),
                    reference,
                    &[rhs.clone(), result.x.clone()],
                    hmatrix_action_tolerance(config.tolerance),
                )?;
                let exact_residual = exact_relative_residual(reference, &result.x, &rhs);
                (result, exact_residual, report.max_relative_error)
            }
            SweepSystem::Mlfmm(operator, reference, _) => {
                let result = solve_sweep_operator(
                    operator.as_ref(),
                    &rhs,
                    initial_guess,
                    gcro_dr_subspace.as_mut(),
                    &gmres_config,
                );
                let report = require_sweep_operator_accuracy(
                    format!("MLFMM frequency sweep at {frequency:.3} Hz"),
                    operator.as_ref(),
                    reference,
                    &[rhs.clone(), result.x.clone()],
                    hmatrix_action_tolerance(config.tolerance),
                )?;
                let exact_residual = exact_relative_residual(reference, &result.x, &rhs);
                (result, exact_residual, report.max_relative_error)
            }
        };

        let residual_tolerance = (config.tolerance * 20.0).max(1e-10);
        if !exact_residual.is_finite() || exact_residual > residual_tolerance {
            return Err(BemError::ResidualCheckFailed {
                label: format!("frequency sweep at {frequency:.3} Hz"),
                residual: exact_residual,
                tolerance: residual_tolerance,
            });
        }

        let solve_elapsed = solve_start.elapsed().as_secs_f64();
        solve_times.push(solve_elapsed);

        log::info!(
            "Frequency sweep [{}/{}]: f = {:.1} Hz, {} iterations, converged = {}, exact residual = {:.3e}, solve = {:.3}s",
            freq_idx + 1,
            num_freqs,
            frequency,
            result.iterations,
            result.converged,
            exact_residual,
            solve_elapsed
        );

        converged_flags.push(result.converged);
        exact_relative_residuals.push(exact_residual);
        operator_accuracy_errors.push(operator_accuracy_error);
        recycled_subspace_eligible.push(can_recycle_subspace);
        iteration_counts.push(result.iterations);

        // Store solution and recycle for next frequency
        let solution = result.x;
        previous_solution = Some(solution.clone());
        previous_frequency = Some(frequency);
        solutions.push(solution);
    }

    let total_elapsed = total_start.elapsed().as_secs_f64();

    Ok(FrequencySweepResult {
        solutions,
        frequencies: config.frequencies.clone(),
        iteration_counts,
        converged: converged_flags,
        exact_relative_residuals,
        operator_accuracy_errors,
        recycled_subspace_eligible,
        total_time_seconds: total_elapsed,
        assembly_times,
        solve_times,
    })
}

/// Compute the unpreconditioned relative residual of a sweep result.
///
/// This deliberately lives outside the Krylov implementation: it certifies
/// the concrete assembled operator selected by the sweep, including the
/// compressed H-matrix path.
fn exact_relative_residual<O: LinearOperator<Complex64>>(
    operator: &O,
    solution: &[Complex64],
    rhs: &[Complex64],
) -> f64 {
    if solution.len() != operator.num_cols() || rhs.len() != operator.num_rows() {
        return f64::NAN;
    }
    let applied = operator.apply(solution);
    if applied.len() != rhs.len() {
        return f64::NAN;
    }
    let residual_squared = applied
        .iter()
        .zip(rhs)
        .map(|(actual, expected)| (*actual - *expected).norm_sqr())
        .sum::<f64>();
    let rhs_squared = rhs.iter().map(|value| value.norm_sqr()).sum::<f64>();
    residual_squared.sqrt() / rhs_squared.sqrt().max(f64::MIN_POSITIVE)
}

/// Keep the compressed-operator action budget materially tighter than the
/// accepted solve residual, while avoiding a false rejection below the stable
/// double-precision floor of the independently assembled reference.
fn hmatrix_action_tolerance(solve_tolerance: f64) -> f64 {
    (solve_tolerance * 10.0).max(1e-10)
}

fn require_sweep_operator_accuracy<A, R>(
    label: String,
    approximate: &A,
    reference: &R,
    probes: &[Vec<Complex64>],
    tolerance: f64,
) -> Result<OperatorAccuracyReport, BemError>
where
    A: LinearOperator<Complex64>,
    R: LinearOperator<Complex64>,
{
    require_operator_accuracy(approximate, reference, probes, tolerance).map_err(|error| {
        BemError::OperatorAccuracyFailed {
            label,
            reason: error.to_string(),
        }
    })
}

/// Assemble the BEM system at a specific frequency, returning (matrix, rhs).
fn assemble_at_frequency<'a>(
    elements: &'a [Element],
    nodes: &'a [[f64; 3]],
    physics: &PhysicsParams,
    assembly_method: AssemblyMethod,
    beta_scale: f64,
) -> Result<SweepSystem<'a>, BemError> {
    let n_dofs = elements
        .iter()
        .filter(|e| !e.property.is_evaluation())
        .map(|e| e.dof_addresses.len())
        .sum::<usize>();
    match assembly_method.resolve(n_dofs) {
        AssemblyMethod::Auto => unreachable!("resolve() never returns Auto"),
        AssemblyMethod::Tbem => {
            let sign = default_sweep_sign(elements, physics);
            let system = build_tbem_system_stabilized(elements, nodes, physics, sign, beta_scale);
            Ok(SweepSystem::Dense(system.matrix, system.rhs))
        }
        AssemblyMethod::Hmatrix => {
            let sign = default_sweep_sign(elements, physics);
            let hmatrix = build_hmatrix_at_frequency(elements, nodes, physics, sign, beta_scale);
            let system = build_tbem_system_stabilized(elements, nodes, physics, sign, beta_scale);
            let reference = ExactTbemSweepOperator {
                elements,
                nodes,
                physics: physics.clone(),
                sign,
                beta_scale,
                num_dofs: system.matrix.nrows(),
            };
            Ok(SweepSystem::Hmatrix(
                Box::new(hmatrix),
                reference,
                system.rhs,
            ))
        }
        AssemblyMethod::Slfmm => {
            let clusters = build_octree_clusters(elements);
            let levels = build_cluster_tree(elements, 32, physics);
            let root = levels.first().ok_or_else(|| {
                BemError::InvalidParameters("SLFMM frequency sweep has no cluster level".into())
            })?;
            let system = build_slfmm_system_stabilized(
                elements,
                nodes,
                &clusters,
                physics,
                root.theta_points,
                root.phi_points,
                root.expansion_terms,
                default_sweep_sign(elements, physics),
                beta_scale,
            );
            let rhs = system.rhs.clone();
            let reference = ExactTbemSweepOperator {
                elements,
                nodes,
                physics: physics.clone(),
                sign: default_sweep_sign(elements, physics),
                beta_scale,
                num_dofs: system.num_dofs,
            };
            Ok(SweepSystem::Slfmm(
                Box::new(SlfmmOperator::new(system)),
                reference,
                rhs,
            ))
        }
        AssemblyMethod::Mlfmm | AssemblyMethod::DirectionalMlfmm => {
            let sign = default_sweep_sign(elements, physics);
            let cluster_levels = build_cluster_tree(elements, 32, physics);
            if cluster_levels.is_empty() {
                return Err(BemError::InvalidParameters(
                    "MLFMM frequency sweep has no cluster levels".into(),
                ));
            }
            let system = if assembly_method == AssemblyMethod::DirectionalMlfmm {
                build_directional_mlfmm_system_stabilized_checked(
                    elements,
                    nodes,
                    cluster_levels,
                    physics,
                    sign,
                    beta_scale,
                    Default::default(),
                )
            } else {
                build_mlfmm_system_stabilized_checked(
                    elements,
                    nodes,
                    cluster_levels,
                    physics,
                    sign,
                    beta_scale,
                )
            }
            .map_err(|error| {
                BemError::InvalidParameters(format!(
                    "MLFMM constant-panel admission rejected: {error}"
                ))
            })?;
            let rhs = system.rhs.clone();
            let reference = ExactTbemSweepOperator {
                elements,
                nodes,
                physics: physics.clone(),
                sign,
                beta_scale,
                num_dofs: system.num_dofs,
            };
            Ok(SweepSystem::Mlfmm(
                Box::new(MlfmmOperator::new(system)),
                reference,
                rhs,
            ))
        }
    }
}

/// Clone element metadata while replacing local impedance admittances.
///
/// The sweep keeps the source slice borrowed by the assembled operator.  A
/// per-frequency clone is therefore required for frequency-dependent
/// boundaries; replacing only `VelocityWithAdmittance` leaves prescribed
/// velocity/pressure and evaluation elements untouched.  Requiring at least
/// one replacement prevents a caller from accidentally supplying an
/// admittance override to a non-impedance problem and silently getting no-op
/// behavior.
fn elements_with_surface_admittance(
    elements: &[Element],
    admittance: Complex64,
) -> Result<Vec<Element>, BemError> {
    let mut updated = elements.to_vec();
    let mut replaced = 0usize;
    for element in &mut updated {
        if let BoundaryCondition::VelocityWithAdmittance {
            admittance: current,
            ..
        } = &mut element.boundary_condition
        {
            *current = admittance;
            replaced += 1;
        }
    }
    if replaced == 0 {
        return Err(BemError::InvalidParameters(
            "frequency-sweep admittance override requires impedance elements".to_string(),
        ));
    }
    Ok(updated)
}

/// Build the flat cluster representation consumed by the single-level FMM.
///
/// H-matrix and SLFMM intentionally share the same octree partition and
/// near/far interaction policy. Keeping this construction in one helper makes
/// frequency-sweep backend comparisons meaningful instead of comparing
/// different clusterizations.
fn build_octree_clusters(elements: &[Element]) -> Vec<Cluster> {
    let centers: Vec<[f64; 3]> = elements.iter().map(|element| element.center).collect();
    let mut octree = Octree::build(&centers, 32, 8);
    octree.compute_interaction_lists(1.2);

    let leaf_indices = octree.leaves();
    let mut node_to_cluster = std::collections::HashMap::new();
    for (cluster_idx, &node_idx) in leaf_indices.iter().enumerate() {
        node_to_cluster.insert(node_idx, cluster_idx);
    }

    leaf_indices
        .iter()
        .map(|&node_idx| {
            let node = &octree.nodes[node_idx];
            let mut cluster = Cluster::new(node.center);
            cluster.element_indices = node.element_indices.clone();
            cluster.num_elements = node.element_indices.len();
            cluster.radius = node.radius();
            cluster.near_clusters = node
                .near_clusters
                .iter()
                .filter_map(|neighbor| node_to_cluster.get(neighbor).copied())
                .collect();
            cluster.far_clusters = node
                .far_clusters
                .iter()
                .filter_map(|neighbor| node_to_cluster.get(neighbor).copied())
                .collect();
            cluster
        })
        .collect()
}

fn build_hmatrix_at_frequency(
    elements: &[Element],
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
) -> HMatrix {
    let clusters = build_octree_clusters(elements);

    let aca_tolerance = 1e-4;
    let max_rank = elements.len() / 4 + 1;
    build_hmatrix_stabilized(
        elements,
        nodes,
        &clusters,
        physics,
        aca_tolerance,
        max_rank,
        sign,
        beta_scale,
    )
}

fn default_sweep_sign(elements: &[Element], physics: &PhysicsParams) -> f64 {
    let mut min_coord = [f64::MAX, f64::MAX, f64::MAX];
    let mut max_coord = [f64::MIN, f64::MIN, f64::MIN];
    for element in elements.iter() {
        if element.property.is_evaluation() {
            continue;
        }
        for d in 0..3 {
            min_coord[d] = min_coord[d].min(element.center[d]);
            max_coord[d] = max_coord[d].max(element.center[d]);
        }
    }
    let characteristic_radius = 0.5
        * (max_coord[0] - min_coord[0])
            .max(max_coord[1] - min_coord[1])
            .max(max_coord[2] - min_coord[2]);
    let ka = physics.wave_number * characteristic_radius;
    PhysicsParams::optimal_formulation_sign(ka)
}

/// Add incident field contribution to the RHS for constant elements.
fn add_incident_field_rhs(
    mut rhs: Vec<Complex64>,
    elements: &[Element],
    _nodes: &[[f64; 3]],
    incident_field: &IncidentField,
    physics: &PhysicsParams,
    _gamma: Complex64,
    _tau: Complex64,
    beta: Complex64,
) -> Vec<Complex64> {
    let centers: Vec<[f64; 3]> = elements.iter().map(|e| e.center).collect();
    let normals: Vec<[f64; 3]> = elements.iter().map(|e| e.normal).collect();

    let incident_rhs = incident_field.compute_rhs_with_beta(&centers, &normals, physics, beta);
    for (r, inc) in rhs.iter_mut().zip(incident_rhs.iter()) {
        *r += *inc;
    }
    rhs
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{BoundaryCondition, ElementProperty, ElementType};

    #[test]
    fn grid_validation_accepts_strictly_increasing_positive_grids() {
        validate_frequency_grid(&[20.0, 100.0, 120.0]).unwrap();
        validate_frequency_grid(&[100.0]).unwrap();
    }

    #[test]
    fn grid_validation_rejects_empty_nonfinite_nonpositive_and_flat_grids() {
        for grid in [
            vec![],
            vec![100.0, f64::NAN],
            vec![100.0, f64::INFINITY],
            vec![100.0, 0.0],
            vec![100.0, -50.0],
            vec![120.0, 120.0],
            vec![200.0, 100.0],
        ] {
            assert!(
                validate_frequency_grid(&grid).is_err(),
                "grid {grid:?} must be rejected"
            );
        }
    }

    #[test]
    fn sequential_dispatch_threads_the_recycled_cache_in_order() {
        let seen_guesses: std::cell::RefCell<Vec<Option<Vec<Complex64>>>> =
            std::cell::RefCell::new(Vec::new());
        let points =
            dispatch_sequential_sweep(&[100.0, 120.0, 140.0], true, |index, freq, guess| {
                seen_guesses.borrow_mut().push(guess.map(|g| g.to_vec()));
                let solution = vec![Complex64::new(freq, index as f64)];
                (solution, index * 10)
            });
        assert_eq!(points.len(), 3);
        // First point sees no cache; later points see the previous solution.
        let guesses = seen_guesses.borrow();
        assert_eq!(guesses[0], None);
        assert_eq!(guesses[1], Some(vec![Complex64::new(100.0, 0.0)]));
        assert_eq!(guesses[2], Some(vec![Complex64::new(120.0, 1.0)]));
        // Points keep index/frequency tags, caller values, and owned solutions.
        assert_eq!(points[2].index, 2);
        assert_eq!(points[2].frequency_hz, 140.0);
        assert_eq!(points[2].value, 20);
        assert_eq!(points[2].solution, vec![Complex64::new(140.0, 2.0)]);
    }

    #[test]
    fn sequential_dispatch_without_recycling_never_shares_solutions() {
        let points = dispatch_sequential_sweep(&[100.0, 120.0], false, |_, freq, guess| {
            assert!(guess.is_none(), "recycling disabled must hide the cache");
            (vec![Complex64::new(freq, 0.0)], ())
        });
        assert_eq!(points.len(), 2);
        assert_eq!(points[0].solution, vec![Complex64::new(100.0, 0.0)]);
    }

    #[test]
    fn sequential_dispatch_of_an_empty_grid_dispatches_nothing() {
        let points: Vec<FrequencySweepPoint<()>> =
            dispatch_sequential_sweep(&[], true, |_, _, _| unreachable!());
        assert!(points.is_empty());
    }

    fn make_sweep_test_elements() -> (Vec<Element>, Vec<[f64; 3]>) {
        let nodes: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.5, 1.0, 0.0],
        ];

        let elem0 = Element {
            connectivity: vec![0, 1, 2],
            element_type: ElementType::Tri3,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0; 3]; 3],
            center: [0.5, 0.33, 0.0],
            area: 0.5,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses: vec![0],
        };

        let elem1 = Element {
            connectivity: vec![1, 3, 2],
            element_type: ElementType::Tri3,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0; 3]; 3],
            center: [1.0, 0.67, 0.0],
            area: 0.5,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses: vec![1],
        };

        (vec![elem0, elem1], nodes)
    }

    struct DiagonalSweepOperator;

    impl LinearOperator<Complex64> for DiagonalSweepOperator {
        fn num_rows(&self) -> usize {
            2
        }

        fn num_cols(&self) -> usize {
            2
        }

        fn apply_into(&self, x: &[Complex64], y: &mut [Complex64]) {
            y[0] = Complex64::new(2.0, 0.0) * x[0];
            y[1] = Complex64::new(3.0, 0.0) * x[1];
        }

        fn apply_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
            self.apply_into(x, y);
        }
    }

    struct IdentitySweepOperator;

    impl LinearOperator<Complex64> for IdentitySweepOperator {
        fn num_rows(&self) -> usize {
            2
        }

        fn num_cols(&self) -> usize {
            2
        }

        fn apply_into(&self, x: &[Complex64], y: &mut [Complex64]) {
            y.copy_from_slice(x);
        }

        fn apply_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
            self.apply_into(x, y);
        }
    }

    #[test]
    fn sweep_operator_accuracy_failure_maps_to_typed_bem_error() {
        let error = require_sweep_operator_accuracy(
            "test compressed operator".to_owned(),
            &DiagonalSweepOperator,
            &IdentitySweepOperator,
            &[vec![Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)]],
            1e-12,
        )
        .expect_err("inaccurate operator must be rejected");

        assert!(matches!(
            error,
            BemError::OperatorAccuracyFailed { label, reason }
                if label == "test compressed operator" && reason.contains("accuracy")
        ));
    }

    #[test]
    fn hmatrix_sweep_operator_accuracy_failure_maps_to_typed_bem_error() {
        use crate::core::assembly::hmatrix::{
            DenseBlock, HMatrixPartition, HMatrixTransposePartition,
        };

        let approximate = HMatrix {
            num_dofs: 1,
            dense_blocks: vec![DenseBlock {
                source_cluster: 0,
                field_cluster: 0,
                matrix: DenseMatrix::from_row_major_vec(1, 1, vec![Complex64::new(2.0, 0.0)]),
            }],
            low_rank_blocks: Vec::new(),
            cluster_dof_indices: vec![vec![0]],
            partitions: vec![HMatrixPartition {
                cluster_indices: vec![0],
                dense_start: 0,
                dense_end: 1,
                low_rank_start: 0,
                low_rank_end: 0,
                num_dofs: 1,
            }],
            transpose_dense_order: vec![0],
            transpose_low_rank_order: Vec::new(),
            transpose_partitions: vec![HMatrixTransposePartition {
                cluster_indices: vec![0],
                dense_start: 0,
                dense_end: 1,
                low_rank_start: 0,
                low_rank_end: 0,
                num_dofs: 1,
            }],
            memory_bytes: 16,
            compression_ratio: 1.0,
        };
        let error = require_sweep_operator_accuracy(
            "H-matrix impossible budget".to_owned(),
            &approximate,
            &IdentitySweepOperator,
            &[vec![Complex64::new(1.0, 0.0)]],
            1.0e-12,
        )
        .expect_err("H-matrix action error must reject impossible budget");
        assert!(matches!(
            error,
            BemError::OperatorAccuracyFailed { label, .. }
                if label == "H-matrix impossible budget"
        ));
    }

    #[test]
    fn test_exact_relative_residual_checks_the_unpreconditioned_operator() {
        let operator = DiagonalSweepOperator;
        let rhs = [Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)];
        let exact = [Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)];
        let bad = [Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)];

        assert!(exact_relative_residual(&operator, &exact, &rhs) < 1e-14);
        assert!(exact_relative_residual(&operator, &bad, &rhs) > 0.9);
        assert!(exact_relative_residual(&operator, &[exact[0]], &rhs).is_nan());
    }

    #[test]
    fn test_frequency_sweep_config_linear() {
        let config = FrequencySweepConfig::linear(100.0, 1000.0, 10);
        assert_eq!(config.frequencies.len(), 10);
        assert!((config.frequencies[0] - 100.0).abs() < 1e-10);
        assert!((config.frequencies[9] - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_frequency_sweep_config_logarithmic() {
        let config = FrequencySweepConfig::logarithmic(100.0, 10000.0, 5);
        assert_eq!(config.frequencies.len(), 5);
        assert!((config.frequencies[0] - 100.0).abs() < 1e-10);
        assert!((config.frequencies[4] - 10000.0).abs() < 1e-6);
        // Check logarithmic spacing: ratios between consecutive frequencies should be equal
        let ratio1 = config.frequencies[1] / config.frequencies[0];
        let ratio2 = config.frequencies[2] / config.frequencies[1];
        assert!(
            (ratio1 - ratio2).abs() < 1e-10,
            "Logarithmic spacing should have equal ratios"
        );
    }

    #[test]
    fn test_frequency_sweep_config_validates_units_and_controls() {
        let config = FrequencySweepConfig::linear(100.0, 1000.0, 3);
        let quantities = config
            .frequency_quantities()
            .expect("positive increasing frequencies are valid");
        assert_eq!(quantities[0], sonium_xem::FrequencyHz::new(100.0));

        let mut invalid = config;
        invalid.frequencies[1] = f64::NAN;
        assert!(matches!(
            invalid.frequency_quantities(),
            Err(BemError::InvalidParameters(_))
        ));
        invalid = FrequencySweepConfig::linear(100.0, 1000.0, 3);
        invalid.gmres_restart = 0;
        assert!(matches!(
            invalid.validate(),
            Err(BemError::InvalidParameters(_))
        ));
        invalid = FrequencySweepConfig::linear(100.0, 1000.0, 3);
        invalid.max_recycle_relative_frequency_delta = -1.0;
        assert!(matches!(
            invalid.validate(),
            Err(BemError::InvalidParameters(_))
        ));
    }

    #[test]
    fn test_assemble_at_frequency() {
        let (elements, nodes) = make_sweep_test_elements();
        let physics = PhysicsParams::new(500.0, 343.0, 1.21, false);

        let result = assemble_at_frequency(&elements, &nodes, &physics, AssemblyMethod::Tbem, 4.0);
        assert!(result.is_ok());
        match result.unwrap() {
            SweepSystem::Dense(matrix, rhs) => {
                assert_eq!(matrix.nrows(), 2);
                assert_eq!(matrix.ncols(), 2);
                assert_eq!(rhs.len(), 2);
            }
            _ => panic!("TBEM assembly should return dense system"),
        }
    }

    #[test]
    fn test_frequency_sweep_result_stats() {
        let result = FrequencySweepResult {
            solutions: vec![vec![Complex64::new(0.0, 0.0); 2]; 3],
            frequencies: vec![100.0, 200.0, 300.0],
            iteration_counts: vec![50, 10, 8],
            converged: vec![true, true, true],
            exact_relative_residuals: vec![1.0e-12, 2.0e-12, 3.0e-12],
            operator_accuracy_errors: vec![0.0, 0.0, 0.0],
            recycled_subspace_eligible: vec![false, true, true],
            total_time_seconds: 1.0,
            assembly_times: vec![0.3, 0.3, 0.3],
            solve_times: vec![0.1, 0.05, 0.04],
        };

        assert_eq!(result.total_iterations(), 68);
        assert!((result.average_iterations() - 68.0 / 3.0).abs() < 1e-10);
        assert_eq!(result.num_converged(), 3);
        result.validate().unwrap();

        let mut stale = result;
        stale.exact_relative_residuals.pop();
        assert!(stale.validate().is_err());
    }

    #[test]
    fn test_frequency_sweep_basic() {
        let (elements, nodes) = make_sweep_test_elements();
        let incident_field = IncidentField::plane_wave_z();
        let config = FrequencySweepConfig {
            frequencies: vec![100.0, 200.0],
            tolerance: 1e-6,
            max_iterations: 200,
            recycle_solution: true,
            gmres_restart: 30,
            beta_scale: 4.0,
            recycle_subspace_size: 0,
            max_recycle_relative_frequency_delta: 0.1,
        };

        let result = frequency_sweep(
            &elements,
            &nodes,
            &incident_field,
            &config,
            AssemblyMethod::Tbem,
            343.0,
            1.21,
        );

        assert!(result.is_ok());
        let sweep_result = result.unwrap();
        assert_eq!(sweep_result.solutions.len(), 2);
        assert_eq!(sweep_result.iteration_counts.len(), 2);
        assert!(sweep_result.converged.iter().all(|&converged| converged));

        // Second frequency should typically need fewer iterations when recycling
        // (though for this tiny problem it might converge quickly anyway)
        for sol in &sweep_result.solutions {
            assert_eq!(sol.len(), 2, "Each solution should have 2 DOFs");
        }
    }

    #[test]
    fn test_frequency_sweep_no_recycling() {
        let (elements, nodes) = make_sweep_test_elements();
        let incident_field = IncidentField::plane_wave_z();
        let config = FrequencySweepConfig {
            frequencies: vec![100.0, 200.0],
            tolerance: 1e-6,
            max_iterations: 200,
            recycle_solution: false,
            gmres_restart: 30,
            beta_scale: 4.0,
            recycle_subspace_size: 0,
            max_recycle_relative_frequency_delta: 0.1,
        };

        let result = frequency_sweep(
            &elements,
            &nodes,
            &incident_field,
            &config,
            AssemblyMethod::Tbem,
            343.0,
            1.21,
        );

        assert!(result.is_ok());
        let sweep_result = result.unwrap();
        assert_eq!(sweep_result.solutions.len(), 2);
    }

    #[test]
    fn test_frequency_sweep_gcro_dr_matches_legacy() {
        // Same problem solved two ways must yield close solutions. With the
        // toy 2-element problem the iteration counts are too small to show
        // the recycling win, but we can validate correctness.
        let (elements, nodes) = make_sweep_test_elements();
        let incident_field = IncidentField::plane_wave_z();
        let frequencies = vec![100.0, 200.0, 300.0, 400.0];
        let common = || FrequencySweepConfig {
            frequencies: frequencies.clone(),
            tolerance: 1e-9,
            max_iterations: 200,
            recycle_solution: true,
            gmres_restart: 30,
            beta_scale: 4.0,
            recycle_subspace_size: 0,
            max_recycle_relative_frequency_delta: 0.1,
        };
        let cfg_legacy = common();
        let cfg_gcro = FrequencySweepConfig {
            recycle_subspace_size: 2,
            ..common()
        };

        let r_legacy = frequency_sweep(
            &elements,
            &nodes,
            &incident_field,
            &cfg_legacy,
            AssemblyMethod::Tbem,
            343.0,
            1.21,
        )
        .unwrap();
        let r_gcro = frequency_sweep(
            &elements,
            &nodes,
            &incident_field,
            &cfg_gcro,
            AssemblyMethod::Tbem,
            343.0,
            1.21,
        )
        .unwrap();

        assert_eq!(r_legacy.solutions.len(), r_gcro.solutions.len());
        for (a, b) in r_legacy.solutions.iter().zip(r_gcro.solutions.iter()) {
            let diff: f64 = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).norm_sqr())
                .sum::<f64>()
                .sqrt();
            let n: f64 = a
                .iter()
                .map(|e| e.norm_sqr())
                .sum::<f64>()
                .sqrt()
                .max(1e-20);
            assert!(
                diff / n < 1e-5,
                "GCRO-DR vs warm-start GMRES disagree: rel diff = {:.3e}",
                diff / n
            );
        }
    }

    #[test]
    fn frequency_sweep_records_and_enforces_recycle_frequency_bound() {
        let (elements, nodes) = make_sweep_test_elements();
        let incident_field = IncidentField::plane_wave_z();
        let config = FrequencySweepConfig {
            frequencies: vec![100.0, 105.0, 150.0],
            tolerance: 1e-8,
            max_iterations: 200,
            recycle_solution: true,
            gmres_restart: 20,
            beta_scale: 4.0,
            recycle_subspace_size: 2,
            max_recycle_relative_frequency_delta: 0.1,
        };
        let result = frequency_sweep(
            &elements,
            &nodes,
            &incident_field,
            &config,
            AssemblyMethod::Tbem,
            343.0,
            1.21,
        )
        .expect("frequency sweep should remain correct when recycling resets");
        assert_eq!(result.recycled_subspace_eligible, vec![false, true, false]);
        result
            .validate()
            .expect("complete reuse evidence is aligned");
    }

    #[test]
    fn test_frequency_sweep_hmatrix_path_runs_operator_directly() {
        let (elements, nodes) = make_sweep_test_elements();
        let incident_field = IncidentField::plane_wave_z();
        let config = FrequencySweepConfig {
            frequencies: vec![100.0, 150.0],
            tolerance: 1e-8,
            max_iterations: 200,
            recycle_solution: true,
            gmres_restart: 20,
            beta_scale: 4.0,
            recycle_subspace_size: 2,
            max_recycle_relative_frequency_delta: 0.1,
        };

        let result = frequency_sweep(
            &elements,
            &nodes,
            &incident_field,
            &config,
            AssemblyMethod::Hmatrix,
            343.0,
            1.21,
        )
        .unwrap();

        assert_eq!(result.solutions.len(), 2);
        assert_eq!(result.num_converged(), 2);
        assert_eq!(result.operator_accuracy_errors.len(), 2);
        assert!(
            result
                .operator_accuracy_errors
                .iter()
                .all(|&error| error <= hmatrix_action_tolerance(config.tolerance)),
            "H-matrix sweep must retain bounded dense-reference action evidence: {:#?}",
            result.operator_accuracy_errors
        );
        assert!(
            result
                .exact_relative_residuals
                .iter()
                .all(|&residual| { residual <= (config.tolerance * 20.0).max(1e-10) }),
            "H-matrix sweep must retain bounded dense-reference residual evidence: {:#?}",
            result.exact_relative_residuals
        );
    }

    #[test]
    fn test_frequency_sweep_assembles_fmm_methods() {
        let (elements, nodes) = make_sweep_test_elements();
        let physics = PhysicsParams::new(500.0, 343.0, 1.21, false);

        for method in [AssemblyMethod::Slfmm, AssemblyMethod::Mlfmm] {
            let result = assemble_at_frequency(&elements, &nodes, &physics, method, 4.0);
            assert!(result.is_ok(), "{method:?} assembly failed");
            assert!(matches!(
                result.unwrap(),
                SweepSystem::Slfmm(_, _, _) | SweepSystem::Mlfmm(_, _, _)
            ));

            let config = FrequencySweepConfig {
                frequencies: vec![500.0],
                tolerance: 1e-6,
                max_iterations: 100,
                recycle_solution: false,
                gmres_restart: 20,
                beta_scale: 4.0,
                recycle_subspace_size: 0,
                max_recycle_relative_frequency_delta: 0.1,
            };
            let sweep = frequency_sweep(
                &elements,
                &nodes,
                &IncidentField::plane_wave_z(),
                &config,
                method,
                343.0,
                1.21,
            );
            assert!(sweep.is_ok(), "{method:?} frequency sweep failed");
        }
    }
}
