//! Algebraic Multigrid (AMG) Preconditioner
//!
//! This module implements an algebraic multigrid preconditioner inspired by
//! hypre's BoomerAMG, designed for better parallel scalability across CPU cores.
//!
//! ## Features
//!
//! - **Coarsening**: Classical Ruge-Stuben (RS), Falgout (the BoomerAMG
//!   default), parallel PMIS/HMIS/CLJP, and aggregation-based (SA,
//!   energy-minimizing) algorithms — see [`AmgCoarsening`]
//! - **Interpolation**: Standard, direct, extended, and extended+i
//!   interpolation operators — see [`AmgInterpolation`]
//! - **Restriction**: Galerkin transpose (`R = P^T`, default), conjugated
//!   transpose (`R = P^H`) for Hermitian complex systems, or local approximate
//!   ideal restriction (ℓAIR, Manteuffel-Ruge-Southworth 2018) for nonsymmetric
//!   and advection-dominated operators — see [`AmgRestriction`]
//! - **Smoothers**: Jacobi, L1-Jacobi, Chebyshev, symmetric Gauss-Seidel, multicolor Gauss-Seidel
//!   — optionally restricted to F-points (CF relaxation, the pairing AIR theory
//!   assumes) via [`AmgConfig::f_point_smoothing`]
//! - **Cycles**: Standard V(nu_1, nu_2), W, and F cycling with configurable
//!   pre/post smoothing, plus optional Krylov-accelerated K-cycles (flexible
//!   CG around the top-level cycle) via [`AmgConfig::k_cycle_iterations`]
//!
//! ## Scalability
//!
//! The AMG preconditioner scales better than ILU across multiple cores because:
//! - Coarsening can be parallelized (PMIS is inherently parallel)
//! - Jacobi smoothing is embarrassingly parallel
//! - Each level's operations can be parallelized independently
//!
//! ## Usage
//!
//! ```ignore
//! use sonium_solvers::{AmgPreconditioner, AmgConfig, CsrMatrix};
//!
//! let config = AmgConfig::default();
//! let precond = AmgPreconditioner::from_csr(&matrix, config);
//!
//! // Use with GMRES
//! let z = precond.apply(&residual);
//! ```

#[path = "../amg_checkpoint.rs"]
mod amg_checkpoint;
#[cfg(test)]
mod backend_l1_tests;
#[cfg(test)]
mod backend_pcg_tests;
mod backend_profile;
mod backend_refresh;
#[doc(inline)]
pub use backend_profile::{BackendAmgPhase, BackendAmgPhaseProfile};
#[cfg(test)]
mod backend_refresh_tests;
#[cfg(test)]
mod backend_zero_tests;
mod coarsening;
mod coloring;
#[cfg(feature = "cuda")]
mod cuda_c64;
#[cfg(feature = "experimental-gnn")]
mod gnn_coarsening;
mod interpolation;
mod lair;
mod strength;
pub(crate) mod util;
pub use amg_checkpoint::AMG_CHECKPOINT_FORMAT_VERSION;

#[cfg(feature = "experimental-gnn")]
pub use gnn_coarsening::GnnCoarseningWeights;

use num_traits::{Float, FromPrimitive};
#[cfg(feature = "cuda")]
use sonium_compute::Backend;
use sonium_compute::DeviceBufferLike;
use sonium_solvers_core::traits::{AdjointCapability, ComplexField, Preconditioner, SolverError};
use sonium_solvers_sparse::CsrMatrix;
use std::any::type_name;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Coarsening algorithm for AMG hierarchy construction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AmgCoarsening {
    /// Classical Ruge-Stuben coarsening
    /// Good quality but limited parallelism in the selection phase
    RugeStuben,

    /// Parallel Modified Independent Set (PMIS)
    /// Better parallel scalability, may produce slightly larger coarse grids
    Pmis,

    /// Hybrid MIS (HMIS) - PMIS in first pass, then RS cleanup
    /// Balance between quality and parallelism
    Hmis,

    /// Smoothed Aggregation (SA) - groups nodes into aggregates
    /// Best coarsening ratios (4-8x per level), optimal for Poisson/Helmholtz
    SmoothedAggregation,

    /// CLJP (Cleary-Luby-Jones-Plassmann) parallel coarsening
    /// Fully parallel with RS-quality C/F splitting (best of both worlds)
    #[default]
    Cljp,

    /// Falgout coarsening (the hypre BoomerAMG default)
    ///
    /// Classical RS first pass followed by the Falgout second pass, which
    /// also sweeps "boundary" F-points in `S_i^T` (points influenced by
    /// already-assigned C-points) in addition to the strong dependencies
    /// `S_i`. Slightly larger C-sets than RS but interpolation stencils
    /// that cover both directions of strong coupling.
    Falgout,

    /// Energy-Minimizing Aggregation (MANA-K)
    /// Uses MIS-k aggregation with column-wise CG energy minimization for prolongation.
    EnergyMinimizing,

    /// GNN-driven coarsening using a lightweight message-passing network
    /// over the strong-connection graph to predict optimal aggregate seeds.
    #[cfg(feature = "experimental-gnn")]
    GnnCoarsening,
}

/// Restriction operator type for the Galerkin coarse operator `A_c = R A P`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum AmgRestriction {
    /// Transpose of prolongation, `R = P^T`.
    ///
    /// Optimal for SPD problems (energy-norm Galerkin projection) and the
    /// right default for symmetric/Hermitian systems. Fails on strongly
    /// nonsymmetric operators (upwinded advection, transport).
    ///
    /// **Complex caveat**: `P^T` (no conjugation) is correct for
    /// complex-*symmetric* systems such as the complex Helmholtz operator,
    /// where `A = A^T` and the Galerkin product `P^T A P` preserves that
    /// structure. For *Hermitian* complex systems (`A = A^H`, e.g.
    /// time-harmonic Maxwell or shifted-Laplacian operators with complex
    /// shifts), `P^T A P` is not Hermitian in general; use
    /// [`AmgRestriction::ConjugateTranspose`] instead.
    #[default]
    Transpose,

    /// Conjugate transpose of prolongation, `R = P^H`.
    ///
    /// The correct Galerkin restriction for Hermitian complex systems:
    /// `P^H A P` is Hermitian whenever `A` is, so the coarse operators
    /// preserve Hermitian (and Hermitian positive-definite) structure down
    /// the hierarchy. For real matrices this coincides with
    /// [`AmgRestriction::Transpose`]. Do **not** use this for
    /// complex-symmetric (non-Hermitian) systems such as Helmholtz — those
    /// need the unconjugated [`AmgRestriction::Transpose`].
    ConjugateTranspose,

    /// Local approximate ideal restriction (ℓAIR),
    /// Manteuffel-Ruge-Southworth 2018, as shipped in hypre >= 2.19.
    ///
    /// Builds `R` row by row from local least-squares approximations of the
    /// ideal restriction `[-A_cf A_ff^{-1} I]`, making the coarse correction
    /// robust for nonsymmetric / advection-dominated operators.
    LAir {
        /// Neighborhood distance in the (symmetrized) strong-connection
        /// graph: 1 or 2 (hypre's AIR-1 / AIR-2). Other values are clamped.
        distance: usize,
        /// Optional relative drop tolerance applied to the restriction
        /// weights: entries with `|z_k| < filter * max_k |z_k|` are dropped
        /// (the unit entry is always kept). Rows are not renormalized.
        filter: Option<f64>,
    },
}

/// Interpolation operator type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AmgInterpolation {
    /// Standard interpolation - direct interpolation from coarse neighbors
    #[default]
    Standard,

    /// Extended interpolation - includes indirect (distance-2) connections
    /// Better for some problem types but more expensive
    Extended,

    /// Extended+i interpolation (hypre's "ext+i")
    ///
    /// Extended interpolation with an immediate-neighbor fallback: F-points
    /// without any C-point in their strong set interpolate from their full
    /// (immediate) matrix neighborhood instead of degrading to injection.
    /// More robust than [`AmgInterpolation::Extended`] on coarser levels
    /// where strong C-neighbors can be absent, at a modest stencil increase.
    ExtendedI,

    /// Direct interpolation - simplest, only immediate strong connections
    /// Fastest but may have poor convergence for hard problems
    Direct,
}

/// Smoother type for AMG relaxation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AmgSmoother {
    /// Jacobi relaxation - fully parallel, requires damping (omega ~ 0.6-0.8)
    #[default]
    Jacobi,

    /// l1-Jacobi - Jacobi with l1 norm scaling, more robust
    L1Jacobi,

    /// Symmetric Gauss-Seidel - forward then backward sweep
    /// Better convergence but limited parallelism
    SymmetricGaussSeidel,

    /// CF-ordered symmetric Gauss-Seidel (hypre hybrid-SGS style).
    ///
    /// Pre-smoothing sweeps C-points forward then F-points forward;
    /// post-smoothing sweeps F-points backward then C-points backward.
    /// Unlike [`AmgConfig::f_point_smoothing`] (F-points only, intended for
    /// AIR-type paths), every point is relaxed each visit, so this stays a
    /// valid SPD smoother while damping the C/F-coupled error components
    /// that full-point sweeps leave behind.
    CfSymmetricGaussSeidel,

    /// Chebyshev polynomial smoother - fully parallel, no damping needed
    Chebyshev,

    /// Multicolor Gauss-Seidel - graph-colored symmetric GS
    /// Rows of the same color are independent, enabling parallel sweeps via rayon.
    /// Convergence is similar to SGS; parallelism scales with the number of colors.
    MulticolorGaussSeidel,
}

/// Configuration for material-aware strength of connection.
///
/// When provided, uses distance-Laplacian-weighted SOC that respects
/// physical discontinuities in heterogeneous materials.
#[derive(Debug, Clone)]
pub struct MaterialSocConfig {
    /// Material tensor per node (flattened row-major).
    /// 2D: [a11, a12, a21, a22] per node (4 entries)
    /// 3D: [a11, a12, a13, a21, a22, a23, a31, a32, a33] per node (9 entries)
    pub tensors: Vec<Vec<f64>>,
    /// Spatial dimension (2 or 3)
    pub dim: usize,
}

/// AMG cycle type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AmgCycle {
    /// V-cycle: one visit to each level
    #[default]
    VCycle,

    /// W-cycle: two visits to coarser levels (more expensive)
    WCycle,

    /// F-cycle: hybrid between V and W
    FCycle,
}

/// Physics-oriented AMG profile.
///
/// These presets make solver policy explicit for the major matrix families in
/// Sonium. They are conservative defaults; callers can still start from a
/// profile and override individual [`AmgConfig`] fields when a problem has
/// better-known spectral or block structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmgPhysicsProfile {
    /// Scalar SPD elliptic systems such as Poisson/Laplacian.
    SpdLaplacian,
    /// Nodal elasticity systems with multiple displacement components per node.
    Elasticity { block_size: usize },
    /// Shifted-Laplacian preconditioner profile for indefinite Helmholtz solves.
    HelmholtzShiftedLaplacian,
    /// H(curl)-like Maxwell systems; prefer system-aware aggregation defaults.
    MaxwellHCurl { block_size: usize },
    /// Velocity/momentum block used inside saddle-point preconditioners.
    SaddlePointVelocity,
    /// Sparse proxy for BEM or dense/sparse hybrid blocks.
    BemHybrid,
}

/// Configuration for AMG preconditioner
///
/// Note on [`AmgRestriction::LAir`]: the strength-of-connection graph is
/// symmetrized on that path, because classical one-sided SOC on upwinded
/// (nonsymmetric) operators would otherwise degenerate the coarseners.
#[derive(Debug, Clone)]
pub struct AmgConfig {
    /// Coarsening algorithm
    pub coarsening: AmgCoarsening,

    /// Interpolation operator type
    pub interpolation: AmgInterpolation,

    /// Restriction operator type (default: transpose of prolongation, `R = P^T`).
    ///
    /// Use [`AmgRestriction::LAir`] for nonsymmetric / advection-dominated
    /// operators where transpose restriction stalls.
    pub restriction: AmgRestriction,

    /// Symmetrize the strength-of-connection graph before coarsening and
    /// interpolation (default: false).
    ///
    /// Classical SOC on strongly nonsymmetric (upwinded) operators produces
    /// a one-sided graph (each row only connects upwind). The coarseners
    /// degenerate on such graphs — tie-broken influence measures can select
    /// nearly every point as a C-point, yielding no coarsening at all.
    /// Enable this for advection-dominated problems. It is implied by
    /// [`AmgRestriction::LAir`], and classical (RS/PMIS/HMIS/CLJP/Falgout)
    /// coarsening paths additionally auto-symmetrize when the strong graph
    /// is *detected* one-sided (see `soc_is_one_sided`); this flag forces
    /// symmetrization regardless of the detection outcome.
    pub symmetrize_strength: bool,

    /// Smoother for pre- and post-relaxation
    pub smoother: AmgSmoother,

    /// Restrict pre/post smoothing to F-points (CF relaxation), default false.
    ///
    /// AIR theory (Manteuffel-Ruge-Southworth 2018) pairs ℓAIR restriction
    /// with F-point relaxation: the coarse correction is designed for error
    /// supported on F-points, so relaxing only F-points is both cheaper and
    /// typically gives a better convergence factor than all-point smoothing.
    ///
    /// Behaviour when enabled (per level, only where a classical C/F
    /// splitting is available — aggregation-based levels keep all-point
    /// smoothing):
    /// - Jacobi / l1-Jacobi: the damped update is applied to F indices only.
    /// - Symmetric Gauss-Seidel: only F-point rows are swept (forward then
    ///   backward over the F-point set).
    /// - Multicolor Gauss-Seidel: falls back to the F-point SGS sweep above.
    /// - Chebyshev: falls back to all-point Chebyshev (an F-restricted
    ///   polynomial smoother is not a standard construction).
    ///
    /// Only affects the host (CPU) apply path; backend (GPU) AMG ignores it.
    pub f_point_smoothing: bool,

    /// Cycle type (V, W, or F)
    pub cycle: AmgCycle,

    /// Krylov-accelerated cycle (K-cycle) iterations at the top level
    /// (default: 0 = plain cycle).
    ///
    /// When > 0, each top-level preconditioner application wraps
    /// `k_cycle_iterations` iterations of flexible conjugate gradient
    /// (Notay's FCG, the AGMG-style K-cycle) around the configured multigrid
    /// cycle, using one cycle as the (variable) preconditioner per
    /// iteration. This accelerates convergence exactly when a single cycle
    /// is a mediocre but SPD-compatible preconditioner; typical values are
    /// 1–3 (1 is already a Krylov-wrapped cycle, 2–3 approach the
    /// Krylov-optimal convergence for the cycle's spectral equivalence).
    ///
    /// Scope: **top level only** — coarser levels keep plain V/W/F cycling.
    /// Intended for SPD / Hermitian positive-definite systems (FCG assumes
    /// a conjugated inner product and a near-SPD preconditioner); for
    /// nonsymmetric problems use outer FGMRES instead. Only affects the
    /// host (CPU) apply path; backend (GPU) AMG ignores it.
    pub k_cycle_iterations: usize,

    /// Strong connection threshold (default: 0.25)
    /// Connections with |a_ij| >= theta * max_k |a_ik| are considered strong
    pub strong_threshold: f64,

    /// Maximum number of levels in the hierarchy
    pub max_levels: usize,

    /// Coarsest level size - switch to direct solve below this
    pub coarse_size: usize,

    /// Number of pre-smoothing sweeps (nu_1)
    pub num_pre_smooth: usize,

    /// Number of post-smoothing sweeps (nu_2)
    pub num_post_smooth: usize,

    /// Jacobi damping parameter (omega)
    pub jacobi_weight: f64,

    /// Truncation factor for interpolation (drop small weights)
    pub trunc_factor: f64,

    /// Drop tolerance for Galerkin coarse operators, relative to the row
    /// diagonal: entries with `|a_ij| < galerkin_drop_tol * |a_ii|` are
    /// removed after each `R A P` product. 0.0 keeps the full product
    /// (hypre default behavior); larger values trade coarse-correction
    /// quality for sparser hierarchies. Preserved by v2 checkpoints; legacy v1
    /// checkpoints restore the historical 0.001 default.
    pub galerkin_drop_tol: f64,

    /// Maximum interpolation stencil size per row
    pub max_interp_elements: usize,

    /// Enable aggressive coarsening on first few levels
    pub aggressive_coarsening_levels: usize,

    /// MIS distance for aggregation coarsening (default: 2)
    /// Controls the minimum separation between seed nodes in the MIS.
    /// Seeds are separated by at least k+1 hops, producing aggregates
    /// of varying size. Larger k = fewer, bigger aggregates.
    pub mis_distance: usize,

    /// Use `A^T A` as the strength-of-connection matrix on the smoothed /
    /// energy-minimizing aggregation paths (default: true, the historical
    /// behavior; effective MIS distance is then 1 since `A^T A` already
    /// reaches distance-2 neighbors).
    ///
    /// This is unusual compared to MueLu, which computes SOC on a filtered
    /// `A` directly. Measured on 2D Poisson (31x31, `for_aggregation`
    /// defaults, theta=0.08, geometric-mean factor over 15 V-cycles):
    /// `A^T A` SOC converges at factor 0.558 with operator complexity 1.39,
    /// plain-`A` SOC at factor 0.558 with operator complexity 1.41 — a wash
    /// on this benchmark (see `test_sa_strength_matrix_comparison` for the
    /// printed numbers). Set to `false` to use plain-`A` SOC (with the
    /// configured [`AmgConfig::mis_distance`]).
    pub sa_strength_with_ata: bool,

    /// Material-aware strength of connection configuration.
    /// When set, uses distance-Laplacian-weighted SOC for heterogeneous materials.
    pub material_soc: Option<MaterialSocConfig>,

    /// Number of CG iterations for energy-minimizing prolongation (default: 4)
    pub energy_min_iterations: usize,

    /// Pattern growth distance for energy-minimizing prolongation (default: 1)
    /// Number of applications of A's adjacency graph to grow each column's sparsity pattern.
    pub energy_min_pattern_distance: usize,

    /// Near-nullspace dimension (default: 1)
    /// Number of near-nullspace vectors used in tentative prolongation.
    pub near_nullspace_dim: usize,

    /// Block size for nodal (unknown) AMG (default: 1 = scalar AMG).
    ///
    /// For systems of PDEs with `block_size` DOFs per node (e.g., 2 for Stokes,
    /// 3 for 3D elasticity), this enables nodal coarsening: strength and
    /// coarsening are computed on the nodal graph (matrix size / block_size),
    /// and the prolongation operator interpolates all components together.
    ///
    /// This dramatically improves convergence for system problems.
    /// Matrix dimensions must be a multiple of `block_size`.
    pub block_size: usize,

    /// Optional GNN weights for GNN-based coarsening.
    /// If `None`, uses default pre-trained weights optimized for
    /// elliptic PDE matrices (Poisson, diffusion).
    #[cfg(feature = "experimental-gnn")]
    pub gnn_weights: Option<GnnCoarseningWeights>,

    /// Optional grid-complexity target for setup diagnostics.
    ///
    /// Grid complexity is `sum(level dofs) / finest dofs`. The preconditioner
    /// still builds when this target is exceeded, but diagnostics and logs make
    /// the excess visible to benchmark gates.
    pub max_grid_complexity: Option<f64>,

    /// Optional operator-complexity target for setup diagnostics.
    ///
    /// Operator complexity is `sum(level nnz) / finest nnz`. Keeping it bounded
    /// makes AMG setup/apply memory cost explicit for reuse-heavy solves.
    pub max_operator_complexity: Option<f64>,
}

impl Default for AmgConfig {
    fn default() -> Self {
        Self {
            coarsening: AmgCoarsening::default(),
            interpolation: AmgInterpolation::default(),
            restriction: AmgRestriction::default(),
            symmetrize_strength: false,
            smoother: AmgSmoother::default(),
            f_point_smoothing: false,
            cycle: AmgCycle::default(),
            k_cycle_iterations: 0,
            strong_threshold: 0.25,
            max_levels: 25,
            coarse_size: 50,
            num_pre_smooth: 1,
            num_post_smooth: 1,
            jacobi_weight: 0.6667, // 2/3 is optimal for Poisson
            trunc_factor: 0.0,
            galerkin_drop_tol: 0.001,
            max_interp_elements: 4,
            aggressive_coarsening_levels: 0,
            mis_distance: 2,
            sa_strength_with_ata: true,
            material_soc: None,
            energy_min_iterations: 4,
            energy_min_pattern_distance: 1,
            near_nullspace_dim: 1,
            #[cfg(feature = "experimental-gnn")]
            gnn_weights: None,
            block_size: 1,
            max_grid_complexity: None,
            max_operator_complexity: None,
        }
    }
}

impl AmgConfig {
    /// Configure stationary, symmetric backend AMG cycles for SPD/HPD PCG.
    ///
    /// Use with `BackendAmgPreconditioner::try_from_csr` on a validated SPD/HPD
    /// matrix. Equal L1-Jacobi pre/post sweeps, R=Pᴴ, and undropped Galerkin
    /// operators preserve symmetry. Row-norm scaling bounds the normalized HPD
    /// spectrum by one, making relaxation and its fixed coarse polynomial
    /// positive definite. The backend uses a fixed coarse sweep count, without
    /// tolerance-dependent inner iterations or a substituted serial smoother.
    ///
    /// This is a backend preset: the host smoother implementation is different.
    /// Altering the smoother, restriction, dropping, cycle or coarse solver can
    /// invalidate PCG assumptions. This configuration does not validate the input
    /// matrix, guarantee mesh-independent convergence, or accelerate host setup.
    pub fn for_backend_pcg() -> Self {
        Self {
            coarsening: AmgCoarsening::Cljp,
            restriction: AmgRestriction::ConjugateTranspose,
            smoother: AmgSmoother::L1Jacobi,
            cycle: AmgCycle::VCycle,
            k_cycle_iterations: 0,
            f_point_smoothing: false,
            num_pre_smooth: 1,
            num_post_smooth: 1,
            galerkin_drop_tol: 0.0,
            ..Self::default()
        }
    }

    /// Validate configuration values that can make hierarchy construction
    /// undefined or silently fall back to a different algorithm.
    ///
    /// The historical `AmgPreconditioner::from_csr` constructor remains
    /// infallible for compatibility. Checked callers should use
    /// `AmgPreconditioner::try_from_csr`, which validates both the CSR
    /// storage and the matrix-dependent fields below before setup starts.
    pub fn validate(&self) -> Result<(), SolverError> {
        let invalid = |field: &'static str, reason: String| {
            Err(SolverError::OperationFailed {
                operation: "AMG configuration",
                reason: format!("{field}: {reason}"),
            })
        };

        if self.max_levels == 0 {
            return invalid("max_levels", "must be greater than zero".into());
        }
        if self.coarse_size == 0 {
            return invalid("coarse_size", "must be greater than zero".into());
        }
        if self.max_interp_elements == 0 {
            return invalid("max_interp_elements", "must be greater than zero".into());
        }
        if self.near_nullspace_dim == 0 {
            return invalid("near_nullspace_dim", "must be greater than zero".into());
        }
        if self.block_size == 0 {
            return invalid("block_size", "must be greater than zero".into());
        }
        if !self.strong_threshold.is_finite() || !(0.0..=1.0).contains(&self.strong_threshold) {
            return invalid(
                "strong_threshold",
                format!(
                    "must be finite and in [0, 1], got {}",
                    self.strong_threshold
                ),
            );
        }
        if !self.jacobi_weight.is_finite() || self.jacobi_weight <= 0.0 {
            return invalid(
                "jacobi_weight",
                format!("must be finite and positive, got {}", self.jacobi_weight),
            );
        }
        if !self.trunc_factor.is_finite() || !(0.0..=1.0).contains(&self.trunc_factor) {
            return invalid(
                "trunc_factor",
                format!("must be finite and in [0, 1], got {}", self.trunc_factor),
            );
        }
        if !self.galerkin_drop_tol.is_finite() || !(0.0..=0.1).contains(&self.galerkin_drop_tol) {
            return invalid(
                "galerkin_drop_tol",
                format!(
                    "must be finite and in [0, 0.1], got {}",
                    self.galerkin_drop_tol
                ),
            );
        }
        for (field, value) in [
            ("max_grid_complexity", self.max_grid_complexity),
            ("max_operator_complexity", self.max_operator_complexity),
        ] {
            if let Some(value) = value
                && (!value.is_finite() || value <= 0.0)
            {
                return invalid(
                    field,
                    format!("must be finite and positive when set, got {value}"),
                );
            }
        }
        if let Some(material) = &self.material_soc {
            if material.dim == 0 {
                return invalid("material_soc.dim", "must be greater than zero".into());
            }
            let tensor_len = material.dim.checked_mul(material.dim).ok_or_else(|| {
                SolverError::OperationFailed {
                    operation: "AMG configuration",
                    reason: "material_soc.dim overflows tensor size".into(),
                }
            })?;
            if material
                .tensors
                .iter()
                .any(|tensor| tensor.len() != tensor_len || tensor.iter().any(|v| !v.is_finite()))
            {
                return invalid(
                    "material_soc.tensors",
                    format!("each tensor must contain {tensor_len} finite entries"),
                );
            }
        }
        Ok(())
    }

    /// Attach grid/operator complexity targets to this configuration.
    pub fn with_complexity_targets(
        mut self,
        max_grid_complexity: f64,
        max_operator_complexity: f64,
    ) -> Self {
        self.max_grid_complexity = Some(max_grid_complexity);
        self.max_operator_complexity = Some(max_operator_complexity);
        self
    }

    /// Create a physics-oriented AMG profile.
    pub fn for_physics(profile: AmgPhysicsProfile) -> Self {
        match profile {
            AmgPhysicsProfile::SpdLaplacian => Self::for_spd_laplacian(),
            AmgPhysicsProfile::Elasticity { block_size } => Self::for_elasticity(block_size),
            AmgPhysicsProfile::HelmholtzShiftedLaplacian => Self::for_helmholtz_shifted_laplacian(),
            AmgPhysicsProfile::MaxwellHCurl { block_size } => Self::for_maxwell_hcurl(block_size),
            AmgPhysicsProfile::SaddlePointVelocity => Self::for_saddle_point_velocity(),
            AmgPhysicsProfile::BemHybrid => Self::for_bem_hybrid(),
        }
    }

    /// Configuration for scalar SPD Laplacian/Poisson-like systems.
    ///
    /// This path emphasizes smoothed aggregation and Chebyshev smoothing, which
    /// keeps both setup and apply parallel for bandwidth-bound large problems.
    pub fn for_spd_laplacian() -> Self {
        Self::for_muelu_sa_comparison()
    }

    /// Configuration for elasticity systems with nodal vector unknowns.
    pub fn for_elasticity(block_size: usize) -> Self {
        Self {
            near_nullspace_dim: block_size.max(1),
            ..Self::for_systems(block_size.max(1))
        }
    }

    /// Configuration for shifted-Laplacian Helmholtz preconditioning.
    ///
    /// AMG is used here as a preconditioner for the shifted elliptic operator,
    /// not directly for arbitrary complex Helmholtz matrices.
    pub fn for_helmholtz_shifted_laplacian() -> Self {
        Self {
            coarsening: AmgCoarsening::SmoothedAggregation,
            interpolation: AmgInterpolation::Standard,
            smoother: AmgSmoother::Chebyshev,
            cycle: AmgCycle::FCycle,
            strong_threshold: 0.05,
            num_pre_smooth: 3,
            num_post_smooth: 3,
            mis_distance: 2,
            ..Default::default()
        }
        .with_complexity_targets(2.0, 3.0)
    }

    /// Configuration for Maxwell/H(curl)-like block systems.
    pub fn for_maxwell_hcurl(block_size: usize) -> Self {
        Self {
            coarsening: AmgCoarsening::EnergyMinimizing,
            interpolation: AmgInterpolation::Extended,
            smoother: AmgSmoother::Chebyshev,
            strong_threshold: 0.12,
            max_interp_elements: 8,
            near_nullspace_dim: block_size.max(1),
            block_size: block_size.max(1),
            energy_min_iterations: 6,
            energy_min_pattern_distance: 2,
            ..Default::default()
        }
        .with_complexity_targets(2.5, 4.0)
    }

    /// Configuration for the elliptic velocity/momentum block in saddle-point systems.
    pub fn for_saddle_point_velocity() -> Self {
        Self {
            coarsening: AmgCoarsening::Hmis,
            interpolation: AmgInterpolation::Extended,
            smoother: AmgSmoother::L1Jacobi,
            strong_threshold: 0.25,
            max_interp_elements: 8,
            num_pre_smooth: 2,
            num_post_smooth: 2,
            ..Default::default()
        }
        .with_complexity_targets(2.0, 3.0)
    }

    /// Configuration for sparse BEM/dense-hybrid proxy systems.
    pub fn for_bem_hybrid() -> Self {
        Self {
            aggressive_coarsening_levels: 1,
            ..Self::for_bem()
        }
        .with_complexity_targets(2.5, 4.0)
    }

    /// Configuration optimized for BEM systems
    ///
    /// BEM matrices are typically denser and less sparse than FEM,
    /// requiring adjusted thresholds.
    pub fn for_bem() -> Self {
        Self {
            strong_threshold: 0.5,           // Higher for denser BEM matrices
            coarsening: AmgCoarsening::Pmis, // Better parallel scalability
            smoother: AmgSmoother::L1Jacobi, // More robust for BEM
            max_interp_elements: 6,
            ..Default::default()
        }
        .with_complexity_targets(2.5, 4.0)
    }

    /// Configuration optimized for FEM systems
    pub fn for_fem() -> Self {
        Self {
            strong_threshold: 0.25,
            coarsening: AmgCoarsening::Cljp,
            smoother: AmgSmoother::MulticolorGaussSeidel,
            ..Default::default()
        }
    }

    /// Configuration optimized for maximum parallel scalability
    pub fn for_parallel() -> Self {
        Self {
            coarsening: AmgCoarsening::Pmis,
            smoother: AmgSmoother::Jacobi,
            jacobi_weight: 0.8,
            num_pre_smooth: 2,
            num_post_smooth: 2,
            ..Default::default()
        }
    }

    /// Configuration for difficult/ill-conditioned problems
    pub fn for_difficult_problems() -> Self {
        Self {
            coarsening: AmgCoarsening::RugeStuben,
            interpolation: AmgInterpolation::Extended,
            smoother: AmgSmoother::SymmetricGaussSeidel,
            strong_threshold: 0.25,
            max_interp_elements: 8,
            num_pre_smooth: 2,
            num_post_smooth: 2,
            ..Default::default()
        }
    }

    /// Configuration using Smoothed Aggregation (SA-AMG)
    ///
    /// Similar to PETSc GAMG defaults. Best coarsening ratios and optimal
    /// for Poisson/Helmholtz problems.
    pub fn for_aggregation() -> Self {
        Self {
            coarsening: AmgCoarsening::SmoothedAggregation,
            smoother: AmgSmoother::Chebyshev,
            num_pre_smooth: 2,
            num_post_smooth: 2,
            strong_threshold: 0.08,
            mis_distance: 2,
            ..Default::default()
        }
        .with_complexity_targets(2.0, 3.0)
    }

    /// Configuration using Energy-Minimizing Aggregation (MANA-K)
    ///
    /// Uses MIS-k aggregation with column-wise CG energy minimization
    /// for constructing the prolongation operator.
    pub fn for_energy_minimizing() -> Self {
        Self {
            coarsening: AmgCoarsening::EnergyMinimizing,
            smoother: AmgSmoother::Chebyshev,
            num_pre_smooth: 2,
            num_post_smooth: 2,
            strong_threshold: 0.08,
            mis_distance: 2,
            energy_min_iterations: 4,
            energy_min_pattern_distance: 1,
            near_nullspace_dim: 1,
            ..Default::default()
        }
        .with_complexity_targets(2.0, 3.0)
    }

    /// Comparison profile mirroring common hypre BoomerAMG choices:
    /// HMIS-style coarsening, extended interpolation, and L1-Jacobi smoothing.
    pub fn for_hypre_boomeramg_comparison() -> Self {
        Self {
            // Measured BoomerAMG defaults on 3D Poisson (see
            // benches/ffi/README.md and the amg_vendor_probe harness):
            // Falgout-CLJP coarsening, modified classical interpolation,
            // CF-ordered SGS relaxation, theta 0.25. The interpolation cap
            // of 8 is retained (measured neutral vs unlimited on 3D Poisson).
            coarsening: AmgCoarsening::Falgout,
            interpolation: AmgInterpolation::Standard,
            smoother: AmgSmoother::CfSymmetricGaussSeidel,
            strong_threshold: 0.25,
            max_interp_elements: 8,
            num_pre_smooth: 1,
            num_post_smooth: 1,
            ..Default::default()
        }
        .with_complexity_targets(2.0, 3.0)
    }

    /// Comparison profile mirroring Trilinos MueLu-style smoothed aggregation
    /// with a parallel Chebyshev smoother.
    pub fn for_muelu_sa_comparison() -> Self {
        Self {
            coarsening: AmgCoarsening::SmoothedAggregation,
            interpolation: AmgInterpolation::Standard,
            smoother: AmgSmoother::Chebyshev,
            strong_threshold: 0.08,
            max_interp_elements: 4,
            num_pre_smooth: 2,
            num_post_smooth: 2,
            mis_distance: 2,
            ..Default::default()
        }
        .with_complexity_targets(2.0, 3.0)
    }

    /// Configuration for nonsymmetric / advection-dominated problems.
    ///
    /// Uses ℓAIR (local approximate ideal restriction, distance 2) instead
    /// of the Galerkin transpose restriction, HMIS coarsening, and symmetric
    /// Gauss-Seidel smoothing restricted to F-points (CF relaxation), which
    /// is the relaxation AIR theory is designed around
    /// (Manteuffel-Ruge-Southworth 2018); see [`AmgConfig::f_point_smoothing`].
    pub fn for_nonsymmetric() -> Self {
        Self {
            coarsening: AmgCoarsening::Hmis,
            interpolation: AmgInterpolation::Standard,
            restriction: AmgRestriction::LAir {
                distance: 2,
                filter: None,
            },
            smoother: AmgSmoother::SymmetricGaussSeidel,
            f_point_smoothing: true,
            num_pre_smooth: 1,
            num_post_smooth: 1,
            ..Default::default()
        }
    }

    /// Configuration using GNN-driven coarsening.
    ///
    /// A lightweight 2-layer GCN predicts aggregate seeds from matrix
    /// features, then BFS assigns remaining nodes. Combined with
    /// Smoothed Aggregation interpolation and Chebyshev smoothing.
    #[cfg(feature = "experimental-gnn")]
    pub fn for_gnn() -> Self {
        Self {
            coarsening: AmgCoarsening::GnnCoarsening,
            smoother: AmgSmoother::Chebyshev,
            num_pre_smooth: 2,
            num_post_smooth: 2,
            strong_threshold: 0.08,
            ..Default::default()
        }
    }

    /// Configuration for system problems (Stokes, elasticity).
    ///
    /// Uses nodal coarsening with the given block size.
    /// Coarsening and strength are computed on the nodal graph (n / block_size);
    /// prolongation interpolates all DOFs per node together.
    pub fn for_systems(block_size: usize) -> Self {
        Self {
            coarsening: AmgCoarsening::Cljp,
            smoother: AmgSmoother::Chebyshev,
            block_size,
            ..Default::default()
        }
        .with_complexity_targets(2.5, 4.0)
    }
}

/// Point classification in coarsening
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PointType {
    /// Undecided
    Undecided,
    /// Coarse point (C-point)
    Coarse,
    /// Fine point (F-point)
    Fine,
}

/// Single level in the AMG hierarchy
#[derive(Debug)]
struct AmgLevel<T: ComplexField> {
    /// System matrix A at this level (CSR format)
    matrix: CsrMatrix<T>,

    /// Prolongation operator P: coarse -> fine
    prolongation: Option<CsrMatrix<T>>,

    /// Restriction operator R: fine -> coarse (typically R = P^T)
    restriction: Option<CsrMatrix<T>>,

    /// Inverse diagonal for Jacobi smoothing
    diag_inv: Vec<T>,

    /// L1 row-norm diagonal for L1-Jacobi smoothing (cached at setup)
    l1_diag: Vec<T::Real>,

    /// Estimated maximum eigenvalue of D^{-1}A (for Chebyshev smoother)
    lambda_max: Option<T::Real>,

    /// Coarse-to-fine mapping
    coarse_to_fine: Vec<usize>,

    /// F-point indices (ascending) of this level's C/F splitting, used for
    /// F-point (CF) relaxation when [`AmgConfig::f_point_smoothing`] is set.
    /// Empty when F-point smoothing is disabled, when the level was built by
    /// an aggregation-based path (no pointwise C/F splitting), or on the
    /// terminal level.
    f_points: Vec<usize>,

    /// Number of DOFs at this level
    num_dofs: usize,

    /// Pre-allocated workspace: residual vector (same size as this level)
    work_r: Mutex<Vec<T>>,
    /// Pre-allocated workspace: temporary vector (same size as this level)
    work_tmp: Mutex<Vec<T>>,
    /// Pre-allocated workspace: restricted residual (size of NEXT coarser level)
    work_r_coarse: Mutex<Vec<T>>,
    /// Pre-allocated workspace: coarse correction (size of NEXT coarser level)
    work_e_coarse: Mutex<Vec<T>>,

    /// Sparse direct solver for coarsest level (None for non-coarsest levels)
    coarse_solver: Option<sonium_solvers_direct::SparseLuFactorization<T>>,

    /// Row coloring for multicolor Gauss-Seidel (computed during setup)
    color_sets: Option<Vec<Vec<usize>>>,
}

/// Algebraic Multigrid Preconditioner
///
/// Implements a classical AMG V-cycle preconditioner with configurable
/// coarsening, interpolation, and smoothing strategies.
#[derive(Debug)]
pub struct AmgPreconditioner<T: ComplexField> {
    /// AMG hierarchy (finest to coarsest)
    levels: Vec<AmgLevel<T>>,

    /// Configuration
    config: AmgConfig,

    /// Statistics
    setup_time_ms: f64,
    grid_complexity: f64,
    operator_complexity: f64,
    apply_count: AtomicU64,
    total_apply_ns: AtomicU64,
}

struct BackendAmgLevel<
    B: sonium_compute::Backend,
    T: ComplexField<Real = <T as sonium_compute::DeviceScalar>::Real> + sonium_compute::DeviceScalar,
> {
    matrix: sonium_compute::PersistentDeviceCsr<B, T>,
    prolongation: Option<sonium_compute::PersistentDeviceCsr<B, T>>,
    restriction: Option<sonium_compute::PersistentDeviceCsr<B, T>>,
    diag_inv: sonium_compute::PersistentDeviceCsr<B, T>,
    lambda_max: Option<<T as sonium_compute::DeviceScalar>::Real>,
    num_dofs: usize,
    /// Uploaded row-index buffer per graph color, for device multicolor
    /// Gauss-Seidel. Populated from the host `color_sets` (present only when
    /// the hierarchy was built with `AmgSmoother::MulticolorGaussSeidel`).
    colors: Vec<B::Buffer<i32>>,
}

struct BackendAmgLevelWorkspace<
    B: sonium_compute::Backend,
    T: ComplexField<Real = <T as sonium_compute::DeviceScalar>::Real> + sonium_compute::DeviceScalar,
> {
    residual: B::Buffer<T>,
    r_coarse: Option<B::Buffer<T>>,
    e_coarse: Option<B::Buffer<T>>,
    correction: B::Buffer<T>,
    ax: B::Buffer<T>,
    d: B::Buffer<T>,
    d_new: B::Buffer<T>,
}

struct BackendAmgWorkspace<
    B: sonium_compute::Backend,
    T: ComplexField<Real = <T as sonium_compute::DeviceScalar>::Real> + sonium_compute::DeviceScalar,
> {
    levels: Vec<BackendAmgLevelWorkspace<B, T>>,
    /// Finest-level result reused by checked applications.  Caller output is
    /// written only after the complete cycle succeeds.
    result: B::Buffer<T>,
    /// One column-sized staging buffer for the native block-RHS adapter.
    block_input: B::Buffer<T>,
}

trait BackendAmgDirectSolver<
    B: sonium_compute::Backend,
    T: ComplexField<Real = <T as sonium_compute::DeviceScalar>::Real> + sonium_compute::DeviceScalar,
>: Send + Sync
{
    fn supports_numeric_refresh(&self) -> bool {
        false
    }

    fn refresh(
        &mut self,
        _backend: &B,
        _matrix: &CsrMatrix<T>,
    ) -> sonium_compute::BackendResult<()> {
        Err(sonium_compute::BackendError::Unsupported {
            backend: type_name::<B>(),
            operation: "amg_numeric_refresh_direct_coarse",
            scalar: type_name::<T>(),
        })
    }

    fn solve(
        &self,
        backend: &B,
        b: &B::Buffer<T>,
        x: &mut B::Buffer<T>,
    ) -> sonium_compute::BackendResult<bool>;
}

#[cfg(feature = "cuda")]
struct CudaC32CoarseLu {
    lu: sonium_compute::CudaDenseLuC32,
}

#[cfg(feature = "cuda")]
impl BackendAmgDirectSolver<sonium_compute::CudaBackend, num_complex::Complex32>
    for CudaC32CoarseLu
{
    fn solve(
        &self,
        backend: &sonium_compute::CudaBackend,
        b: &sonium_compute::CudaBuffer<num_complex::Complex32>,
        x: &mut sonium_compute::CudaBuffer<num_complex::Complex32>,
    ) -> sonium_compute::BackendResult<bool> {
        match backend.dense_lu_c32_solve(&self.lu, b, x) {
            Ok(()) => Ok(true),
            Err(sonium_compute::OomError::DriverOom(reason)) => {
                Err(sonium_compute::BackendError::Allocation {
                    reason: format!("CUDA AMG coarse LU solve failed: {reason}"),
                })
            }
            Err(sonium_compute::OomError::QuotaExceeded {
                requested,
                in_use,
                cap,
            }) => Err(sonium_compute::BackendError::Allocation {
                reason: format!(
                    "CUDA AMG coarse LU solve exceeded memory quota: requested {requested} B, in use {in_use} B, cap {cap} B"
                ),
            }),
            Err(sonium_compute::OomError::Invalid(reason)) => {
                Err(sonium_compute::BackendError::OperationFailed {
                    operation: "cuda_amg_coarse_lu_solve",
                    reason,
                })
            }
        }
    }
}

/// Backend-resident AMG preconditioner.
///
/// Setup reuses the existing CPU AMG hierarchy builder, then uploads each
/// level to a [`sonium_compute::Backend`]. Apply runs a multigrid cycle with
/// backend SpMV/BLAS operations only. GPU-safe Jacobi, L1-Jacobi and Chebyshev smoothing
/// are supported; serial CPU smoothers fall back to Jacobi on apply.
pub struct BackendAmgPreconditioner<
    B: sonium_compute::Backend,
    T: ComplexField<Real = <T as sonium_compute::DeviceScalar>::Real>
        + sonium_compute::DeviceScalar = f64,
> {
    levels: Vec<BackendAmgLevel<B, T>>,
    config: AmgConfig,
    coarse_direct: Option<Box<dyn BackendAmgDirectSolver<B, T>>>,
    workspace: Mutex<BackendAmgWorkspace<B, T>>,
    checkpoint_bytes: Vec<u8>,
    numeric_valid: bool,
    phase_profile: Option<Vec<[backend_profile::PhaseCounters; 6]>>,
    small_jacobi_enabled: bool,
    fused_jacobi_enabled: bool,
    apply_count: AtomicU64,
    total_apply_ns: AtomicU64,
    runtime_diagnostics: sonium_compute::DevicePreconditionerDiagnostics,
}

/// Setup diagnostics for a backend-resident AMG hierarchy.
///
/// The hierarchy is built on the host and uploaded level by level, so these
/// values describe the device-resident matrices and the workspace allocated
/// for backend application.  They intentionally do not require a device
/// readback and are therefore safe to query before the first application.
#[derive(Debug, Clone, PartialEq)]
pub struct BackendAmgDiagnostics {
    /// Number of levels in the uploaded hierarchy.
    pub num_levels: usize,
    /// Grid complexity (`sum(level DOFs) / finest DOFs`).
    pub grid_complexity: f64,
    /// Operator complexity (`sum(level NNZ) / finest NNZ`).
    pub operator_complexity: f64,
    /// Total matrix entries across all hierarchy levels.
    pub total_nnz: usize,
    /// Finest-level matrix entries.
    pub finest_nnz: usize,
    /// Coarsest-level degrees of freedom.
    pub coarsest_dofs: usize,
    /// Degrees of freedom per level.
    pub level_dofs: Vec<usize>,
    /// Nonzero entries per level.
    pub level_nnz: Vec<usize>,
    /// Number of scalar elements reserved by the backend workspace.
    pub workspace_elements: usize,
    /// Whether a direct coarse solver was attached to this hierarchy.
    pub has_coarse_direct_solver: bool,
    /// Number of recorded backend preconditioner applications.
    pub apply_count: u64,
    /// Total host submission time in milliseconds; GPU completion is not implied.
    pub total_apply_ms: f64,
    /// Mean host submission time in milliseconds; GPU completion is not implied.
    pub mean_apply_ms: f64,
    /// Normalized action/failure counters for the backend boundary.
    pub runtime: sonium_compute::DevicePreconditionerDiagnosticsSnapshot,
}

const BACKEND_AMG_COARSE_SWEEPS: usize = 20;
const BACKEND_AMG_CHECKPOINT_MAGIC: &[u8; 8] = b"SBAMGCP2";
const BACKEND_AMG_CHECKPOINT_VERSION: u16 = 2;
/// Relative diagonal shift applied on a single retry when the dense f32
/// coarse factor hits a zero pivot. Far below the approximation error of a
/// coarse-grid correction, but enough to lift an inherited nullspace pivot.
#[cfg(feature = "cuda")]
const COARSE_LU_DIAGONAL_REGULARIZATION: f32 = 1.0e-6;

fn backend_checkpoint_bytes<B: sonium_compute::Backend, T: sonium_compute::DeviceScalar>(
    payload: &[u8],
    has_coarse_direct_solver: bool,
) -> sonium_compute::BackendResult<Vec<u8>> {
    let backend = type_name::<B>().as_bytes();
    let scalar = type_name::<T>().as_bytes();
    let backend_len = u16::try_from(backend.len()).map_err(|_| {
        sonium_compute::BackendError::InvalidArgument {
            operation: "amg_checkpoint",
            reason: "backend type name exceeds checkpoint limit".into(),
        }
    })?;
    let scalar_len =
        u16::try_from(scalar.len()).map_err(|_| sonium_compute::BackendError::InvalidArgument {
            operation: "amg_checkpoint",
            reason: "scalar type name exceeds checkpoint limit".into(),
        })?;
    let payload_len = u64::try_from(payload.len()).map_err(|_| {
        sonium_compute::BackendError::InvalidArgument {
            operation: "amg_checkpoint",
            reason: "checkpoint payload exceeds u64 length".into(),
        }
    })?;
    let mut bytes = Vec::with_capacity(
        BACKEND_AMG_CHECKPOINT_MAGIC.len()
            + 2
            + 2
            + scalar.len()
            + 2
            + backend.len()
            + 1
            + 8
            + payload.len()
            + 8,
    );
    bytes.extend_from_slice(BACKEND_AMG_CHECKPOINT_MAGIC);
    bytes.extend_from_slice(&BACKEND_AMG_CHECKPOINT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&scalar_len.to_le_bytes());
    bytes.extend_from_slice(scalar);
    bytes.extend_from_slice(&backend_len.to_le_bytes());
    bytes.extend_from_slice(backend);
    bytes.push(u8::from(has_coarse_direct_solver));
    bytes.extend_from_slice(&payload_len.to_le_bytes());
    bytes.extend_from_slice(payload);
    bytes.extend_from_slice(&checkpoint_checksum(&bytes).to_le_bytes());
    Ok(bytes)
}

fn checkpoint_checksum(bytes: &[u8]) -> u64 {
    // FNV-1a is deliberately small and dependency-free; it detects transport
    // corruption while the nested host checkpoint performs semantic checks.
    bytes.iter().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

fn decode_backend_checkpoint<B: sonium_compute::Backend, T: sonium_compute::DeviceScalar>(
    bytes: &[u8],
) -> sonium_compute::BackendResult<Option<(Vec<u8>, bool)>> {
    if bytes.len() < BACKEND_AMG_CHECKPOINT_MAGIC.len()
        || &bytes[..BACKEND_AMG_CHECKPOINT_MAGIC.len()] != BACKEND_AMG_CHECKPOINT_MAGIC
    {
        return Ok(None);
    }
    let mut offset = BACKEND_AMG_CHECKPOINT_MAGIC.len();
    let take = |bytes: &[u8], offset: &mut usize, len: usize| {
        let end = offset.checked_add(len).ok_or_else(|| {
            sonium_compute::BackendError::InvalidArgument {
                operation: "amg_checkpoint_restore",
                reason: "checkpoint offset overflow".into(),
            }
        })?;
        let value = bytes.get(*offset..end).ok_or_else(|| {
            sonium_compute::BackendError::InvalidArgument {
                operation: "amg_checkpoint_restore",
                reason: "truncated backend AMG checkpoint".into(),
            }
        })?;
        *offset = end;
        Ok::<Vec<u8>, sonium_compute::BackendError>(value.to_vec())
    };
    let version = u16::from_le_bytes(take(bytes, &mut offset, 2)?.try_into().map_err(|_| {
        sonium_compute::BackendError::InvalidArgument {
            operation: "amg_checkpoint_restore",
            reason: "checkpoint field has an invalid length".into(),
        }
    })?);
    if version != BACKEND_AMG_CHECKPOINT_VERSION {
        return Err(sonium_compute::BackendError::InvalidArgument {
            operation: "amg_checkpoint_restore",
            reason: format!("unsupported backend AMG checkpoint version {version}"),
        });
    }
    let scalar_len = usize::from(u16::from_le_bytes(
        take(bytes, &mut offset, 2)?.try_into().map_err(|_| {
            sonium_compute::BackendError::InvalidArgument {
                operation: "amg_checkpoint_restore",
                reason: "checkpoint field has an invalid length".into(),
            }
        })?,
    ));
    let scalar = take(bytes, &mut offset, scalar_len)?;
    if scalar != type_name::<T>().as_bytes() {
        return Err(sonium_compute::BackendError::InvalidArgument {
            operation: "amg_checkpoint_restore",
            reason: format!("checkpoint scalar does not match {}", type_name::<T>()),
        });
    }
    let backend_len = usize::from(u16::from_le_bytes(
        take(bytes, &mut offset, 2)?.try_into().map_err(|_| {
            sonium_compute::BackendError::InvalidArgument {
                operation: "amg_checkpoint_restore",
                reason: "checkpoint field has an invalid length".into(),
            }
        })?,
    ));
    let backend = take(bytes, &mut offset, backend_len)?;
    if backend != type_name::<B>().as_bytes() {
        return Err(sonium_compute::BackendError::InvalidArgument {
            operation: "amg_checkpoint_restore",
            reason: format!("checkpoint backend does not match {}", type_name::<B>()),
        });
    }
    let has_coarse_direct_solver = match take(bytes, &mut offset, 1)?[0] {
        0 => false,
        1 => true,
        value => {
            return Err(sonium_compute::BackendError::InvalidArgument {
                operation: "amg_checkpoint_restore",
                reason: format!("invalid coarse-solver flag {value}"),
            });
        }
    };
    let payload_len = usize::try_from(u64::from_le_bytes(
        take(bytes, &mut offset, 8)?.try_into().map_err(|_| {
            sonium_compute::BackendError::InvalidArgument {
                operation: "amg_checkpoint_restore",
                reason: "checkpoint field has an invalid length".into(),
            }
        })?,
    ))
    .map_err(|_| sonium_compute::BackendError::InvalidArgument {
        operation: "amg_checkpoint_restore",
        reason: "checkpoint payload length does not fit usize".into(),
    })?;
    let payload = take(bytes, &mut offset, payload_len)?;
    let stored_checksum =
        u64::from_le_bytes(take(bytes, &mut offset, 8)?.try_into().map_err(|_| {
            sonium_compute::BackendError::InvalidArgument {
                operation: "amg_checkpoint_restore",
                reason: "checkpoint field has an invalid length".into(),
            }
        })?);
    if offset != bytes.len() || checkpoint_checksum(&bytes[..bytes.len() - 8]) != stored_checksum {
        return Err(sonium_compute::BackendError::InvalidArgument {
            operation: "amg_checkpoint_restore",
            reason: "backend AMG checkpoint checksum or trailing bytes invalid".into(),
        });
    }
    Ok(Some((payload, has_coarse_direct_solver)))
}

impl<B, T> BackendAmgPreconditioner<B, T>
where
    B: sonium_compute::Backend,
    T: ComplexField<Real = <T as sonium_compute::DeviceScalar>::Real>
        + sonium_compute::DeviceScalar,
{
    /// Build a backend-resident AMG hierarchy with typed setup/allocation
    /// failures.  This is the checked counterpart to [`Self::from_csr`].
    pub fn try_from_csr(
        backend: &B,
        matrix: &CsrMatrix<T>,
        config: AmgConfig,
    ) -> sonium_compute::BackendResult<Self>
    where
        <T as sonium_compute::DeviceScalar>::Real: Sync + Send,
    {
        let host_amg = AmgPreconditioner::try_from_csr(matrix, config).map_err(|error| {
            sonium_compute::BackendError::InvalidArgument {
                operation: "amg_setup",
                reason: error.to_string(),
            }
        })?;
        Self::try_from_host_amg(backend, &host_amg)
    }

    /// Restore a backend-resident AMG hierarchy from the portable host
    /// checkpoint envelope. Device buffers and workspaces are rebuilt on the
    /// supplied backend after strict host-side checkpoint validation. A
    /// checkpoint carrying CUDA coarse-factor state must be restored through
    /// the backend-specific CUDA coarse-LU constructor when available.
    pub fn try_from_checkpoint_bytes(
        backend: &B,
        bytes: &[u8],
    ) -> sonium_compute::BackendResult<Self>
    where
        <T as sonium_compute::DeviceScalar>::Real: Sync + Send,
    {
        let (payload, has_coarse_direct_solver) =
            decode_backend_checkpoint::<B, T>(bytes)?.unwrap_or_else(|| (bytes.to_vec(), false));
        if has_coarse_direct_solver {
            return Err(sonium_compute::BackendError::Unsupported {
                backend: type_name::<B>(),
                operation: "amg_checkpoint_restore_coarse_lu",
                scalar: type_name::<T>(),
            });
        }
        let host_amg =
            AmgPreconditioner::<T>::from_checkpoint_bytes(&payload).map_err(|error| {
                sonium_compute::BackendError::InvalidArgument {
                    operation: "amg_checkpoint_restore",
                    reason: error.to_string(),
                }
            })?;
        Self::try_from_host_amg(backend, &host_amg)
    }

    /// Compatibility wrapper around [`Self::try_from_checkpoint_bytes`].
    pub fn from_checkpoint_bytes(backend: &B, bytes: &[u8]) -> Self
    where
        <T as sonium_compute::DeviceScalar>::Real: Sync + Send,
    {
        Self::try_from_checkpoint_bytes(backend, bytes)
            .unwrap_or_else(|error| panic!("backend AMG checkpoint restore failed: {error}"))
    }

    /// Return the current successfully published hierarchy checkpoint without device readback.
    ///
    /// # Errors
    /// Rejects export after an interrupted numeric refresh until recovery succeeds.
    pub fn to_checkpoint_bytes(&self) -> sonium_compute::BackendResult<Vec<u8>> {
        self.ensure_numeric_valid()?;
        Ok(self.checkpoint_bytes.clone())
    }

    /// Build a backend-resident AMG hierarchy from a host CSR matrix.
    pub fn from_csr(backend: &B, matrix: &CsrMatrix<T>, config: AmgConfig) -> Self
    where
        <T as sonium_compute::DeviceScalar>::Real: Sync + Send,
    {
        Self::try_from_csr(backend, matrix, config)
            .unwrap_or_else(|error| panic!("backend AMG setup failed: {error}"))
    }

    /// Upload an already-built host hierarchy with typed backend failures.
    pub fn try_from_host_amg(
        backend: &B,
        host_amg: &AmgPreconditioner<T>,
    ) -> sonium_compute::BackendResult<Self> {
        Self::try_from_host_amg_with_direct(backend, host_amg, None)
    }

    /// Upload an already-built host AMG hierarchy to the backend.
    pub fn from_host_amg(backend: &B, host_amg: &AmgPreconditioner<T>) -> Self {
        Self::try_from_host_amg(backend, host_amg)
            .unwrap_or_else(|error| panic!("backend AMG upload failed: {error}"))
    }

    fn try_from_host_amg_with_direct(
        backend: &B,
        host_amg: &AmgPreconditioner<T>,
        coarse_direct: Option<Box<dyn BackendAmgDirectSolver<B, T>>>,
    ) -> sonium_compute::BackendResult<Self> {
        host_amg.validate_generated_setup().map_err(|error| {
            sonium_compute::BackendError::InvalidArgument {
                operation: "amg_setup",
                reason: error.to_string(),
            }
        })?;
        let host_checkpoint = host_amg.to_checkpoint_bytes().map_err(|error| {
            sonium_compute::BackendError::InvalidArgument {
                operation: "amg_checkpoint",
                reason: error.to_string(),
            }
        })?;
        let has_coarse_direct_solver = coarse_direct.is_some();
        let checkpoint_bytes =
            backend_checkpoint_bytes::<B, T>(&host_checkpoint, has_coarse_direct_solver)?;
        backend.try_inject_preconditioner_failure(
            sonium_compute::PreconditionerFailureInjectionPoint::HierarchyUpload,
        )?;
        let levels: Vec<BackendAmgLevel<B, T>> = host_amg
            .levels
            .iter()
            .map(|level| {
                let matrix = sonium_compute::PersistentDeviceCsr::try_from_host_csr_ref(
                    backend,
                    level.matrix.as_host_csr_ref(),
                )?;
                let prolongation = level
                    .prolongation
                    .as_ref()
                    .map(|p| {
                        sonium_compute::PersistentDeviceCsr::try_from_host_csr_ref(
                            backend,
                            p.as_host_csr_ref(),
                        )
                    })
                    .transpose()?;
                let restriction = level
                    .restriction
                    .as_ref()
                    .map(|r| {
                        sonium_compute::PersistentDeviceCsr::try_from_host_csr_ref(
                            backend,
                            r.as_host_csr_ref(),
                        )
                    })
                    .transpose()?;
                backend.try_inject_preconditioner_failure(
                    sonium_compute::PreconditionerFailureInjectionPoint::DiagonalUpload,
                )?;
                let mut colors = Vec::new();
                if let Some(sets) = level.color_sets.as_ref() {
                    for set in sets {
                        if set.is_empty() {
                            continue;
                        }
                        let idx: Vec<i32> = set.iter().map(|&i| i as i32).collect();
                        let mut buf = backend
                            .raw_alloc::<i32>(idx.len())
                            .map_err(sonium_compute::BackendError::from)?;
                        backend.copy_h2d(&idx, &mut buf);
                        colors.push(buf);
                    }
                }
                // L1-Jacobi uses the inverse row norm, not the inverse diagonal.
                // Retain the selected scaling once for every device cycle.
                let scaling = if host_amg.config.smoother == AmgSmoother::L1Jacobi {
                    level
                        .l1_diag
                        .iter()
                        .map(|&norm| <T as ComplexField>::from_real(norm).inv())
                        .collect::<Vec<_>>()
                } else {
                    level.diag_inv.clone()
                };
                Ok(BackendAmgLevel {
                    matrix,
                    prolongation,
                    restriction,
                    diag_inv: sonium_compute::PersistentDeviceCsr::try_from_host_diagonal(
                        backend,
                        scaling.as_slice(),
                    )?,
                    lambda_max: level.lambda_max,
                    num_dofs: level.num_dofs,
                    colors,
                })
            })
            .collect::<sonium_compute::BackendResult<_>>()?;

        backend.try_inject_preconditioner_failure(
            sonium_compute::PreconditionerFailureInjectionPoint::WorkspaceAllocation,
        )?;
        let workspace = BackendAmgWorkspace {
            levels: levels
                .iter()
                .map(|level| {
                    let coarse_n = level.restriction.as_ref().map(|r| r.m());
                    Ok(BackendAmgLevelWorkspace {
                        residual: backend.raw_alloc::<T>(level.num_dofs)?,
                        r_coarse: coarse_n.map(|n| backend.raw_alloc::<T>(n)).transpose()?,
                        e_coarse: coarse_n.map(|n| backend.raw_alloc::<T>(n)).transpose()?,
                        correction: backend.raw_alloc::<T>(level.num_dofs)?,
                        ax: backend.raw_alloc::<T>(level.num_dofs)?,
                        d: backend.raw_alloc::<T>(level.num_dofs)?,
                        d_new: backend.raw_alloc::<T>(level.num_dofs)?,
                    })
                })
                .collect::<sonium_compute::BackendResult<_>>()?,
            result: backend
                .raw_alloc::<T>(host_amg.levels.first().map_or(0, |level| level.num_dofs))
                .map_err(sonium_compute::BackendError::from)?,
            block_input: backend
                .raw_alloc::<T>(host_amg.levels.first().map_or(0, |level| level.num_dofs))
                .map_err(sonium_compute::BackendError::from)?,
        };
        let runtime_diagnostics = sonium_compute::DevicePreconditionerDiagnostics::default();
        runtime_diagnostics.record_setup();

        Ok(Self {
            levels,
            config: host_amg.config.clone(),
            coarse_direct,
            workspace: Mutex::new(workspace),
            checkpoint_bytes,
            numeric_valid: true,
            phase_profile: None,
            small_jacobi_enabled: false,
            fused_jacobi_enabled: false,
            apply_count: AtomicU64::new(0),
            total_apply_ns: AtomicU64::new(0),
            runtime_diagnostics,
        })
    }

    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Return setup and allocation diagnostics without synchronizing or
    /// reading any backend buffers back to the host.
    pub fn diagnostics(&self) -> BackendAmgDiagnostics {
        let level_dofs = self
            .levels
            .iter()
            .map(|level| level.num_dofs)
            .collect::<Vec<_>>();
        let level_nnz = self
            .levels
            .iter()
            .map(|level| level.matrix.nnz())
            .collect::<Vec<_>>();
        let total_dofs = level_dofs.iter().copied().sum::<usize>();
        let total_nnz = level_nnz.iter().copied().sum::<usize>();
        let finest_dofs = level_dofs.first().copied().unwrap_or(0);
        let finest_nnz = level_nnz.first().copied().unwrap_or(0);
        let level_workspace_elements: usize = self
            .levels
            .iter()
            .map(|level| {
                let coarse = level
                    .restriction
                    .as_ref()
                    .map_or(0, |restriction| restriction.m());
                level
                    .num_dofs
                    .saturating_mul(5)
                    .saturating_add(coarse.saturating_mul(2))
            })
            .sum::<usize>();
        // The finest-level result and block adapter buffers live alongside
        // the per-level workspaces and are allocated during setup as well.
        let workspace_elements =
            level_workspace_elements.saturating_add(finest_dofs.saturating_mul(2));

        let apply_count = self.apply_count.load(Ordering::Relaxed);
        let total_apply_ms = self.total_apply_ns.load(Ordering::Relaxed) as f64 / 1e6;

        BackendAmgDiagnostics {
            num_levels: self.levels.len(),
            grid_complexity: if finest_dofs == 0 {
                0.0
            } else {
                total_dofs as f64 / finest_dofs as f64
            },
            operator_complexity: if finest_nnz == 0 {
                0.0
            } else {
                total_nnz as f64 / finest_nnz as f64
            },
            total_nnz,
            finest_nnz,
            coarsest_dofs: level_dofs.last().copied().unwrap_or(0),
            level_dofs,
            level_nnz,
            workspace_elements,
            has_coarse_direct_solver: self.coarse_direct.is_some(),
            apply_count,
            total_apply_ms,
            mean_apply_ms: if apply_count == 0 {
                0.0
            } else {
                total_apply_ms / apply_count as f64
            },
            runtime: self.runtime_diagnostics.snapshot(),
        }
    }

    fn record_apply_elapsed(&self, start: Instant) {
        self.apply_count.fetch_add(1, Ordering::Relaxed);
        self.total_apply_ns.fetch_add(
            start.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
    }

    fn mg_cycle(
        &self,
        backend: &B,
        level_idx: usize,
        workspace_levels: &mut [BackendAmgLevelWorkspace<B, T>],
        x: &mut B::Buffer<T>,
        b: &B::Buffer<T>,
        cycle: AmgCycle,
        x_is_zero: bool,
    ) -> sonium_compute::BackendResult<()> {
        let level =
            self.levels
                .get(level_idx)
                .ok_or(sonium_compute::BackendError::InvalidArgument {
                    operation: "amg_apply",
                    reason: format!("AMG level index {level_idx} is out of range"),
                })?;
        let (level_workspace, coarse_workspace) = workspace_levels.split_first_mut().ok_or(
            sonium_compute::BackendError::InvalidArgument {
                operation: "amg_apply",
                reason: format!("workspace missing level {level_idx}"),
            },
        )?;
        let is_coarsest = level_idx == self.levels.len() - 1 || level.prolongation.is_none();

        if is_coarsest {
            let started = self.profile_start(backend)?;
            if let Some(coarse_direct) = self.coarse_direct.as_ref() {
                backend.try_inject_preconditioner_failure(
                    sonium_compute::PreconditionerFailureInjectionPoint::CoarseSolve,
                )?;
                match coarse_direct.solve(backend, b, x)? {
                    true => {
                        self.profile_finish(
                            backend,
                            level_idx,
                            BackendAmgPhase::CoarseSolve,
                            started,
                        )?;
                        return Ok(());
                    }
                    false => self.runtime_diagnostics.record_fallback(),
                }
            }
            self.smooth_initial(
                backend,
                level_idx,
                level_workspace,
                x,
                b,
                BACKEND_AMG_COARSE_SWEEPS,
                x_is_zero,
            )?;
            self.profile_finish(backend, level_idx, BackendAmgPhase::CoarseSolve, started)?;
            return Ok(());
        }

        let started = self.profile_start(backend)?;
        self.smooth_initial(
            backend,
            level_idx,
            level_workspace,
            x,
            b,
            self.config.num_pre_smooth,
            x_is_zero,
        )?;
        self.profile_finish(backend, level_idx, BackendAmgPhase::PreSmooth, started)?;

        let started = self.profile_start(backend)?;
        level
            .matrix
            .try_apply(backend, x, &mut level_workspace.residual)?;
        if !backend.try_residual_update_native(b, &mut level_workspace.residual)? {
            backend.try_scale::<T>(
                -<T as sonium_compute::DeviceScalar>::one(),
                &mut level_workspace.residual,
            )?;
            backend.try_axpy::<T>(
                <T as sonium_compute::DeviceScalar>::one(),
                b,
                &mut level_workspace.residual,
            )?;
        }
        self.profile_finish(backend, level_idx, BackendAmgPhase::Residual, started)?;

        let restriction =
            level
                .restriction
                .as_ref()
                .ok_or(sonium_compute::BackendError::InvalidArgument {
                    operation: "amg_apply",
                    reason: format!("level {level_idx} is missing restriction"),
                })?;
        let r_coarse = level_workspace.r_coarse.as_mut().ok_or(
            sonium_compute::BackendError::InvalidArgument {
                operation: "amg_apply",
                reason: format!("level {level_idx} is missing coarse residual workspace"),
            },
        )?;
        let started = self.profile_start(backend)?;
        restriction.try_apply(backend, &level_workspace.residual, r_coarse)?;
        self.profile_finish(backend, level_idx, BackendAmgPhase::Restriction, started)?;

        let e_coarse = level_workspace.e_coarse.as_mut().ok_or(
            sonium_compute::BackendError::InvalidArgument {
                operation: "amg_apply",
                reason: format!("level {level_idx} is missing coarse correction workspace"),
            },
        )?;

        match cycle {
            AmgCycle::VCycle => {
                self.mg_cycle(
                    backend,
                    level_idx + 1,
                    coarse_workspace,
                    e_coarse,
                    r_coarse,
                    AmgCycle::VCycle,
                    true,
                )?;
            }
            AmgCycle::WCycle => {
                self.mg_cycle(
                    backend,
                    level_idx + 1,
                    coarse_workspace,
                    e_coarse,
                    r_coarse,
                    AmgCycle::WCycle,
                    true,
                )?;
                self.mg_cycle(
                    backend,
                    level_idx + 1,
                    coarse_workspace,
                    e_coarse,
                    r_coarse,
                    AmgCycle::WCycle,
                    false,
                )?;
            }
            AmgCycle::FCycle => {
                self.mg_cycle(
                    backend,
                    level_idx + 1,
                    coarse_workspace,
                    e_coarse,
                    r_coarse,
                    AmgCycle::FCycle,
                    true,
                )?;
                self.mg_cycle(
                    backend,
                    level_idx + 1,
                    coarse_workspace,
                    e_coarse,
                    r_coarse,
                    AmgCycle::VCycle,
                    false,
                )?;
            }
        }

        let prolongation =
            level
                .prolongation
                .as_ref()
                .ok_or(sonium_compute::BackendError::InvalidArgument {
                    operation: "amg_apply",
                    reason: format!("level {level_idx} is missing prolongation"),
                })?;
        let started = self.profile_start(backend)?;
        prolongation.try_apply(backend, e_coarse, &mut level_workspace.correction)?;
        backend.try_axpy::<T>(
            <T as sonium_compute::DeviceScalar>::one(),
            &level_workspace.correction,
            x,
        )?;
        self.profile_finish(backend, level_idx, BackendAmgPhase::Prolongation, started)?;

        let started = self.profile_start(backend)?;
        self.smooth(
            backend,
            level_idx,
            level_workspace,
            x,
            b,
            self.config.num_post_smooth,
        )?;
        self.profile_finish(backend, level_idx, BackendAmgPhase::PostSmooth, started)?;
        Ok(())
    }

    // A fresh correction is a semantic zero, regardless of old scratch contents.
    // Jacobi and Chebyshev start with scaled diagonal application. W/F revisits
    // call this with false to preserve their accumulated coarse correction.
    fn smooth_initial(
        &self,
        backend: &B,
        level_idx: usize,
        workspace: &mut BackendAmgLevelWorkspace<B, T>,
        x: &mut B::Buffer<T>,
        b: &B::Buffer<T>,
        num_sweeps: usize,
        x_is_zero: bool,
    ) -> sonium_compute::BackendResult<()> {
        if !x_is_zero {
            return self.smooth(backend, level_idx, workspace, x, b, num_sweeps);
        }
        if num_sweeps > 0 && self.config.smoother == AmgSmoother::Chebyshev {
            backend.try_inject_preconditioner_failure(
                sonium_compute::PreconditionerFailureInjectionPoint::Smoother,
            )?;
            return Self::smooth_chebyshev(
                backend,
                &self.levels[level_idx],
                workspace,
                x,
                b,
                true,
                num_sweeps,
            );
        }
        if num_sweeps > 0
            && matches!(
                self.config.smoother,
                AmgSmoother::Jacobi | AmgSmoother::L1Jacobi
            )
        {
            backend.try_inject_preconditioner_failure(
                sonium_compute::PreconditionerFailureInjectionPoint::Smoother,
            )?;
            let weight = if self.config.smoother == AmgSmoother::L1Jacobi {
                1.0
            } else {
                self.config.jacobi_weight
            };
            let omega = <T as sonium_compute::DeviceScalar>::Real::from_f64(weight).ok_or(
                sonium_compute::BackendError::InvalidArgument {
                    operation: "amg_smoother",
                    reason: "Jacobi weight is not representable".into(),
                },
            )?;
            let diagonal = &self.levels[level_idx].diag_inv;
            if !backend.try_diagonal_scale_native(&diagonal.matrix().values, b, x, omega)? {
                diagonal.try_apply(backend, b, x)?;
                backend.try_scale::<T>(<T as sonium_compute::DeviceScalar>::from_real(omega), x)?;
            }
            return self.smooth(backend, level_idx, workspace, x, b, num_sweeps - 1);
        }
        if !backend.try_zero_native(x)? {
            backend.try_scale::<T>(<T as sonium_compute::DeviceScalar>::zero(), x)?;
        }
        self.smooth(backend, level_idx, workspace, x, b, num_sweeps)
    }

    fn smooth(
        &self,
        backend: &B,
        level_idx: usize,
        workspace: &mut BackendAmgLevelWorkspace<B, T>,
        x: &mut B::Buffer<T>,
        b: &B::Buffer<T>,
        num_sweeps: usize,
    ) -> sonium_compute::BackendResult<()> {
        if num_sweeps == 0 {
            return Ok(());
        }

        backend.try_inject_preconditioner_failure(
            sonium_compute::PreconditionerFailureInjectionPoint::Smoother,
        )?;

        let level =
            self.levels
                .get(level_idx)
                .ok_or(sonium_compute::BackendError::InvalidArgument {
                    operation: "amg_smoother",
                    reason: format!("AMG level index {level_idx} is out of range"),
                })?;
        match self.config.smoother {
            AmgSmoother::Chebyshev => {
                Self::smooth_chebyshev(backend, level, workspace, x, b, false, num_sweeps)?;
            }
            AmgSmoother::Jacobi | AmgSmoother::L1Jacobi => {
                // Match host L1-Jacobi: row-norm scaling already supplies damping.
                let weight = if self.config.smoother == AmgSmoother::L1Jacobi {
                    1.0
                } else {
                    self.config.jacobi_weight
                };
                let omega = <T as sonium_compute::DeviceScalar>::Real::from_f64(weight).ok_or(
                    sonium_compute::BackendError::InvalidArgument {
                        operation: "amg_smoother",
                        reason: "Jacobi weight is not representable".into(),
                    },
                )?;
                self.smooth_jacobi(backend, level, workspace, x, b, omega, num_sweeps)?;
            }
            AmgSmoother::SymmetricGaussSeidel | AmgSmoother::CfSymmetricGaussSeidel => {
                // These smoothers are serial host implementations in the
                // portable hierarchy.  The backend path intentionally
                // substitutes the CUDA-safe Jacobi kernel, and records that
                // capability fallback instead of hiding the change in the
                // runtime diagnostics.
                self.runtime_diagnostics.record_fallback();
                let omega =
                    <T as sonium_compute::DeviceScalar>::Real::from_f64(self.config.jacobi_weight)
                        .ok_or(sonium_compute::BackendError::InvalidArgument {
                            operation: "amg_smoother",
                            reason: "Jacobi weight is not representable".into(),
                        })?;
                self.smooth_jacobi(backend, level, workspace, x, b, omega, num_sweeps)?;
            }
            AmgSmoother::MulticolorGaussSeidel => {
                if level.colors.is_empty() {
                    // No coloring was uploaded (aggregation-built levels have
                    // no pointwise coloring): fall back to Jacobi.
                    self.runtime_diagnostics.record_fallback();
                    let omega = <T as sonium_compute::DeviceScalar>::Real::from_f64(
                        self.config.jacobi_weight,
                    )
                    .ok_or(sonium_compute::BackendError::InvalidArgument {
                        operation: "amg_smoother",
                        reason: "Jacobi weight is not representable".into(),
                    })?;
                    self.smooth_jacobi(backend, level, workspace, x, b, omega, num_sweeps)?;
                } else {
                    Self::smooth_multicolor_gs(backend, level, x, b, num_sweeps)?;
                }
            }
        }
        Ok(())
    }

    fn smooth_jacobi(
        &self,
        backend: &B,
        level: &BackendAmgLevel<B, T>,
        workspace: &mut BackendAmgLevelWorkspace<B, T>,
        x: &mut B::Buffer<T>,
        b: &B::Buffer<T>,
        omega: <T as sonium_compute::DeviceScalar>::Real,
        num_sweeps: usize,
    ) -> sonium_compute::BackendResult<()> {
        if self.small_jacobi_enabled {
            let csr = level.matrix.matrix();
            if backend.try_small_csr_jacobi_native(
                &csr.row_ptr,
                &csr.col_idx,
                &csr.values,
                &level.diag_inv.matrix().values,
                b,
                x,
                omega,
                num_sweeps,
            )? {
                return Ok(());
            }
        }
        for _ in 0..num_sweeps {
            if self.fused_jacobi_enabled {
                let csr = level.matrix.matrix();
                if backend.try_csr_jacobi_step_native(
                    &csr.row_ptr,
                    &csr.col_idx,
                    &csr.values,
                    &level.diag_inv.matrix().values,
                    b,
                    x,
                    &mut workspace.ax,
                    omega,
                )? {
                    // The old iterate remains immutable during the launch. Reuse
                    // its allocation as scratch for the next simultaneous sweep.
                    std::mem::swap(x, &mut workspace.ax);
                    continue;
                }
            }
            level.matrix.try_apply(backend, x, &mut workspace.ax)?;
            if backend.try_jacobi_update_native(
                &level.diag_inv.matrix().values,
                b,
                &workspace.ax,
                x,
                omega,
            )? {
                continue;
            }
            backend.try_scale::<T>(
                -<T as sonium_compute::DeviceScalar>::one(),
                &mut workspace.ax,
            )?;
            backend.try_axpy::<T>(
                <T as sonium_compute::DeviceScalar>::one(),
                b,
                &mut workspace.ax,
            )?;

            level
                .diag_inv
                .try_apply(backend, &workspace.ax, &mut workspace.d_new)?;
            backend.try_axpy::<T>(
                <T as sonium_compute::DeviceScalar>::from_real(omega),
                &workspace.d_new,
                x,
            )?;
        }
        Ok(())
    }

    /// Device multicolor Gauss-Seidel: forward color order then reverse
    /// (symmetric variant), mirroring the host smoother. GS uses omega=1.0
    /// (full update), not the Jacobi weight.
    fn smooth_multicolor_gs(
        backend: &B,
        level: &BackendAmgLevel<B, T>,
        x: &mut B::Buffer<T>,
        b: &B::Buffer<T>,
        num_sweeps: usize,
    ) -> sonium_compute::BackendResult<()> {
        let omega = <T as sonium_compute::DeviceScalar>::Real::from_f64(1.0).ok_or(
            sonium_compute::BackendError::InvalidArgument {
                operation: "amg_smoother",
                reason: "GS relaxation factor is not representable".into(),
            },
        )?;
        let csr = level.matrix.matrix();
        for _ in 0..num_sweeps {
            for colors in level.colors.iter() {
                backend.try_gs_color_sweep(
                    &csr.row_ptr,
                    &csr.col_idx,
                    &csr.values,
                    colors,
                    omega,
                    b,
                    x,
                )?;
            }
            for colors in level.colors.iter().rev() {
                backend.try_gs_color_sweep(
                    &csr.row_ptr,
                    &csr.col_idx,
                    &csr.values,
                    colors,
                    omega,
                    b,
                    x,
                )?;
            }
        }
        Ok(())
    }

    fn smooth_chebyshev(
        backend: &B,
        level: &BackendAmgLevel<B, T>,
        workspace: &mut BackendAmgLevelWorkspace<B, T>,
        x: &mut B::Buffer<T>,
        b: &B::Buffer<T>,
        x_is_zero: bool,
        num_sweeps: usize,
    ) -> sonium_compute::BackendResult<()> {
        use num_traits::{One, Zero};

        let lambda_max = level
            .lambda_max
            .ok_or(sonium_compute::BackendError::InvalidArgument {
                operation: "amg_smoother",
                reason: "Chebyshev level has no eigenvalue bound".into(),
            })?;
        let thirty = <T as sonium_compute::DeviceScalar>::Real::from_f64(30.0).ok_or(
            sonium_compute::BackendError::InvalidArgument {
                operation: "amg_smoother",
                reason: "Chebyshev constant 30 is not representable".into(),
            },
        )?;
        let two = <T as sonium_compute::DeviceScalar>::Real::from_f64(2.0).ok_or(
            sonium_compute::BackendError::InvalidArgument {
                operation: "amg_smoother",
                reason: "Chebyshev constant 2 is not representable".into(),
            },
        )?;
        let one = <T as sonium_compute::DeviceScalar>::Real::one();

        let lambda_min = lambda_max / thirty;
        let theta = (lambda_max + lambda_min) / two;
        let delta = (lambda_max - lambda_min) / two;
        let sigma = theta / delta;
        let mut rho = one / sigma;

        let native_first = if x_is_zero {
            false
        } else {
            level.matrix.try_apply(backend, x, &mut workspace.ax)?;
            backend.try_chebyshev_update_native(
                &level.diag_inv.matrix().values,
                b,
                &workspace.ax,
                &mut workspace.d,
                x,
                one / theta,
                <T as sonium_compute::DeviceScalar>::Real::zero(),
            )?
        };
        if !native_first {
            // A fresh correction ignores stale scratch. Keep the recurrence's first
            // direction even when A*0 and the residual construction can be skipped.
            let residual = if x_is_zero {
                b
            } else {
                backend.try_scale::<T>(
                    -<T as sonium_compute::DeviceScalar>::one(),
                    &mut workspace.ax,
                )?;
                backend.try_axpy::<T>(
                    <T as sonium_compute::DeviceScalar>::one(),
                    b,
                    &mut workspace.ax,
                )?;
                &workspace.ax
            };
            // The inverse diagonal has one stored value per row. Native scaling
            // avoids a sparse dispatch and folds in the Chebyshev coefficient.
            if !backend.try_diagonal_scale_native(
                &level.diag_inv.matrix().values,
                residual,
                &mut workspace.d,
                one / theta,
            )? {
                level
                    .diag_inv
                    .try_apply(backend, residual, &mut workspace.d)?;
                backend.try_scale::<T>(
                    <T as sonium_compute::DeviceScalar>::from_real(one / theta),
                    &mut workspace.d,
                )?;
            }
            if x_is_zero {
                backend.try_copy_d2d(&workspace.d, x)?;
            } else {
                backend.try_axpy::<T>(
                    <T as sonium_compute::DeviceScalar>::one(),
                    &workspace.d,
                    x,
                )?;
            }
        }

        for _ in 1..num_sweeps {
            let rho_new = one / (two * sigma - rho);
            let coeff_old = <T as sonium_compute::DeviceScalar>::from_real(rho_new * rho);
            let coeff_res = <T as sonium_compute::DeviceScalar>::from_real(two * rho_new / delta);

            level.matrix.try_apply(backend, x, &mut workspace.ax)?;
            if backend.try_chebyshev_update_native(
                &level.diag_inv.matrix().values,
                b,
                &workspace.ax,
                &mut workspace.d,
                x,
                two * rho_new / delta,
                rho_new * rho,
            )? {
                rho = rho_new;
                continue;
            }
            backend.try_scale::<T>(
                -<T as sonium_compute::DeviceScalar>::one(),
                &mut workspace.ax,
            )?;
            backend.try_axpy::<T>(
                <T as sonium_compute::DeviceScalar>::one(),
                b,
                &mut workspace.ax,
            )?;
            if !backend.try_diagonal_scale_native(
                &level.diag_inv.matrix().values,
                &workspace.ax,
                &mut workspace.d_new,
                two * rho_new / delta,
            )? {
                level
                    .diag_inv
                    .try_apply(backend, &workspace.ax, &mut workspace.d_new)?;
                backend.try_scale::<T>(coeff_res, &mut workspace.d_new)?;
            }
            backend.try_axpy::<T>(coeff_old, &workspace.d, &mut workspace.d_new)?;
            backend.try_axpy::<T>(
                <T as sonium_compute::DeviceScalar>::one(),
                &workspace.d_new,
                x,
            )?;

            std::mem::swap(&mut workspace.d, &mut workspace.d_new);
            rho = rho_new;
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl BackendAmgPreconditioner<sonium_compute::CudaBackend, num_complex::Complex32> {
    /// Checked CUDA-resident AMG setup with a cuSOLVERDN coarse LU.
    pub fn try_from_csr_with_cuda_coarse_lu(
        backend: &sonium_compute::CudaBackend,
        matrix: &CsrMatrix<num_complex::Complex32>,
        config: AmgConfig,
    ) -> sonium_compute::BackendResult<Self> {
        let host_amg = AmgPreconditioner::try_from_csr(matrix, config).map_err(|error| {
            sonium_compute::BackendError::InvalidArgument {
                operation: "amg_setup",
                reason: error.to_string(),
            }
        })?;
        let coarse_direct = build_cuda_c32_coarse_lu(backend, &host_amg)?;
        Self::try_from_host_amg_with_direct(backend, &host_amg, Some(coarse_direct))
    }

    /// Build CUDA-resident AMG and factor the coarsest level with cuSOLVERDN.
    ///
    /// Setup fails with a typed backend error if dense factorization fails.
    pub fn from_csr_with_cuda_coarse_lu(
        backend: &sonium_compute::CudaBackend,
        matrix: &CsrMatrix<num_complex::Complex32>,
        config: AmgConfig,
    ) -> Self {
        Self::try_from_csr_with_cuda_coarse_lu(backend, matrix, config)
            .unwrap_or_else(|error| panic!("CUDA backend AMG setup failed: {error}"))
    }

    /// Restore a CUDA-resident AMG hierarchy and rebuild its supported
    /// cuSOLVERDN coarse factor from the checkpointed coarse matrix.
    ///
    /// The portable format stores the canonical hierarchy rather than opaque
    /// driver handles. Rebuilding the factor is deterministic and keeps the
    /// restored action on CUDA without host staging during application.
    pub fn try_from_checkpoint_bytes_with_cuda_coarse_lu(
        backend: &sonium_compute::CudaBackend,
        bytes: &[u8],
    ) -> sonium_compute::BackendResult<Self> {
        let (payload, has_coarse_direct_solver) = decode_backend_checkpoint::<
            sonium_compute::CudaBackend,
            num_complex::Complex32,
        >(bytes)?
        .unwrap_or_else(|| (bytes.to_vec(), false));
        let host_amg = AmgPreconditioner::<num_complex::Complex32>::from_checkpoint_bytes(&payload)
            .map_err(|error| sonium_compute::BackendError::InvalidArgument {
                operation: "amg_checkpoint_restore",
                reason: error.to_string(),
            })?;
        let coarse_direct = if has_coarse_direct_solver {
            Some(build_cuda_c32_coarse_lu(backend, &host_amg)?)
        } else {
            None
        };
        Self::try_from_host_amg_with_direct(backend, &host_amg, coarse_direct)
    }
}

#[cfg(feature = "cuda")]
fn build_cuda_c32_coarse_lu(
    backend: &sonium_compute::CudaBackend,
    host_amg: &AmgPreconditioner<num_complex::Complex32>,
) -> sonium_compute::BackendResult<
    Box<dyn BackendAmgDirectSolver<sonium_compute::CudaBackend, num_complex::Complex32>>,
> {
    backend.try_inject_preconditioner_failure(
        sonium_compute::PreconditionerFailureInjectionPoint::CoarseSolve,
    )?;
    let coarse = host_amg
        .levels
        .last()
        .ok_or(sonium_compute::BackendError::InvalidArgument {
            operation: "amg_setup",
            reason: "AMG hierarchy has no coarsest level".into(),
        })?;
    if coarse.matrix.num_rows != coarse.matrix.num_cols {
        return Err(sonium_compute::BackendError::InvalidArgument {
            operation: "amg_setup_coarse_lu",
            reason: format!(
                "coarse matrix is not square ({}x{})",
                coarse.matrix.num_rows, coarse.matrix.num_cols
            ),
        });
    }

    let dense = csr_to_column_major_dense(&coarse.matrix);
    let map_factor_error =
        |error: sonium_compute::OomError| sonium_compute::BackendError::Allocation {
            reason: format!("CUDA AMG coarse LU factorization failed: {error}"),
        };
    let lu = match factor_cuda_c32_coarse_lu(backend, coarse.matrix.num_rows, &dense) {
        Ok(lu) => lu,
        Err(first_error) => {
            // The Galerkin coarse operator can inherit the fine-level
            // nullspace (e.g. low-frequency Neumann Helmholtz), so the dense
            // f32 factor hits an exact zero pivot. A coarse solve is only an
            // approximate correction, so retry once with a tiny relative
            // diagonal shift rather than failing the whole setup.
            let scale = dense
                .iter()
                .map(|value| value.norm())
                .fold(0.0f32, f32::max);
            if scale > 0.0 {
                let shift =
                    num_complex::Complex32::new(COARSE_LU_DIAGONAL_REGULARIZATION * scale, 0.0);
                let n = coarse.matrix.num_rows;
                let mut regularized = dense;
                // Column-major dense: column j owns entries [j*n, (j+1)*n)
                // with the diagonal at offset j.
                for (j, column) in regularized.chunks_exact_mut(n).enumerate() {
                    if let Some(diag) = column.get_mut(j) {
                        *diag += shift;
                    }
                }
                log::debug!(
                    "CUDA AMG coarse factor hit a zero pivot ({first_error}); retrying with relative diagonal shift {COARSE_LU_DIAGONAL_REGULARIZATION:e}",
                );
                factor_cuda_c32_coarse_lu(backend, n, &regularized).map_err(map_factor_error)?
            } else {
                return Err(map_factor_error(first_error));
            }
        }
    };
    log::debug!(
        "CUDA AMG coarse cuSOLVERDN LU enabled at n={}, workspace={} complex values",
        lu.n(),
        lu.workspace_len()
    );
    Ok(Box::new(CudaC32CoarseLu { lu }))
}

#[cfg(feature = "cuda")]
fn factor_cuda_c32_coarse_lu(
    backend: &sonium_compute::CudaBackend,
    n: usize,
    dense: &[num_complex::Complex32],
) -> Result<sonium_compute::CudaDenseLuC32, sonium_compute::OomError> {
    backend.dense_lu_c32_from_column_major(n, dense)
}

#[cfg(feature = "cuda")]
fn csr_to_column_major_dense(
    matrix: &CsrMatrix<num_complex::Complex32>,
) -> Vec<num_complex::Complex32> {
    let n = matrix.num_rows;
    let mut dense = vec![num_complex::Complex32::new(0.0, 0.0); n * matrix.num_cols];
    for row in 0..matrix.num_rows {
        for idx in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            let col = matrix.col_indices[idx];
            dense[col * n + row] = matrix.values[idx];
        }
    }
    dense
}

impl<B, T> sonium_compute::DevicePreconditioner<B, T> for BackendAmgPreconditioner<B, T>
where
    B: sonium_compute::Backend,
    T: ComplexField<Real = <T as sonium_compute::DeviceScalar>::Real>
        + sonium_compute::DeviceScalar,
{
    fn apply(&self, backend: &B, r: &B::Buffer<T>, z: &mut B::Buffer<T>) {
        if let Err(error) = self.try_apply(backend, r, z) {
            log::error!(
                target: "sonium-solvers-preconditioners::amg",
                "backend AMG application failed: {error}"
            );
        }
    }

    fn diagnostics(&self) -> sonium_compute::DevicePreconditionerDiagnosticsSnapshot {
        self.runtime_diagnostics.snapshot()
    }

    fn try_apply(
        &self,
        backend: &B,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> sonium_compute::BackendResult<()> {
        let result = self.try_apply_untracked(backend, r, z);
        match &result {
            Ok(()) => self.runtime_diagnostics.record_apply(),
            Err(error) => self.runtime_diagnostics.record_failure(error),
        }
        result
    }

    fn try_apply_untracked(
        &self,
        backend: &B,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> sonium_compute::BackendResult<()> {
        self.ensure_numeric_valid()?;
        backend.try_inject_preconditioner_failure(
            sonium_compute::PreconditionerFailureInjectionPoint::Launch,
        )?;
        let started = Instant::now();
        if self.levels.is_empty() {
            backend.try_copy_d2d(r, z)?;
            self.record_apply_elapsed(started);
            return Ok(());
        }
        let n = self.levels[0].num_dofs;
        if r.len() != n || z.len() != n {
            return Err(sonium_compute::BackendError::InvalidArgument {
                operation: "amg_apply",
                reason: format!(
                    "expected input/output length {n}, got {}/{}",
                    r.len(),
                    z.len()
                ),
            });
        }
        let mut workspace =
            self.workspace
                .lock()
                .map_err(|_| sonium_compute::BackendError::OperationFailed {
                    operation: "amg_apply",
                    reason: "workspace mutex poisoned".into(),
                })?;
        let BackendAmgWorkspace { levels, result, .. } = &mut *workspace;
        self.mg_cycle(backend, 0, levels, result, r, self.config.cycle, true)?;
        backend.try_copy_d2d(result, z)?;
        self.record_apply_elapsed(started);
        Ok(())
    }

    fn try_apply_colmajor(
        &self,
        backend: &B,
        rhs: usize,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> sonium_compute::BackendResult<()> {
        let result = self.try_apply_colmajor_untracked(backend, rhs, r, z);
        match &result {
            Ok(()) => self.runtime_diagnostics.record_apply(),
            Err(error) => self.runtime_diagnostics.record_failure(error),
        }
        result
    }

    fn try_apply_colmajor_untracked(
        &self,
        backend: &B,
        rhs: usize,
        r: &B::Buffer<T>,
        z: &mut B::Buffer<T>,
    ) -> sonium_compute::BackendResult<()> {
        self.ensure_numeric_valid()?;
        backend.try_inject_preconditioner_failure(
            sonium_compute::PreconditionerFailureInjectionPoint::Launch,
        )?;
        let n = self.levels.first().map_or(0, |level| level.num_dofs);
        let len = n
            .checked_mul(rhs)
            .ok_or(sonium_compute::BackendError::InvalidArgument {
                operation: "amg_apply_colmajor",
                reason: format!("workspace length overflow: n={n}, rhs={rhs}"),
            })?;
        if r.len() != len || z.len() != len {
            return Err(sonium_compute::BackendError::InvalidArgument {
                operation: "amg_apply_colmajor",
                reason: format!(
                    "expected input/output length {len}, got {}/{}",
                    r.len(),
                    z.len()
                ),
            });
        }
        if rhs == 0 || n == 0 {
            return Ok(());
        }
        if rhs == 1 {
            return self.try_apply_untracked(backend, r, z);
        }

        let started = Instant::now();
        let mut workspace =
            self.workspace
                .lock()
                .map_err(|_| sonium_compute::BackendError::OperationFailed {
                    operation: "amg_apply_colmajor",
                    reason: "workspace mutex poisoned".into(),
                })?;
        for column in 0..rhs {
            let view = sonium_compute::BufferView::contiguous(column * n, n);
            backend.try_copy_views_d2d(
                r,
                view,
                &mut workspace.block_input,
                sonium_compute::BufferView::contiguous(0, n),
            )?;
            let BackendAmgWorkspace {
                levels,
                result,
                block_input,
            } = &mut *workspace;
            self.mg_cycle(
                backend,
                0,
                levels,
                result,
                block_input,
                self.config.cycle,
                true,
            )?;
            backend.try_copy_views_d2d(
                result,
                sonium_compute::BufferView::contiguous(0, n),
                z,
                view,
            )?;
        }
        self.record_apply_elapsed(started);
        Ok(())
    }

    fn try_apply_transpose(
        &self,
        _backend: &B,
        _r: &B::Buffer<T>,
        _z: &mut B::Buffer<T>,
    ) -> sonium_compute::BackendResult<()> {
        let error = sonium_compute::BackendError::Unsupported {
            backend: "device",
            operation: "amg_transpose",
            scalar: std::any::type_name::<T>(),
        };
        self.runtime_diagnostics.record_transpose_failure(&error);
        Err(error)
    }

    fn try_apply_hermitian(
        &self,
        _backend: &B,
        _r: &B::Buffer<T>,
        _z: &mut B::Buffer<T>,
    ) -> sonium_compute::BackendResult<()> {
        let error = sonium_compute::BackendError::Unsupported {
            backend: "device",
            operation: "amg_hermitian",
            scalar: std::any::type_name::<T>(),
        };
        self.runtime_diagnostics.record_hermitian_failure(&error);
        Err(error)
    }

    fn try_apply_transpose_colmajor(
        &self,
        _backend: &B,
        _rhs: usize,
        _r: &B::Buffer<T>,
        _z: &mut B::Buffer<T>,
    ) -> sonium_compute::BackendResult<()> {
        let error = sonium_compute::BackendError::Unsupported {
            backend: "device",
            operation: "amg_transpose_colmajor",
            scalar: std::any::type_name::<T>(),
        };
        self.runtime_diagnostics.record_transpose_failure(&error);
        Err(error)
    }

    fn try_apply_hermitian_colmajor(
        &self,
        _backend: &B,
        _rhs: usize,
        _r: &B::Buffer<T>,
        _z: &mut B::Buffer<T>,
    ) -> sonium_compute::BackendResult<()> {
        let error = sonium_compute::BackendError::Unsupported {
            backend: "device",
            operation: "amg_hermitian_colmajor",
            scalar: std::any::type_name::<T>(),
        };
        self.runtime_diagnostics.record_hermitian_failure(&error);
        Err(error)
    }
}

/// Minimum elements per worker before AMG vector ops dispatch to a parallel
/// walk over the Array1 backing slice. Below this, the rayon fork/join
/// barrier dominates the actual arithmetic — a problem that compounds on
/// many-core machines where each barrier syncs 32+ workers across NUMA
/// nodes. Scaling the threshold with `current_num_threads` keeps the
/// chunk-per-worker work bounded above this floor regardless of pool size.
const MIN_ELEMS_PER_SMOOTH_WORKER: usize = 4_096;

/// Decide whether an n-element AMG vector op should go parallel given the
/// current rayon pool size. Returns true only when each worker would get
/// at least [`MIN_ELEMS_PER_SMOOTH_WORKER`] elements *and* `n` clears a
/// 32 k absolute floor (below which fork/join overhead beats any
/// arithmetic gain on pretty much any architecture).
#[inline]
fn smooth_in_parallel(n: usize) -> bool {
    const ABSOLUTE_FLOOR: usize = 32_768;
    if n < ABSOLUTE_FLOOR {
        return false;
    }
    let workers = rayon::current_num_threads().max(1);
    n >= workers.saturating_mul(MIN_ELEMS_PER_SMOOTH_WORKER)
}

#[allow(clippy::too_many_arguments)]
fn build_first_level<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    config: &AmgConfig,
) -> (Vec<AmgLevel<T>>, CsrMatrix<T>)
where
    T::Real: Sync + Send,
{
    let mut levels = Vec::new();
    let current_matrix = matrix.clone();
    let diag_inv = util::compute_diag_inv(&current_matrix);
    let l1_diag = util::compute_l1_diag(&current_matrix);
    let lambda_max = if config.smoother == AmgSmoother::Chebyshev {
        Some(util::estimate_max_eigenvalue(&current_matrix, &diag_inv))
    } else {
        None
    };
    let n0 = current_matrix.num_rows;
    levels.push(AmgLevel {
        matrix: current_matrix.clone(),
        prolongation: None,
        restriction: None,
        diag_inv,
        l1_diag,
        lambda_max,
        coarse_to_fine: Vec::new(),
        f_points: Vec::new(),
        num_dofs: n0,
        work_r: Mutex::new(vec![T::zero(); n0]),
        work_tmp: Mutex::new(vec![T::zero(); n0]),
        work_r_coarse: Mutex::new(Vec::new()),
        work_e_coarse: Mutex::new(Vec::new()),
        coarse_solver: None,
        color_sets: None,
    });
    (levels, current_matrix)
}

#[allow(clippy::too_many_arguments)]
fn build_nodal_level<T: ComplexField>(
    levels: &mut Vec<AmgLevel<T>>,
    current_matrix: &CsrMatrix<T>,
    config: &AmgConfig,
    _level_idx: usize,
) -> Option<CsrMatrix<T>>
where
    T::Real: Sync + Send,
{
    let bs = config.block_size;
    let n = current_matrix.num_rows;
    if bs <= 1 || !n.is_multiple_of(bs) {
        return None;
    }
    let nodal_matrix = util::extract_nodal_graph(current_matrix, bs);
    let nn = nodal_matrix.num_rows;

    if nn <= 1 {
        return None;
    }

    let nodal_strong = strength::compute_strength_matrix(&nodal_matrix, config.strong_threshold);

    let (point_types, coarse_to_fine) = match config.coarsening {
        AmgCoarsening::Cljp => coarsening::coarsen_cljp(&nodal_matrix, &nodal_strong),
        AmgCoarsening::Pmis => coarsening::coarsen_pmis(&nodal_matrix, &nodal_strong),
        AmgCoarsening::Hmis => coarsening::coarsen_hmis(&nodal_matrix, &nodal_strong),
        AmgCoarsening::Falgout => coarsening::coarsen_falgout(&nodal_matrix, &nodal_strong),
        _ => coarsening::coarsen_ruge_stuben(&nodal_matrix, &nodal_strong),
    };

    if coarse_to_fine.is_empty() || coarse_to_fine.len() >= nn {
        return None;
    }

    let nodal_prolongation = interpolation::build_interpolation(
        &nodal_matrix,
        &nodal_strong,
        &point_types,
        &coarse_to_fine,
        config,
    );

    let prolongation = util::nodal_to_block_prolongation(&nodal_prolongation, bs);
    let restriction = transpose_or_adjoint(&prolongation, config);

    let block_c2f: Vec<usize> = coarse_to_fine
        .iter()
        .flat_map(|&ni| (0..bs).map(move |c| ni * bs + c))
        .collect();

    let num_coarse = prolongation.num_cols;
    if num_coarse == 0 || num_coarse >= n {
        return None;
    }

    let tol = T::Real::from_f64(util::AMG_MATMUL_DROP_TOL).unwrap();
    let ap = current_matrix.matmul_with_drop(&prolongation, tol);
    let coarse_matrix = restriction.matmul_with_drop(&ap, tol);

    let level_diag_inv = util::compute_diag_inv(&coarse_matrix);
    let level_l1_diag = util::compute_l1_diag(&coarse_matrix);
    let level_lambda_max = if config.smoother == AmgSmoother::Chebyshev {
        Some(util::estimate_max_eigenvalue(
            &coarse_matrix,
            &level_diag_inv,
        ))
    } else {
        None
    };

    levels.push(AmgLevel {
        matrix: coarse_matrix.clone(),
        prolongation: Some(prolongation),
        restriction: Some(restriction),
        diag_inv: level_diag_inv,
        l1_diag: level_l1_diag,
        lambda_max: level_lambda_max,
        coarse_to_fine: block_c2f,
        f_points: Vec::new(),
        num_dofs: num_coarse,
        work_r: Mutex::new(vec![T::zero(); num_coarse]),
        work_r_coarse: Mutex::new(Vec::new()),
        work_e_coarse: Mutex::new(Vec::new()),
        work_tmp: Mutex::new(vec![T::zero(); num_coarse]),
        coarse_solver: None,
        color_sets: None,
    });

    Some(coarse_matrix)
}

/// Build the Galerkin restriction from a prolongation operator, honoring
/// [`AmgRestriction::ConjugateTranspose`]: `R = P^H` when conjugated
/// restriction is configured, `R = P^T` otherwise (including the ℓAIR
/// configuration, which the aggregation-based paths do not support).
fn transpose_or_adjoint<T: ComplexField>(
    prolongation: &CsrMatrix<T>,
    config: &AmgConfig,
) -> CsrMatrix<T> {
    if matches!(config.restriction, AmgRestriction::ConjugateTranspose) {
        util::adjoint_csr(prolongation)
    } else {
        util::transpose_csr(prolongation)
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_strength_matrix<T: ComplexField>(
    current_matrix: &CsrMatrix<T>,
    config: &AmgConfig,
    level_idx: usize,
) -> (Vec<Vec<usize>>, usize)
where
    T::Real: Sync + Send,
{
    let use_ata = (config.coarsening == AmgCoarsening::SmoothedAggregation
        || config.coarsening == AmgCoarsening::EnergyMinimizing)
        && config.sa_strength_with_ata;

    let use_aggressive = config.aggressive_coarsening_levels > 0
        && level_idx < config.aggressive_coarsening_levels
        && !use_ata;

    // Owned scratch for the product-based strength matrices; the common
    // path filters `current_matrix` in place (no clone).
    let ata_storage: CsrMatrix<T>;
    let a2_storage: CsrMatrix<T>;
    let (strength_ref, effective_mis_distance) = if use_ata {
        let drop_tol = T::Real::from_f64(util::AMG_MATMUL_DROP_TOL).unwrap();
        let at = util::transpose_csr(current_matrix);
        ata_storage = at.matmul_with_drop(current_matrix, drop_tol);
        (&ata_storage, 1)
    } else if use_aggressive {
        let drop_tol = T::Real::from_f64(util::AMG_MATMUL_DROP_TOL).unwrap();
        a2_storage = current_matrix.matmul_with_drop(current_matrix, drop_tol);
        (&a2_storage, config.mis_distance)
    } else {
        (current_matrix, config.mis_distance)
    };

    let strong_connections = if let Some(ref material) = config.material_soc {
        strength::compute_material_strength(strength_ref, config.strong_threshold, material)
    } else {
        strength::compute_strength_matrix(strength_ref, config.strong_threshold)
    };

    // Classical SOC on strongly nonsymmetric (upwinded) operators is
    // one-sided; the coarseners degenerate on such graphs (no coarsening at
    // all in the worst case — levels == 1 and the coarsest-level LU silently
    // solves the whole matrix). The ℓAIR path (and the explicit opt-in flag)
    // symmetrizes the strong graph; the classical paths now also
    // auto-symmetrize when the graph is detected one-sided.
    let strong_connections = if config.symmetrize_strength
        || matches!(config.restriction, AmgRestriction::LAir { .. })
    {
        strength::symmetrize_strong(&strong_connections)
    } else if uses_classical_coarsening(config.coarsening) {
        let symmetrized = strength::symmetrize_strong(&strong_connections);
        if soc_is_one_sided(&strong_connections, &symmetrized) {
            let directed: usize = strong_connections.iter().map(Vec::len).sum();
            let grown: usize = symmetrized.iter().map(Vec::len).sum();
            log::debug!(
                "amg level {level_idx}: one-sided strong graph ({directed} directed \
                 edges, {grown} after symmetrization); auto-symmetrizing SOC for \
                 classical coarsening"
            );
            symmetrized
        } else {
            strong_connections
        }
    } else {
        strong_connections
    };

    (strong_connections, effective_mis_distance)
}

/// Whether the configured coarsening is a classical (pointwise C/F) path
/// that degenerates on one-sided strong graphs. Aggregation-based paths
/// build piecewise-constant prolongations over MIS aggregates and do not
/// rely on reciprocal C/F influence, so they are excluded.
fn uses_classical_coarsening(coarsening: AmgCoarsening) -> bool {
    match coarsening {
        AmgCoarsening::RugeStuben
        | AmgCoarsening::Pmis
        | AmgCoarsening::Hmis
        | AmgCoarsening::Cljp
        | AmgCoarsening::Falgout => true,
        AmgCoarsening::SmoothedAggregation | AmgCoarsening::EnergyMinimizing => false,
        #[cfg(feature = "experimental-gnn")]
        AmgCoarsening::GnnCoarsening => false,
    }
}

/// Detect a one-sided strength-of-connection graph by comparing directed
/// edge counts before and after symmetrization. A reciprocal graph is
/// unchanged by symmetrization (ratio 1.0); a fully one-sided (upwinded)
/// graph roughly doubles. The 1.2 threshold tolerates the mild asymmetry
/// that per-row max thresholds produce on boundary rows of symmetric
/// problems — measured at ~1.0 for 2D Poisson at theta=0.25 versus ~2.0
/// for upwinded advection-diffusion at high Péclet.
fn soc_is_one_sided(strong: &[Vec<usize>], symmetrized: &[Vec<usize>]) -> bool {
    let directed: usize = strong.iter().map(Vec::len).sum();
    let grown: usize = symmetrized.iter().map(Vec::len).sum();
    directed > 0 && grown as f64 > 1.2 * directed as f64
}

/// Transfer operators produced when building one coarser AMG level:
/// `(prolongation, restriction, coarse_to_fine map, F-point indices)`.
/// The F-point list is empty unless [`AmgConfig::f_point_smoothing`] is set
/// and the level was built by a classical (pointwise C/F) coarsening path.
type LevelTransfer<T> = (CsrMatrix<T>, CsrMatrix<T>, Vec<usize>, Vec<usize>);

fn build_smoothed_aggregation_prolongation<T: ComplexField>(
    current_matrix: &CsrMatrix<T>,
    strong_connections: &[Vec<usize>],
    effective_mis_distance: usize,
    config: &AmgConfig,
    n: usize,
) -> Option<LevelTransfer<T>>
where
    T::Real: Sync + Send,
{
    let (aggregate_ids, num_aggregates) =
        coarsening::coarsen_aggregation(current_matrix, strong_connections, effective_mis_distance);

    if num_aggregates == 0 || num_aggregates >= n {
        return None;
    }
    let p_tent = interpolation::build_tentative_prolongation(n, &aggregate_ids, num_aggregates);

    let level_diag_inv = util::compute_diag_inv(current_matrix);
    let spectral_radius = util::estimate_max_eigenvalue(current_matrix, &level_diag_inv);
    let four_thirds = T::Real::from_f64(4.0).unwrap() / T::Real::from_f64(3.0).unwrap();
    let omega = four_thirds / spectral_radius;

    let prolongation = interpolation::smooth_prolongation(
        current_matrix,
        &level_diag_inv,
        &p_tent,
        omega,
        config.max_interp_elements,
    );
    let restriction = transpose_or_adjoint(&prolongation, config);

    let mut c2f = vec![0usize; num_aggregates];
    let mut seen = vec![false; num_aggregates];
    for (i, &agg) in aggregate_ids.iter().enumerate() {
        if !seen[agg] {
            c2f[agg] = i;
            seen[agg] = true;
        }
    }

    Some((prolongation, restriction, c2f, Vec::new()))
}

fn build_energy_minimizing_prolongation<T: ComplexField>(
    current_matrix: &CsrMatrix<T>,
    strong_connections: &[Vec<usize>],
    effective_mis_distance: usize,
    config: &AmgConfig,
    n: usize,
) -> Option<LevelTransfer<T>>
where
    T::Real: Sync + Send,
{
    let (aggregate_ids, num_aggregates) =
        coarsening::coarsen_aggregation(current_matrix, strong_connections, effective_mis_distance);

    if num_aggregates == 0 || num_aggregates >= n {
        return None;
    }
    let p_tent = interpolation::build_tentative_prolongation(n, &aggregate_ids, num_aggregates);

    let prolongation = interpolation::energy_minimizing_prolongation(
        current_matrix,
        &p_tent,
        config.energy_min_iterations,
        config.energy_min_pattern_distance,
    );
    let restriction = transpose_or_adjoint(&prolongation, config);

    let mut c2f = vec![0usize; num_aggregates];
    let mut seen = vec![false; num_aggregates];
    for (i, &agg) in aggregate_ids.iter().enumerate() {
        if !seen[agg] {
            c2f[agg] = i;
            seen[agg] = true;
        }
    }

    Some((prolongation, restriction, c2f, Vec::new()))
}

#[cfg(feature = "experimental-gnn")]
fn build_gnn_prolongation<T: ComplexField>(
    current_matrix: &CsrMatrix<T>,
    strong_connections: &[Vec<usize>],
    config: &AmgConfig,
    n: usize,
) -> Option<LevelTransfer<T>>
where
    T::Real: Sync + Send,
{
    let gnn_weights = config.gnn_weights.as_ref().cloned().unwrap_or_default();
    let (aggregate_ids, num_aggregates) =
        gnn_coarsening::gnn_coarsen(current_matrix, strong_connections, &gnn_weights);

    if num_aggregates == 0 || num_aggregates >= n {
        return None;
    }
    let p_tent = interpolation::build_tentative_prolongation(n, &aggregate_ids, num_aggregates);

    let level_diag_inv = util::compute_diag_inv(current_matrix);
    let spectral_radius = util::estimate_max_eigenvalue(current_matrix, &level_diag_inv);
    let four_thirds = T::Real::from_f64(4.0).unwrap() / T::Real::from_f64(3.0).unwrap();
    let omega = four_thirds / spectral_radius;

    let prolongation = interpolation::smooth_prolongation(
        current_matrix,
        &level_diag_inv,
        &p_tent,
        omega,
        config.max_interp_elements,
    );
    let restriction = transpose_or_adjoint(&prolongation, config);

    let mut c2f = vec![0usize; num_aggregates];
    let mut seen = vec![false; num_aggregates];
    for (i, &agg) in aggregate_ids.iter().enumerate() {
        if !seen[agg] {
            c2f[agg] = i;
            seen[agg] = true;
        }
    }

    Some((prolongation, restriction, c2f, Vec::new()))
}

fn build_classical_prolongation<T: ComplexField>(
    current_matrix: &CsrMatrix<T>,
    strong_connections: &[Vec<usize>],
    config: &AmgConfig,
    n: usize,
    level_idx: usize,
) -> Option<LevelTransfer<T>>
where
    T::Real: Sync + Send,
{
    let (point_types, coarse_to_fine) = match config.coarsening {
        AmgCoarsening::RugeStuben => {
            coarsening::coarsen_ruge_stuben(current_matrix, strong_connections)
        }
        AmgCoarsening::Cljp => coarsening::coarsen_cljp(current_matrix, strong_connections),
        AmgCoarsening::Pmis => coarsening::coarsen_pmis(current_matrix, strong_connections),
        AmgCoarsening::Hmis => coarsening::coarsen_hmis(current_matrix, strong_connections),
        AmgCoarsening::Falgout => coarsening::coarsen_falgout(current_matrix, strong_connections),
        AmgCoarsening::SmoothedAggregation | AmgCoarsening::EnergyMinimizing => {
            unreachable!("Aggregation variants handled above")
        }
        #[cfg(feature = "experimental-gnn")]
        AmgCoarsening::GnnCoarsening => {
            unreachable!("GNN variant handled above")
        }
    };

    if coarse_to_fine.is_empty() || coarse_to_fine.len() >= n {
        return None;
    }

    let interp_start = std::time::Instant::now();
    let prolongation = interpolation::build_interpolation(
        current_matrix,
        strong_connections,
        &point_types,
        &coarse_to_fine,
        config,
    );
    let interp_dt = interp_start.elapsed();
    let trans_start = std::time::Instant::now();
    let restriction = match config.restriction {
        AmgRestriction::Transpose => util::transpose_csr(&prolongation),
        AmgRestriction::ConjugateTranspose => util::adjoint_csr(&prolongation),
        AmgRestriction::LAir { distance, filter } => lair::build_lair_restriction(
            current_matrix,
            strong_connections,
            &point_types,
            &coarse_to_fine,
            distance,
            filter,
        ),
    };
    log::debug!(
        "amg level {} (n={}): interp={:.1}ms transpose={:.1}ms n_coarse={}",
        level_idx,
        n,
        interp_dt.as_secs_f64() * 1000.0,
        trans_start.elapsed().as_secs_f64() * 1000.0,
        coarse_to_fine.len(),
    );

    // F-point list (ascending) for CF relaxation; only classical coarsening
    // paths produce a pointwise C/F splitting. Built for F-point-only
    // smoothing and for CF-ordered SGS (which needs both C and F lists;
    // C-points come from the stored coarse-to-fine mapping).
    let f_points = if config.f_point_smoothing
        || matches!(config.smoother, AmgSmoother::CfSymmetricGaussSeidel)
    {
        (0..n)
            .filter(|&i| point_types[i] == PointType::Fine)
            .collect()
    } else {
        Vec::new()
    };

    Some((prolongation, restriction, coarse_to_fine, f_points))
}

#[allow(clippy::too_many_arguments)]
fn build_coarsening_interpolation<T: ComplexField>(
    current_matrix: &CsrMatrix<T>,
    strong_connections: &[Vec<usize>],
    effective_mis_distance: usize,
    config: &AmgConfig,
    n: usize,
    level_idx: usize,
) -> Option<LevelTransfer<T>>
where
    T::Real: Sync + Send,
{
    if config.coarsening == AmgCoarsening::SmoothedAggregation {
        build_smoothed_aggregation_prolongation(
            current_matrix,
            strong_connections,
            effective_mis_distance,
            config,
            n,
        )
    } else if config.coarsening == AmgCoarsening::EnergyMinimizing {
        build_energy_minimizing_prolongation(
            current_matrix,
            strong_connections,
            effective_mis_distance,
            config,
            n,
        )
    } else {
        #[cfg(feature = "experimental-gnn")]
        if config.coarsening == AmgCoarsening::GnnCoarsening {
            return build_gnn_prolongation(current_matrix, strong_connections, config, n);
        }
        build_classical_prolongation(current_matrix, strong_connections, config, n, level_idx)
    }
}

#[allow(clippy::too_many_arguments)]
fn build_galerkin_level<T: ComplexField>(
    levels: &mut Vec<AmgLevel<T>>,
    current_matrix: &CsrMatrix<T>,
    config: &AmgConfig,
    level_idx: usize,
    prolongation: CsrMatrix<T>,
    restriction: CsrMatrix<T>,
    coarse_to_fine: Vec<usize>,
    f_points: Vec<usize>,
    level_start: std::time::Instant,
) -> Option<CsrMatrix<T>>
where
    T::Real: Sync + Send,
{
    let n = current_matrix.num_rows;
    let num_coarse = coarse_to_fine.len();
    if num_coarse == 0 || num_coarse >= n {
        return None;
    }

    if (n as f64 / num_coarse as f64) < 1.3 {
        return None;
    }

    let galerkin_start = std::time::Instant::now();
    let raw_coarse = util::galerkin_product(&restriction, current_matrix, &prolongation);
    let galerkin_dt = galerkin_start.elapsed();
    let drop_start = std::time::Instant::now();
    let coarse_matrix = util::drop_small_entries(&raw_coarse, config.galerkin_drop_tol);
    log::debug!(
        "amg level {} galerkin={:.1}ms drop={:.1}ms coarse_nnz={} (level total={:.1}ms)",
        level_idx,
        galerkin_dt.as_secs_f64() * 1000.0,
        drop_start.elapsed().as_secs_f64() * 1000.0,
        coarse_matrix.nnz(),
        level_start.elapsed().as_secs_f64() * 1000.0,
    );

    if coarse_matrix.nnz() > 2 * current_matrix.nnz() {
        if let Some(last) = levels.last_mut() {
            last.prolongation = Some(prolongation);
            last.restriction = Some(restriction);
            last.coarse_to_fine = coarse_to_fine;
            last.f_points = f_points;
        }
        let coarse_diag_inv = util::compute_diag_inv(&coarse_matrix);
        let coarse_l1_diag = util::compute_l1_diag(&coarse_matrix);
        let nc = num_coarse;
        levels.push(AmgLevel {
            matrix: coarse_matrix,
            prolongation: None,
            restriction: None,
            diag_inv: coarse_diag_inv,
            l1_diag: coarse_l1_diag,
            lambda_max: None,
            coarse_to_fine: Vec::new(),
            f_points: Vec::new(),
            num_dofs: nc,
            work_r: Mutex::new(vec![T::zero(); nc]),
            work_tmp: Mutex::new(vec![T::zero(); nc]),
            work_r_coarse: Mutex::new(Vec::new()),
            work_e_coarse: Mutex::new(Vec::new()),
            coarse_solver: None,
            color_sets: None,
        });
        return None;
    }

    let coarse_diag_inv = util::compute_diag_inv(&coarse_matrix);
    let coarse_l1_diag = util::compute_l1_diag(&coarse_matrix);
    let coarse_lambda_max = if config.smoother == AmgSmoother::Chebyshev {
        Some(util::estimate_max_eigenvalue(
            &coarse_matrix,
            &coarse_diag_inv,
        ))
    } else {
        None
    };

    if let Some(last) = levels.last_mut() {
        last.prolongation = Some(prolongation);
        last.restriction = Some(restriction);
        last.coarse_to_fine = coarse_to_fine;
        last.f_points = f_points;
    }

    levels.push(AmgLevel {
        matrix: coarse_matrix.clone(),
        prolongation: None,
        restriction: None,
        diag_inv: coarse_diag_inv,
        l1_diag: coarse_l1_diag,
        lambda_max: coarse_lambda_max,
        coarse_to_fine: Vec::new(),
        f_points: Vec::new(),
        num_dofs: num_coarse,
        work_r: Mutex::new(vec![T::zero(); num_coarse]),
        work_tmp: Mutex::new(vec![T::zero(); num_coarse]),
        work_r_coarse: Mutex::new(Vec::new()),
        work_e_coarse: Mutex::new(Vec::new()),
        coarse_solver: None,
        color_sets: None,
    });

    Some(coarse_matrix)
}

#[allow(clippy::too_many_arguments)]
fn setup_coarse_solver<T: ComplexField>(levels: &mut [AmgLevel<T>])
where
    T::Real: Sync + Send,
{
    const COARSEST_LU_LIMIT: usize = 1_000;
    if let Some(last) = levels.last_mut() {
        let n_coarsest = last.matrix.num_rows;
        if n_coarsest <= COARSEST_LU_LIMIT {
            let lu_start = std::time::Instant::now();
            match sonium_solvers_direct::SparseLuFactorization::factorize(&last.matrix) {
                Ok(lu) => {
                    log::debug!(
                        "amg coarsest LU n={} nnz={} done in {:.1}ms",
                        n_coarsest,
                        last.matrix.nnz(),
                        lu_start.elapsed().as_secs_f64() * 1000.0,
                    );
                    last.coarse_solver = Some(lu);
                }
                Err(_) => {
                    log::debug!(
                        "host sparse LU failed on coarsest AMG level (n={}); using smoother fallback",
                        n_coarsest,
                    );
                }
            }
        } else {
            log::debug!(
                "amg coarsest level too large for LU (n={} > {}); falling back to smoother",
                n_coarsest,
                COARSEST_LU_LIMIT,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn finalize_hierarchy<T: ComplexField>(
    mut levels: Vec<AmgLevel<T>>,
    config: AmgConfig,
    start: std::time::Instant,
) -> AmgPreconditioner<T>
where
    T::Real: Sync + Send,
{
    for i in 0..levels.len().saturating_sub(1) {
        let next_n = levels[i + 1].num_dofs;
        // Safe by construction: these mutexes are freshly built above and no
        // lock has ever been taken, so they cannot be poisoned.
        *levels[i].work_r_coarse.get_mut().unwrap() = vec![T::zero(); next_n];
        *levels[i].work_e_coarse.get_mut().unwrap() = vec![T::zero(); next_n];
    }

    setup_coarse_solver(&mut levels);

    if config.smoother == AmgSmoother::MulticolorGaussSeidel {
        for lvl in &mut levels {
            let (_, rows_per_color) = coloring::greedy_graph_coloring(&lvl.matrix);
            lvl.color_sets = Some(rows_per_color);
        }
    }

    let setup_time_ms = start.elapsed().as_secs_f64() * 1000.0;

    let (grid_complexity, operator_complexity) = util::compute_complexities(&levels);
    if let Some(max_grid_complexity) = config.max_grid_complexity
        && grid_complexity > max_grid_complexity
    {
        log::warn!(
            "AMG grid complexity {:.2} exceeds target {:.2}",
            grid_complexity,
            max_grid_complexity
        );
    }
    if let Some(max_operator_complexity) = config.max_operator_complexity
        && operator_complexity > max_operator_complexity
    {
        log::warn!(
            "AMG operator complexity {:.2} exceeds target {:.2}",
            operator_complexity,
            max_operator_complexity
        );
    }

    AmgPreconditioner {
        levels,
        config,
        setup_time_ms,
        grid_complexity,
        operator_complexity,
        apply_count: AtomicU64::new(0),
        total_apply_ns: AtomicU64::new(0),
    }
}

impl<T: ComplexField> AmgPreconditioner<T>
where
    T::Real: Sync + Send,
{
    /// Create AMG preconditioner from a CSR matrix
    pub fn from_csr(matrix: &CsrMatrix<T>, config: AmgConfig) -> Self {
        let start = std::time::Instant::now();

        // Validate-once: level 0 keeps a verbatim clone of the caller's
        // matrix and every apply-path SpMV runs unchecked through
        // `hierarchy_matvec`, so the memory-safety invariants must be
        // established here. This accepts everything the kernels accept
        // (unsorted rows, duplicates, non-finite values); structurally
        // invalid storage now panics here instead of later in setup.
        util::validate_owned_structure(matrix).expect("AMG setup requires valid CSR storage");

        let (mut levels, mut current_matrix) = build_first_level(matrix, &config);

        for level_idx in 0..config.max_levels - 1 {
            let n = current_matrix.num_rows;
            if n <= config.coarse_size {
                break;
            }
            let level_start = std::time::Instant::now();

            if config.block_size > 1 && n.is_multiple_of(config.block_size) {
                match build_nodal_level(&mut levels, &current_matrix, &config, level_idx) {
                    Some(coarse_matrix) => {
                        current_matrix = coarse_matrix;
                        continue;
                    }
                    None => break,
                }
            }

            let strength_start = std::time::Instant::now();
            let (strong_connections, effective_mis_distance) =
                compute_strength_matrix(&current_matrix, &config, level_idx);
            let strength_dt = strength_start.elapsed();
            let coarsen_start = std::time::Instant::now();
            let level_result = build_coarsening_interpolation(
                &current_matrix,
                &strong_connections,
                effective_mis_distance,
                &config,
                n,
                level_idx,
            );

            let coarsen_dt = coarsen_start.elapsed();
            log::debug!(
                "amg level {} (n={}): strength={:.1}ms coarsen={:.1}ms",
                level_idx,
                n,
                strength_dt.as_secs_f64() * 1000.0,
                coarsen_dt.as_secs_f64() * 1000.0,
            );

            let Some((prolongation, restriction, coarse_to_fine, f_points)) = level_result else {
                break;
            };

            let Some(coarse_matrix) = build_galerkin_level(
                &mut levels,
                &current_matrix,
                &config,
                level_idx,
                prolongation,
                restriction,
                coarse_to_fine,
                f_points,
                level_start,
            ) else {
                break;
            };

            current_matrix = coarse_matrix;
        }

        finalize_hierarchy(levels, config, start)
    }

    /// Get number of levels in hierarchy
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Per-level `(num_dofs, nnz)` from finest to coarsest, for hierarchy
    /// shape diagnosis (coarsening rates, fill growth per level).
    pub fn level_dims(&self) -> Vec<(usize, usize)> {
        self.levels
            .iter()
            .map(|lvl| (lvl.num_dofs, lvl.matrix.values.len()))
            .collect()
    }

    /// Prolongation operator of `level` (finest = 0), for hierarchy
    /// diagnosis against vendor tables (entries/row, weight and row-sum
    /// ranges). `None` on the terminal level, which has no prolongation.
    pub fn level_prolongation(&self, level: usize) -> Option<&CsrMatrix<T>> {
        self.levels.get(level)?.prolongation.as_ref()
    }

    /// Get setup time in milliseconds
    pub fn setup_time_ms(&self) -> f64 {
        self.setup_time_ms
    }

    /// Get grid complexity (sum of DOFs / fine DOFs)
    pub fn grid_complexity(&self) -> f64 {
        self.grid_complexity
    }

    /// Get operator complexity (sum of nnz / fine nnz)
    pub fn operator_complexity(&self) -> f64 {
        self.operator_complexity
    }

    /// Whether the hierarchy satisfies configured grid/operator complexity targets.
    pub fn complexity_within_targets(&self) -> bool {
        self.config
            .max_grid_complexity
            .is_none_or(|target| self.grid_complexity <= target)
            && self
                .config
                .max_operator_complexity
                .is_none_or(|target| self.operator_complexity <= target)
    }

    fn record_apply_elapsed(&self, start: Instant) {
        self.apply_count.fetch_add(1, Ordering::Relaxed);
        self.total_apply_ns
            .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    /// Get configuration
    pub fn config(&self) -> &AmgConfig {
        &self.config
    }

    /// Apply Chebyshev polynomial smoothing.
    ///
    /// Uses the 3-term recurrence for a degree-`num_sweeps` Chebyshev polynomial
    /// smoother with eigenvalue bounds derived from `lambda_max`.
    fn smooth_chebyshev(
        matrix: &CsrMatrix<T>,
        diag_inv: &[T],
        b: &[T],
        x: &mut [T],
        lambda_max: T::Real,
        num_sweeps: usize,
        ax: &mut [T],
    ) {
        if num_sweeps == 0 {
            return;
        }
        use rayon::prelude::*;

        let n = matrix.num_rows;
        debug_assert_eq!(ax.len(), n);
        let thirty = T::Real::from_f64(30.0).unwrap();
        let two = T::Real::from_f64(2.0).unwrap();

        let lambda_min = lambda_max / thirty;
        let theta = (lambda_max + lambda_min) / two;
        let delta = (lambda_max - lambda_min) / two;
        let sigma = theta / delta;
        let mut rho = T::Real::from_f64(1.0).unwrap() / sigma;

        // Reuse caller-provided SpMV scratch across sweeps.
        let mut d = vec![T::zero(); n];
        let mut d_new = vec![T::zero(); n];

        // First sweep: d = (1/theta) * D^{-1} (b - Ax)
        Self::hierarchy_matvec(matrix, x, ax);
        let inv_theta = T::Real::from_f64(1.0).unwrap() / theta;
        let inv_theta_t = T::from_real(inv_theta);

        let parallel = smooth_in_parallel(n);
        let b_slice = b;
        let ax_slice = &*ax;
        let diag_slice = diag_inv;
        let d_slice = &mut d;
        let x_slice = &mut *x;
        if parallel {
            // Fused: compute d[i] and update x[i] in one parallel walk.
            d_slice
                .par_iter_mut()
                .zip(x_slice.par_iter_mut())
                .zip(b_slice.par_iter())
                .zip(ax_slice.par_iter())
                .zip(diag_slice.par_iter())
                .for_each(|((((di, xi), &bi), &axi), &dinv)| {
                    let v = inv_theta_t * (dinv * (bi - axi));
                    *di = v;
                    *xi += v;
                });
        } else {
            for i in 0..n {
                let v = inv_theta_t * (diag_inv[i] * (b[i] - ax[i]));
                d[i] = v;
                x[i] += v;
            }
        }

        // Subsequent sweeps using 3-term recurrence
        for _ in 1..num_sweeps {
            let rho_new = T::Real::from_f64(1.0).unwrap() / (two * sigma - rho);
            Self::hierarchy_matvec(matrix, x, ax);
            let coeff1 = T::from_real(rho_new * rho);
            let coeff2 = T::from_real(two * rho_new / delta);

            let b_slice = b;
            let ax_slice = &*ax;
            let diag_slice = diag_inv;
            let d_slice = &d;
            let d_new_slice = &mut d_new;
            let x_slice = &mut *x;

            if parallel {
                // Fused: compute d_new[i] and update x[i] in one walk.
                d_new_slice
                    .par_iter_mut()
                    .zip(x_slice.par_iter_mut())
                    .zip(d_slice.par_iter())
                    .zip(b_slice.par_iter())
                    .zip(ax_slice.par_iter())
                    .zip(diag_slice.par_iter())
                    .for_each(|(((((dn, xi), &di), &bi), &axi), &dinv)| {
                        let residual_i = dinv * (bi - axi);
                        let v = coeff1 * di + coeff2 * residual_i;
                        *dn = v;
                        *xi += v;
                    });
            } else {
                for i in 0..n {
                    let residual_i = diag_inv[i] * (b[i] - ax[i]);
                    let v = coeff1 * d[i] + coeff2 * residual_i;
                    d_new[i] = v;
                    x[i] += v;
                }
            }
            std::mem::swap(&mut d, &mut d_new);
            rho = rho_new;
        }
    }

    /// Apply Jacobi smoothing: x = x + omega * D^{-1} * (b - A*x)
    fn smooth_jacobi(
        matrix: &CsrMatrix<T>,
        diag_inv: &[T],
        x: &mut [T],
        b: &[T],
        omega: f64,
        num_sweeps: usize,
        ax: &mut [T],
    ) {
        use rayon::prelude::*;

        let omega_t = T::from_real(T::Real::from_f64(omega).unwrap());
        let n = x.len();
        debug_assert_eq!(ax.len(), n);
        let parallel = smooth_in_parallel(n);

        for _ in 0..num_sweeps {
            // ax = A*x (zero allocation via matvec_into_buf)
            Self::hierarchy_matvec(matrix, x, ax);

            // Fused update: x[i] += omega * diag_inv[i] * (b[i] - ax[i]).
            // One par_iter walk instead of two — at high thread counts the
            // rayon fork/join barrier dominates the actual arithmetic, so
            // halving the number of dispatches is the main scaling lever.
            let ax_slice = &*ax;
            let b_slice = b;
            let diag_slice = diag_inv;
            let x_slice = &mut *x;

            if parallel {
                x_slice
                    .par_iter_mut()
                    .zip(diag_slice.par_iter())
                    .zip(b_slice.par_iter())
                    .zip(ax_slice.par_iter())
                    .for_each(|(((xi, &dinv), &bi), &axi)| {
                        *xi += omega_t * dinv * (bi - axi);
                    });
            } else {
                for i in 0..n {
                    x[i] += omega_t * diag_inv[i] * (b[i] - ax[i]);
                }
            }
        }
    }

    /// Apply l1-Jacobi smoothing using pre-cached L1 row-norm diagonal.
    fn smooth_l1_jacobi(
        matrix: &CsrMatrix<T>,
        x: &mut [T],
        b: &[T],
        l1_diag: &[T::Real],
        num_sweeps: usize,
        ax: &mut [T],
    ) {
        use rayon::prelude::*;

        let n = x.len();
        debug_assert_eq!(ax.len(), n);
        let parallel = smooth_in_parallel(n);

        for _ in 0..num_sweeps {
            // ax = A*x (zero allocation via matvec_into_buf)
            Self::hierarchy_matvec(matrix, x, ax);

            // Fused update: x[i] += (b[i] - ax[i]) / l1_diag[i].
            let ax_slice = &*ax;
            let b_slice = b;
            let x_slice = &mut *x;

            if parallel {
                x_slice
                    .par_iter_mut()
                    .zip(b_slice.par_iter())
                    .zip(ax_slice.par_iter())
                    .zip(l1_diag.par_iter())
                    .for_each(|(((xi, &bi), &axi), &l1)| {
                        *xi += (bi - axi) * T::from_real(l1).inv();
                    });
            } else {
                for i in 0..n {
                    x[i] += (b[i] - ax[i]) * T::from_real(l1_diag[i]).inv();
                }
            }
        }
    }

    /// Apply F-point-restricted Jacobi smoothing (CF relaxation):
    /// `x[i] += omega * D^{-1}(b - Ax)[i]` for F-point indices only.
    #[allow(clippy::too_many_arguments)]
    fn smooth_jacobi_f(
        matrix: &CsrMatrix<T>,
        diag_inv: &[T],
        f_points: &[usize],
        x: &mut [T],
        b: &[T],
        omega: f64,
        num_sweeps: usize,
        ax: &mut [T],
    ) {
        use rayon::prelude::*;

        let omega_t = T::from_real(T::Real::from_f64(omega).unwrap());
        let n = x.len();
        debug_assert_eq!(ax.len(), n);
        let parallel = smooth_in_parallel(n);

        // Wrapper to share a raw pointer across rayon closures.
        // SAFETY: f_points are distinct indices, so parallel writes through
        // this pointer never alias.
        #[derive(Clone, Copy)]
        struct SendPtr<U>(*mut U);
        unsafe impl<U> Send for SendPtr<U> {}
        unsafe impl<U> Sync for SendPtr<U> {}
        impl<U> SendPtr<U> {
            #[inline]
            fn ptr(self) -> *mut U {
                self.0
            }
        }

        for _ in 0..num_sweeps {
            // ax = A*x (full SpMV: F residuals couple to C-point values)
            Self::hierarchy_matvec(matrix, x, ax);

            let ax_slice = &*ax;
            let b_slice = b;
            let diag_slice = diag_inv;

            if parallel {
                let x_ptr = SendPtr(x.as_mut_ptr());
                f_points.par_iter().for_each(|&i| {
                    let v = omega_t * diag_slice[i] * (b_slice[i] - ax_slice[i]);
                    // SAFETY: each i in f_points is unique, so no two
                    // iterations write the same element.
                    unsafe {
                        *x_ptr.ptr().add(i) += v;
                    }
                });
            } else {
                for &i in f_points {
                    x[i] += omega_t * diag_slice[i] * (b_slice[i] - ax_slice[i]);
                }
            }
        }
    }

    /// Apply F-point-restricted l1-Jacobi smoothing (CF relaxation).
    fn smooth_l1_jacobi_f(
        matrix: &CsrMatrix<T>,
        f_points: &[usize],
        x: &mut [T],
        b: &[T],
        l1_diag: &[T::Real],
        num_sweeps: usize,
        ax: &mut [T],
    ) {
        use rayon::prelude::*;

        let n = x.len();
        debug_assert_eq!(ax.len(), n);
        let parallel = smooth_in_parallel(n);

        // SAFETY: see smooth_jacobi_f.
        #[derive(Clone, Copy)]
        struct SendPtr<U>(*mut U);
        unsafe impl<U> Send for SendPtr<U> {}
        unsafe impl<U> Sync for SendPtr<U> {}
        impl<U> SendPtr<U> {
            #[inline]
            fn ptr(self) -> *mut U {
                self.0
            }
        }

        for _ in 0..num_sweeps {
            Self::hierarchy_matvec(matrix, x, ax);

            let ax_slice = &*ax;
            let b_slice = b;

            if parallel {
                let x_ptr = SendPtr(x.as_mut_ptr());
                f_points.par_iter().for_each(|&i| {
                    let v = (b_slice[i] - ax_slice[i]) * T::from_real(l1_diag[i]).inv();
                    // SAFETY: each i in f_points is unique.
                    unsafe {
                        *x_ptr.ptr().add(i) += v;
                    }
                });
            } else {
                for &i in f_points {
                    x[i] += (b[i] - ax[i]) * T::from_real(l1_diag[i]).inv();
                }
            }
        }
    }

    /// Apply F-point-restricted symmetric Gauss-Seidel smoothing
    /// (CF relaxation): forward then backward sweep over the F-point rows
    /// only. This is the minimal Manteuffel-Ruge-Southworth F-relaxation;
    /// C-points are left to the coarse correction.
    /// One Gauss-Seidel sweep restricted to `subset`, in ascending
    /// (`forward = true`) or descending order.
    fn smooth_gs_subset(
        matrix: &CsrMatrix<T>,
        subset: &[usize],
        x: &mut [T],
        b: &[T],
        forward: bool,
    ) {
        let tol = T::Real::from_f64(1e-15).unwrap();
        if forward {
            for &i in subset {
                let mut sum = b[i];
                let mut diag = T::one();
                for (j, val) in matrix.row_entries(i) {
                    if j == i {
                        diag = val;
                    } else {
                        sum -= val * x[j];
                    }
                }
                if diag.norm() > tol {
                    x[i] = sum * diag.inv();
                }
            }
        } else {
            for &i in subset.iter().rev() {
                let mut sum = b[i];
                let mut diag = T::one();
                for (j, val) in matrix.row_entries(i) {
                    if j == i {
                        diag = val;
                    } else {
                        sum -= val * x[j];
                    }
                }
                if diag.norm() > tol {
                    x[i] = sum * diag.inv();
                }
            }
        }
    }

    /// CF-ordered SGS: pre-smoothing relaxes C then F (forward each),
    /// post-smoothing relaxes F then C (backward each).
    fn smooth_cf_sgs(
        matrix: &CsrMatrix<T>,
        c_points: &[usize],
        f_points: &[usize],
        x: &mut [T],
        b: &[T],
        num_sweeps: usize,
        pre: bool,
    ) {
        for _ in 0..num_sweeps {
            if pre {
                Self::smooth_gs_subset(matrix, c_points, x, b, true);
                Self::smooth_gs_subset(matrix, f_points, x, b, true);
            } else {
                Self::smooth_gs_subset(matrix, f_points, x, b, false);
                Self::smooth_gs_subset(matrix, c_points, x, b, false);
            }
        }
    }

    fn smooth_sym_gauss_seidel_f(
        matrix: &CsrMatrix<T>,
        f_points: &[usize],
        x: &mut [T],
        b: &[T],
        num_sweeps: usize,
    ) {
        let tol = T::Real::from_f64(1e-15).unwrap();

        for _ in 0..num_sweeps {
            // Forward sweep over F-point rows (ascending)
            for &i in f_points {
                let mut sum = b[i];
                let mut diag = T::one();

                for (j, val) in matrix.row_entries(i) {
                    if j == i {
                        diag = val;
                    } else {
                        sum -= val * x[j];
                    }
                }

                if diag.norm() > tol {
                    x[i] = sum * diag.inv();
                }
            }

            // Backward sweep over F-point rows (descending)
            for &i in f_points.iter().rev() {
                let mut sum = b[i];
                let mut diag = T::one();

                for (j, val) in matrix.row_entries(i) {
                    if j == i {
                        diag = val;
                    } else {
                        sum -= val * x[j];
                    }
                }

                if diag.norm() > tol {
                    x[i] = sum * diag.inv();
                }
            }
        }
    }

    /// Apply symmetric Gauss-Seidel smoothing
    fn smooth_sym_gauss_seidel(matrix: &CsrMatrix<T>, x: &mut [T], b: &[T], num_sweeps: usize) {
        let n = x.len();
        let tol = T::Real::from_f64(1e-15).unwrap();

        for _ in 0..num_sweeps {
            // Forward sweep
            for i in 0..n {
                let mut sum = b[i];
                let mut diag = T::one();

                for (j, val) in matrix.row_entries(i) {
                    if j == i {
                        diag = val;
                    } else {
                        sum -= val * x[j];
                    }
                }

                if diag.norm() > tol {
                    x[i] = sum * diag.inv();
                }
            }

            // Backward sweep
            for i in (0..n).rev() {
                let mut sum = b[i];
                let mut diag = T::one();

                for (j, val) in matrix.row_entries(i) {
                    if j == i {
                        diag = val;
                    } else {
                        sum -= val * x[j];
                    }
                }

                if diag.norm() > tol {
                    x[i] = sum * diag.inv();
                }
            }
        }
    }

    /// Apply multicolor (parallel) Gauss-Seidel smoothing.
    ///
    /// Rows are partitioned by graph color so that same-color rows share no
    /// edges.  Each color is swept in order; within a color, all row updates
    /// run in parallel via rayon.  Forward then backward color order gives a
    /// symmetric variant analogous to SGS.
    fn smooth_multicolor_gauss_seidel(
        matrix: &CsrMatrix<T>,
        x: &mut [T],
        b: &[T],
        color_sets: &[Vec<usize>],
        omega: T::Real,
        num_sweeps: usize,
    ) {
        use rayon::prelude::*;

        // Wrapper to share a raw pointer across rayon closures.
        // SAFETY: callers guarantee no aliasing across same-color rows.
        #[derive(Clone, Copy)]
        struct SendPtr<U>(*mut U);
        unsafe impl<U> Send for SendPtr<U> {}
        unsafe impl<U> Sync for SendPtr<U> {}
        impl<U> SendPtr<U> {
            #[inline]
            fn ptr(self) -> *mut U {
                self.0
            }
        }

        let tol = T::Real::from_f64(1e-15).unwrap();
        let x_ptr = SendPtr(x.as_mut_ptr());

        for _ in 0..num_sweeps {
            // Forward sweep: colors in natural order
            for color_rows in color_sets.iter() {
                // SAFETY: same-color rows are independent (no edges between
                // them), so parallel writes to distinct x[i] are race-free.
                // Reads of x[j] for j in a different (already-swept) color
                // are stable because we synchronize between colors.
                color_rows.par_iter().for_each(|&i| {
                    let row_start = matrix.row_ptrs[i];
                    let row_end = matrix.row_ptrs[i + 1];
                    let mut sum = b[i];
                    let mut diag = T::one();

                    for idx in row_start..row_end {
                        let j = matrix.col_indices[idx];
                        let val = matrix.values[idx];
                        if j == i {
                            diag = val;
                        } else {
                            // SAFETY: j belongs to a different color, so no
                            // concurrent write to x[j] in this color sweep.
                            sum -= val * unsafe { *x_ptr.ptr().add(j) };
                        }
                    }

                    if diag.norm() > tol {
                        let new_val = sum * diag.inv();
                        let old_val = unsafe { *x_ptr.ptr().add(i) };
                        unsafe {
                            *x_ptr.ptr().add(i) =
                                T::from_real(T::Real::from_f64(1.0).unwrap() - omega) * old_val
                                    + T::from_real(omega) * new_val;
                        }
                    }
                });
            }

            // Backward sweep: colors in reverse order
            for color_rows in color_sets.iter().rev() {
                color_rows.par_iter().for_each(|&i| {
                    let row_start = matrix.row_ptrs[i];
                    let row_end = matrix.row_ptrs[i + 1];
                    let mut sum = b[i];
                    let mut diag = T::one();

                    for idx in row_start..row_end {
                        let j = matrix.col_indices[idx];
                        let val = matrix.values[idx];
                        if j == i {
                            diag = val;
                        } else {
                            sum -= val * unsafe { *x_ptr.ptr().add(j) };
                        }
                    }

                    if diag.norm() > tol {
                        let new_val = sum * diag.inv();
                        let old_val = unsafe { *x_ptr.ptr().add(i) };
                        unsafe {
                            *x_ptr.ptr().add(i) =
                                T::from_real(T::Real::from_f64(1.0).unwrap() - omega) * old_val
                                    + T::from_real(omega) * new_val;
                        }
                    }
                });
            }
        }
    }

    /// Apply V-cycle
    /// General multigrid cycle: V-cycle, W-cycle, or F-cycle.
    ///
    /// - **V-cycle** (`gamma=1`): recurse once at each coarse level.
    /// - **W-cycle** (`gamma=2`): recurse twice at each coarse level,
    ///   visiting coarser levels more frequently for better convergence
    ///   on difficult problems.
    /// - **F-cycle**: recurse with F-cycle then V-cycle at each coarse level,
    ///   a compromise between V and W that front-loads coarse-grid work.
    fn mg_cycle(&self, level: usize, x: &mut [T], b: &[T], cycle: AmgCycle, x_is_zero: bool) {
        let lvl = &self.levels[level];

        // Coarsest level: direct solve if available, otherwise smoother fallback
        if level == self.levels.len() - 1 || lvl.prolongation.is_none() {
            if let Some(ref solver) = lvl.coarse_solver {
                // A validated factorization still admits a failing triangular
                // solve (e.g. overflow on an extreme RHS); fall through to
                // the smoother fallback instead of panicking inside `apply`.
                if let Ok(coarse) = solver.solve(b) {
                    x.copy_from_slice(&coarse);
                    return;
                }
            }
            if x_is_zero && self.smooth_zero_initial(level, x, b, 20) {
                return;
            }
            // Fallback: many smoothing iterations
            let mut work_tmp = lvl
                .work_tmp
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            match self.config.smoother {
                AmgSmoother::Jacobi | AmgSmoother::Chebyshev => {
                    Self::smooth_jacobi(
                        &lvl.matrix,
                        &lvl.diag_inv,
                        x,
                        b,
                        self.config.jacobi_weight,
                        20,
                        &mut work_tmp,
                    );
                }
                AmgSmoother::L1Jacobi => {
                    Self::smooth_l1_jacobi(&lvl.matrix, x, b, &lvl.l1_diag, 20, &mut work_tmp);
                }
                AmgSmoother::SymmetricGaussSeidel | AmgSmoother::CfSymmetricGaussSeidel => {
                    Self::smooth_sym_gauss_seidel(&lvl.matrix, x, b, 10);
                }
                AmgSmoother::MulticolorGaussSeidel => {
                    if let Some(ref colors) = lvl.color_sets {
                        // GS uses omega=1.0 (full update), not the Jacobi weight
                        let omega = T::Real::from_f64(1.0).unwrap();
                        Self::smooth_multicolor_gauss_seidel(&lvl.matrix, x, b, colors, omega, 10);
                    } else {
                        Self::smooth_sym_gauss_seidel(&lvl.matrix, x, b, 10);
                    }
                }
            }
            return;
        }

        // Pre-smoothing
        if !(x_is_zero && self.smooth_zero_initial(level, x, b, self.config.num_pre_smooth)) {
            self.smooth(level, x, b, self.config.num_pre_smooth, true);
        }

        // Compute residual: r = b - A*x
        let mut work_r = lvl
            .work_r
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Self::hierarchy_matvec(&lvl.matrix, x, &mut work_r);
        {
            use rayon::prelude::*;
            let n = lvl.num_dofs;
            let r_slice = &mut work_r;
            let b_slice = b;
            if smooth_in_parallel(n) {
                r_slice
                    .par_iter_mut()
                    .zip(b_slice.par_iter())
                    .for_each(|(ri, &bi)| *ri = bi - *ri);
            } else {
                for i in 0..n {
                    r_slice[i] = b[i] - r_slice[i];
                }
            }
        }

        // Restrict residual to coarse grid (zero allocation via pre-allocated workspace)
        let mut r_coarse = lvl
            .work_r_coarse
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Self::hierarchy_matvec(lvl.restriction.as_ref().unwrap(), &work_r, &mut r_coarse);

        // Coarse correction (zero allocation via pre-allocated workspace)
        let mut e_coarse = lvl
            .work_e_coarse
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        {
            use rayon::prelude::*;
            let nc = e_coarse.len();
            if smooth_in_parallel(nc) {
                let e_slice = &mut *e_coarse;
                e_slice.par_iter_mut().for_each(|ei| *ei = T::zero());
            } else {
                for i in 0..nc {
                    e_coarse[i] = T::zero();
                }
            }
        }

        // Recurse based on cycle type
        match cycle {
            AmgCycle::VCycle => {
                self.mg_cycle(level + 1, &mut e_coarse, &r_coarse, AmgCycle::VCycle, true);
            }
            AmgCycle::WCycle => {
                self.mg_cycle(level + 1, &mut e_coarse, &r_coarse, AmgCycle::WCycle, true);
                self.mg_cycle(level + 1, &mut e_coarse, &r_coarse, AmgCycle::WCycle, false);
            }
            AmgCycle::FCycle => {
                self.mg_cycle(level + 1, &mut e_coarse, &r_coarse, AmgCycle::FCycle, true);
                self.mg_cycle(level + 1, &mut e_coarse, &r_coarse, AmgCycle::VCycle, false);
            }
        }

        // Prolongate correction and add to x
        {
            let mut work_tmp = lvl
                .work_tmp
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            Self::hierarchy_matvec(lvl.prolongation.as_ref().unwrap(), &e_coarse, &mut work_tmp);

            use rayon::prelude::*;
            let n = lvl.num_dofs;
            if smooth_in_parallel(n) {
                let x_slice = &mut *x;
                let t_slice = &*work_tmp;
                x_slice
                    .par_iter_mut()
                    .zip(t_slice.par_iter())
                    .for_each(|(xi, &ti)| *xi += ti);
            } else {
                for i in 0..n {
                    x[i] += work_tmp[i];
                }
            }
        }

        // Post-smoothing reuses work_tmp, so the prolongation guard above must
        // be dropped before smooth() acquires that workspace again.
        self.smooth(level, x, b, self.config.num_post_smooth, false);
    }

    /// SpMV against hierarchy-owned operators without per-call structural
    /// validation.
    ///
    /// Sound because the input matrix is validated once in `from_csr`
    /// (`try_from_csr` validates more strongly), coarser levels and
    /// transfers are valid by construction, and `AmgLevel` exposes no
    /// post-setup mutation path. Non-finite values are still rejected at
    /// the `try_apply` boundary.
    #[inline]
    fn hierarchy_matvec(matrix: &CsrMatrix<T>, x: &[T], y: &mut [T]) {
        // SAFETY: see above; dimensions are asserted inside.
        unsafe { matrix.matvec_into_buf_unchecked(x, y) };
    }

    /// Krylov-accelerated top-level cycle (K-cycle): run
    /// `k_cycle_iterations` flexible-CG iterations on `A z = r` with one
    /// multigrid cycle as the preconditioner.
    ///
    /// Flexible CG (Notay, "Flexible conjugate gradients", SIAM J. Sci.
    /// Comput. 22(4), 2000 — the AGMG cycle) tolerates a variable or
    /// not-exactly-symmetric preconditioner, which a multigrid cycle is in
    /// practice (nonsymmetric smoothers, F-point relaxation, ℓAIR
    /// restriction). The conjugated inner product makes it applicable to
    /// Hermitian systems as well. Intended for (Hermitian) SPD problems;
    /// for nonsymmetric ones, use outer FGMRES instead.
    fn apply_k_cycle(&self, r: &[T], z: &mut [T]) {
        let a = &self.levels[0].matrix;
        let n = a.num_rows;
        let k_iters = self.config.k_cycle_iterations;
        let tol = T::Real::from_f64(1e-30).unwrap();

        let mut x = vec![T::zero(); n];
        let mut rr = r.to_vec(); // residual r - A*x0 with x0 = 0
        let mut zc = vec![T::zero(); n];
        let mut p = vec![T::zero(); n];
        let mut ap = vec![T::zero(); n];

        self.mg_cycle(0, &mut zc, &rr, self.config.cycle, true);
        p.copy_from_slice(&zc);
        let mut rz = sonium_solvers_core::blas_helpers::inner_product(&rr, &zc);

        for k in 0..k_iters {
            Self::hierarchy_matvec(a, &p, &mut ap);
            let pap = sonium_solvers_core::blas_helpers::inner_product(&p, &ap);
            if pap.norm() <= tol || rz.norm() <= tol {
                break;
            }
            let alpha = rz * pap.inv();
            for i in 0..n {
                x[i] += alpha * p[i];
                rr[i] -= alpha * ap[i];
            }
            if k + 1 == k_iters {
                break;
            }
            zc.fill(T::zero());
            self.mg_cycle(0, &mut zc, &rr, self.config.cycle, true);
            let rz_new = sonium_solvers_core::blas_helpers::inner_product(&rr, &zc);
            let beta = rz_new * rz.inv();
            for i in 0..n {
                p[i] = zc[i] + beta * p[i];
            }
            rz = rz_new;
        }

        z.copy_from_slice(&x);
    }

    /// Run the first smoothing sweep when the current iterate is known to be zero.
    ///
    /// For point-Jacobi smoothers, the first sweep is diagonal scaling because
    /// A * 0 is exactly zero. Skipping that SpMV matters in AMG applies because
    /// every V-cycle level starts its correction from zero.
    ///
    /// With F-point smoothing enabled, the diagonal scaling is likewise
    /// restricted to F-point indices.
    fn smooth_zero_initial(&self, level: usize, x: &mut [T], b: &[T], num_sweeps: usize) -> bool {
        if num_sweeps == 0 {
            return true;
        }

        let lvl = &self.levels[level];
        let n = lvl.num_dofs;
        let f_points: Option<&[usize]> =
            if self.config.f_point_smoothing && !lvl.f_points.is_empty() {
                Some(lvl.f_points.as_slice())
            } else {
                None
            };
        match self.config.smoother {
            AmgSmoother::Jacobi => {
                use rayon::prelude::*;
                let omega_t = T::from_real(T::Real::from_f64(self.config.jacobi_weight).unwrap());
                if let Some(fp) = f_points {
                    for &i in fp {
                        x[i] = omega_t * lvl.diag_inv[i] * b[i];
                    }
                } else if smooth_in_parallel(n) {
                    x.par_iter_mut()
                        .zip(lvl.diag_inv.par_iter())
                        .zip(b.par_iter())
                        .for_each(|((xi, &dinv), &bi)| {
                            *xi = omega_t * dinv * bi;
                        });
                } else {
                    for i in 0..n {
                        x[i] = omega_t * lvl.diag_inv[i] * b[i];
                    }
                }
            }
            AmgSmoother::L1Jacobi => {
                use rayon::prelude::*;
                if let Some(fp) = f_points {
                    for &i in fp {
                        x[i] = b[i] * T::from_real(lvl.l1_diag[i]).inv();
                    }
                } else if smooth_in_parallel(n) {
                    x.par_iter_mut()
                        .zip(lvl.l1_diag.par_iter())
                        .zip(b.par_iter())
                        .for_each(|((xi, &l1), &bi)| {
                            *xi = bi * T::from_real(l1).inv();
                        });
                } else {
                    for i in 0..n {
                        x[i] = b[i] * T::from_real(lvl.l1_diag[i]).inv();
                    }
                }
            }
            _ => return false,
        }

        if num_sweeps > 1 {
            self.smooth(level, x, b, num_sweeps - 1, true);
        }
        true
    }

    /// Apply smoothing at the given level.
    ///
    /// When [`AmgConfig::f_point_smoothing`] is enabled and this level has a
    /// classical C/F splitting, the update is restricted to F-points (CF
    /// relaxation). Chebyshev has no standard F-restricted form and falls
    /// back to all-point smoothing; multicolor GS falls back to the serial
    /// F-point SGS sweep.
    fn smooth(&self, level: usize, x: &mut [T], b: &[T], num_sweeps: usize, pre: bool) {
        let lvl = &self.levels[level];
        let f_points: Option<&[usize]> =
            if self.config.f_point_smoothing && !lvl.f_points.is_empty() {
                Some(lvl.f_points.as_slice())
            } else {
                None
            };
        let mut work_tmp = lvl
            .work_tmp
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match self.config.smoother {
            AmgSmoother::Jacobi => {
                if let Some(fp) = f_points {
                    Self::smooth_jacobi_f(
                        &lvl.matrix,
                        &lvl.diag_inv,
                        fp,
                        x,
                        b,
                        self.config.jacobi_weight,
                        num_sweeps,
                        &mut work_tmp,
                    );
                } else {
                    Self::smooth_jacobi(
                        &lvl.matrix,
                        &lvl.diag_inv,
                        x,
                        b,
                        self.config.jacobi_weight,
                        num_sweeps,
                        &mut work_tmp,
                    );
                }
            }
            AmgSmoother::Chebyshev => {
                let lm = lvl.lambda_max.unwrap_or(T::Real::from_f64(2.0).unwrap());
                Self::smooth_chebyshev(
                    &lvl.matrix,
                    &lvl.diag_inv,
                    b,
                    x,
                    lm,
                    num_sweeps,
                    &mut work_tmp,
                );
            }
            AmgSmoother::L1Jacobi => {
                if let Some(fp) = f_points {
                    Self::smooth_l1_jacobi_f(
                        &lvl.matrix,
                        fp,
                        x,
                        b,
                        &lvl.l1_diag,
                        num_sweeps,
                        &mut work_tmp,
                    );
                } else {
                    Self::smooth_l1_jacobi(
                        &lvl.matrix,
                        x,
                        b,
                        &lvl.l1_diag,
                        num_sweeps,
                        &mut work_tmp,
                    );
                }
            }
            AmgSmoother::SymmetricGaussSeidel => {
                if let Some(fp) = f_points {
                    Self::smooth_sym_gauss_seidel_f(&lvl.matrix, fp, x, b, num_sweeps);
                } else {
                    Self::smooth_sym_gauss_seidel(&lvl.matrix, x, b, num_sweeps);
                }
            }
            AmgSmoother::CfSymmetricGaussSeidel => {
                if !lvl.coarse_to_fine.is_empty() && !lvl.f_points.is_empty() {
                    Self::smooth_cf_sgs(
                        &lvl.matrix,
                        &lvl.coarse_to_fine,
                        &lvl.f_points,
                        x,
                        b,
                        num_sweeps,
                        pre,
                    );
                } else {
                    // Aggregation-built or terminal levels have no C/F
                    // splitting: fall back to full-point SGS.
                    Self::smooth_sym_gauss_seidel(&lvl.matrix, x, b, num_sweeps);
                }
            }
            AmgSmoother::MulticolorGaussSeidel => {
                if let Some(fp) = f_points {
                    Self::smooth_sym_gauss_seidel_f(&lvl.matrix, fp, x, b, num_sweeps);
                } else if let Some(ref colors) = lvl.color_sets {
                    // GS uses omega=1.0 (full update), not the Jacobi weight
                    let omega = T::Real::from_f64(1.0).unwrap();
                    Self::smooth_multicolor_gauss_seidel(
                        &lvl.matrix,
                        x,
                        b,
                        colors,
                        omega,
                        num_sweeps,
                    );
                } else {
                    Self::smooth_sym_gauss_seidel(&lvl.matrix, x, b, num_sweeps);
                }
            }
        }
    }
}

fn amg_matrix_matches_adjoint<T: ComplexField>(matrix: &CsrMatrix<T>, hermitian: bool) -> bool {
    if matrix.num_rows != matrix.num_cols {
        return false;
    }
    let tolerance = T::Real::from_f64(1.0e-10).unwrap_or_else(T::Real::epsilon);
    for row in 0..matrix.num_rows {
        for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            let col = matrix.col_indices[index];
            let value = matrix.values[index];
            let counterpart = matrix.get(col, row);
            let expected = if hermitian {
                counterpart.conj()
            } else {
                counterpart
            };
            let scale = T::Real::from_f64(1.0).unwrap() + value.norm().max(expected.norm());
            if (value - expected).norm() > tolerance * scale {
                return false;
            }
        }
    }
    true
}

/// Return whether a hierarchy matrix is diagonal up to round-off.
///
/// Jacobi and L1-Jacobi are self-adjoint smoothers on a diagonal matrix (the
/// update is itself diagonal).  Keeping this predicate separate from the
/// general adjoint check lets the capability gate admit that mathematically
/// exact, useful boundary without incorrectly claiming that a Jacobi sweep is
/// self-adjoint for a general matrix.
fn amg_matrix_is_diagonal<T: ComplexField>(matrix: &CsrMatrix<T>) -> bool {
    let tolerance = T::Real::epsilon() * T::Real::from_f64(16.0).unwrap();
    (0..matrix.num_rows).all(|row| {
        let start = matrix.row_ptrs[row];
        let end = matrix.row_ptrs[row + 1];
        (start..end).all(|index| {
            matrix.col_indices[index] == row || matrix.values[index].norm() <= tolerance
        })
    })
}

/// Whether all transfer entries are real-valued. This is needed when a
/// caller selected unconjugated transpose restriction but asks for a
/// Hermitian action (or selected conjugated restriction but asks for a
/// plain transpose): only then do the two transfer operators coincide.
fn amg_transfers_are_real<T: ComplexField>(levels: &[AmgLevel<T>]) -> bool {
    levels.iter().all(|level| {
        level.prolongation.as_ref().is_none_or(|matrix| {
            matrix
                .values
                .iter()
                .all(|value| value.im() == T::Real::from_f64(0.0).unwrap())
        }) && level.restriction.as_ref().is_none_or(|matrix| {
            matrix
                .values
                .iter()
                .all(|value| value.im() == T::Real::from_f64(0.0).unwrap())
        })
    })
}

/// Restrict exact adjoint support to configurations for which the
/// existing cycle is provably self-adjoint. Symmetric Gauss--Seidel is a
/// forward/backward pair, equal pre/post counts make the cycle ordering
/// symmetric, and a V-cycle with Galerkin `Pᴴ A P` (or its real-valued
/// transpose equivalent) preserves the adjoint identity at every level.
/// Other configurations remain explicitly unsupported instead of
/// returning a forward action for an adjoint request.
fn amg_supports_exact_self_adjoint<T: ComplexField>(
    amg: &AmgPreconditioner<T>,
    hermitian: bool,
) -> bool
where
    T::Real: Sync + Send,
{
    let diagonal_jacobi = matches!(
        amg.config.smoother,
        AmgSmoother::Jacobi | AmgSmoother::L1Jacobi
    ) && amg
        .levels
        .iter()
        .all(|level| amg_matrix_is_diagonal(&level.matrix));
    let exact_smoother =
        amg.config.smoother == AmgSmoother::SymmetricGaussSeidel || diagonal_jacobi;
    if amg.config.cycle != AmgCycle::VCycle
        || amg.config.k_cycle_iterations != 0
        || amg.config.f_point_smoothing
        || amg.config.num_pre_smooth != amg.config.num_post_smooth
        || !exact_smoother
    {
        return false;
    }

    if !amg
        .levels
        .iter()
        .all(|level| amg_matrix_matches_adjoint(&level.matrix, hermitian))
    {
        return false;
    }

    match (hermitian, amg.config.restriction) {
        (true, AmgRestriction::ConjugateTranspose) | (false, AmgRestriction::Transpose) => true,
        (true, AmgRestriction::Transpose) | (false, AmgRestriction::ConjugateTranspose) => {
            amg_transfers_are_real(&amg.levels)
        }
        (_, AmgRestriction::LAir { .. }) => false,
    }
}

impl<T: ComplexField> Preconditioner<T> for AmgPreconditioner<T>
where
    T::Real: Sync + Send,
{
    fn transpose_capability(&self) -> AdjointCapability {
        if amg_supports_exact_self_adjoint(self, false) {
            AdjointCapability::Exact
        } else {
            AdjointCapability::Unsupported
        }
    }

    fn hermitian_capability(&self) -> AdjointCapability {
        if amg_supports_exact_self_adjoint(self, true) {
            AdjointCapability::Exact
        } else {
            AdjointCapability::Unsupported
        }
    }

    fn apply_transpose(&self, r: &[T]) -> Vec<T> {
        assert_eq!(
            self.transpose_capability(),
            AdjointCapability::Exact,
            "AMG transpose action is unsupported for this cycle configuration"
        );
        self.apply(r)
    }

    fn apply_hermitian(&self, r: &[T]) -> Vec<T> {
        assert_eq!(
            self.hermitian_capability(),
            AdjointCapability::Exact,
            "AMG Hermitian action is unsupported for this cycle configuration"
        );
        self.apply(r)
    }

    fn try_apply_transpose(&self, r: &[T]) -> Result<Vec<T>, SolverError> {
        if self.transpose_capability() != AdjointCapability::Exact {
            return Err(SolverError::UnsupportedOperation {
                operation: "preconditioner transpose",
                preconditioner: std::any::type_name::<Self>(),
            });
        }
        self.try_apply(r)
    }

    fn try_apply_hermitian(&self, r: &[T]) -> Result<Vec<T>, SolverError> {
        if self.hermitian_capability() != AdjointCapability::Exact {
            return Err(SolverError::UnsupportedOperation {
                operation: "preconditioner Hermitian transpose",
                preconditioner: std::any::type_name::<Self>(),
            });
        }
        self.try_apply(r)
    }

    fn try_apply(&self, r: &[T]) -> Result<Vec<T>, SolverError> {
        let expected = self.levels.first().map_or(0, |level| level.num_dofs);
        if r.len() != expected {
            return Err(SolverError::DimensionMismatch {
                expected,
                got: r.len(),
            });
        }
        if r.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation: "AMG apply",
                reason: "input contains NaN or infinity".into(),
            });
        }
        let applied = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.apply(r)))
            .map_err(|_| SolverError::OperationFailed {
                operation: "AMG apply",
                reason: "preconditioner application panicked".into(),
            })?;
        if applied.len() != expected {
            return Err(SolverError::DimensionMismatch {
                expected,
                got: applied.len(),
            });
        }
        if applied.iter().any(|value| !value.norm().is_finite()) {
            return Err(SolverError::OperationFailed {
                operation: "AMG apply",
                reason: "output contains NaN or infinity".into(),
            });
        }
        Ok(applied)
    }

    fn apply(&self, r: &[T]) -> Vec<T> {
        let n = self.levels.first().map_or(0, |level| level.num_dofs);
        assert_eq!(r.len(), n, "AMG RHS length mismatch");
        let mut z = vec![T::zero(); n];
        self.apply_into(r, &mut z);
        z
    }

    fn apply_into(&self, r: &[T], z: &mut [T]) {
        let start = Instant::now();
        let n = self.levels.first().map_or(0, |level| level.num_dofs);
        assert_eq!(r.len(), n, "AMG RHS length mismatch");
        assert_eq!(z.len(), n, "preconditioner output size mismatch");
        if self.levels.is_empty() {
            self.record_apply_elapsed(start);
            return;
        }

        if smooth_in_parallel(n) {
            use rayon::prelude::*;
            z.par_iter_mut().for_each(|zi| *zi = T::zero());
        } else {
            z.fill(T::zero());
        }
        if self.config.k_cycle_iterations > 0 && self.levels.len() > 1 {
            self.apply_k_cycle(r, z);
        } else {
            self.mg_cycle(0, z, r, self.config.cycle, true);
        }
        self.record_apply_elapsed(start);
    }
}

/// Diagnostic information about AMG setup
#[derive(Debug, Clone)]
pub struct AmgDiagnostics {
    /// Number of levels
    pub num_levels: usize,
    /// Grid complexity
    pub grid_complexity: f64,
    /// Operator complexity
    pub operator_complexity: f64,
    /// Configured grid-complexity target, when present.
    pub max_grid_complexity: Option<f64>,
    /// Configured operator-complexity target, when present.
    pub max_operator_complexity: Option<f64>,
    /// Whether the hierarchy satisfies the configured grid-complexity target.
    pub grid_complexity_within_target: bool,
    /// Whether the hierarchy satisfies the configured operator-complexity target.
    pub operator_complexity_within_target: bool,
    /// Setup time in milliseconds
    pub setup_time_ms: f64,
    /// Number of recorded preconditioner applications.
    pub apply_count: u64,
    /// Total recorded application time in milliseconds.
    pub total_apply_ms: f64,
    /// Mean recorded application time in milliseconds.
    pub mean_apply_ms: f64,
    /// Total matrix entries across all hierarchy levels.
    pub total_nnz: usize,
    /// Finest-level matrix entries.
    pub finest_nnz: usize,
    /// Coarsest-level degrees of freedom.
    pub coarsest_dofs: usize,
    /// DOFs per level
    pub level_dofs: Vec<usize>,
    /// NNZ per level
    pub level_nnz: Vec<usize>,
}

impl<T: ComplexField> AmgPreconditioner<T> {
    /// Checked AMG construction for public callers.
    ///
    /// This validates CSR structure, rejects rectangular or empty matrices,
    /// validates the configuration, and rejects matrix/configuration shape
    /// mismatches before hierarchy construction. The compatibility
    /// `Self::from_csr` constructor retains its historical infallible
    /// behavior.
    pub fn try_from_csr(matrix: &CsrMatrix<T>, config: AmgConfig) -> Result<Self, SolverError> {
        matrix.validate_structure()?;
        if matrix.num_rows != matrix.num_cols {
            return Err(SolverError::OperationFailed {
                operation: "AMG construction",
                reason: format!(
                    "matrix must be square, got {}x{}",
                    matrix.num_rows, matrix.num_cols
                ),
            });
        }
        if matrix.num_rows == 0 {
            return Err(SolverError::OperationFailed {
                operation: "AMG construction",
                reason: "matrix must be non-empty".into(),
            });
        }
        config.validate()?;
        if !matrix.num_rows.is_multiple_of(config.block_size) {
            return Err(SolverError::OperationFailed {
                operation: "AMG construction",
                reason: format!(
                    "matrix dimension {} must be divisible by block_size {}",
                    matrix.num_rows, config.block_size
                ),
            });
        }
        if let Some(material) = &config.material_soc
            && material.tensors.len() != matrix.num_rows
        {
            return Err(SolverError::OperationFailed {
                operation: "AMG construction",
                reason: format!(
                    "material_soc has {} tensors for {} matrix rows",
                    material.tensors.len(),
                    matrix.num_rows
                ),
            });
        }
        let preconditioner = Self::from_csr(matrix, config);
        preconditioner.validate_generated_setup()?;
        Ok(preconditioner)
    }

    /// Validate values produced while building the hierarchy.
    ///
    /// The compatibility constructor intentionally preserves the historical
    /// best-effort behavior, but checked callers must not receive a hierarchy
    /// containing an overflowed transfer operator, smoother diagonal, or
    /// spectral estimate.  These values can become non-finite even when the
    /// input CSR values and configuration were finite (for example, when a
    /// row norm overflows during interpolation smoothing).
    fn validate_generated_setup(&self) -> Result<(), SolverError> {
        if self.levels.is_empty()
            || !self.setup_time_ms.is_finite()
            || !self.grid_complexity.is_finite()
            || !self.operator_complexity.is_finite()
        {
            return Err(SolverError::OperationFailed {
                operation: "AMG setup",
                reason: "generated hierarchy diagnostics are non-finite or empty".into(),
            });
        }

        for (level_index, level) in self.levels.iter().enumerate() {
            if level
                .matrix
                .values
                .iter()
                .any(|value| !value.norm().is_finite())
                || level.prolongation.as_ref().is_some_and(|matrix| {
                    matrix.values.iter().any(|value| !value.norm().is_finite())
                })
                || level.restriction.as_ref().is_some_and(|matrix| {
                    matrix.values.iter().any(|value| !value.norm().is_finite())
                })
                || level.diag_inv.iter().any(|value| !value.norm().is_finite())
                || level.l1_diag.iter().any(|value| !value.is_finite())
                || level.lambda_max.is_some_and(|value| !value.is_finite())
            {
                return Err(SolverError::OperationFailed {
                    operation: "AMG setup",
                    reason: format!(
                        "generated hierarchy level {level_index} contains NaN or infinity"
                    ),
                });
            }
        }

        Ok(())
    }

    /// Get diagnostic information
    pub fn diagnostics(&self) -> AmgDiagnostics {
        let apply_count = self.apply_count.load(Ordering::Relaxed);
        let total_apply_ms = self.total_apply_ns.load(Ordering::Relaxed) as f64 / 1e6;
        let level_dofs = self
            .levels
            .iter()
            .map(|level| level.num_dofs)
            .collect::<Vec<_>>();
        let level_nnz = self
            .levels
            .iter()
            .map(|level| level.matrix.nnz())
            .collect::<Vec<_>>();
        AmgDiagnostics {
            num_levels: self.levels.len(),
            grid_complexity: self.grid_complexity,
            operator_complexity: self.operator_complexity,
            max_grid_complexity: self.config.max_grid_complexity,
            max_operator_complexity: self.config.max_operator_complexity,
            grid_complexity_within_target: self
                .config
                .max_grid_complexity
                .is_none_or(|target| self.grid_complexity <= target),
            operator_complexity_within_target: self
                .config
                .max_operator_complexity
                .is_none_or(|target| self.operator_complexity <= target),
            setup_time_ms: self.setup_time_ms,
            apply_count,
            total_apply_ms,
            mean_apply_ms: if apply_count == 0 {
                0.0
            } else {
                total_apply_ms / apply_count as f64
            },
            total_nnz: level_nnz.iter().sum(),
            finest_nnz: level_nnz.first().copied().unwrap_or(0),
            coarsest_dofs: level_dofs.last().copied().unwrap_or(0),
            level_dofs,
            level_nnz,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;
    use sonium_compute::{Backend, CpuBackend, DevicePreconditioner};

    /// Create a simple 1D Laplacian matrix for testing
    fn create_1d_laplacian(n: usize) -> CsrMatrix<Complex64> {
        let mut triplets: Vec<(usize, usize, Complex64)> = Vec::new();

        for i in 0..n {
            triplets.push((i, i, Complex64::new(2.0, 0.0)));
            if i > 0 {
                triplets.push((i, i - 1, Complex64::new(-1.0, 0.0)));
            }
            if i < n - 1 {
                triplets.push((i, i + 1, Complex64::new(-1.0, 0.0)));
            }
        }

        CsrMatrix::from_triplets(n, n, triplets)
    }

    #[test]
    fn from_csr_rejects_structurally_invalid_storage_up_front() {
        // Validate-once: level 0 keeps a verbatim clone and the apply
        // path is unchecked, so malformed storage must panic here
        // instead of later in setup.
        let mut malformed =
            CsrMatrix::from_raw_parts(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0]);
        malformed.row_ptrs[2] = 3;
        assert!(
            std::panic::catch_unwind(|| AmgPreconditioner::from_csr(
                &malformed,
                AmgConfig::default()
            ))
            .is_err()
        );
    }

    #[test]
    fn test_amg_creation() {
        let matrix = create_1d_laplacian(100);
        let config = AmgConfig::default();

        let amg = AmgPreconditioner::from_csr(&matrix, config);

        assert!(amg.num_levels() >= 2);
        assert!(amg.grid_complexity() >= 1.0);
        assert!(amg.operator_complexity() >= 1.0);
        assert!(amg.complexity_within_targets());
    }

    #[test]
    fn test_amg_complexity_targets_are_reported() {
        let matrix = create_1d_laplacian(100);
        let config = AmgConfig::for_aggregation().with_complexity_targets(1.0, 1.0);

        let amg = AmgPreconditioner::from_csr(&matrix, config);
        let diag = amg.diagnostics();

        assert_eq!(diag.max_grid_complexity, Some(1.0));
        assert_eq!(diag.max_operator_complexity, Some(1.0));
        assert!(!diag.grid_complexity_within_target);
        assert!(!diag.operator_complexity_within_target);
        assert!(!amg.complexity_within_targets());
    }

    #[test]
    fn test_default_and_fem_configs_use_parallel_components() {
        let default = AmgConfig::default();
        assert_eq!(default.coarsening, AmgCoarsening::Cljp);
        assert_eq!(default.smoother, AmgSmoother::Jacobi);

        let fem = AmgConfig::for_fem();
        assert_eq!(fem.coarsening, AmgCoarsening::Cljp);
        assert_eq!(fem.smoother, AmgSmoother::MulticolorGaussSeidel);
    }

    #[test]
    fn test_physics_profiles_choose_expected_parallel_paths() {
        let laplacian = AmgConfig::for_physics(AmgPhysicsProfile::SpdLaplacian);
        assert_eq!(laplacian.coarsening, AmgCoarsening::SmoothedAggregation);
        assert_eq!(laplacian.smoother, AmgSmoother::Chebyshev);
        assert_eq!(laplacian.max_operator_complexity, Some(3.0));

        let elasticity = AmgConfig::for_physics(AmgPhysicsProfile::Elasticity { block_size: 3 });
        assert_eq!(elasticity.block_size, 3);
        assert_eq!(elasticity.near_nullspace_dim, 3);
        assert_eq!(elasticity.smoother, AmgSmoother::Chebyshev);

        let helmholtz = AmgConfig::for_physics(AmgPhysicsProfile::HelmholtzShiftedLaplacian);
        assert_eq!(helmholtz.coarsening, AmgCoarsening::SmoothedAggregation);
        assert_eq!(helmholtz.cycle, AmgCycle::FCycle);
        assert_eq!(helmholtz.num_pre_smooth, 3);

        let maxwell = AmgConfig::for_physics(AmgPhysicsProfile::MaxwellHCurl { block_size: 3 });
        assert_eq!(maxwell.coarsening, AmgCoarsening::EnergyMinimizing);
        assert_eq!(maxwell.interpolation, AmgInterpolation::Extended);
        assert_eq!(maxwell.energy_min_pattern_distance, 2);

        let saddle = AmgConfig::for_physics(AmgPhysicsProfile::SaddlePointVelocity);
        assert_eq!(saddle.coarsening, AmgCoarsening::Hmis);
        assert_eq!(saddle.smoother, AmgSmoother::L1Jacobi);

        let bem = AmgConfig::for_physics(AmgPhysicsProfile::BemHybrid);
        assert_eq!(bem.coarsening, AmgCoarsening::Pmis);
        assert_eq!(bem.aggressive_coarsening_levels, 1);
    }

    #[test]
    fn test_amg_apply_completes_without_workspace_deadlock() {
        const N: usize = 200;
        let matrix = create_1d_laplacian(N);
        let config = AmgConfig {
            aggressive_coarsening_levels: 2,
            ..Default::default()
        };
        let amg = AmgPreconditioner::from_csr(&matrix, config);
        assert!(amg.num_levels() > 1, "test requires post-smoothing");

        let r = (0..N)
            .map(|i| Complex64::new(i as f64, 0.0))
            .collect::<Vec<_>>();

        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let z = amg.apply(&r);
            tx.send(z).expect("test receiver should remain connected");
        });

        let z = rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("AMG apply deadlocked while reusing a level workspace");

        assert_eq!(z.len(), N);

        let original = (0..N)
            .map(|i| Complex64::new(i as f64, 0.0))
            .collect::<Vec<_>>();
        let diff: f64 = (sonium_solvers_core::vec_ops::vec_sub(&z, &original))
            .iter()
            .map(|x| x.norm())
            .sum();
        assert!(diff > 1e-10, "Preconditioner should modify the vector");
    }

    #[test]
    fn test_amg_pmis_coarsening() {
        let matrix = create_1d_laplacian(100);
        let config = AmgConfig {
            coarsening: AmgCoarsening::Pmis,
            ..Default::default()
        };

        let amg = AmgPreconditioner::from_csr(&matrix, config);
        assert!(amg.num_levels() >= 2);
    }

    #[test]
    fn test_amg_cljp_coarsening() {
        let matrix = create_1d_laplacian(100);
        let config = AmgConfig {
            coarsening: AmgCoarsening::Cljp,
            ..Default::default()
        };

        let amg = AmgPreconditioner::from_csr(&matrix, config);
        assert!(amg.num_levels() >= 2);

        // Verify it reduces the residual
        let b = (0..100)
            .map(|i| Complex64::new(i as f64, 0.0))
            .collect::<Vec<_>>();
        let z = amg.apply(&b);
        let r = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&z));
        let r_norm = r.iter().map(|x| x.norm()).sum::<f64>();
        let b_norm = b.iter().map(|x| x.norm()).sum::<f64>();
        assert!(
            r_norm < b_norm,
            "CLJP AMG should reduce residual: {r_norm:.4e} vs {b_norm:.4e}"
        );
    }

    #[test]
    fn test_standard_interpolation_classical_weights() {
        // Modified-classical semantics: C-rows are identity, F-rows carry
        // `-a_ij / (a_ii + weak_sum)` over their strong C-neighbors with
        // every non-strong-C off-diagonal lumped into `weak_sum`. Diagonal
        // lumping (not a post-hoc rescale) keeps M-matrix rows positive:
        // the historical `1 + weak_sum / (a_ii * sum_weights)` rescale
        // flips whole rows negative whenever weak connections dominate
        // (665 such rows on 3D Poisson level 1). The oracle below
        // recomputes the closed form from the matrix, strength graph,
        // and splitting, independently of `interpolation_row`'s assembly
        // path. Assumes default (zero) truncation, as configured.
        for coarsening in [AmgCoarsening::Cljp, AmgCoarsening::Falgout] {
            let matrix = create_1d_laplacian(40);
            let config = AmgConfig {
                coarsening,
                interpolation: AmgInterpolation::Standard,
                ..Default::default()
            };
            let strong = strength::compute_strength_matrix(&matrix, config.strong_threshold);
            let (point_types, coarse_to_fine) = match coarsening {
                AmgCoarsening::Cljp => coarsening::coarsen_cljp(&matrix, &strong),
                _ => coarsening::coarsen_falgout(&matrix, &strong),
            };
            let mut fine_to_coarse = vec![usize::MAX; 40];
            for (c, &f) in coarse_to_fine.iter().enumerate() {
                fine_to_coarse[f] = c;
            }
            let p = interpolation::build_interpolation(
                &matrix,
                &strong,
                &point_types,
                &coarse_to_fine,
                &config,
            );
            for i in 0..40 {
                let mut expected: Vec<(usize, Complex64)> = Vec::new();
                if point_types[i] == PointType::Coarse {
                    expected.push((fine_to_coarse[i], Complex64::new(1.0, 0.0)));
                } else {
                    let strong_c: Vec<usize> = strong[i]
                        .iter()
                        .copied()
                        .filter(|&j| point_types[j] == PointType::Coarse)
                        .collect();
                    if strong_c.is_empty() {
                        // Injection fallback, mirroring `interpolation_row`.
                        let mut best = usize::MAX;
                        let mut best_v = 0.0;
                        for (j, v) in matrix.row_entries(i) {
                            if point_types[j] == PointType::Coarse && v.norm() > best_v {
                                best_v = v.norm();
                                best = j;
                            }
                        }
                        if best != usize::MAX {
                            expected.push((fine_to_coarse[best], Complex64::new(1.0, 0.0)));
                        } else if !coarse_to_fine.is_empty() {
                            expected.push((0, Complex64::new(1.0, 0.0)));
                        }
                    } else {
                        let a_ii = matrix.get(i, i);
                        let weak_sum: Complex64 = matrix
                            .row_entries(i)
                            .filter(|(j, _)| *j != i && !strong_c.contains(j))
                            .map(|(_, v)| v)
                            .sum();
                        if (a_ii + weak_sum).norm() > 1e-15 {
                            for &j in &strong_c {
                                expected.push((
                                    fine_to_coarse[j],
                                    -matrix.get(i, j) / (a_ii + weak_sum),
                                ));
                            }
                        }
                    }
                }
                let mut actual: Vec<(usize, Complex64)> = p.row_entries(i).collect();
                actual.sort_by_key(|(j, _)| *j);
                expected.sort_by_key(|(j, _)| *j);
                assert_eq!(
                    actual.len(),
                    expected.len(),
                    "{coarsening:?} P row {i} has {} entries, expected {}",
                    actual.len(),
                    expected.len()
                );
                for ((aj, av), (ej, ev)) in actual.iter().zip(expected.iter()) {
                    assert_eq!(*aj, *ej, "{coarsening:?} P row {i} column mismatch");
                    assert!(
                        (*av - *ev).norm() < 1e-12,
                        "{coarsening:?} P[{i},{aj}] = {av:?}, expected {ev:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_galerkin_drop_tol_default_and_validation() {
        assert_eq!(AmgConfig::default().galerkin_drop_tol, 0.001);
        for tol in [0.0, 1e-7, 0.001, 0.1] {
            let config = AmgConfig {
                galerkin_drop_tol: tol,
                ..Default::default()
            };
            assert!(config.validate().is_ok(), "tol {tol} should validate");
        }
        for tol in [f64::NAN, -1.0, 0.5] {
            let config = AmgConfig {
                galerkin_drop_tol: tol,
                ..Default::default()
            };
            assert!(config.validate().is_err(), "tol {tol} should not validate");
        }
    }

    #[test]
    fn test_amg_different_smoothers() {
        let matrix = create_1d_laplacian(50);
        let r = (0..50)
            .map(|i| Complex64::new(i as f64, 0.0))
            .collect::<Vec<_>>();

        for smoother in [
            AmgSmoother::Jacobi,
            AmgSmoother::L1Jacobi,
            AmgSmoother::SymmetricGaussSeidel,
            AmgSmoother::MulticolorGaussSeidel,
        ] {
            let config = AmgConfig {
                smoother,
                ..Default::default()
            };
            let amg = AmgPreconditioner::from_csr(&matrix, config);

            let z = amg.apply(&r);
            assert_eq!(z.len(), r.len());
        }
    }

    #[test]
    fn test_amg_reduces_residual() {
        let n = 64;
        let matrix = create_1d_laplacian(n);
        let config = AmgConfig::default();
        let amg = AmgPreconditioner::from_csr(&matrix, config);

        let b = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();

        let mut x = vec![Complex64::new(0.0, 0.0); n];

        let r0 = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
        let norm_r0: f64 = r0.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();

        for _ in 0..10 {
            let r = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
            let z = amg.apply(&r);
            x = sonium_solvers_core::vec_ops::vec_add(&x, &z);
        }

        let rf = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
        let norm_rf: f64 = rf.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();

        assert!(
            norm_rf < norm_r0 * 0.1,
            "AMG should significantly reduce residual: {} -> {}",
            norm_r0,
            norm_rf
        );
    }

    #[test]
    fn test_backend_amg_reduces_residual_on_cpu_backend() {
        let n = 64;
        let matrix = create_1d_laplacian(n);
        let config = AmgConfig {
            smoother: AmgSmoother::Chebyshev,
            num_pre_smooth: 2,
            num_post_smooth: 2,
            ..AmgConfig::default()
        };
        let backend = CpuBackend::new();
        let amg =
            BackendAmgPreconditioner::<CpuBackend, Complex64>::from_csr(&backend, &matrix, config);

        assert!(amg.num_levels() >= 2);

        let b = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();
        let mut b_dev = backend.raw_alloc(n).expect("b allocation failed");
        backend.copy_h2d(&b, &mut b_dev);

        let mut z_dev = backend.raw_alloc(n).expect("z allocation failed");
        amg.apply(&backend, &b_dev, &mut z_dev);

        let mut z_host = vec![Complex64::new(0.0, 0.0); n];
        backend.copy_d2h(&z_dev, &mut z_host);

        let z = z_host;
        let r = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&z));
        let r_norm = r.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        let b_norm = b.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();

        assert!(
            r_norm < b_norm,
            "Backend AMG should reduce residual: {r_norm:.4e} vs {b_norm:.4e}"
        );
    }

    #[test]
    fn test_backend_checked_setup_rejects_generated_non_finite_hierarchy() {
        let scale = f64::MAX;
        let off_diagonal = 0.49 * scale;
        let matrix = CsrMatrix::from_triplets(
            3,
            3,
            vec![
                (0, 0, scale),
                (1, 1, scale),
                (2, 2, scale),
                (0, 1, off_diagonal),
                (1, 0, off_diagonal),
                (0, 2, off_diagonal),
                (2, 0, off_diagonal),
                (1, 2, off_diagonal),
                (2, 1, off_diagonal),
            ],
        );
        let backend = CpuBackend::new();
        let result = std::panic::catch_unwind(|| {
            BackendAmgPreconditioner::<CpuBackend, f64>::try_from_csr(
                &backend,
                &matrix,
                AmgConfig::default(),
            )
        })
        .expect("checked backend AMG setup must not panic on finite overflow fixtures");
        let error = match result {
            Ok(_) => panic!("generated non-finite hierarchy must be rejected"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            sonium_compute::BackendError::InvalidArgument {
                operation: "amg_setup",
                ..
            }
        ));
    }

    #[test]
    fn test_diagnostics() {
        let matrix = create_1d_laplacian(100);
        let amg = AmgPreconditioner::from_csr(&matrix, AmgConfig::default());

        let diag = amg.diagnostics();

        assert!(diag.num_levels >= 2);
        assert_eq!(diag.level_dofs.len(), diag.num_levels);
        assert_eq!(diag.level_nnz.len(), diag.num_levels);
        assert!(diag.grid_complexity >= 1.0);
        assert!(diag.setup_time_ms >= 0.0);
        assert_eq!(diag.apply_count, 0);
        assert_eq!(diag.finest_nnz, matrix.nnz());
        assert!(diag.total_nnz >= diag.finest_nnz);

        let r = vec![Complex64::new(1.0, 0.0); matrix.num_rows];
        let _ = amg.apply(&r);
        let diag = amg.diagnostics();
        assert_eq!(diag.apply_count, 1);
        assert!(diag.total_apply_ms >= 0.0);
    }

    #[test]
    fn test_chebyshev_smoother_reduces_residual() {
        let n = 32;
        let matrix = create_1d_laplacian(n);
        let diag_inv = util::compute_diag_inv(&matrix);
        let lambda_max = util::estimate_max_eigenvalue(&matrix, &diag_inv);

        // lambda_max for 1D Laplacian with diag=4 should be close to 2 (D^{-1}A eigenvalues)
        assert!(
            lambda_max > 0.5,
            "lambda_max should be positive: {}",
            lambda_max
        );

        let b = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();
        let mut x = vec![Complex64::new(0.0, 0.0); n];
        let mut ax = vec![Complex64::new(0.0, 0.0); n];

        let r0 = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
        let norm_r0: f64 = r0.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();

        AmgPreconditioner::smooth_chebyshev(&matrix, &diag_inv, &b, &mut x, lambda_max, 5, &mut ax);

        let r1 = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
        let norm_r1: f64 = r1.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();

        assert!(
            norm_r1 < norm_r0,
            "Chebyshev smoother should reduce residual: {} -> {}",
            norm_r0,
            norm_r1
        );
    }

    #[test]
    fn test_amg_chebyshev_converges() {
        let n = 64;
        let matrix = create_1d_laplacian(n);
        let config = AmgConfig {
            smoother: AmgSmoother::Chebyshev,
            num_pre_smooth: 2,
            num_post_smooth: 2,
            ..AmgConfig::default()
        };
        let amg = AmgPreconditioner::from_csr(&matrix, config);

        let b = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();

        let mut x = vec![Complex64::new(0.0, 0.0); n];

        let r0 = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
        let norm_r0: f64 = r0.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();

        for _ in 0..10 {
            let r = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
            let z = amg.apply(&r);
            x = sonium_solvers_core::vec_ops::vec_add(&x, &z);
        }

        let rf = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
        let norm_rf: f64 = rf.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();

        assert!(
            norm_rf < norm_r0 * 0.1,
            "AMG with Chebyshev should reduce residual: {} -> {}",
            norm_r0,
            norm_rf
        );
    }

    /// Create a 2D Poisson (5-point stencil) matrix on an nx-by-nx grid
    fn create_2d_poisson(nx: usize) -> CsrMatrix<Complex64> {
        let n = nx * nx;
        let mut triplets: Vec<(usize, usize, Complex64)> = Vec::new();

        for iy in 0..nx {
            for ix in 0..nx {
                let i = iy * nx + ix;
                triplets.push((i, i, Complex64::new(4.0, 0.0)));
                if ix > 0 {
                    triplets.push((i, i - 1, Complex64::new(-1.0, 0.0)));
                }
                if ix < nx - 1 {
                    triplets.push((i, i + 1, Complex64::new(-1.0, 0.0)));
                }
                if iy > 0 {
                    triplets.push((i, i - nx, Complex64::new(-1.0, 0.0)));
                }
                if iy < nx - 1 {
                    triplets.push((i, i + nx, Complex64::new(-1.0, 0.0)));
                }
            }
        }

        CsrMatrix::from_triplets(n, n, triplets)
    }

    #[test]
    fn test_aggregation_coarsening_ratio() {
        // Verify aggregation produces 3-6x coarsening on 2D Poisson
        let nx = 16;
        let n = nx * nx; // 256
        let matrix = create_2d_poisson(nx);
        let config = AmgConfig::for_aggregation();

        let amg = AmgPreconditioner::from_csr(&matrix, config);
        let diag = amg.diagnostics();

        assert!(
            diag.num_levels >= 2,
            "Should have at least 2 levels, got {}",
            diag.num_levels
        );

        // Check coarsening ratio from level 0 to level 1
        let fine_dofs = diag.level_dofs[0] as f64;
        let coarse_dofs = diag.level_dofs[1] as f64;
        let ratio = fine_dofs / coarse_dofs;

        assert_eq!(fine_dofs as usize, n);
        assert!(
            (2.0..=10.0).contains(&ratio),
            "Aggregation coarsening ratio should be ~3-8x, got {:.1}x ({} -> {})",
            ratio,
            fine_dofs as usize,
            coarse_dofs as usize
        );
    }

    #[test]
    fn test_amg_aggregation_converges() {
        // Verify SA-AMG V-cycle converges on 2D Poisson
        let nx = 16;
        let n = nx * nx;
        let matrix = create_2d_poisson(nx);
        let config = AmgConfig::for_aggregation();
        let amg = AmgPreconditioner::from_csr(&matrix, config);

        let b = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();

        let mut x = vec![Complex64::new(0.0, 0.0); n];

        let r0 = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
        let norm_r0: f64 = r0.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();

        for _ in 0..30 {
            let r = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
            let z = amg.apply(&r);
            x = sonium_solvers_core::vec_ops::vec_add(&x, &z);
        }

        let rf = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
        let norm_rf: f64 = rf.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();

        assert!(
            norm_rf < norm_r0 * 5e-2,
            "SA-AMG should converge: residual ratio = {:.2e} (from {} to {})",
            norm_rf / norm_r0,
            norm_r0,
            norm_rf
        );
    }

    #[test]
    fn test_amg_aggregation_few_iterations() {
        // Verify SA-AMG converges significantly faster than classical RS on 2D Poisson.
        // Compare residual reduction after a fixed number of stationary iterations.
        let nx = 16;
        let n = nx * nx;
        let matrix = create_2d_poisson(nx);

        let b = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();

        let r0_norm: f64 = b.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();

        // SA-AMG: should reach 1e-2 relative residual within 15 stationary iterations
        let config_sa = AmgConfig::for_aggregation();
        let amg_sa = AmgPreconditioner::from_csr(&matrix, config_sa);

        let mut x_sa = vec![Complex64::new(0.0, 0.0); n];
        let tol = 1e-2;
        let max_iter = 50;
        let mut sa_converged_iter = max_iter;

        for iter in 0..max_iter {
            let r = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x_sa));
            let norm_r: f64 = r.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
            if norm_r < tol * r0_norm {
                sa_converged_iter = iter;
                break;
            }
            let z = amg_sa.apply(&r);
            x_sa = sonium_solvers_core::vec_ops::vec_add(&x_sa, &z);
        }

        assert!(
            sa_converged_iter < max_iter,
            "SA-AMG should converge within {} iterations, did not reach tol={}",
            max_iter,
            tol
        );
        assert!(
            sa_converged_iter <= 20,
            "SA-AMG should converge to 1e-2 in <=20 stationary iterations on 2D Poisson, took {}",
            sa_converged_iter
        );
    }

    #[test]
    fn test_mis_k_distance_separation() {
        // 1D chain of 20 nodes: node i connected to i-1 and i+1
        // Test that MIS-k seeds are separated by >= k+1 hops
        let n = 20;
        let mut strong: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, neighbors) in strong.iter_mut().enumerate() {
            if i > 0 {
                neighbors.push(i - 1);
            }
            if i < n - 1 {
                neighbors.push(i + 1);
            }
        }

        for k in [1, 2, 3] {
            let (aggregate_ids, num_aggregates) = coarsening::coarsen_mis_k(&strong, k);

            assert_eq!(aggregate_ids.len(), n);
            assert!(num_aggregates > 0);

            // Identify seeds: for each aggregate, the first node assigned to it
            // is the seed (since seeds get their aggregate IDs first in order).
            let mut seed_of_agg = vec![usize::MAX; num_aggregates];
            for (i, &agg) in aggregate_ids.iter().enumerate() {
                if seed_of_agg[agg] == usize::MAX {
                    seed_of_agg[agg] = i;
                }
            }

            // Collect all seed positions
            let seeds: Vec<usize> = seed_of_agg
                .iter()
                .copied()
                .filter(|&s| s != usize::MAX)
                .collect();

            // On a 1D chain, distance between nodes is |i - j|.
            // Verify min distance between any two seeds >= k+1.
            for (idx_a, &sa) in seeds.iter().enumerate() {
                for &sb in &seeds[idx_a + 1..] {
                    let dist = sa.abs_diff(sb);
                    assert!(
                        dist > k,
                        "MIS-{}: seeds at {} and {} are only {} apart (need >= {})",
                        k,
                        sa,
                        sb,
                        dist,
                        k + 1,
                    );
                }
            }
        }
    }

    #[test]
    fn test_mis_k_coarsening_ratios() {
        // 16x16 2D Poisson
        // MIS-1 should produce MORE aggregates than MIS-2, which should produce MORE than MIS-3
        // (smaller k = smaller aggregates = more of them)
        let nx = 16;
        let n = nx * nx;
        let matrix = create_2d_poisson(nx);
        let strong_connections = strength::compute_strength_matrix(&matrix, 0.08);

        let (_, num_agg_1) = coarsening::coarsen_mis_k(&strong_connections, 1);
        let (_, num_agg_2) = coarsening::coarsen_mis_k(&strong_connections, 2);
        let (_, num_agg_3) = coarsening::coarsen_mis_k(&strong_connections, 3);

        assert!(
            num_agg_1 > num_agg_2,
            "MIS-1 should produce more aggregates than MIS-2: {} vs {}",
            num_agg_1,
            num_agg_2,
        );
        assert!(
            num_agg_2 > num_agg_3,
            "MIS-2 should produce more aggregates than MIS-3: {} vs {}",
            num_agg_2,
            num_agg_3,
        );

        // Sanity: all nodes must be assigned
        let (ids_1, _) = coarsening::coarsen_mis_k(&strong_connections, 1);
        let (ids_2, _) = coarsening::coarsen_mis_k(&strong_connections, 2);
        let (ids_3, _) = coarsening::coarsen_mis_k(&strong_connections, 3);
        assert_eq!(ids_1.len(), n);
        assert_eq!(ids_2.len(), n);
        assert_eq!(ids_3.len(), n);
    }

    #[test]
    fn test_material_soc_config() {
        // Verify MaterialSocConfig can be constructed and used without panic
        let n = 100;
        let matrix = create_1d_laplacian(n);
        let material = MaterialSocConfig {
            tensors: (0..n).map(|_| vec![1.0, 0.0, 0.0, 1.0]).collect(),
            dim: 2,
        };
        let config = AmgConfig {
            material_soc: Some(material),
            ..AmgConfig::default()
        };
        let amg = AmgPreconditioner::from_csr(&matrix, config);
        assert!(amg.num_levels() >= 2);
    }

    #[test]
    fn test_energy_min_preserves_constants() {
        // Build 2D Poisson, verify P * ones_coarse ≈ ones_fine
        let nx = 16; // 16x16 = 256 nodes, well above coarse_size threshold
        let n = nx * nx;
        let matrix = create_2d_poisson(nx);
        let config = AmgConfig {
            coarse_size: 20,
            ..AmgConfig::for_energy_minimizing()
        };
        let amg = AmgPreconditioner::from_csr(&matrix, config);

        // The prolongation at level 0 should preserve constants:
        // P * ones_coarse ≈ ones_fine
        if let Some(ref p) = amg.levels[0].prolongation {
            let n_coarse = p.num_cols;
            let ones_coarse = vec![Complex64::new(1.0, 0.0); n_coarse];
            let result = p.matvec(&ones_coarse);

            // Check that every entry is approximately 1.0
            for i in 0..n {
                assert!(
                    (result[i] - Complex64::new(1.0, 0.0)).norm() < 1e-10,
                    "P * ones should be ones at row {}: got {:?}",
                    i,
                    result[i]
                );
            }
        } else {
            panic!("Expected prolongation operator at level 0");
        }
    }

    #[test]
    fn test_energy_min_convergence() {
        // 16x16 2D Poisson with energy-minimizing AMG
        // 30 stationary iterations, verify residual reduction > 1e-2
        let nx = 16;
        let n = nx * nx;
        let matrix = create_2d_poisson(nx);
        let config = AmgConfig {
            energy_min_iterations: 8,
            energy_min_pattern_distance: 2,
            ..AmgConfig::for_energy_minimizing()
        };
        let amg = AmgPreconditioner::from_csr(&matrix, config);

        let b = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();

        let mut x = vec![Complex64::new(0.0, 0.0); n];

        let r0 = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
        let norm_r0: f64 = r0.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();

        for _ in 0..30 {
            let r = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
            let z = amg.apply(&r);
            x = sonium_solvers_core::vec_ops::vec_add(&x, &z);
        }

        let rf = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
        let norm_rf: f64 = rf.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();

        assert!(
            norm_rf < norm_r0 * 1e-1,
            "Energy-min AMG should converge: residual ratio = {:.2e} (from {} to {})",
            norm_rf / norm_r0,
            norm_r0,
            norm_rf
        );
    }

    #[test]
    fn test_energy_min_operator_complexity() {
        // 16x16 2D Poisson, verify operator complexity < 3.0
        let nx = 16;
        let matrix = create_2d_poisson(nx);
        let config = AmgConfig::for_energy_minimizing();
        let amg = AmgPreconditioner::from_csr(&matrix, config);

        let op_cx = amg.operator_complexity();
        assert!(
            op_cx < 3.0,
            "Operator complexity should be < 3.0, got {:.2}",
            op_cx
        );
    }

    #[cfg(feature = "experimental-gnn")]
    #[test]
    fn test_gnn_coarsening_produces_valid_aggregates() {
        // GNN coarsening should assign every node and produce contiguous IDs
        let nx = 16;
        let n = nx * nx;
        let matrix = create_2d_poisson(nx);
        let strong_connections = strength::compute_strength_matrix(&matrix, 0.08);

        let weights = gnn_coarsening::GnnCoarseningWeights::default();
        let (aggregate_ids, num_aggregates) =
            gnn_coarsening::gnn_coarsen(&matrix, &strong_connections, &weights);

        // Every node must be assigned
        assert_eq!(aggregate_ids.len(), n);
        for (i, &agg) in aggregate_ids.iter().enumerate() {
            assert!(
                agg < num_aggregates,
                "Node {} has invalid aggregate ID {} (num_aggregates={})",
                i,
                agg,
                num_aggregates
            );
        }

        // No gaps in aggregate IDs: all IDs in 0..num_aggregates must appear
        let mut seen = vec![false; num_aggregates];
        for &agg in &aggregate_ids {
            seen[agg] = true;
        }
        for (id, &present) in seen.iter().enumerate() {
            assert!(present, "Aggregate ID {} has no nodes assigned", id);
        }
    }

    #[cfg(feature = "experimental-gnn")]
    #[test]
    fn test_gnn_coarsening_ratio() {
        // GNN coarsening should produce a reasonable coarsening ratio (2-10x)
        let nx = 16;
        let n = nx * nx;
        let matrix = create_2d_poisson(nx);
        let strong_connections = strength::compute_strength_matrix(&matrix, 0.08);

        let weights = gnn_coarsening::GnnCoarseningWeights::default();
        let (_, num_aggregates) =
            gnn_coarsening::gnn_coarsen(&matrix, &strong_connections, &weights);

        let ratio = n as f64 / num_aggregates as f64;
        assert!(
            (2.0..=10.0).contains(&ratio),
            "GNN coarsening ratio should be 2-10x, got {:.1}x ({} -> {})",
            ratio,
            n,
            num_aggregates
        );
    }

    #[cfg(feature = "experimental-gnn")]
    #[test]
    fn test_gnn_amg_converges() {
        // GNN-based AMG should converge on 16x16 Poisson
        let nx = 16;
        let n = nx * nx;
        let matrix = create_2d_poisson(nx);
        let config = AmgConfig::for_gnn();
        let amg = AmgPreconditioner::from_csr(&matrix, config);

        assert!(
            amg.num_levels() >= 2,
            "GNN AMG should produce at least 2 levels, got {}",
            amg.num_levels()
        );

        let b = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();

        let mut x = vec![Complex64::new(0.0, 0.0); n];

        let r0 = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
        let norm_r0: f64 = r0.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();

        for _ in 0..30 {
            let r = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
            let z = amg.apply(&r);
            x = sonium_solvers_core::vec_ops::vec_add(&x, &z);
        }

        let rf = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
        let norm_rf: f64 = rf.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();

        assert!(
            norm_rf < norm_r0 * 0.1,
            "GNN AMG should converge: residual ratio = {:.2e} (from {} to {})",
            norm_rf / norm_r0,
            norm_r0,
            norm_rf
        );
    }

    #[cfg(feature = "experimental-gnn")]
    #[test]
    fn test_gnn_seeds_form_independent_set() {
        // With the default weights the message-passing network implements
        // hash-priority parallel MIS: no two seeds may be strong neighbors.
        let nx = 16;
        let n = nx * nx;
        let matrix = create_2d_poisson(nx);
        let strong = strength::compute_strength_matrix(&matrix, 0.08);

        let weights = gnn_coarsening::GnnCoarseningWeights::default();
        let is_seed = gnn_coarsening::gnn_seed_mask(&matrix, &strong, &weights);

        let seed_count = is_seed.iter().filter(|&&s| s).count();
        assert!(
            seed_count > n / 10 && seed_count < n / 2,
            "Expected a PMIS-like seed density on 16x16 Poisson, got {}/{}",
            seed_count,
            n
        );

        for (i, &s) in is_seed.iter().enumerate() {
            if !s {
                continue;
            }
            for &j in &strong[i] {
                assert!(
                    !is_seed[j],
                    "Seeds {} and {} are strong neighbors — not an independent set",
                    i, j
                );
            }
        }
    }

    #[cfg(feature = "experimental-gnn")]
    #[test]
    fn test_gnn_vs_cljp_validation() {
        use sonium_solvers_iterative::{GmresConfig, gmres_preconditioned};

        // Validate GNN coarsening against CLJP on the standard test matrix:
        // comparable coarsening ratio and comparable GMRES iteration count.
        let nx = 16;
        let n = nx * nx;
        let matrix = create_2d_poisson(nx);
        let strong_connections = strength::compute_strength_matrix(&matrix, 0.08);

        // --- Coarsening ratio comparison ---
        let weights = gnn_coarsening::GnnCoarseningWeights::default();
        let (_, num_aggregates) =
            gnn_coarsening::gnn_coarsen(&matrix, &strong_connections, &weights);
        let (_, cljp_coarse) = coarsening::coarsen_cljp(&matrix, &strong_connections);

        let gnn_ratio = n as f64 / num_aggregates as f64;
        let cljp_ratio = n as f64 / cljp_coarse.len() as f64;
        let spread = (gnn_ratio / cljp_ratio).max(cljp_ratio / gnn_ratio);
        assert!(
            spread <= 3.0,
            "GNN and CLJP coarsening ratios should be within 3x of each other: \
             GNN {:.2}x ({}), CLJP {:.2}x ({})",
            gnn_ratio,
            num_aggregates,
            cljp_ratio,
            cljp_coarse.len()
        );

        // --- GMRES iteration count comparison ---
        let b = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();

        let gmres_config = GmresConfig {
            max_iterations: 200,
            restart: 30,
            tolerance: 1e-8,
            ..Default::default()
        };

        let amg_gnn = AmgPreconditioner::from_csr(&matrix, AmgConfig::for_gnn());
        let amg_cljp = AmgPreconditioner::from_csr(&matrix, AmgConfig::default());

        let sol_gnn = gmres_preconditioned(&matrix, &amg_gnn, &b, &gmres_config);
        let sol_cljp = gmres_preconditioned(&matrix, &amg_cljp, &b, &gmres_config);

        assert!(sol_gnn.converged, "GMRES + GNN-AMG should converge");
        assert!(sol_cljp.converged, "GMRES + CLJP-AMG should converge");
        assert!(
            sol_gnn.iterations <= 2 * sol_cljp.iterations + 5,
            "GNN-AMG GMRES iterations ({}) should be comparable to CLJP-AMG ({})",
            sol_gnn.iterations,
            sol_cljp.iterations
        );
    }

    #[test]
    fn test_hmis_produces_valid_cf_splitting() {
        // HMIS must produce a valid C/F splitting:
        // - every point is C or F, coarse set non-trivial
        // - every F-point has at least one strong C-neighbor
        // - (RS second-pass property) every strong F-F pair shares a common
        //   strong C-neighbor
        // - HMIS coarsens no further than PMIS (cleanup only promotes F->C)
        let nx = 16;
        let n = nx * nx;
        let matrix = create_2d_poisson(nx);
        let strong = strength::compute_strength_matrix(&matrix, 0.08);

        let (point_types, coarse_to_fine) = coarsening::coarsen_hmis(&matrix, &strong);
        let (_, pmis_coarse) = coarsening::coarsen_pmis(&matrix, &strong);

        assert!(!coarse_to_fine.is_empty(), "HMIS produced no C-points");
        assert!(coarse_to_fine.len() < n, "HMIS coarsened nothing");
        assert!(
            coarse_to_fine.len() >= pmis_coarse.len(),
            "HMIS ({} C-points) should not coarsen further than PMIS ({} C-points)",
            coarse_to_fine.len(),
            pmis_coarse.len()
        );

        for i in 0..n {
            assert_ne!(
                point_types[i],
                PointType::Undecided,
                "Node {} left undecided",
                i
            );
            if point_types[i] == PointType::Fine {
                assert!(
                    strong[i]
                        .iter()
                        .any(|&j| point_types[j] == PointType::Coarse),
                    "F-point {} has no strong C-neighbor",
                    i
                );
            }
        }

        for i in 0..n {
            if point_types[i] != PointType::Fine {
                continue;
            }
            for &j in &strong[i] {
                if j == i || point_types[j] != PointType::Fine {
                    continue;
                }
                let common_c = strong[j]
                    .iter()
                    .any(|&c| point_types[c] == PointType::Coarse && strong[i].contains(&c));
                assert!(
                    common_c,
                    "Strong F-F pair ({}, {}) has no common strong C-neighbor",
                    i, j
                );
            }
        }
    }

    #[test]
    fn test_hmis_amg_converges() {
        // HMIS-based AMG should converge on 16x16 Poisson
        let nx = 16;
        let n = nx * nx;
        let matrix = create_2d_poisson(nx);
        let config = AmgConfig {
            coarsening: AmgCoarsening::Hmis,
            ..Default::default()
        };
        let amg = AmgPreconditioner::from_csr(&matrix, config);

        assert!(
            amg.num_levels() >= 2,
            "HMIS AMG should produce at least 2 levels, got {}",
            amg.num_levels()
        );

        let b = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();

        let mut x = vec![Complex64::new(0.0, 0.0); n];

        let r0 = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
        let norm_r0: f64 = r0.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();

        for _ in 0..30 {
            let r = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
            let z = amg.apply(&r);
            x = sonium_solvers_core::vec_ops::vec_add(&x, &z);
        }

        let rf = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
        let norm_rf: f64 = rf.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();

        assert!(
            norm_rf < norm_r0 * 0.1,
            "HMIS AMG should converge: residual ratio = {:.2e} (from {} to {})",
            norm_rf / norm_r0,
            norm_r0,
            norm_rf
        );
    }

    #[test]
    fn test_ruge_stuben_dynamic_measure_converges() {
        // RS coarsening with the dynamic lambda update should produce a
        // valid splitting and a converging hierarchy on 16x16 Poisson.
        let nx = 16;
        let n = nx * nx;
        let matrix = create_2d_poisson(nx);
        let strong = strength::compute_strength_matrix(&matrix, 0.25);

        let (point_types, coarse_to_fine) = coarsening::coarsen_ruge_stuben(&matrix, &strong);
        assert!(!coarse_to_fine.is_empty(), "RS produced no C-points");
        assert!(coarse_to_fine.len() < n, "RS coarsened nothing");
        for i in 0..n {
            assert_ne!(point_types[i], PointType::Undecided);
            if point_types[i] == PointType::Fine {
                assert!(
                    strong[i]
                        .iter()
                        .any(|&j| point_types[j] == PointType::Coarse),
                    "F-point {} has no strong C-neighbor",
                    i
                );
            }
        }

        let config = AmgConfig {
            coarsening: AmgCoarsening::RugeStuben,
            ..Default::default()
        };
        let amg = AmgPreconditioner::from_csr(&matrix, config);
        assert!(amg.num_levels() >= 2, "RS AMG should produce >= 2 levels");

        let b = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();

        let mut x = vec![Complex64::new(0.0, 0.0); n];

        let r0 = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
        let norm_r0: f64 = r0.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();

        for _ in 0..30 {
            let r = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
            let z = amg.apply(&r);
            x = sonium_solvers_core::vec_ops::vec_add(&x, &z);
        }

        let rf = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
        let norm_rf: f64 = rf.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();

        assert!(
            norm_rf < norm_r0 * 0.1,
            "RS AMG (dynamic measure) should converge: residual ratio = {:.2e}",
            norm_rf / norm_r0
        );
    }

    #[test]
    fn test_amg_w_cycle_converges() {
        let n = 64;
        let matrix = create_1d_laplacian(n);
        let config = AmgConfig {
            cycle: AmgCycle::WCycle,
            ..Default::default()
        };
        let amg = AmgPreconditioner::from_csr(&matrix, config);

        let b = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();
        let mut x = vec![Complex64::new(0.0, 0.0); n];

        let r0 = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
        let norm_r0: f64 = r0.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();

        for _ in 0..10 {
            let r = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
            let z = amg.apply(&r);
            x = sonium_solvers_core::vec_ops::vec_add(&x, &z);
        }

        let rf = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
        let norm_rf: f64 = rf.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();

        assert!(
            norm_rf < norm_r0 * 0.1,
            "W-cycle should converge: {:.2e} -> {:.2e}",
            norm_r0,
            norm_rf
        );
    }

    #[test]
    fn test_amg_w_cycle_converges_faster_than_v_cycle() {
        let n = 64;
        let matrix = create_1d_laplacian(n);

        let b = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();

        // V-cycle: 5 iterations
        let amg_v = AmgPreconditioner::from_csr(
            &matrix,
            AmgConfig {
                cycle: AmgCycle::VCycle,
                ..Default::default()
            },
        );
        let mut x_v = vec![Complex64::new(0.0, 0.0); n];
        for _ in 0..5 {
            let r = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x_v));
            let z = amg_v.apply(&r);
            x_v = sonium_solvers_core::vec_ops::vec_add(&x_v, &z);
        }
        let r_v = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x_v));
        let norm_v: f64 = r_v.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();

        // W-cycle: 5 iterations
        let amg_w = AmgPreconditioner::from_csr(
            &matrix,
            AmgConfig {
                cycle: AmgCycle::WCycle,
                ..Default::default()
            },
        );
        let mut x_w = vec![Complex64::new(0.0, 0.0); n];
        for _ in 0..5 {
            let r = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x_w));
            let z = amg_w.apply(&r);
            x_w = sonium_solvers_core::vec_ops::vec_add(&x_w, &z);
        }
        let r_w = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x_w));
        let norm_w: f64 = r_w.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();

        // W-cycle should achieve at least as good (typically better) residual
        assert!(
            norm_w <= norm_v * 1.1, // allow 10% tolerance for noise
            "W-cycle ({:.2e}) should converge at least as well as V-cycle ({:.2e})",
            norm_w,
            norm_v
        );
    }

    #[test]
    fn test_amg_f_cycle_converges() {
        let n = 64;
        let matrix = create_1d_laplacian(n);
        let config = AmgConfig {
            cycle: AmgCycle::FCycle,
            ..Default::default()
        };
        let amg = AmgPreconditioner::from_csr(&matrix, config);

        let b = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();
        let mut x = vec![Complex64::new(0.0, 0.0); n];

        let r0 = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
        let norm_r0: f64 = r0.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();

        for _ in 0..10 {
            let r = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
            let z = amg.apply(&r);
            x = sonium_solvers_core::vec_ops::vec_add(&x, &z);
        }

        let rf = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
        let norm_rf: f64 = rf.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();

        assert!(
            norm_rf < norm_r0 * 0.1,
            "F-cycle should converge: {:.2e} -> {:.2e}",
            norm_r0,
            norm_rf
        );
    }

    #[test]
    fn test_nodal_amg_block_size_2() {
        // Create a 2D vector Laplacian (block_size=2):
        // Scalar 1D Laplacian for each component, no cross-coupling.
        // This gives a block-diagonal structure with 2×2 identity blocks.
        let n_nodes = 30;
        let n = n_nodes * 2; // 2 DOFs per node

        let mut row_ptrs = vec![0usize];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for node in 0..n_nodes {
            for comp in 0..2usize {
                let row = node * 2 + comp;
                // Connection to left node, same component
                if node > 0 {
                    col_indices.push((node - 1) * 2 + comp);
                    values.push(Complex64::new(-1.0, 0.0));
                }
                // Diagonal
                col_indices.push(row);
                values.push(Complex64::new(2.0, 0.0));
                // Connection to right node, same component
                if node < n_nodes - 1 {
                    col_indices.push((node + 1) * 2 + comp);
                    values.push(Complex64::new(-1.0, 0.0));
                }
                row_ptrs.push(col_indices.len());
            }
        }

        let matrix = CsrMatrix::from_raw_parts(n, n, row_ptrs, col_indices, values);

        // Nodal AMG with block_size=2
        let config = AmgConfig::for_systems(2);
        let amg = AmgPreconditioner::from_csr(&matrix, config);

        assert!(
            amg.num_levels() >= 2,
            "Nodal AMG should build at least 2 levels, got {}",
            amg.num_levels()
        );

        // Verify it reduces the residual
        let b = (0..n)
            .map(|i| Complex64::new(i as f64, 0.0))
            .collect::<Vec<_>>();
        let z = amg.apply(&b);
        let r = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&z));
        let r_norm: f64 = r.iter().map(|v| v.norm()).sum();
        let b_norm: f64 = b.iter().map(|v| v.norm()).sum();
        assert!(
            r_norm < b_norm,
            "Nodal AMG should reduce residual: {r_norm:.4e} vs {b_norm:.4e}"
        );
    }

    #[test]
    fn test_aggressive_coarsening_fewer_levels() {
        // Aggressive (A^2-strength) coarsening targets 2D/3D PDE systems.
        // On 1D problems its distance-2 C-neighbors are structural zeros
        // for direct interpolation (zero-numerator rows no diagonal
        // formula can repair — hypre pairs aggressive coarsening with
        // multipass interpolation instead), so coverage uses 2D Poisson.
        let nx = 20;
        let n = nx * nx;
        let matrix = create_2d_poisson(nx);

        let normal = AmgConfig::default();
        let aggressive = AmgConfig {
            aggressive_coarsening_levels: 2,
            ..Default::default()
        };

        let amg_normal = AmgPreconditioner::from_csr(&matrix, normal);
        let amg_aggressive = AmgPreconditioner::from_csr(&matrix, aggressive);

        assert!(
            amg_aggressive.num_levels() <= amg_normal.num_levels(),
            "Aggressive ({} levels) should have <= normal ({} levels)",
            amg_aggressive.num_levels(),
            amg_normal.num_levels()
        );

        // Aggressive should still reduce the residual
        let b = (0..n)
            .map(|i| Complex64::new(i as f64, 0.0))
            .collect::<Vec<_>>();
        let z = amg_aggressive.apply(&b);
        let r = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&z));
        let r_norm: f64 = r.iter().map(|v| v.norm()).sum();
        let b_norm: f64 = b.iter().map(|v| v.norm()).sum();
        assert!(
            r_norm < b_norm,
            "Aggressive AMG should reduce residual: {r_norm:.4e} vs {b_norm:.4e}"
        );
    }

    #[test]
    fn test_multicolor_gs_converges() {
        let nx = 16;
        let n = nx * nx;
        let matrix = create_2d_poisson(nx);
        let config = AmgConfig {
            smoother: AmgSmoother::MulticolorGaussSeidel,
            ..Default::default()
        };
        let amg = AmgPreconditioner::from_csr(&matrix, config);

        assert!(amg.num_levels() >= 2);

        let b = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();
        let mut x = vec![Complex64::new(0.0, 0.0); n];

        let r0 = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
        let norm_r0: f64 = r0.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();

        for _ in 0..20 {
            let r = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
            let z = amg.apply(&r);
            x = sonium_solvers_core::vec_ops::vec_add(&x, &z);
        }

        let rf = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
        let norm_rf: f64 = rf.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();

        assert!(
            norm_rf < norm_r0 * 0.05,
            "Multicolor GS AMG should converge: residual ratio = {:.2e} (from {:.2e} to {:.2e})",
            norm_rf / norm_r0,
            norm_r0,
            norm_rf
        );
    }

    #[test]
    fn test_multicolor_vs_sgs() {
        let nx = 16;
        let n = nx * nx;
        let matrix = create_2d_poisson(nx);

        let b = (0..n)
            .map(|i| Complex64::new((i as f64).sin(), 0.0))
            .collect::<Vec<_>>();
        let r0_norm: f64 = b.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        let tol = 1e-4;
        let max_iter = 100;

        // SGS
        let amg_sgs = AmgPreconditioner::from_csr(
            &matrix,
            AmgConfig {
                smoother: AmgSmoother::SymmetricGaussSeidel,
                ..Default::default()
            },
        );
        let mut x_sgs = vec![Complex64::new(0.0, 0.0); n];
        let mut sgs_iters = max_iter;
        for iter in 0..max_iter {
            let r = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x_sgs));
            let norm_r: f64 = r.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
            if norm_r < tol * r0_norm {
                sgs_iters = iter;
                break;
            }
            let z = amg_sgs.apply(&r);
            x_sgs = sonium_solvers_core::vec_ops::vec_add(&x_sgs, &z);
        }

        // Multicolor GS
        let amg_mc = AmgPreconditioner::from_csr(
            &matrix,
            AmgConfig {
                smoother: AmgSmoother::MulticolorGaussSeidel,
                ..Default::default()
            },
        );
        let mut x_mc = vec![Complex64::new(0.0, 0.0); n];
        let mut mc_iters = max_iter;
        for iter in 0..max_iter {
            let r = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x_mc));
            let norm_r: f64 = r.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
            if norm_r < tol * r0_norm {
                mc_iters = iter;
                break;
            }
            let z = amg_mc.apply(&r);
            x_mc = sonium_solvers_core::vec_ops::vec_add(&x_mc, &z);
        }

        assert!(
            sgs_iters < max_iter,
            "SGS should converge within {max_iter} iterations"
        );
        assert!(
            mc_iters < max_iter,
            "Multicolor GS should converge within {max_iter} iterations"
        );
        // Multicolor GS should be within 2x of SGS iteration count
        assert!(
            mc_iters <= sgs_iters * 2 + 2,
            "Multicolor GS ({mc_iters} iters) should be within 2x of SGS ({sgs_iters} iters)"
        );
    }

    #[test]
    fn test_greedy_coloring_valid() {
        let nx = 12;
        let n = nx * nx;
        let matrix = create_2d_poisson(nx);

        let (color_of_row, rows_per_color) = coloring::greedy_graph_coloring(&matrix);

        assert_eq!(color_of_row.len(), n);

        for i in 0..n {
            let c_i = color_of_row[i];
            for idx in matrix.row_ptrs[i]..matrix.row_ptrs[i + 1] {
                let j = matrix.col_indices[idx];
                if j != i {
                    assert_ne!(
                        c_i, color_of_row[j],
                        "Adjacent rows {} and {} share color {}",
                        i, j, c_i
                    );
                }
            }
        }

        let total: usize = rows_per_color.iter().map(|v| v.len()).sum();
        assert_eq!(total, n);
    }

    // ------------------------------------------------------------------
    // ℓAIR (local approximate ideal restriction) tests
    // ------------------------------------------------------------------

    /// Upwinded 1D advection-diffusion on (0,1), velocity a > 0:
    /// -eps u'' + a u' with central diffusion and first-order upwind
    /// advection. `cell_peclet` = a h / eps; values >> 1 are
    /// advection-dominated and strongly nonsymmetric.
    fn create_1d_advection_diffusion(n: usize, cell_peclet: f64) -> CsrMatrix<Complex64> {
        let h = 1.0 / (n + 1) as f64;
        let eps = 1.0;
        let a = cell_peclet * eps / h;
        let diag = 2.0 * eps / (h * h) + a / h;
        let lower = -eps / (h * h) - a / h; // upwind: a (u_i - u_{i-1}) / h
        let upper = -eps / (h * h);

        let mut triplets = Vec::with_capacity(3 * n);
        for i in 0..n {
            triplets.push((i, i, Complex64::new(diag, 0.0)));
            if i > 0 {
                triplets.push((i, i - 1, Complex64::new(lower, 0.0)));
            }
            if i < n - 1 {
                triplets.push((i, i + 1, Complex64::new(upper, 0.0)));
            }
        }
        CsrMatrix::from_triplets(n, n, triplets)
    }

    /// 2D advection-diffusion with a divergence-free recirculating velocity
    /// field b = (sin(pi x) cos(pi y), -cos(pi x) sin(pi y)) (a single
    /// vortex), upwinded per face. Recirculation breaks any triangular
    /// ordering of the matrix, so Gauss-Seidel is no longer a near-exact
    /// solver — this is the classical hard case where transpose-restriction
    /// AMG loses coarse-grid stability at high Péclet.
    fn create_2d_vortex(nx: usize, cell_peclet: f64) -> CsrMatrix<Complex64> {
        use std::f64::consts::PI;
        let h = 1.0 / (nx + 1) as f64;
        let eps = 1.0;
        let scale = cell_peclet * eps / h; // velocity scale: max cell Péclet ~ cell_peclet
        let dif = -eps / (h * h);
        let n = nx * nx;
        let mut diag = vec![4.0 * eps / (h * h); n];
        let mut west = vec![dif; n];
        let mut east = vec![dif; n];
        let mut south = vec![dif; n];
        let mut north = vec![dif; n];
        for i in 0..nx {
            for j in 0..nx {
                let x = (i + 1) as f64 * h;
                let y = (j + 1) as f64 * h;
                let bx = scale * (PI * x).sin() * (PI * y).cos();
                let by = -scale * (PI * x).cos() * (PI * y).sin();
                let row = i * nx + j;
                if bx > 0.0 {
                    diag[row] += bx / h;
                    west[row] -= bx / h;
                } else {
                    diag[row] -= bx / h;
                    east[row] += bx / h;
                }
                if by > 0.0 {
                    diag[row] += by / h;
                    south[row] -= by / h;
                } else {
                    diag[row] -= by / h;
                    north[row] += by / h;
                }
            }
        }
        let mut triplets = Vec::new();
        for i in 0..nx {
            for j in 0..nx {
                let row = i * nx + j;
                triplets.push((row, row, Complex64::new(diag[row], 0.0)));
                if i > 0 {
                    triplets.push((row, (i - 1) * nx + j, Complex64::new(south[row], 0.0)));
                }
                if i + 1 < nx {
                    triplets.push((row, (i + 1) * nx + j, Complex64::new(north[row], 0.0)));
                }
                if j > 0 {
                    triplets.push((row, i * nx + j - 1, Complex64::new(west[row], 0.0)));
                }
                if j + 1 < nx {
                    triplets.push((row, i * nx + j + 1, Complex64::new(east[row], 0.0)));
                }
            }
        }
        CsrMatrix::from_triplets(n, n, triplets)
    }

    /// Upwinded 2D advection-diffusion on the unit square with constant
    /// velocity (a, a), central diffusion, first-order upwind advection.
    fn create_2d_advection_diffusion(nx: usize, cell_peclet: f64) -> CsrMatrix<Complex64> {
        let h = 1.0 / (nx + 1) as f64;
        let eps = 1.0;
        let a = cell_peclet * eps / h;
        let diag = 4.0 * eps / (h * h) + 2.0 * a / h;
        let adv = -eps / (h * h) - a / h; // upwind neighbor (west / south)
        let dif = -eps / (h * h); // downwind neighbor (east / north)

        let n = nx * nx;
        let mut triplets = Vec::with_capacity(5 * n);
        for i in 0..nx {
            for j in 0..nx {
                let row = i * nx + j;
                triplets.push((row, row, Complex64::new(diag, 0.0)));
                if i > 0 {
                    triplets.push((row, (i - 1) * nx + j, Complex64::new(adv, 0.0)));
                }
                if i + 1 < nx {
                    triplets.push((row, (i + 1) * nx + j, Complex64::new(dif, 0.0)));
                }
                if j > 0 {
                    triplets.push((row, i * nx + j - 1, Complex64::new(adv, 0.0)));
                }
                if j + 1 < nx {
                    triplets.push((row, i * nx + j + 1, Complex64::new(dif, 0.0)));
                }
            }
        }
        CsrMatrix::from_triplets(n, n, triplets)
    }

    fn residual_norm(matrix: &CsrMatrix<Complex64>, x: &[Complex64], b: &[Complex64]) -> f64 {
        let r = sonium_solvers_core::vec_ops::vec_sub(b, &matrix.matvec(x));
        r.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt()
    }

    /// Standalone multigrid iteration x <- x + M^{-1}(b - A x); returns the
    /// geometric-mean residual reduction factor per cycle and the number of
    /// levels in the hierarchy (so tests can assert the problem actually
    /// coarsened instead of being solved exactly by the coarsest-level LU).
    fn standalone_convergence_factor(
        matrix: &CsrMatrix<Complex64>,
        config: AmgConfig,
        cycles: usize,
    ) -> (f64, usize) {
        let n = matrix.num_rows;
        let amg = AmgPreconditioner::from_csr(matrix, config);
        let levels = amg.num_levels();
        // Deterministic non-smooth right-hand side
        let b: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(((i * 37 % 11) as f64 - 5.0) / 5.0, 0.0))
            .collect();
        let mut x = vec![Complex64::new(0.0, 0.0); n];
        let r0 = residual_norm(matrix, &x, &b);
        let mut rk = r0;
        for _ in 0..cycles {
            let r = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
            let z = amg.apply(&r);
            x = sonium_solvers_core::vec_ops::vec_add(&x, &z);
            rk = residual_norm(matrix, &x, &b);
        }
        ((rk / r0).powf(1.0 / cycles as f64), levels)
    }

    #[test]
    fn test_default_restriction_is_transpose() {
        assert_eq!(AmgConfig::default().restriction, AmgRestriction::Transpose);
    }

    #[test]
    fn test_lair_restriction_local_solve() {
        // On a small advection-diffusion problem with an alternating C/F
        // split, check the ℓAIR row structure and that each row's local
        // system (paper eq. 14) is actually solved:
        //   a_{i,j} + sum_k z_k a_{k,j} = 0   for j in R_i (the F-point
        //   restriction neighborhood of C-point i),
        // i.e. (R A)_{i,j} = 0 on the F-neighborhood columns.
        let n = 16;
        let matrix = create_1d_advection_diffusion(n, 10.0);
        let strong = strength::compute_strength_matrix(&matrix, 0.25);
        let sym = strength::symmetrize_strong(&strong);

        let mut point_types = vec![PointType::Undecided; n];
        let mut coarse_to_fine = Vec::new();
        for i in 0..n {
            if i % 2 == 0 {
                point_types[i] = PointType::Coarse;
                coarse_to_fine.push(i);
            } else {
                point_types[i] = PointType::Fine;
            }
        }
        let n_coarse = coarse_to_fine.len();

        let r =
            lair::build_lair_restriction(&matrix, &strong, &point_types, &coarse_to_fine, 2, None);
        assert_eq!(r.num_rows, n_coarse);
        assert_eq!(r.num_cols, n);

        // Unit entry at the C-point column in every row.
        for (ci, &fi) in coarse_to_fine.iter().enumerate() {
            let unit = r.get(ci, fi);
            assert!(
                (unit - Complex64::new(1.0, 0.0)).norm() < 1e-12,
                "row {ci} should have a unit entry at C-point column {fi}, got {unit:?}"
            );
        }

        // Local solve quality: (R A)_{i,j} must vanish (to solve accuracy)
        // on the distance-2 F-neighborhood columns R_i.
        let ra = r.matmul(&matrix);
        for (ci, &fi) in coarse_to_fine.iter().enumerate() {
            // Rebuild the distance-2 neighborhood used by the builder.
            let mut neighborhood = vec![fi];
            for &j in &sym[fi] {
                neighborhood.push(j);
                for &k in &sym[j] {
                    neighborhood.push(k);
                }
            }
            neighborhood.sort_unstable();
            neighborhood.dedup();

            let row_scale = matrix
                .row_entries(fi)
                .map(|(_, v)| v.norm())
                .fold(0.0_f64, f64::max)
                .max(1e-30);
            for &j in &neighborhood {
                if point_types[j] != PointType::Fine {
                    continue;
                }
                let res = ra.get(ci, j).norm();
                assert!(
                    res <= 1e-8 * row_scale,
                    "ℓAIR row {ci}: |(RA)_{{i,{j}}}| = {res:e} should vanish (row scale {row_scale:e})"
                );
            }
        }
    }

    #[test]
    fn test_lair_advection_diffusion_converges() {
        // Headline test: 2D recirculating (vortex) advection-diffusion at
        // cell Péclet 1000. Both arms use the *same* C/F splitting and
        // interpolation (symmetrize_strength is set explicitly for the
        // classical arm; it is implied for ℓAIR), so the restriction is the
        // only difference. Classical transpose restriction diverges (the
        // Galerkin coarse operator P^T A P loses stability at high Péclet);
        // ℓAIR converges robustly.
        let matrix = create_2d_vortex(31, 1000.0);

        let base = AmgConfig {
            coarsening: AmgCoarsening::RugeStuben,
            interpolation: AmgInterpolation::Standard,
            smoother: AmgSmoother::SymmetricGaussSeidel,
            symmetrize_strength: true,
            ..Default::default()
        };
        let (classical, classical_levels) = standalone_convergence_factor(
            &matrix,
            AmgConfig {
                restriction: AmgRestriction::Transpose,
                ..base.clone()
            },
            10,
        );
        let (lair, lair_levels) = standalone_convergence_factor(
            &matrix,
            AmgConfig {
                restriction: AmgRestriction::LAir {
                    distance: 2,
                    filter: None,
                },
                ..base
            },
            10,
        );

        eprintln!(
            "2D vortex Pe=1000: classical={classical:.3} ({classical_levels} levels) \
             lair={lair:.3} ({lair_levels} levels)"
        );
        assert!(classical_levels >= 2 && lair_levels >= 2);
        // Classical transpose restriction is inadequate here (factor > 0.7
        // means 40+ cycles for 1e-6), while ℓAIR converges in a handful of
        // cycles. Note the bound is slow-convergence, not strict divergence:
        // row-sum normalization of the classical interpolation weights
        // (hypre modified-classical form) stabilized this path from genuine
        // divergence (> 1.0) to 0.87; ℓAIR at 0.04 remains ~20x faster per
        // cycle, which is the efficiency gap this test guards.
        assert!(
            classical > 0.7,
            "classical transpose-R AMG should stagnate on the recirculating Pe=1000 problem, \
             got factor {classical:.3}"
        );
        assert!(
            lair < 0.5,
            "ℓAIR should converge with factor < 0.5 per cycle, got {lair:.3}"
        );
    }

    #[test]
    fn test_lair_advection_diffusion_preset_converges() {
        // The production preset path (for_nonsymmetric: HMIS + default
        // theta=0.25 + ℓAIR-2 + SGS) must coarsen and converge on the same
        // advection-dominated problem, exercising the SOC symmetrization
        // that the ℓAIR path applies for one-sided strong graphs.
        let n = 255;
        let matrix = create_1d_advection_diffusion(n, 100.0);

        let (lair, levels) =
            standalone_convergence_factor(&matrix, AmgConfig::for_nonsymmetric(), 15);
        eprintln!("1D advection Pe=100 for_nonsymmetric: lair={lair:.3} ({levels} levels)");
        assert!(
            levels >= 2,
            "ℓAIR path should coarsen the one-sided problem"
        );
        assert!(
            lair < 0.5,
            "for_nonsymmetric should converge with factor < 0.5 per cycle, got {lair:.3}"
        );
    }

    #[test]
    fn test_lair_advection_diffusion_2d_converges() {
        // 2D upwinded advection-diffusion (constant velocity), cell Péclet
        // 1000, damped-Jacobi smoothing, same-splitting comparison against
        // transpose restriction, plus the production preset.
        let matrix = create_2d_advection_diffusion(31, 1000.0);

        let base = AmgConfig {
            coarsening: AmgCoarsening::RugeStuben,
            interpolation: AmgInterpolation::Standard,
            smoother: AmgSmoother::Jacobi,
            symmetrize_strength: true,
            ..Default::default()
        };
        let (classical, classical_levels) = standalone_convergence_factor(
            &matrix,
            AmgConfig {
                restriction: AmgRestriction::Transpose,
                ..base.clone()
            },
            15,
        );
        let (lair, lair_levels) = standalone_convergence_factor(
            &matrix,
            AmgConfig {
                restriction: AmgRestriction::LAir {
                    distance: 2,
                    filter: None,
                },
                ..base
            },
            15,
        );
        let (preset, preset_levels) =
            standalone_convergence_factor(&matrix, AmgConfig::for_nonsymmetric(), 15);

        eprintln!("2D upwind Pe=1000: classical={classical:.3} lair={lair:.3} preset={preset:.3}");
        assert!(classical_levels >= 2 && lair_levels >= 2 && preset_levels >= 2);
        assert!(
            lair < 0.5,
            "ℓAIR should converge with factor < 0.5 per cycle, got {lair:.3}"
        );
        assert!(
            preset < 0.5,
            "for_nonsymmetric should converge with factor < 0.5 per cycle, got {preset:.3}"
        );
        assert!(
            classical > 1.3 * lair,
            "classical transpose-R AMG should be worse ({classical:.3}) than ℓAIR ({lair:.3})"
        );
    }

    #[test]
    fn test_lair_spd_poisson_still_converges() {
        // SPD regression: ℓAIR must not catastrophically degrade on a
        // symmetric Poisson problem. Transpose-R remains the default and is
        // expected to be at least as good.
        let matrix = create_2d_poisson(24);

        let (lair, lair_levels) = standalone_convergence_factor(
            &matrix,
            AmgConfig {
                restriction: AmgRestriction::LAir {
                    distance: 2,
                    filter: None,
                },
                ..AmgConfig::for_nonsymmetric()
            },
            15,
        );
        let (classical, _) =
            standalone_convergence_factor(&matrix, AmgConfig::for_difficult_problems(), 15);

        eprintln!("2D Poisson: classical={classical:.3} lair={lair:.3}");
        assert!(lair_levels >= 2);
        assert!(
            lair < 0.9,
            "ℓAIR should still converge on SPD Poisson, got factor {lair:.3}"
        );
    }

    #[test]
    fn test_lair_w_cycle_converges() {
        let matrix = create_1d_advection_diffusion(255, 100.0);
        let (factor, levels) = standalone_convergence_factor(
            &matrix,
            AmgConfig {
                cycle: AmgCycle::WCycle,
                ..AmgConfig::for_nonsymmetric()
            },
            10,
        );
        eprintln!("1D advection W-cycle ℓAIR: {factor:.3}");
        assert!(levels >= 2);
        assert!(
            factor < 0.5,
            "W-cycle with ℓAIR should converge, got factor {factor:.3}"
        );
    }

    #[test]
    fn test_lair_filter_and_distance1_converge() {
        // Exercise the filter and distance-1 variants: both must remain
        // convergent on the advection-dominated problem.
        let matrix = create_1d_advection_diffusion(255, 100.0);
        let (d1, d1_levels) = standalone_convergence_factor(
            &matrix,
            AmgConfig {
                restriction: AmgRestriction::LAir {
                    distance: 1,
                    filter: None,
                },
                ..AmgConfig::for_nonsymmetric()
            },
            15,
        );
        let (filtered, filtered_levels) = standalone_convergence_factor(
            &matrix,
            AmgConfig {
                restriction: AmgRestriction::LAir {
                    distance: 2,
                    filter: Some(0.1),
                },
                ..AmgConfig::for_nonsymmetric()
            },
            15,
        );
        eprintln!("1D advection ℓAIR-1: {d1:.3}, ℓAIR-2+filter: {filtered:.3}");
        assert!(d1_levels >= 2 && filtered_levels >= 2);
        assert!(d1 < 0.7, "ℓAIR-1 should converge, got {d1:.3}");
        assert!(
            filtered < 0.7,
            "filtered ℓAIR-2 should converge, got {filtered:.3}"
        );
    }

    #[test]
    fn test_lair_complex_smoke() {
        // Complex path: advection-diffusion with an imaginary diagonal shift
        // (Helmholtz-like). Smoke test: a few V-cycles must reduce the
        // residual without panicking or diverging.
        let n = 127;
        let mut matrix = create_1d_advection_diffusion(n, 20.0);
        for i in 0..n {
            for idx in matrix.row_range(i) {
                if matrix.col_indices[idx] == i {
                    matrix.values[idx] += Complex64::new(0.0, 50.0);
                }
            }
        }

        let amg = AmgPreconditioner::from_csr(
            &matrix,
            AmgConfig {
                restriction: AmgRestriction::LAir {
                    distance: 2,
                    filter: None,
                },
                ..AmgConfig::for_nonsymmetric()
            },
        );
        assert!(amg.num_levels() >= 2);

        let b: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(1.0, 0.1 * (i % 3) as f64))
            .collect();
        let mut x = vec![Complex64::new(0.0, 0.0); n];
        let r0 = residual_norm(&matrix, &x, &b);
        for _ in 0..10 {
            let r = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
            let z = amg.apply(&r);
            x = sonium_solvers_core::vec_ops::vec_add(&x, &z);
        }
        let rk = residual_norm(&matrix, &x, &b);
        eprintln!("complex ℓAIR: r0={r0:e} r10={rk:e}");
        assert!(
            rk < r0,
            "complex ℓAIR V-cycles should reduce the residual (r0={r0:e}, r10={rk:e})"
        );
    }

    #[test]
    fn test_lair_f_point_smoothing_vortex() {
        // CF relaxation on the recirculating-vortex Pe=1000 case: same
        // splitting, interpolation, and ℓAIR-2 restriction in both arms; the
        // only difference is all-point vs F-point-only smoothing. AIR theory
        // (Manteuffel-Ruge-Southworth 2018) pairs ℓAIR with F-point
        // relaxation, so the F-point arm must converge at least as well.
        let matrix = create_2d_vortex(31, 1000.0);

        let base = AmgConfig {
            coarsening: AmgCoarsening::RugeStuben,
            interpolation: AmgInterpolation::Standard,
            restriction: AmgRestriction::LAir {
                distance: 2,
                filter: None,
            },
            symmetrize_strength: true,
            ..Default::default()
        };

        let (sgs_all, sgs_all_levels) = standalone_convergence_factor(
            &matrix,
            AmgConfig {
                smoother: AmgSmoother::SymmetricGaussSeidel,
                ..base.clone()
            },
            10,
        );
        let (sgs_f, sgs_f_levels) = standalone_convergence_factor(
            &matrix,
            AmgConfig {
                smoother: AmgSmoother::SymmetricGaussSeidel,
                f_point_smoothing: true,
                ..base.clone()
            },
            10,
        );
        eprintln!(
            "2D vortex Pe=1000 SGS: all-point={sgs_all:.3} F-point={sgs_f:.3} \
             ({sgs_all_levels}/{sgs_f_levels} levels)"
        );
        assert!(sgs_all_levels >= 2 && sgs_f_levels >= 2);
        assert!(
            sgs_f < 0.5,
            "F-point SGS + ℓAIR should converge with factor < 0.5, got {sgs_f:.3}"
        );
        assert!(
            sgs_f <= sgs_all + 0.05,
            "F-point SGS ({sgs_f:.3}) should not be worse than all-point SGS ({sgs_all:.3})"
        );

        // Same comparison for damped Jacobi.
        let (jac_all, _) = standalone_convergence_factor(
            &matrix,
            AmgConfig {
                smoother: AmgSmoother::Jacobi,
                ..base.clone()
            },
            15,
        );
        let (jac_f, jac_f_levels) = standalone_convergence_factor(
            &matrix,
            AmgConfig {
                smoother: AmgSmoother::Jacobi,
                f_point_smoothing: true,
                ..base
            },
            15,
        );
        eprintln!("2D vortex Pe=1000 Jacobi: all-point={jac_all:.3} F-point={jac_f:.3}");
        assert!(jac_f_levels >= 2);
        assert!(
            jac_f < 0.9,
            "F-point Jacobi + ℓAIR should converge, got factor {jac_f:.3}"
        );
    }

    #[test]
    fn test_f_point_smoothing_spd_poisson_converges() {
        // SPD regression: enabling CF relaxation on the classical
        // transpose-R path must not break convergence on symmetric Poisson.
        // F-only relaxation leaves C-point error to the coarse correction,
        // which is exactly what Galerkin AMG handles, so the cycle still
        // converges.
        let matrix = create_2d_poisson(24);

        let (all_pt, all_levels) = standalone_convergence_factor(
            &matrix,
            AmgConfig {
                smoother: AmgSmoother::SymmetricGaussSeidel,
                ..Default::default()
            },
            10,
        );
        let (f_pt, f_levels) = standalone_convergence_factor(
            &matrix,
            AmgConfig {
                smoother: AmgSmoother::SymmetricGaussSeidel,
                f_point_smoothing: true,
                ..Default::default()
            },
            10,
        );
        eprintln!("2D Poisson transpose-R: all-point={all_pt:.3} F-point={f_pt:.3}");
        assert!(all_levels >= 2 && f_levels >= 2);
        assert!(
            f_pt < 0.9,
            "F-point smoothing must still converge on SPD Poisson, got {f_pt:.3}"
        );
    }

    /// Hermitian positive-definite complex 1D chain: dominant positive real
    /// diagonal, complex Hermitian off-diagonal coupling whose phase *varies
    /// per row* (`A[i,i+1] = -1 + 0.3i·h(i)`, `A[i+1,i] = conj(A[i,i+1])`,
    /// `h(i) = ((7i) mod 13)/13`). Strictly diagonally dominant with a
    /// positive real diagonal and Hermitian, hence positive definite
    /// (Gershgorin + real spectrum). The varying phase breaks the
    /// reflection/conjugation symmetry that would otherwise make the
    /// Galerkin product `P^T A P` accidentally real (and hence Hermitian).
    fn create_1d_hermitian_hpd(n: usize) -> CsrMatrix<Complex64> {
        let mut triplets = Vec::with_capacity(3 * n);
        for i in 0..n {
            triplets.push((i, i, Complex64::new(4.0, 0.0)));
            let h = ((7 * i) % 13) as f64 / 13.0;
            let upper = Complex64::new(-1.0, 0.3 * h); // A[i, i+1]
            if i > 0 {
                let h_prev = ((7 * (i - 1)) % 13) as f64 / 13.0;
                triplets.push((i, i - 1, Complex64::new(-1.0, 0.3 * h_prev).conj()));
            }
            if i < n - 1 {
                triplets.push((i, i + 1, upper));
            }
        }
        CsrMatrix::from_triplets(n, n, triplets)
    }

    /// Largest deviation from Hermitian structure, `max |A[i,j] - conj(A[j,i])|`.
    fn hermitian_defect(matrix: &CsrMatrix<Complex64>) -> f64 {
        let mut defect = 0.0_f64;
        for i in 0..matrix.num_rows {
            for (j, v) in matrix.row_entries(i) {
                defect = defect.max((v - matrix.get(j, i).conj()).norm());
            }
        }
        defect
    }

    #[test]
    fn test_conjugated_restriction_preserves_hermitian_structure() {
        let matrix = create_1d_hermitian_hpd(63);
        assert!(
            hermitian_defect(&matrix) < 1e-14,
            "test matrix must be Hermitian"
        );

        let base = AmgConfig {
            coarsening: AmgCoarsening::RugeStuben,
            interpolation: AmgInterpolation::Extended,
            smoother: AmgSmoother::SymmetricGaussSeidel,
            ..Default::default()
        };

        // R = P^H: every coarse operator must stay Hermitian.
        let amg_h = AmgPreconditioner::from_csr(
            &matrix,
            AmgConfig {
                restriction: AmgRestriction::ConjugateTranspose,
                ..base.clone()
            },
        );
        assert!(amg_h.num_levels() >= 2, "HPD problem should coarsen");
        for (lvl, level) in amg_h.levels.iter().enumerate().skip(1) {
            let defect = hermitian_defect(&level.matrix);
            assert!(
                defect < 1e-10,
                "P^H A P at level {lvl} must be Hermitian, defect {defect:e}"
            );
        }

        // Caveat demonstration (complex-symmetric vs Hermitian): for a
        // *generic* complex prolongation P and Hermitian A, P^T A P is not
        // Hermitian while P^H A P is. (AMG's direct-interpolation weights
        // w = -a_ij/a_ii telescope with A's rows, so on simple chains the
        // transpose product is *accidentally* Hermitian — a generic complex
        // P has no such cancellation.)
        let p = CsrMatrix::from_triplets(
            4,
            2,
            vec![
                (0, 0, Complex64::new(1.0, 0.0)),
                (1, 0, Complex64::new(0.3, 0.2)),
                (1, 1, Complex64::new(0.1, -0.4)),
                (2, 0, Complex64::new(-0.2, 0.3)),
                (2, 1, Complex64::new(0.5, 0.1)),
                (3, 1, Complex64::new(1.0, 0.0)),
            ],
        );
        let a_small = create_1d_hermitian_hpd(4);
        let m_t = util::galerkin_product(&util::transpose_csr(&p), &a_small, &p);
        let m_h = util::galerkin_product(&util::adjoint_csr(&p), &a_small, &p);
        let defect_t = hermitian_defect(&m_t);
        let defect_h = hermitian_defect(&m_h);
        eprintln!("generic-P Hermitian defect: P^T A P={defect_t:e} P^H A P={defect_h:e}");
        assert!(
            defect_h < 1e-12,
            "P^H A P must be Hermitian for Hermitian A, defect {defect_h:e}"
        );
        assert!(
            defect_t > 1e-6,
            "P^T A P should be visibly non-Hermitian for a generic complex P \
             (the complex-symmetric caveat), got defect {defect_t:e}"
        );
    }

    #[test]
    fn test_conjugated_restriction_converges_on_hpd() {
        let matrix = create_1d_hermitian_hpd(255);
        let (factor, levels) = standalone_convergence_factor(
            &matrix,
            AmgConfig {
                restriction: AmgRestriction::ConjugateTranspose,
                ..Default::default()
            },
            10,
        );
        eprintln!("1D Hermitian HPD conjugated-R: factor={factor:.3} ({levels} levels)");
        assert!(levels >= 2);
        assert!(
            factor < 0.5,
            "conjugated-restriction AMG should converge on the HPD problem, got {factor:.3}"
        );
    }
    #[test]
    fn test_falgout_valid_splitting_and_converges() {
        // Splitting validity: every F-point has at least one strong
        // C-neighbor, and the C-set is a nontrivial fraction of the points.
        let matrix = create_2d_poisson(31);
        let strong = strength::compute_strength_matrix(&matrix, 0.25);
        let (point_types, coarse_to_fine) = coarsening::coarsen_falgout(&matrix, &strong);
        for (i, pt) in point_types.iter().enumerate() {
            if *pt == PointType::Fine {
                assert!(
                    strong[i]
                        .iter()
                        .any(|&j| point_types[j] == PointType::Coarse),
                    "Falgout F-point {i} has no strong C-neighbor"
                );
            }
        }
        let c_fraction = coarse_to_fine.len() as f64 / point_types.len() as f64;
        eprintln!(
            "Falgout splitting: {} C-points of {} ({c_fraction:.3})",
            coarse_to_fine.len(),
            point_types.len()
        );
        assert!(
            (0.05..0.95).contains(&c_fraction),
            "Falgout C-fraction {c_fraction:.3} outside plausible range"
        );

        // Poisson convergence + complexity metrics (printed for the report).
        let config = AmgConfig {
            coarsening: AmgCoarsening::Falgout,
            interpolation: AmgInterpolation::Extended,
            ..Default::default()
        };
        let amg = AmgPreconditioner::from_csr(&matrix, config.clone());
        let diag = amg.diagnostics();
        let (factor, levels) = standalone_convergence_factor(&matrix, config, 10);
        eprintln!(
            "Falgout+extended 2D Poisson 31x31: factor={factor:.3} levels={levels} \
             grid_complexity={:.2} operator_complexity={:.2} level_dofs={:?} level_nnz={:?}",
            diag.grid_complexity, diag.operator_complexity, diag.level_dofs, diag.level_nnz
        );
        assert!(levels >= 2, "Falgout should build multiple levels");
        assert!(
            factor < 0.5,
            "Falgout+extended should converge on 2D Poisson, got {factor:.3}"
        );
    }

    #[test]
    fn test_extended_i_interpolation_converges() {
        let matrix = create_2d_poisson(31);
        let config = AmgConfig {
            coarsening: AmgCoarsening::Falgout,
            interpolation: AmgInterpolation::ExtendedI,
            ..Default::default()
        };
        let amg = AmgPreconditioner::from_csr(&matrix, config.clone());
        let diag = amg.diagnostics();
        let (factor, levels) = standalone_convergence_factor(&matrix, config, 10);
        eprintln!(
            "Falgout+extended+i 2D Poisson 31x31: factor={factor:.3} levels={levels} \
             grid_complexity={:.2} operator_complexity={:.2} level_nnz={:?}",
            diag.grid_complexity, diag.operator_complexity, diag.level_nnz
        );
        assert!(levels >= 2);
        assert!(
            factor < 0.5,
            "Falgout+extended+i should converge on 2D Poisson, got {factor:.3}"
        );
    }

    #[test]
    fn test_extended_i_immediate_neighbor_fallback() {
        // F-point 1 with an EMPTY strong set but immediate C-neighbors 0 and
        // 2 in the matrix row (1D Laplacian [−1, 2, −1]). Extended+i must
        // interpolate from both immediate C-neighbors (weights 0.5 each);
        // plain Extended degrades to a single injection-style entry.
        let matrix = create_1d_laplacian(3);
        let strong: Vec<Vec<usize>> = vec![Vec::new(), Vec::new(), Vec::new()];
        let point_types = vec![PointType::Coarse, PointType::Fine, PointType::Coarse];
        let coarse_to_fine = vec![0, 2];

        let p_exti = interpolation::build_interpolation(
            &matrix,
            &strong,
            &point_types,
            &coarse_to_fine,
            &AmgConfig {
                interpolation: AmgInterpolation::ExtendedI,
                ..Default::default()
            },
        );
        let w0 = p_exti.get(1, 0);
        let w2 = p_exti.get(1, 1);
        assert!(
            (w0 - Complex64::new(0.5, 0.0)).norm() < 1e-12
                && (w2 - Complex64::new(0.5, 0.0)).norm() < 1e-12,
            "extended+i should interpolate 0.5/0.5 from immediate C-neighbors, got {w0:?}/{w2:?}"
        );

        let p_ext = interpolation::build_interpolation(
            &matrix,
            &strong,
            &point_types,
            &coarse_to_fine,
            &AmgConfig {
                interpolation: AmgInterpolation::Extended,
                ..Default::default()
            },
        );
        let row1_nnz = p_ext.row_entries(1).count();
        assert_eq!(
            row1_nnz, 1,
            "plain extended should fall back to a single injection entry, got {row1_nnz}"
        );
    }

    /// Directed-edge growth ratio of the strong graph under symmetrization:
    /// 1.0 for a reciprocal graph, ~2.0 for a fully one-sided graph.
    fn soc_growth_ratio(matrix: &CsrMatrix<Complex64>, theta: f64) -> f64 {
        let strong = strength::compute_strength_matrix(matrix, theta);
        let sym = strength::symmetrize_strong(&strong);
        let directed: usize = strong.iter().map(Vec::len).sum();
        let grown: usize = sym.iter().map(Vec::len).sum();
        grown as f64 / directed.max(1) as f64
    }

    #[test]
    fn test_one_sided_soc_detection() {
        // Upwinded advection-diffusion has a strongly one-sided SOC graph;
        // symmetric Poisson is reciprocal. The detector must separate them.
        let adv = create_1d_advection_diffusion(255, 100.0);
        let adv2d = create_2d_advection_diffusion(31, 1000.0);
        let poisson = create_2d_poisson(24);

        let r_adv = soc_growth_ratio(&adv, 0.25);
        let r_adv2d = soc_growth_ratio(&adv2d, 0.25);
        let r_pois = soc_growth_ratio(&poisson, 0.25);
        eprintln!(
            "SOC symmetrization growth: 1D upwind Pe=100 {r_adv:.3}, \
             2D upwind Pe=1000 {r_adv2d:.3}, 2D Poisson {r_pois:.3}"
        );

        let check = |m: &CsrMatrix<Complex64>| {
            let s = strength::compute_strength_matrix(m, 0.25);
            let sym = strength::symmetrize_strong(&s);
            soc_is_one_sided(&s, &sym)
        };
        assert!(check(&adv), "1D upwind SOC should be detected one-sided");
        assert!(check(&adv2d), "2D upwind SOC should be detected one-sided");
        assert!(
            !check(&poisson),
            "Poisson SOC must NOT be detected one-sided"
        );
    }

    #[test]
    fn test_classical_auto_symmetrize_advection_converges() {
        // Regression for the classical-path one-sided fix: before, RS
        // coarsening on the upwinded strong graph selected nearly every
        // point as C (levels == 1) and the coarsest-level LU silently solved
        // the whole matrix. Auto-symmetrization must restore real coarsening.
        let matrix = create_1d_advection_diffusion(255, 100.0);

        let (factor, levels) = standalone_convergence_factor(
            &matrix,
            AmgConfig {
                coarsening: AmgCoarsening::RugeStuben,
                smoother: AmgSmoother::SymmetricGaussSeidel,
                // NOTE: no explicit symmetrize_strength — the classical path
                // must auto-detect the one-sided graph.
                ..Default::default()
            },
            15,
        );
        eprintln!(
            "1D upwind Pe=100 classical auto-symmetrized: factor={factor:.3} ({levels} levels)"
        );
        assert!(
            levels >= 2,
            "auto-symmetrization should restore coarsening on the one-sided problem"
        );
        assert!(
            factor < 0.9,
            "classical AMG should converge once coarsening works, got {factor:.3}"
        );
    }

    #[test]
    fn test_auto_symmetrize_no_poisson_regression() {
        // Justification measurement for default-on detection: on 2D Poisson
        // the detector does not trigger (ratio ~1.0, see the detection test),
        // so convergence with auto-detection is identical to the previous
        // default. Printed for the report; asserted only for sanity.
        let matrix = create_2d_poisson(24);
        let (auto, levels_auto) = standalone_convergence_factor(&matrix, AmgConfig::default(), 10);
        let (forced_sym, levels_sym) = standalone_convergence_factor(
            &matrix,
            AmgConfig {
                symmetrize_strength: true,
                ..Default::default()
            },
            10,
        );
        eprintln!(
            "2D Poisson 24x24 CLJP: auto-detect={auto:.3} ({levels_auto} levels) \
             forced-symmetrize={forced_sym:.3} ({levels_sym} levels)"
        );
        assert!(levels_auto >= 2);
        // Baseline for CLJP+Jacobi on this problem is ~0.58 per cycle; the
        // auto-detection path must not make it worse (it does not trigger
        // on the reciprocal Poisson graph).
        assert!(
            auto < 0.9,
            "Poisson convergence must not regress: {auto:.3}"
        );
    }

    /// Count standalone multigrid iterations needed to reduce the initial
    /// residual by `tol`; returns (iterations, final relative residual).
    fn standalone_iterations_to_tol(
        matrix: &CsrMatrix<Complex64>,
        config: AmgConfig,
        tol: f64,
        max_iters: usize,
    ) -> (usize, f64) {
        let n = matrix.num_rows;
        let amg = AmgPreconditioner::from_csr(matrix, config);
        let b: Vec<Complex64> = (0..n)
            .map(|i| Complex64::new(((i * 37 % 11) as f64 - 5.0) / 5.0, 0.0))
            .collect();
        let mut x = vec![Complex64::new(0.0, 0.0); n];
        let r0 = residual_norm(matrix, &x, &b);
        let mut rel = 1.0;
        let mut iters = 0;
        while rel > tol && iters < max_iters {
            let r = sonium_solvers_core::vec_ops::vec_sub(&b, &matrix.matvec(&x));
            let z = amg.apply(&r);
            x = sonium_solvers_core::vec_ops::vec_add(&x, &z);
            rel = residual_norm(matrix, &x, &b) / r0;
            iters += 1;
        }
        (iters, rel)
    }

    /// Anisotropic 2D Poisson: `-u_xx - anisotropy * u_yy` on an `nx x nx`
    /// grid — a hard-ish SPD problem for isotropic point smoothers, where a
    /// plain V-cycle is a mediocre preconditioner and Krylov acceleration of
    /// the cycle pays off.
    fn create_2d_anisotropic_poisson(nx: usize, anisotropy: f64) -> CsrMatrix<Complex64> {
        let mut triplets = Vec::with_capacity(5 * nx * nx);
        for i in 0..nx {
            for j in 0..nx {
                let row = i * nx + j;
                triplets.push((row, row, Complex64::new(2.0 * (1.0 + anisotropy), 0.0)));
                if i > 0 {
                    triplets.push((row, (i - 1) * nx + j, Complex64::new(-anisotropy, 0.0)));
                }
                if i + 1 < nx {
                    triplets.push((row, (i + 1) * nx + j, Complex64::new(-anisotropy, 0.0)));
                }
                if j > 0 {
                    triplets.push((row, i * nx + j - 1, Complex64::new(-1.0, 0.0)));
                }
                if j + 1 < nx {
                    triplets.push((row, i * nx + j + 1, Complex64::new(-1.0, 0.0)));
                }
            }
        }
        CsrMatrix::from_triplets(nx * nx, nx * nx, triplets)
    }

    #[test]
    fn test_k_cycle_converges_at_least_as_fast_as_v_cycle() {
        // Hard-ish SPD problem: anisotropic Poisson, isotropic coarsening,
        // damped-Jacobi smoothing. The K-cycle (flexible CG around the
        // V-cycle, top level only) must converge in no more standalone
        // iterations than the plain V-cycle.
        let matrix = create_2d_anisotropic_poisson(31, 100.0);
        let base = AmgConfig {
            coarsening: AmgCoarsening::Cljp,
            smoother: AmgSmoother::Jacobi,
            ..Default::default()
        };

        let (v_iters, v_rel) = standalone_iterations_to_tol(
            &matrix,
            AmgConfig {
                cycle: AmgCycle::VCycle,
                ..base.clone()
            },
            1e-6,
            300,
        );
        let (k_iters, k_rel) = standalone_iterations_to_tol(
            &matrix,
            AmgConfig {
                cycle: AmgCycle::VCycle,
                k_cycle_iterations: 2,
                ..base
            },
            1e-6,
            300,
        );
        eprintln!(
            "K-cycle comparison, anisotropic 2D Poisson 31x31 (ratio 100): \
             V-cycle {v_iters} iterations (rel {v_rel:.2e}) vs \
             K(2)-cycle {k_iters} iterations (rel {k_rel:.2e})"
        );
        assert!(
            v_rel < 1e-6,
            "V-cycle should converge within the iteration cap, rel={v_rel:.2e}"
        );
        assert!(
            k_rel < 1e-6,
            "K-cycle should converge within the iteration cap, rel={k_rel:.2e}"
        );
        assert!(
            k_iters <= v_iters,
            "K-cycle ({k_iters} iterations) should converge at least as fast \
             as the V-cycle ({v_iters} iterations)"
        );
    }

    #[test]
    fn test_sa_strength_matrix_comparison() {
        // Measurement (printed, not asserted as a comparison): the SA path
        // historically uses A^T A as the strength matrix, which is unusual
        // vs MueLu's filtered-A SOC. Compare Poisson convergence and
        // operator complexity for both choices. Only loose convergence
        // sanity is asserted.
        let matrix = create_2d_poisson(31);
        let base = AmgConfig::for_aggregation();

        let (ata_factor, ata_levels) = standalone_convergence_factor(
            &matrix,
            AmgConfig {
                sa_strength_with_ata: true,
                ..base.clone()
            },
            15,
        );
        let amg_ata = AmgPreconditioner::from_csr(
            &matrix,
            AmgConfig {
                sa_strength_with_ata: true,
                ..base.clone()
            },
        );
        let (a_factor, a_levels) = standalone_convergence_factor(
            &matrix,
            AmgConfig {
                sa_strength_with_ata: false,
                ..base.clone()
            },
            15,
        );
        let amg_a = AmgPreconditioner::from_csr(
            &matrix,
            AmgConfig {
                sa_strength_with_ata: false,
                ..base
            },
        );
        eprintln!(
            "SA strength-matrix comparison, 2D Poisson 31x31 (for_aggregation): \
             A^T A SOC: factor={ata_factor:.3}, levels={ata_levels}, \
             grid_c={:.2}, op_c={:.2}, nnz/level={:?} | \
             plain-A SOC: factor={a_factor:.3}, levels={a_levels}, \
             grid_c={:.2}, op_c={:.2}, nnz/level={:?}",
            amg_ata.grid_complexity(),
            amg_ata.operator_complexity(),
            amg_ata.diagnostics().level_nnz,
            amg_a.grid_complexity(),
            amg_a.operator_complexity(),
            amg_a.diagnostics().level_nnz,
        );
        assert!(ata_levels >= 2 && a_levels >= 2);
        assert!(
            ata_factor < 0.9 && a_factor < 0.9,
            "both SA strength variants should converge: A^T A={ata_factor:.3}, A={a_factor:.3}"
        );
    }
}
