//! Preconditioners for iterative solvers
//!
//! Preconditioners approximate A^(-1) to accelerate convergence of iterative methods.
//!
//! # Available Preconditioners
//!
//! - **DiagonalPreconditioner** (Jacobi): Simple diagonal scaling, fully parallel
//! - **IluPreconditioner**: Sequential ILU(0), best convergence
//! - **HybridIluPreconditioner**: Probe-gated low-rank augmentation of ILU(0)
//! - **IluColoringPreconditioner**: Parallel ILU via level scheduling
//! - **IluFixedPointPreconditioner**: Parallel ILU via Jacobi iteration
//! - **ParIlutPreconditioner**: ParILUT — parallel fixed-point ILU with threshold (Chow/Anzt)
//! - **HiluPreconditioner**: Hierarchical ILU with parallel setup via nested dissection

//! - **BlockJacobiPreconditioner**: Block Jacobi with ILUT subdomain solves (parallel, no overlap)
//! - **AdditiveSchwarzPreconditioner**: Domain decomposition with overlap (parallel)
//! - **AmgPreconditioner**: Algebraic multigrid with parallel coarsening and smoothing
//! - **SaddlePointPreconditioner**: Block preconditioner for saddle point systems
//!
//! # Which preconditioner for which physics
//!
//! Mirrors the workspace solver-selection philosophy (see the workspace
//! `AGENTS.md` performance section):
//!
//! | Physics / operator class | Recommended preconditioner | Solver pairing |
//! |---|---|---|
//! | SPD (elasticity, Laplacian, diffusion) | `AmgPreconditioner` (or sparse Cholesky direct) | CG + AMG |
//! | Indefinite Helmholtz (FEM) | Shifted-Laplacian + `IluPreconditioner`/`ParIlutPreconditioner`, or AMG with the Helmholtz-tuned config | FGMRES + shifted-Laplacian ILU, or AMG-preconditioned GMRES |
//! | BEM dense, N < 1000 | none (direct LU) | direct LU |
//! | BEM medium, N < 5000 | `IluPreconditioner` on the dense block | GMRES + ILU |
//! | BEM large, N > 5000 (FMM) | `AdditiveSchwarzPreconditioner` / `BlockJacobiPreconditioner` | FMM + GMRES + Schwarz |
//! | Saddle point (Stokes) | `SaddlePointPreconditioner` (PCD/LSC Schur strategy) | block-preconditioned Krylov |
//! | Nonsymmetric advection-dominated | `AdditiveSchwarzPreconditioner`, AMG with ℓAIR-style coarsening | GMRES + Schwarz |
//!
//! An unpreconditioned GMRES on an indefinite Helmholtz system will stall;
//! the preconditioner is the algorithm choice that matters, not an accessory.

mod amg;
mod block_jacobi;
mod chebyshev;
#[cfg(feature = "cuda")]
mod cuda_ilu;
mod diagonal;
mod hybrid_ilu;
mod ilu;
#[cfg(feature = "cuda")]
#[doc(inline)]
pub use cuda_ilu::CudaIluPreconditioner;
mod ilu_hierarchical;
mod ilu_parallel;
pub(crate) mod ilut;
#[cfg(feature = "cuda")]
#[doc(inline)]
pub use ilut::CudaIlutPreconditioner;
mod saddle_point;
mod schwarz;
#[cfg(feature = "cuda")]
#[doc(inline)]
pub use schwarz::CudaSchwarzPreconditioner;

#[cfg(feature = "experimental-gnn")]
pub use amg::GnnCoarseningWeights;
pub use amg::{
    AMG_CHECKPOINT_FORMAT_VERSION, AmgCoarsening, AmgConfig, AmgCycle, AmgDiagnostics,
    AmgInterpolation, AmgPhysicsProfile, AmgPreconditioner, AmgRestriction, AmgSmoother,
    BackendAmgDiagnostics, BackendAmgPhase, BackendAmgPhaseProfile, BackendAmgPreconditioner,
    MaterialSocConfig,
};
pub use block_jacobi::{
    BLOCK_JACOBI_CHECKPOINT_FORMAT_VERSION, BlockJacobiDiagnostics, BlockJacobiPreconditioner,
    BlockJacobiRuntimeDiagnostics, BlockPartitioning,
};
pub use chebyshev::{
    CHEBYSHEV_CHECKPOINT_FORMAT_VERSION, ChebyshevConfig, ChebyshevError, ChebyshevPreconditioner,
};
pub use diagonal::DiagonalPreconditioner;
pub use hybrid_ilu::{
    HybridIluConfig, HybridIluDiagnostics, HybridIluPreconditioner, MAX_HYBRID_ILU_RANK,
};
pub use ilu::{
    ILU_CHECKPOINT_FORMAT_VERSION, IluConfig, IluDiagnostics, IluPreconditioner, IluReordering,
};
pub use ilu_hierarchical::{HILU_CHECKPOINT_FORMAT_VERSION, HiluConfig, HiluPreconditioner};
pub use ilu_parallel::{
    ILU_COLORING_CHECKPOINT_FORMAT_VERSION, ILU_FIXED_POINT_CHECKPOINT_FORMAT_VERSION,
    IluColoringPreconditioner, IluFixedPointPreconditioner, PARILUT_CHECKPOINT_FORMAT_VERSION,
    ParIlutConfig, ParIlutPreconditioner,
};
pub use ilut::{ILUT_CHECKPOINT_FORMAT_VERSION, IlutConfig, IlutDiagnostics, IlutPreconditioner};
pub use saddle_point::{
    BlockPreconditionerConfig, LscBlocks, PcdBlocks, SADDLE_POINT_CHECKPOINT_FORMAT_VERSION,
    SaddlePointBlockAssumption, SaddlePointBlockAssumptions, SaddlePointBlockProvenance,
    SaddlePointBlockValidation, SaddlePointConfig, SaddlePointPreconditioner, SchurStrategy,
    btdiagb_schur_from_monolithic, has_saddle_point_structure,
};
pub use schwarz::{
    AdditiveSchwarzPreconditioner, GeneoConfig, GeneoSetupError, SCHWARZ_CHECKPOINT_FORMAT_VERSION,
    SchwarzCoarseSetup, SchwarzRuntimeDiagnostics, SweepingSchwarzPreconditioner,
    coordinate_layers,
};
pub use sonium_compute::preconditioner::{
    BlockJacobiBackend, ChebyshevBackend, ChebyshevBackendF32, DEFAULT_BLOCK_SIZE,
    PointJacobiBackend, PolynomialJacobiBackend,
};

// Re-export IdentityPreconditioner from traits
pub use sonium_solvers_core::traits::IdentityPreconditioner;
