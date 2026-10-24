//! High-performance linear solvers for BEM and FEM
//!
//! This crate provides a collection of iterative and direct solvers for linear systems,
//! along with sparse matrix representations and preconditioners.
//!
//! # Features
//!
//! - **Iterative Solvers**: GMRES, BiCGSTAB, CGS, CG
//! - **Direct Solvers**: LU decomposition (with BLAS and pure-Rust fallbacks)
//! - **Preconditioners**: Jacobi, ILU, block diagonal
//! - **Sparse Matrices**: CSR format with efficient matrix-vector products
//! - **Generic Scalar Types**: Works with Complex64, Complex32, f64, f32
//!
//! # Example
//!
//! ```ignore
//! use sonium_solvers::{CsrMatrix, gmres, GmresConfig};
//! use num_complex::Complex64;
//!
//! // Create a sparse system matrix
//! let matrix = CsrMatrix::from_dense(&dense_matrix, 1e-10);
//!
//! // Solve with GMRES
//! let config = GmresConfig::default();
//! let solution = gmres(&matrix, &rhs, &config)?;
//! ```

// Canonical `array!` lives in `sonium-solvers-core`; re-export it here so the
// facade keeps a single definition (previously duplicated verbatim).
pub use sonium_solvers_core::array;

#[cfg(test)]
pub mod test_array {
    pub use crate::array;
    pub use std::vec;
}

pub use sonium_solvers_core::dense;
pub mod checkpoint_manifest;
pub mod conformance;
pub use checkpoint_manifest::{
    BackendCheckpointBuffer, BackendCheckpointRestoreError, CheckpointManifestError,
    CheckpointResidence, CheckpointShard, DeviceCheckpointState, DistributedCheckpoint,
    SolverCheckpointBundle,
};
pub mod differentiable;
pub mod differentiable_contract;
pub mod matrix_market;
pub mod validation_corpus;
pub use sonium_solvers_direct as direct;
pub use sonium_solvers_eigen as eigen;
pub use sonium_solvers_iterative as iterative;
pub use sonium_solvers_sys::lanes;
pub mod mixed_precision;
pub use sonium_solvers_nonlinear as nonlinear;
pub use sonium_solvers_optimization as optimization;
pub use sonium_solvers_sys::parallel;
#[cfg(feature = "cli")]
pub mod perf;
pub use sonium_solvers_preconditioners as preconditioners;
pub use sonium_solvers_sys::runtime;
pub mod solve;
pub use sonium_solvers_core::traits;
pub use sonium_solvers_core::vec_ops;
pub use sonium_solvers_sparse as sparse;
pub use sonium_solvers_sys::topology;
pub use sonium_solvers_time_integration as time_integration;

pub use conformance::{
    OPTIMIZATION_CONFORMANCE_ALGORITHMS, ReferenceKind, SOLVER_CONFORMANCE, SolverConformance,
    SolverMaturity,
};
pub use matrix_market::{
    MatrixMarketError, ParsedComplexMatrixMarket, ParsedMatrixMarket, parse_complex_matrix_market,
    parse_matrix_market,
};
pub use validation_corpus::{
    CorpusCase, CorpusClass, SOLVER_VALIDATION_CORPUS, convection_diffusion_1d_fixture,
    poisson_1d_fixture,
};

pub use sonium_compute::sme2;

pub use differentiable::{
    AdjointSolve, DifferentiableBlockSolveResult, DifferentiableLinearSolve,
    DifferentiableSolveResult,
};
pub use differentiable_contract::{
    BlockDifferentiableSolve, CancellationToken, CheckpointReplayReport,
    ClosureDifferentiableSolve, ContractDimensions, ContractValidation,
    DifferentiableContractValue, DifferentiableSolveContract, DifferentiableSystemKind,
    EigenValueDifferentiableSolve, EigenVectorDifferentiableSolve, ReplayContractReport,
    SteadyAdjointSeed, TaylorContractReport, TransientDifferentiableSolve, block_contract,
    check_taylor_remainder, eigen_value_contract, eigen_vector_contract, transient_contract,
    verify_forward_checkpoint_replay, verify_forward_replay,
};

// Re-export optimisation types
pub use optimization::{
    // Constrained
    AdvancedSqpConfig,
    AugmentedLagrangianConfig,
    // Unconstrained
    BfgsConfig,
    // Core types
    Bound,
    Bounds,
    CgVariant,
    Constraint,
    ConstraintKind,
    CvarResult,
    EpsilonConstraintConfig,
    HessianRegularisation,
    InnerSolver,
    KktDiagnostics,
    KktMultipliers,
    KktReport,
    KsResult,
    LbfgsConfig,
    LbfgsbConfig,
    LineSearchMethod,
    NelderMeadConfig,
    NewtonCgConfig,
    NonlinearCgConfig,
    OptimError,
    OptimResult,
    OptimStatus,
    ParetoResult,
    PenaltyConfig,
    PenaltyType,
    QuasiNewtonConfig,
    QuasiNewtonUpdate,
    RealScalar,
    RiskAggregation,
    RiskAggregationError,
    RiskObjective,
    RiskObjectiveEvaluation,
    ScenarioEvaluation,
    SlsqpConfig,
    SqpConfig,
    TrustRegionConstrainedConfig,
    augmented_lagrangian,
    bfgs,
    bfgs_risk,
    chance_constraint_satisfied,
    empirical_cvar,
    empirical_cvar_with_gradients,
    epsilon_constraint,
    ks_aggregate,
    ks_aggregate_with_gradients,
    lbfgs,
    lbfgsb,
    nelder_mead,
    newton_cg,
    nonlinear_cg,
    penalty_method,
    quasi_newton,
    slsqp,
    sqp,
    sqp_advanced,
    trust_region_constrained,
};
pub use optimization::{
    CmaEsConfig, DifferentialEvolutionConfig, IpmConfig, MmaConfig, MmaVariant, RandomSearchConfig,
    SimulatedAnnealingConfig, cma_es, differential_evolution, ipm, mma, random_search,
    simulated_annealing,
};
// Deprecated direction-set Powell (kept for API compatibility).
#[allow(deprecated)]
pub use optimization::{PowellConfig, powell};

// Re-export main types
pub use sparse::{
    BsrMatrix, CompositeOperator, CsrBuilder, CsrMatrix, FnOperator, ScaledOperator, SumOperator,
    TripletMatrix, ValidatedCsr,
};
pub use sparse::{
    BsrPlanAnalysis, CompactCsrMatrix, RowBucketPlanAnalysis, RowBucketedCsrMatrix,
    SellCSigmaMatrix, SellCSigmaPlanAnalysis, SpmvAnalysis, SpmvKernel, SpmvPlan, SpmvPlanConfig,
    SpmvPlanFormat, SpmvPlanPreference, SpmvPlanReport, detected_spmv_features,
    selected_spmv_kernel,
};
pub use sparse::{
    CsrPartition, InputVectorPlacement, NumaPartitionedCsr, NumaPartitioningReport,
    NumaPlacementStatus, NumaSpmvError, NumaSpmvPolicy, NumaStorageEstimate, PartitionStats,
    detect_numa_partitions,
};
pub use traits::{
    AccuracyStatus, AdjointCapability, ComplexField, LinearOperator, Preconditioner,
    SOLVER_RESULT_CONTRACT_VERSION, SolverError, SolverErrorKind, SolverFailure,
    SolverResultContract, SolverStatus,
};

// Re-export iterative solvers
pub use iterative::CancellationToken as BackendCancellationToken;
#[allow(deprecated)] // `cgs` is deprecated but kept re-exported for API compatibility
pub use iterative::{
    BLOCK_FGMRES_CHECKPOINT_FORMAT_VERSION, BLOCK_FGMRES_CHECKPOINT_MAGIC, BackendKrylovCapability,
    BackendKrylovConfig, BackendKrylovError, BackendKrylovFailureDiagnostic, BackendKrylovMethod,
    BackendKrylovScalar, BackendKrylovStatus, BackendKrylovWorkspace, BackendStoppingReason,
    BiCgstabConfig, BiCgstabSolution, BlockFgmresConfig, BlockFgmresSolution, BlockGmresConfig,
    BlockGmresSolution, CaCgConfig, CaCgSolution, CgConfig, CgSolution, CgsConfig, CgsSolution,
    DEFLATED_CHECKPOINT_FORMAT_VERSION, DeflatedCheckpoint, DeflationSubspace, FgmresConfig,
    FgmresSolution, GCRO_DR_CHECKPOINT_FORMAT_VERSION, GMRES_BASIS_CHECKPOINT_FORMAT_VERSION,
    GMRES_CHECKPOINT_FORMAT_VERSION, GcroDrCheckpoint, GcroDrConfig, GcroDrSubspace,
    GmresBasisCheckpoint, GmresBasisRun, GmresCheckpoint, GmresConfig, GmresDrConfig,
    GmresSolution, IdrsConfig, IdrsSolution, IdrsStabConfig, IterativeResultContract,
    KrylovOpCounts, RECYCLED_CHECKPOINT_FORMAT_VERSION, RecycledSubspace,
    backend_krylov_capability_matrix, bicgstab, block_fgmres_preconditioned, block_gmres,
    block_gmres_preconditioned, block_gmres_preconditioned_with_guess, block_gmres_with_guess,
    ca_cg, ca_pcg, cg, cg_with_guess, cgs, fgmres, fgmres_preconditioned, gcro_dr,
    gcro_dr_preconditioned, gmres, gmres_deflated, gmres_deflated_preconditioned,
    gmres_preconditioned, gmres_preconditioned_with_guess, gmres_with_guess, idrs,
    idrs_preconditioned, idrs_stab, idrs_stab_preconditioned, pcg, pcg_with_guess, try_bicgstab,
    try_bicgstab_preconditioned, try_bicgstab_preconditioned_with_guess, try_block_fgmres_backend,
    try_block_fgmres_preconditioned, try_block_gmres, try_block_gmres_backend,
    try_block_gmres_preconditioned_with_abort, try_ca_cg, try_ca_pcg, try_cg, try_cg_with_guess,
    try_cgs, try_fgmres, try_fgmres_backend_preconditioned, try_fgmres_preconditioned, try_gcro_dr,
    try_gcro_dr_preconditioned, try_gmres, try_gmres_backend, try_gmres_deflated,
    try_gmres_deflated_preconditioned, try_gmres_dr_preconditioned, try_gmres_preconditioned,
    try_gmres_preconditioned_with_guess, try_gmres_recycled, try_gmres_with_guess, try_idrs,
    try_idrs_preconditioned, try_idrs_preconditioned_with_residual_replacement, try_idrs_stab,
    try_idrs_stab_preconditioned, try_idrs_with_residual_replacement, try_pcg, try_pcg_with_guess,
};
pub use iterative::{
    CaCgBasis, MinresConfig, MinresSolution, TfqmrConfig, TfqmrSolution, minres,
    minres_preconditioned, minres_preconditioned_with_guess, minres_with_guess, tfqmr,
    tfqmr_preconditioned, tfqmr_preconditioned_with_guess, try_minres, try_minres_preconditioned,
    try_minres_preconditioned_with_guess, try_minres_with_guess, try_tfqmr,
    try_tfqmr_preconditioned, try_tfqmr_preconditioned_with_guess,
};
pub use iterative::{
    RESIDUAL_REPLACEMENT_INTERVAL, bicgstab_pipelined, bicgstab_pipelined_preconditioned,
    bicgstab_pipelined_preconditioned_with_guess,
    bicgstab_pipelined_preconditioned_with_residual_replacement,
    bicgstab_pipelined_with_residual_replacement, gmres_pipelined, try_bicgstab_pipelined,
    try_bicgstab_pipelined_preconditioned, try_bicgstab_pipelined_preconditioned_with_guess,
    try_bicgstab_pipelined_preconditioned_with_residual_replacement,
    try_bicgstab_pipelined_with_residual_replacement, try_gmres_pipelined,
};

// Re-export direct solvers
pub use direct::{
    CholeskyError, CholeskyFactorization, CholeskyOrdering, DenseConfiguredFactorization,
    DenseDirectConfig, DenseDirectError, DenseDirectMethod, DenseFactorization,
    DenseRefinementResult, DenseSvd, DirectFactorization, DirectSolveResult, ExpmError, LuError,
    LuFactorization, LuWorkspace, MatrixClass, MixedLuSolution, Ordering, QrError, QrFactorization,
    QrWorkspace, SPARSE_CHOLESKY_CHECKPOINT_FORMAT_VERSION, SPARSE_LU_CHECKPOINT_FORMAT_VERSION,
    SparseCholeskyError, SparseCholeskyFactorization, SparseCholeskySymbolic,
    SparseConfiguredFactorization, SparseConfiguredRefinementResult, SparseDirectConfig,
    SparseDirectError, SparseDirectMethod, SparseFactorization, SparseLuError,
    SparseLuFactorization, SparseSymbolic, analyze_sparse, cholesky_factorize,
    cholesky_factorize_hermitian, cholesky_factorize_in_place,
    cholesky_factorize_in_place_hermitian, dense_backward_error, dense_svd, expm, factorize_dense,
    factorize_dense_configured, factorize_sparse, factorize_sparse_configured, lu_factorize,
    lu_factorize_in_place, lu_factorize_with_workspace, lu_solve, lu_solve_conj_transpose,
    lu_solve_mixed, lu_solve_transpose, qr_factorize_with_workspace,
};

// Re-export mixed-precision types
pub use mixed_precision::{
    MIXED_PRECISION_BLOCK_CHECKPOINT_FORMAT_VERSION, MIXED_PRECISION_BLOCK_CHECKPOINT_MAGIC,
    MIXED_PRECISION_CHECKPOINT_FORMAT_VERSION, MIXED_PRECISION_CHECKPOINT_MAGIC,
    MixedPrecisionBlockCheckpoint, MixedPrecisionCheckpoint, MixedPrecisionPreconditioner,
    PrecisionCast, gmres_ir_complex32, gmres_ir_f32, gmres_ir_ilu, gmres_ir_ilu_with_guess,
    gmres_ir_lu, gmres_ir_lu_with_guess, mixed_precision_block_solve,
    mixed_precision_block_solve_with_guess, mixed_precision_deflated_solve,
    mixed_precision_deflated_solve_with_guess, mixed_precision_gcro_dr_solve,
    mixed_precision_gcro_dr_solve_with_guess, mixed_precision_recycled_solve,
    mixed_precision_recycled_solve_with_guess, mixed_precision_solve,
    mixed_precision_solve_with_guess, try_gmres_ir_complex32, try_gmres_ir_f32, try_gmres_ir_ilu,
    try_gmres_ir_ilu_with_guess, try_gmres_ir_lu, try_gmres_ir_lu_with_guess,
    try_mixed_precision_block_solve, try_mixed_precision_block_solve_with_guess,
    try_mixed_precision_deflated_solve, try_mixed_precision_deflated_solve_with_guess,
    try_mixed_precision_gcro_dr_solve, try_mixed_precision_gcro_dr_solve_with_guess,
    try_mixed_precision_recycled_solve, try_mixed_precision_recycled_solve_with_guess,
    try_mixed_precision_solve, try_mixed_precision_solve_with_guess,
};

// Re-export nonlinear solvers
pub use nonlinear::{
    AndersonConfig, AndersonError, AndersonSolution, ContinuationStrategy, ForcingTerm,
    InexactNewtonConfig, LineSearch, LineSearchFailure, NewtonConfig, NewtonError, NewtonSolution,
    TrustRegionConfig, anderson_acceleration, newton_raphson, newton_raphson_continuation,
    newton_raphson_inexact,
};

// Re-export time integration
pub use time_integration::{
    GeneralizedAlphaAdjointResult, GeneralizedAlphaConfig, NEWMARK_CHECKPOINT_FORMAT_VERSION,
    NEWMARK_CHECKPOINT_MAGIC, NewmarkAdjointResult, NewmarkCoefficients, NewmarkConfig,
    NewmarkInitial, NewmarkResult, REVOLVE_CHECKPOINT_FORMAT_VERSION, REVOLVE_CHECKPOINT_MAGIC,
    RevolveAction, RevolveError, RevolveSchedule, TRBDF2_GAMMA, TrBdf2Config, TrBdf2Result,
    TrBdf2StepRecord, acceleration_effective_scalar, correct_displacement_from_acceleration,
    correct_from_displacement_slice, correct_velocity_from_acceleration, generalized_alpha_adjoint,
    generalized_alpha_integrate, mass_proportional_damping_power, newmark_adjoint,
    newmark_integrate, predict_displacement_scalar, predict_displacement_slice,
    predict_velocity_scalar, predict_velocity_slice, trapezoidal_force_work_increment,
    trapezoidal_power_increment, trbdf2_integrate, try_generalized_alpha_integrate,
    try_newmark_integrate,
};

// Re-export high-level solver
#[cfg(feature = "cuda")]
#[doc(inline)]
pub use solve::{CudaLinearSolveSession, CudaLinearSolveStatus, CudaLinearUpdateStatus};
pub use solve::{
    AssumptionProvenance, LinearSolveBackend, LinearSolveBackendProvenance, LinearSolveConfig,
    LinearSolveError, LinearSolveFallback, LinearSolveResult, LinearSolverType,
    LinearSystemAssumption, SaddlePointAssumptionReport, solve, solve_complex,
    solve_complex_with_backend, solve_real, solve_real_with_assumption, solve_real_with_backend,
    solve_real_with_guess, solve_real_with_guess_and_assumption, solve_saddle_point,
    solve_saddle_point_with_assumptions, solve_saddle_point_with_guess,
    solve_saddle_point_with_guess_and_assumptions, solve_with_assumption, solve_with_backend,
    solve_with_backend_and_assumption, solve_with_guess, solve_with_guess_and_assumption,
};

// Re-export preconditioners
#[cfg(feature = "experimental-gnn")]
pub use preconditioners::GnnCoarseningWeights;
pub use preconditioners::{
    AdditiveSchwarzPreconditioner, AmgCoarsening, AmgConfig, AmgCycle, AmgDiagnostics,
    AmgInterpolation, AmgPhysicsProfile, AmgPreconditioner, AmgRestriction, AmgSmoother,
    BackendAmgPhase, BackendAmgPhaseProfile, BackendAmgPreconditioner, BlockJacobiBackend,
    BlockJacobiPreconditioner, BlockJacobiRuntimeDiagnostics, BlockPreconditionerConfig,
    ChebyshevBackend, ChebyshevBackendF32, ChebyshevConfig, ChebyshevError,
    ChebyshevPreconditioner, DiagonalPreconditioner, GeneoSetupError, HiluConfig,
    HiluPreconditioner, IdentityPreconditioner, IluColoringPreconditioner, IluConfig,
    IluFixedPointPreconditioner, IluPreconditioner, IlutConfig, IlutPreconditioner, LscBlocks,
    MaterialSocConfig, ParIlutConfig, ParIlutPreconditioner, PcdBlocks, PointJacobiBackend,
    PolynomialJacobiBackend, SADDLE_POINT_CHECKPOINT_FORMAT_VERSION,
    SCHWARZ_CHECKPOINT_FORMAT_VERSION, SaddlePointBlockAssumption, SaddlePointBlockAssumptions,
    SaddlePointBlockProvenance, SaddlePointBlockValidation, SaddlePointConfig,
    SaddlePointPreconditioner, SchurStrategy, SchwarzCoarseSetup, SchwarzRuntimeDiagnostics,
    SweepingSchwarzPreconditioner, btdiagb_schur_from_monolithic, coordinate_layers,
    has_saddle_point_structure,
};

// Re-export eigenvalue solvers
pub use eigen::{
    ComplexArnoldiConfig, ComplexTargetedEigenResult, EIGEN_CHECKPOINT_FORMAT_VERSION,
    EIGEN_CHECKPOINT_MAGIC, EigenContinuationConfig, EigenContinuationEvent,
    EigenContinuationPoint, EigenContinuationPrediction, EigenContinuationStep,
    EigenPseudoArclengthPrediction, EigenPseudoArclengthStep, FeastConfig, LobpcgConfig,
    LobpcgResult, QuadraticEigenConfig, QuadraticEigenResult, ShiftInvertConfig,
    TargetedEigenResult, classify_eigen_continuation_event,
    complex_shift_invert_arnoldi_generalized, continue_generalized_eigenpair,
    continue_generalized_eigenpair_pseudo_arclength, eigenpairs_in_interval,
    eigenvalue_count_in_interval, feast_generalized, generalized_eigenvalue_sensitivity,
    generalized_eigenvector_sensitivity, lobpcg, lobpcg_generalized,
    lobpcg_generalized_preconditioned, lobpcg_preconditioned, nonsymmetric_eigenvalues,
    nonsymmetric_eigenvalues_f32, orthonormal_basis, predict_generalized_eigenpair,
    quadratic_eigen_damped, shift_invert_lanczos_generalized, subspace_distance,
};

// Re-export runtime configuration
pub use runtime::{SolverRuntime, configure_rayon};
pub use topology::{
    AffinityStatus, MachineTopology, NumaNode, PlacementPlan, SocketInfo, WorkerPlacement,
    current_thread_affinity, pin_current_thread_to,
};
