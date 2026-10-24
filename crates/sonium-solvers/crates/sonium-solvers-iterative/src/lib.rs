//! Iterative solvers for linear systems
//!
//! This module provides Krylov subspace methods for solving large sparse systems:
//! - [`gmres`]: GMRES(m) with restart - best for general non-symmetric systems
//! - [`bicgstab`]: BiCGSTAB - good alternative to GMRES
//! - [`idrs`]: IDR(s) - modern alternative, often fewer iterations than BiCGSTAB
//! - [`idrs_stab`]: IDR(s)stab(ℓ) - IDR(s) with degree-ℓ stabilization, the
//!   recommended IDR variant for Helmholtz-type indefinite systems
//! - [`tfqmr`]: TFQMR - transpose-free QMR, smoother convergence than BiCGSTAB
//! - [`cg`]: Conjugate Gradient - for symmetric positive definite systems
//! - [`minres`]: MINRES - for Hermitian (possibly indefinite) systems
//! - [`cgs`]: CGS - **deprecated**, superseded by BiCGSTAB/IDR(s)
//! - [`gmres_recycled`]: recycled GMRES - **deprecated**, use [`gcro_dr`]

mod bicgstab;
pub mod bicgstab_pipelined;
mod block_fgmres;
mod block_gmres;
mod ca_cg;
mod cg;
mod cgs;
mod cgs_orth;
mod checked_callbacks;
mod checkpoint;
mod deflated_checkpoint;
mod fgmres;
pub mod gcro_dr;
mod gcro_dr_checkpoint;
mod gmres;
mod gmres_basis_checkpoint;
mod gmres_checkpoint;
mod gmres_common;
pub mod gmres_deflated;
pub mod gmres_dr;
pub mod gmres_pipelined;
pub mod gmres_recycled;
pub mod host_operator;
mod idrs;
pub mod idrs_stab;
mod minres;
mod result_contract;
mod tfqmr;
mod typed_block_checkpoint;

pub mod backend;

pub use checkpoint::{BLOCK_FGMRES_CHECKPOINT_FORMAT_VERSION, BLOCK_FGMRES_CHECKPOINT_MAGIC};

#[doc(inline)]
pub use backend::{
    BackendKrylovCapability, BackendKrylovConfig, BackendKrylovError,
    BackendKrylovFailureDiagnostic, BackendKrylovMethod, BackendKrylovScalar, BackendKrylovStatus,
    BackendKrylovWorkspace, BackendStoppingReason, CancellationToken, DeviceGmresSession,
    DeviceInitialGuess, DevicePcgAssumption, DevicePcgSession, DevicePcgStatus,
    backend_krylov_capability_matrix, try_block_fgmres_backend, try_block_gmres_backend,
    try_fgmres_backend_preconditioned, try_gmres_backend, validate_backend_krylov,
    validate_backend_pcg,
};
pub use bicgstab::{
    BiCgstabConfig, BiCgstabSolution, bicgstab, bicgstab_preconditioned,
    bicgstab_preconditioned_with_guess, try_bicgstab, try_bicgstab_preconditioned,
    try_bicgstab_preconditioned_with_guess,
};
pub use bicgstab_pipelined::{
    RESIDUAL_REPLACEMENT_INTERVAL, bicgstab_pipelined, bicgstab_pipelined_preconditioned,
    bicgstab_pipelined_preconditioned_with_guess,
    bicgstab_pipelined_preconditioned_with_residual_replacement,
    bicgstab_pipelined_with_residual_replacement, try_bicgstab_pipelined,
    try_bicgstab_pipelined_preconditioned, try_bicgstab_pipelined_preconditioned_with_guess,
    try_bicgstab_pipelined_preconditioned_with_residual_replacement,
    try_bicgstab_pipelined_with_residual_replacement,
};
pub use block_fgmres::{
    BlockFgmresConfig, BlockFgmresSolution, block_fgmres_preconditioned,
    block_fgmres_preconditioned_with_guess, try_block_fgmres_preconditioned,
};
pub use block_gmres::{
    BlockGmresConfig, BlockGmresSolution, block_gmres, block_gmres_preconditioned,
    block_gmres_preconditioned_with_guess, block_gmres_with_guess, try_block_gmres,
    try_block_gmres_preconditioned, try_block_gmres_preconditioned_with_abort,
};
pub use ca_cg::{CaCgBasis, CaCgConfig, CaCgSolution, ca_cg, ca_pcg, try_ca_cg, try_ca_pcg};
pub use cg::{
    CgConfig, CgSolution, cg, cg_with_guess, pcg, pcg_with_guess, try_cg, try_cg_with_guess,
    try_pcg, try_pcg_with_guess,
};
#[allow(deprecated)]
pub use cgs::{CgsConfig, CgsSolution, cgs, try_cgs};
pub use cgs_orth::KrylovOpCounts;
pub use deflated_checkpoint::{DEFLATED_CHECKPOINT_FORMAT_VERSION, DeflatedCheckpoint};
pub use fgmres::{
    FgmresConfig, FgmresSolution, fgmres, fgmres_preconditioned, try_fgmres,
    try_fgmres_preconditioned, try_fgmres_preconditioned_with_abort,
};
pub use gcro_dr::{
    GcroDrConfig, GcroDrSubspace, gcro_dr, gcro_dr_preconditioned, try_gcro_dr,
    try_gcro_dr_preconditioned,
};
pub use gcro_dr_checkpoint::{GCRO_DR_CHECKPOINT_FORMAT_VERSION, GcroDrCheckpoint};
pub use gmres::{
    GmresConfig, GmresSolution, gmres, gmres_preconditioned, gmres_preconditioned_with_guess,
    gmres_with_guess, try_gmres, try_gmres_preconditioned, try_gmres_preconditioned_with_guess,
};
pub use gmres_basis_checkpoint::{
    GMRES_BASIS_CHECKPOINT_FORMAT_VERSION, GmresBasisCheckpoint, GmresBasisRun,
};
pub use gmres_checkpoint::{
    GMRES_CHECKPOINT_FORMAT_VERSION, GmresCheckpoint, try_gmres_with_guess,
};
pub use gmres_deflated::{
    DeflationSubspace, gmres_deflated, gmres_deflated_preconditioned, try_gmres_deflated,
    try_gmres_deflated_preconditioned,
};
pub use gmres_dr::{
    GmresDrConfig, gmres_dr, gmres_dr_preconditioned, gmres_dr_preconditioned_with_guess,
    gmres_dr_with_guess, try_gmres_dr_preconditioned,
};
pub use gmres_pipelined::{gmres_pipelined, try_gmres_pipelined};
#[allow(deprecated)]
pub use gmres_recycled::{
    RECYCLED_CHECKPOINT_FORMAT_VERSION, RecycledSubspace, gmres_recycled,
    gmres_recycled_preconditioned, try_gmres_recycled, try_gmres_recycled_preconditioned,
};
pub use idrs::{
    IDR_RESIDUAL_REPLACEMENT_INTERVAL, IdrsConfig, IdrsSolution, idrs, idrs_preconditioned,
    idrs_preconditioned_with_residual_replacement, idrs_with_residual_replacement, try_idrs,
    try_idrs_preconditioned, try_idrs_preconditioned_with_residual_replacement,
    try_idrs_with_residual_replacement,
};
pub use idrs_stab::{
    IdrsStabConfig, idrs_stab, idrs_stab_preconditioned, try_idrs_stab,
    try_idrs_stab_preconditioned,
};
pub use minres::{
    MinresConfig, MinresSolution, minres, minres_preconditioned, minres_preconditioned_with_guess,
    minres_with_guess, try_minres, try_minres_preconditioned, try_minres_preconditioned_with_guess,
    try_minres_with_guess,
};
pub use result_contract::IterativeResultContract;
pub use tfqmr::{
    TfqmrConfig, TfqmrSolution, tfqmr, tfqmr_preconditioned, tfqmr_preconditioned_with_guess,
    try_tfqmr, try_tfqmr_preconditioned, try_tfqmr_preconditioned_with_guess,
};
