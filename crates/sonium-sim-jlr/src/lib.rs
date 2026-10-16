//! Jet loop reactor LBM replication of Melchior et al. (2026).
//!
//! Contains a D3Q19 lattice Boltzmann solver (BGK + Smagorinsky LES,
//! halfway bounce-back, two-level refinement), the analytic JLR geometry
//! of the RPTU laboratory reactor, the three published operating points,
//! and the comparison gates against the paper's Tables 4.1/4.2/A.4.
//! Start with [`lbm::Lattice`] for the solver, [`geometry::JlrGeometry`]
//! for the reactor, and [`case::run_case_cpu`] for a full case.

// Rust guideline compliant 2026-02-21

pub mod case;
pub mod compare;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod geometry;
pub mod lbm;
pub mod refinement;
