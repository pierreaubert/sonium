//! Generic multi-physics finite element method library
//!
//! Provides a trait-based FEM framework supporting multiple physics domains:
//! acoustics (Helmholtz), electromagnetics (Maxwell), solid mechanics, and coupled problems.
//!
//! # Architecture
//!
//! - **Layer 1**: `mesh`, `basis`, `quadrature` — geometry and shape functions
//! - **Layer 2**: `spaces`, `forms`, `assembly`, `boundary`, `solver` — generic FEM infrastructure
//! - **Layer 3**: `physics` — domain-specific implementations (acoustics, elasticity, Maxwell)

// Layer 1: Geometry and shape functions
pub mod backend;
pub mod backend_selection;
pub mod basis;
pub mod mesh;
pub mod quadrature;

// Layer 2: Generic FEM infrastructure
pub mod assembly;
pub mod boundary;
pub mod conventions;
pub mod forms;
pub mod parallelism;
pub mod solver;
pub mod spaces;
pub mod sweep_schedule;
pub mod validation;

// Layer 2.5: Multigrid (generic, but still references Helmholtz types — Phase 2 cleanup)
pub mod multigrid;

// Layer 3: Physics-specific modules
pub mod physics;

pub use backend::{
    FemAction, FemActualBackend, FemAdmission, FemAdmissionError, FemAssemblyMode, FemBackend,
    FemBackendProvenance, FemBackendRequest, FemCapabilityRow, FemCapabilityStatus, FemElement,
    FemOrder, FemPhysics, FemScalar, FemSpace, FemWorkload, admit_fem_cuda, admit_fem_workload,
    fem_cuda_capability_rows, fem_cuda_magnetic_scalar_capability_rows,
    fem_cuda_maxwell_capability_rows, fem_cuda_stokes_capability_rows,
    fem_cuda_structural_capability_rows, fem_cuda_thermal_capability_rows,
};
pub use backend_selection::{
    BackendPreconditionerKind, BackendPressureSchurPreconditionerKind, CHEBYSHEV_EIGENVALUE_RATIO,
    FemPreconditionerKind, GpuPrecision, GpuPrecond, LinearSolverBackend,
    LinearSolverCpuPreconditionerKind, LinearSolverKind,
};
pub use conventions::{ACOUSTIC_CONVENTION, FemConvention, FemConventionError, FemProblemError};
pub use solver::{FemResultError, MatrixFreeError};
pub use sweep_schedule::{
    ResourceBudget, ScheduleError, SolveLane, SolveSchedule, SolveWorkItem, SolveWorkerAssignment,
    WorkloadShape, partition_solve_schedule, plan_solve_schedule,
};

// Layer 4: Optimization
pub mod optimization;

// I/O: VTK export for visualization
pub mod io;

// Backward-compat re-exports from physics::acoustics::pressure
pub use physics::acoustics::pressure::neural_multigrid;
pub use physics::acoustics::pressure::schwarz_pml;
pub use physics::acoustics::pressure::waveholtz;

/// Library version
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!version().is_empty());
    }
}
