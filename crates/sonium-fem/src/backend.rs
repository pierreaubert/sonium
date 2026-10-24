//! FEM backend capability and admission contracts.
//!
//! Backend selection must happen before a mesh is assembled.  In particular,
//! an accelerator entry point must not build a CPU matrix and only then find
//! out that the requested cell is unsupported.  This module is deliberately
//! independent of the runtime CUDA feature so applications and evidence
//! producers can exercise the same fail-closed contract on every host.

use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Backend requested by a FEM workload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FemBackend {
    Cpu,
    Cuda,
}

impl fmt::Display for FemBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        })
    }
}

/// Backend actually used by an admitted workload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FemActualBackend {
    Cpu,
    Cuda,
    Unavailable,
    /// No assembly or iteration was permitted after admission rejected it.
    #[serde(rename = "none")]
    NotExecuted,
}

impl fmt::Display for FemActualBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Unavailable => "unavailable",
            Self::NotExecuted => "none",
        })
    }
}

/// Scalar type used by a FEM operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FemScalar {
    F64,
    Complex64,
    F32,
    Complex32,
}

impl FemScalar {
    pub const fn id_component(self) -> &'static str {
        match self {
            Self::F64 => "f64",
            Self::Complex64 => "Complex64",
            Self::F32 => "f32",
            Self::Complex32 => "Complex32",
        }
    }
}

/// FEM function space in the capability matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FemSpace {
    H1,
    Hcurl,
    Hdiv,
    Dg,
}

impl FemSpace {
    const fn id_component(self) -> &'static str {
        match self {
            Self::H1 => "h1",
            Self::Hcurl => "hcurl",
            Self::Hdiv => "hdiv",
            Self::Dg => "dg",
        }
    }
}

/// Cell family requested by an accelerator workload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FemElement {
    Tri3,
    Quad,
    Tetrahedron,
    Hexahedron,
}

impl FemElement {
    const fn id_component(self) -> &'static str {
        match self {
            Self::Tri3 => "tri3",
            Self::Quad => "quad",
            Self::Tetrahedron => "tetrahedron",
            Self::Hexahedron => "hexahedron",
        }
    }
}

/// Polynomial order bucket used by the backend contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FemOrder {
    P1,
    /// Quadratic order, covered by the bounded FEM1 production cell.
    P2,
    P2OrHigher,
    /// Cubic order (Q3 for tensor-product hexahedra).
    P3,
}

impl FemOrder {
    const fn id_component(self) -> &'static str {
        match self {
            Self::P1 => "p1",
            Self::P2 => "p2",
            Self::P3 => "p3",
            Self::P2OrHigher => "p2-or-higher",
        }
    }
}

/// Physics form represented by a workload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FemPhysics {
    Helmholtz,
    /// Small native-CUDA linear-elasticity CSR action cell.
    LinearElasticity,
    /// Small native-CUDA H(curl) Maxwell CSR action cell.
    Maxwell,
    /// Small native-CUDA Taylor--Hood Stokes CSR action cell.
    Stokes,
    Other,
    /// Steady scalar heat conduction with host assembly and retained CUDA solves.
    HeatConduction,
    /// Linear reduced scalar magnetic potential with positive permeability.
    MagneticScalarPotential,
    /// Linear scalar electrostatic potential with positive permittivity.
    ElectrostaticPotential,
}

impl FemPhysics {
    const fn id_component(self) -> &'static str {
        match self {
            Self::Helmholtz => "helmholtz",
            Self::LinearElasticity => "linear-elasticity",
            Self::Maxwell => "maxwell",
            Self::Stokes => "stokes",
            Self::Other => "other-form",
            Self::HeatConduction => "heat-conduction",
            Self::MagneticScalarPotential => "magnetic-scalar-potential",
            Self::ElectrostaticPotential => "electrostatic-potential",
        }
    }
}

/// Assembly strategy requested by a workload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FemAssemblyMode {
    Assembled,
    PartialAssembly,
    MatrixFree,
}

impl FemAssemblyMode {
    const fn id_component(self) -> &'static str {
        match self {
            Self::Assembled => "assembled",
            Self::PartialAssembly => "partial-assembly",
            Self::MatrixFree => "matrix-free",
        }
    }
}

/// Matrix action requested from the backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FemAction {
    Forward,
    Transpose,
    Hermitian,
}

impl FemAction {
    const fn id_component(self) -> &'static str {
        match self {
            Self::Forward => "forward",
            Self::Transpose => "transpose",
            Self::Hermitian => "hermitian",
        }
    }
}

/// Complete description required before FEM backend dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FemWorkload {
    pub space: FemSpace,
    pub element: FemElement,
    pub order: FemOrder,
    pub physics: FemPhysics,
    pub mode: FemAssemblyMode,
    pub scalar: FemScalar,
    pub action: FemAction,
}

impl FemWorkload {
    /// The bounded native CUDA FEM1 cell used by the production producer.
    pub const fn native_cuda(scalar: FemScalar, action: FemAction) -> Self {
        Self {
            space: FemSpace::H1,
            element: FemElement::Tri3,
            order: FemOrder::P1,
            physics: FemPhysics::Helmholtz,
            mode: FemAssemblyMode::Assembled,
            scalar,
            action,
        }
    }

    /// The bounded native CUDA H1/Tri3/P2 Helmholtz action cell.
    ///
    /// The producer still assembles the production matrix on the host; this
    /// constructor only admits the checked device CSR action for exactly P2.
    pub const fn native_cuda_p2(scalar: FemScalar, action: FemAction) -> Self {
        Self {
            space: FemSpace::H1,
            element: FemElement::Tri3,
            order: FemOrder::P2,
            physics: FemPhysics::Helmholtz,
            mode: FemAssemblyMode::Assembled,
            scalar,
            action,
        }
    }

    /// Describe a steady assembled f64 heat-conduction system.
    ///
    /// Tri3 and tetrahedron P1/P2 are admitted for all actions. Host form
    /// assembly and boundary elimination feed the retained CUDA solver. This
    /// does not admit thermal advection, nonlinear radiation or time stepping.
    /// PCG requires positive conductivity and constraints removing the constant
    /// nullspace; a pure Neumann system needs explicit nullspace treatment.
    pub const fn native_cuda_heat_conduction(
        element: FemElement,
        order: FemOrder,
        action: FemAction,
    ) -> Self {
        Self {
            space: FemSpace::H1,
            element,
            order,
            physics: FemPhysics::HeatConduction,
            mode: FemAssemblyMode::Assembled,
            scalar: FemScalar::F64,
            action,
        }
    }

    /// Describe an assembled linear scalar magnetic-potential CUDA solve.
    ///
    /// Admits f64 H1 Tri3/tetrahedron P1/P2 for all actions. Host permeability,
    /// remanence assembly and constraints feed the retained CUDA solver. PCG
    /// requires positive permeability and a constrained constant nullspace.
    /// Nonlinear B-H, vector/axisymmetric formulations and device field recovery
    /// have separate contracts and are not covered by this workload.
    pub const fn native_cuda_magnetic_scalar_potential(
        element: FemElement,
        order: FemOrder,
        action: FemAction,
    ) -> Self {
        Self {
            physics: FemPhysics::MagneticScalarPotential,
            ..Self::native_cuda_heat_conduction(element, order, action)
        }
    }

    /// Describe an assembled linear electrostatic-potential CUDA solve.
    ///
    /// Admits f64 H1 Tri3/tetrahedron P1/P2 for all actions. Host forms and
    /// symmetric boundary elimination feed the retained solver. PCG requires
    /// positive permittivity and constraints removing the constant nullspace.
    /// Nonlinear dielectric, floating-conductor and electromechanical coupling
    /// require separate admission; this does not provide GPU form assembly.
    pub const fn native_cuda_electrostatic_potential(
        element: FemElement,
        order: FemOrder,
        action: FemAction,
    ) -> Self {
        Self {
            physics: FemPhysics::ElectrostaticPotential,
            ..Self::native_cuda_heat_conduction(element, order, action)
        }
    }

    /// Describe an assembled Complex64 P1/P2 tetrahedral Helmholtz solve.
    ///
    /// Host assembly and constraint elimination feed `CudaFemSolveSession`.
    /// Admission rejects `P2OrHigher`; this is not a GPU assembly capability.
    pub const fn native_cuda_tet(order: FemOrder, action: FemAction) -> Self {
        Self {
            space: FemSpace::H1,
            element: FemElement::Tetrahedron,
            order,
            physics: FemPhysics::Helmholtz,
            mode: FemAssemblyMode::Assembled,
            scalar: FemScalar::Complex64,
            action,
        }
    }

    /// Describe an assembled Complex64 Q1/Q2/Q3 hexahedral Helmholtz solve.
    ///
    /// `P1`, `P2` and `P3` denote tensor-product Q1, Q2 and Q3 here.
    /// Host assembly and constraints feed `CudaFemSolveSession`; this contract
    /// does not admit generic partial-assembly callbacks or `P2OrHigher`.
    pub const fn native_cuda_hex(order: FemOrder, action: FemAction) -> Self {
        Self {
            space: FemSpace::H1,
            element: FemElement::Hexahedron,
            order,
            physics: FemPhysics::Helmholtz,
            mode: FemAssemblyMode::Assembled,
            scalar: FemScalar::Complex64,
            action,
        }
    }

    /// The bounded native CUDA 3-D linear-elasticity CSR action cell.
    pub const fn native_cuda_structural(scalar: FemScalar, action: FemAction) -> Self {
        Self {
            space: FemSpace::H1,
            element: FemElement::Tetrahedron,
            order: FemOrder::P1,
            physics: FemPhysics::LinearElasticity,
            mode: FemAssemblyMode::Assembled,
            scalar,
            action,
        }
    }

    /// The bounded native CUDA H(curl)/Tri3/P1 Maxwell CSR action cell.
    pub const fn native_cuda_maxwell(scalar: FemScalar, action: FemAction) -> Self {
        Self {
            space: FemSpace::Hcurl,
            element: FemElement::Tri3,
            order: FemOrder::P1,
            physics: FemPhysics::Maxwell,
            mode: FemAssemblyMode::Assembled,
            scalar,
            action,
        }
    }

    /// The bounded native CUDA Taylor--Hood Stokes saddle-point action cell.
    ///
    /// The production cell uses a vector H1/P2 velocity block and scalar
    /// H1/P1 pressure block, flattened to a checked monolithic CSR operator.
    pub const fn native_cuda_stokes(action: FemAction) -> Self {
        Self {
            space: FemSpace::H1,
            element: FemElement::Tri3,
            order: FemOrder::P2,
            physics: FemPhysics::Stokes,
            mode: FemAssemblyMode::Assembled,
            scalar: FemScalar::F64,
            action,
        }
    }

    /// The bounded native CUDA matrix-free H1/Tri3/P1 Helmholtz cell.
    ///
    /// Element tensors are prepared by the production FEM assembly layer;
    /// the matrix-free action itself is a resident batched local contraction.
    pub const fn native_cuda_matrix_free(scalar: FemScalar, action: FemAction) -> Self {
        Self {
            space: FemSpace::H1,
            element: FemElement::Tri3,
            order: FemOrder::P1,
            physics: FemPhysics::Helmholtz,
            mode: FemAssemblyMode::MatrixFree,
            scalar,
            action,
        }
    }

    /// Stable capability-matrix identifier for this workload.
    pub fn id(self) -> String {
        format!(
            "{}.{}.{}.{}.{}.{}.{}",
            self.space.id_component(),
            self.element.id_component(),
            self.order.id_component(),
            self.physics.id_component(),
            self.mode.id_component(),
            self.scalar.id_component(),
            self.action.id_component(),
        )
    }
}

impl fmt::Display for FemWorkload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.id())
    }
}

/// A backend request.  CPU fallback is opt-in and is part of the request so
/// an admission decision can preserve its provenance in a result envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FemBackendRequest {
    pub requested: FemBackend,
    pub fallback: Option<FemBackend>,
}

impl FemBackendRequest {
    pub const fn required(requested: FemBackend) -> Self {
        Self {
            requested,
            fallback: None,
        }
    }

    pub const fn with_cpu_fallback() -> Self {
        Self {
            requested: FemBackend::Cuda,
            fallback: Some(FemBackend::Cpu),
        }
    }
}

/// Backend provenance attached to an admission decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FemBackendProvenance {
    pub requested_backend: FemBackend,
    pub actual_backend: FemActualBackend,
    pub fallback: String,
    pub fallback_reason: String,
}

impl FemBackendProvenance {
    fn native(requested: FemBackend, actual: FemActualBackend) -> Self {
        Self {
            requested_backend: requested,
            actual_backend: actual,
            fallback: "none".into(),
            fallback_reason: "not_applicable".into(),
        }
    }
}

/// Successful admission, carrying the exact workload and backend provenance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FemAdmission {
    pub workload: FemWorkload,
    pub provenance: FemBackendProvenance,
}

/// Admission failure.  The provenance is intentionally retained on the
/// error, including actual backend `none`, so callers cannot accidentally
/// report an unsupported CPU assembly as a CUDA execution.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum FemAdmissionError {
    #[error(
        "FEM workload {workload} rejected before assembly: requested_backend={requested_backend}, actual_backend={actual_backend}, fallback={fallback}, fallback_reason={fallback_reason}"
    )]
    Unsupported {
        workload: FemWorkload,
        requested_backend: FemBackend,
        actual_backend: FemActualBackend,
        fallback: String,
        fallback_reason: String,
    },
    #[error("invalid FEM backend request: {0}")]
    InvalidRequest(String),
}

/// Admit one FEM workload before assembly or iteration zero.
pub fn admit_fem_workload(
    workload: FemWorkload,
    request: FemBackendRequest,
) -> Result<FemAdmission, FemAdmissionError> {
    if request
        .fallback
        .is_some_and(|backend| backend != FemBackend::Cpu)
    {
        return Err(FemAdmissionError::InvalidRequest(
            "only an explicit CPU fallback is supported".into(),
        ));
    }

    if request.requested == FemBackend::Cpu {
        return Ok(FemAdmission {
            workload,
            provenance: FemBackendProvenance::native(FemBackend::Cpu, FemActualBackend::Cpu),
        });
    }

    if cuda_supports(workload) {
        return Ok(FemAdmission {
            workload,
            provenance: FemBackendProvenance::native(FemBackend::Cuda, FemActualBackend::Cuda),
        });
    }

    let reason = cuda_unsupported_reason(workload);
    if request.fallback == Some(FemBackend::Cpu) {
        return Ok(FemAdmission {
            workload,
            provenance: FemBackendProvenance {
                requested_backend: FemBackend::Cuda,
                actual_backend: FemActualBackend::Cpu,
                fallback: "cpu".into(),
                fallback_reason: format!("CUDA workload unsupported: {reason}"),
            },
        });
    }

    Err(FemAdmissionError::Unsupported {
        workload,
        requested_backend: FemBackend::Cuda,
        actual_backend: FemActualBackend::NotExecuted,
        fallback: "none".into(),
        fallback_reason: format!("rejected before assembly; no fallback requested; {reason}"),
    })
}

/// Convenience wrapper for required CUDA workloads.
pub fn admit_fem_cuda(workload: FemWorkload) -> Result<FemAdmission, FemAdmissionError> {
    admit_fem_workload(workload, FemBackendRequest::required(FemBackend::Cuda))
}

fn cuda_supports(workload: FemWorkload) -> bool {
    let acoustic = workload.space == FemSpace::H1
        && workload.element == FemElement::Tri3
        && workload.order == FemOrder::P1
        && workload.physics == FemPhysics::Helmholtz
        && workload.mode == FemAssemblyMode::Assembled
        && matches!(workload.scalar, FemScalar::F64 | FemScalar::Complex64);
    let acoustic_tet = workload.space == FemSpace::H1
        && workload.element == FemElement::Tetrahedron
        && matches!(workload.order, FemOrder::P1 | FemOrder::P2)
        && workload.physics == FemPhysics::Helmholtz
        && workload.mode == FemAssemblyMode::Assembled
        && workload.scalar == FemScalar::Complex64;
    let acoustic_hex = workload.space == FemSpace::H1
        && workload.element == FemElement::Hexahedron
        && matches!(workload.order, FemOrder::P1 | FemOrder::P2 | FemOrder::P3)
        && workload.physics == FemPhysics::Helmholtz
        && workload.mode == FemAssemblyMode::Assembled
        && workload.scalar == FemScalar::Complex64;
    let acoustic_matrix_free = workload.space == FemSpace::H1
        && workload.element == FemElement::Tri3
        && workload.order == FemOrder::P1
        && workload.physics == FemPhysics::Helmholtz
        && workload.mode == FemAssemblyMode::MatrixFree
        && matches!(workload.scalar, FemScalar::F64 | FemScalar::Complex64)
        && matches!(
            workload.action,
            FemAction::Forward | FemAction::Transpose | FemAction::Hermitian
        );
    let acoustic_p2 = workload.space == FemSpace::H1
        && workload.element == FemElement::Tri3
        && workload.order == FemOrder::P2
        && workload.physics == FemPhysics::Helmholtz
        && workload.mode == FemAssemblyMode::Assembled
        && matches!(workload.scalar, FemScalar::F64 | FemScalar::Complex64);
    let structural = workload.space == FemSpace::H1
        && workload.element == FemElement::Tetrahedron
        && workload.order == FemOrder::P1
        && workload.physics == FemPhysics::LinearElasticity
        && workload.mode == FemAssemblyMode::Assembled
        && workload.scalar == FemScalar::F64
        && matches!(
            workload.action,
            FemAction::Forward | FemAction::Transpose | FemAction::Hermitian
        );
    let maxwell = workload.space == FemSpace::Hcurl
        && matches!(workload.element, FemElement::Tri3 | FemElement::Tetrahedron)
        && workload.order == FemOrder::P1
        && workload.physics == FemPhysics::Maxwell
        && workload.mode == FemAssemblyMode::Assembled
        && workload.scalar == FemScalar::Complex64
        && matches!(
            workload.action,
            FemAction::Forward | FemAction::Transpose | FemAction::Hermitian
        );
    let stokes = workload.space == FemSpace::H1
        && workload.element == FemElement::Tri3
        && workload.order == FemOrder::P2
        && workload.physics == FemPhysics::Stokes
        && workload.mode == FemAssemblyMode::Assembled
        && workload.scalar == FemScalar::F64
        && matches!(
            workload.action,
            FemAction::Forward | FemAction::Transpose | FemAction::Hermitian
        );
    let scalar_diffusion = matches!(
        workload.physics,
        FemPhysics::HeatConduction
            | FemPhysics::MagneticScalarPotential
            | FemPhysics::ElectrostaticPotential
    ) && workload.space == FemSpace::H1
        && matches!(workload.element, FemElement::Tri3 | FemElement::Tetrahedron)
        && matches!(workload.order, FemOrder::P1 | FemOrder::P2)
        && workload.mode == FemAssemblyMode::Assembled
        && workload.scalar == FemScalar::F64;
    scalar_diffusion
        || acoustic
        || acoustic_tet
        || acoustic_hex
        || acoustic_p2
        || acoustic_matrix_free
        || structural
        || maxwell
        || stokes
}

fn cuda_unsupported_reason(workload: FemWorkload) -> &'static str {
    if workload.physics == FemPhysics::ElectrostaticPotential {
        return "linear electrostatic-potential CUDA requires assembled f64 H1 Tri3/tetrahedron P1/P2; nonlinear, coupled and other representations are not admitted";
    }
    if workload.physics == FemPhysics::MagneticScalarPotential {
        return "linear scalar magnetic-potential CUDA requires assembled f64 H1 Tri3/tetrahedron P1/P2; other formulations, spaces, scalars, orders and modes are not admitted";
    }
    if workload.physics == FemPhysics::HeatConduction {
        return "steady heat-conduction CUDA requires assembled f64 H1 Tri3/tetrahedron P1/P2; other spaces, scalars, orders and modes are not admitted";
    }
    if workload.physics == FemPhysics::Helmholtz && workload.element == FemElement::Hexahedron {
        if workload.space != FemSpace::H1 {
            return "hexahedral Helmholtz CUDA requires H1 space";
        }
        if workload.scalar != FemScalar::Complex64 {
            return "assembled hexahedral Helmholtz CUDA is bounded to Complex64";
        }
        if !matches!(workload.order, FemOrder::P1 | FemOrder::P2 | FemOrder::P3) {
            return "assembled hexahedral Helmholtz CUDA requires explicit Q1/Q2/Q3 order";
        }
        return "generic hexahedral Helmholtz CUDA requires an assembled operator; specialized partial-assembly producers have separate contracts";
    }
    if workload.physics == FemPhysics::Helmholtz && workload.element == FemElement::Tetrahedron {
        if workload.space != FemSpace::H1 {
            return "tetrahedral Helmholtz CUDA requires H1 space";
        }
        if workload.scalar != FemScalar::Complex64 {
            return "assembled tetrahedral Helmholtz CUDA is bounded to Complex64";
        }
        if !matches!(workload.order, FemOrder::P1 | FemOrder::P2) {
            return "assembled tetrahedral Helmholtz CUDA is bounded to P1/P2";
        }
        return "generic tetrahedral Helmholtz CUDA requires an assembled operator; specialized element producers have separate contracts";
    }
    if workload.physics == FemPhysics::Maxwell {
        if workload.scalar != FemScalar::Complex64 {
            return "Maxwell CUDA is bounded to Complex64";
        }
        if workload.space != FemSpace::Hcurl {
            return "Maxwell CUDA requires H(curl) space";
        }
        if !matches!(workload.element, FemElement::Tri3 | FemElement::Tetrahedron) {
            return "Maxwell CUDA supports Tri3 and tetrahedron assembled operators";
        }
        if workload.order != FemOrder::P1 {
            return "Maxwell CUDA supports lowest-order edge elements";
        }
        return "Maxwell CUDA requires assembled operators";
    }
    if !matches!(workload.scalar, FemScalar::F64 | FemScalar::Complex64) {
        return "requested scalar path is not implemented by the CUDA FEM entrypoint";
    }
    if workload.space != FemSpace::H1 {
        return "requested FEM function-space dispatcher is not implemented by CUDA";
    }
    if workload.element != FemElement::Tri3 {
        return "requested CUDA element assembly kernel is not implemented";
    }
    if !matches!(workload.order, FemOrder::P1 | FemOrder::P2) {
        return "higher-order CUDA element assembly kernel is not implemented";
    }
    if workload.physics == FemPhysics::LinearElasticity {
        if workload.element != FemElement::Tetrahedron {
            return "linear-elasticity CUDA is bounded to tetrahedron P1";
        }
        if workload.scalar != FemScalar::F64 {
            return "linear-elasticity CUDA is bounded to f64";
        }
    } else if !matches!(
        workload.physics,
        FemPhysics::Helmholtz | FemPhysics::Maxwell | FemPhysics::Stokes
    ) {
        return "requested FEM physics form is not wired to CUDA";
    }
    if workload.mode != FemAssemblyMode::Assembled {
        return "partial-assembly and matrix-free CUDA kernels are not implemented";
    }
    "requested CUDA FEM action is not implemented"
}

/// Status of one row in a serialized CUDA capability matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FemCapabilityStatus {
    Native,
    NativeRealEquivalent,
    UnsupportedBeforeAssembly,
}

/// Machine-readable capability row shared by evidence producers and strict
/// checkers.  Every row carries dispatch and fallback provenance, including
/// unsupported rows whose actual backend is `none`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FemCapabilityRow {
    pub id: String,
    pub status: FemCapabilityStatus,
    pub backend: FemBackend,
    pub mode: FemAssemblyMode,
    pub dispatch: bool,
    pub requested_backend: FemBackend,
    pub actual_backend: FemActualBackend,
    pub fallback: String,
    pub fallback_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evidence: Option<String>,
}

fn native_row(
    workload: FemWorkload,
    status: FemCapabilityStatus,
    evidence: &'static str,
) -> FemCapabilityRow {
    FemCapabilityRow {
        id: workload.id(),
        status,
        backend: FemBackend::Cuda,
        mode: workload.mode,
        dispatch: true,
        requested_backend: FemBackend::Cuda,
        actual_backend: FemActualBackend::Cuda,
        fallback: "none".into(),
        fallback_reason: "not_applicable".into(),
        reason: None,
        evidence: Some(evidence.into()),
    }
}

fn unsupported_row(id: &'static str, reason: &'static str) -> FemCapabilityRow {
    FemCapabilityRow {
        id: id.into(),
        status: FemCapabilityStatus::UnsupportedBeforeAssembly,
        backend: FemBackend::Cuda,
        mode: FemAssemblyMode::Assembled,
        dispatch: false,
        requested_backend: FemBackend::Cuda,
        actual_backend: FemActualBackend::NotExecuted,
        fallback: "none".into(),
        fallback_reason: "no fallback requested; rejected before assembly".into(),
        reason: Some(reason.into()),
        evidence: None,
    }
}

fn unsupported_workload_row(workload: FemWorkload, reason: &'static str) -> FemCapabilityRow {
    FemCapabilityRow {
        id: workload.id(),
        status: FemCapabilityStatus::UnsupportedBeforeAssembly,
        backend: FemBackend::Cuda,
        mode: workload.mode,
        dispatch: false,
        requested_backend: FemBackend::Cuda,
        actual_backend: FemActualBackend::NotExecuted,
        fallback: "none".into(),
        fallback_reason: "no fallback requested; rejected before assembly".into(),
        reason: Some(reason.into()),
        evidence: None,
    }
}

/// Return the bounded CUDA matrix used by the FEM1 artifact.
pub fn fem_cuda_capability_rows() -> Vec<FemCapabilityRow> {
    let mut rows = Vec::with_capacity(44);
    for scalar in [FemScalar::F64, FemScalar::Complex64] {
        rows.push(native_row(
            FemWorkload::native_cuda(scalar, FemAction::Forward),
            FemCapabilityStatus::Native,
            "production DeviceCsr forward action",
        ));
        rows.push(native_row(
            FemWorkload::native_cuda(scalar, FemAction::Hermitian),
            FemCapabilityStatus::Native,
            "production AdjointDeviceCsr Hermitian action",
        ));
    }
    rows.push(native_row(
        FemWorkload::native_cuda(FemScalar::F64, FemAction::Transpose),
        FemCapabilityStatus::NativeRealEquivalent,
        "real f64 Hermitian action is transpose-equivalent",
    ));
    for scalar in [FemScalar::F64, FemScalar::Complex64] {
        rows.push(native_row(
            FemWorkload::native_cuda_p2(scalar, FemAction::Forward),
            FemCapabilityStatus::Native,
            "production P2 Helmholtz assembly plus DeviceCsr forward action",
        ));
        rows.push(native_row(
            FemWorkload::native_cuda_p2(scalar, FemAction::Hermitian),
            FemCapabilityStatus::Native,
            "production P2 Helmholtz assembly plus AdjointDeviceCsr Hermitian action",
        ));
    }
    rows.push(native_row(
        FemWorkload::native_cuda_p2(FemScalar::F64, FemAction::Transpose),
        FemCapabilityStatus::NativeRealEquivalent,
        "real f64 Hermitian action is transpose-equivalent",
    ));
    for workload in [
        FemWorkload::native_cuda(FemScalar::Complex64, FemAction::Transpose),
        FemWorkload::native_cuda_p2(FemScalar::Complex64, FemAction::Transpose),
    ] {
        rows.push(native_row(
            workload,
            FemCapabilityStatus::Native,
            "host assembly and transpose canonicalization; retained CUDA CSR solve with caller-prepared preconditioning",
        ));
    }
    for scalar in [FemScalar::F64, FemScalar::Complex64] {
        for action in [
            FemAction::Forward,
            FemAction::Transpose,
            FemAction::Hermitian,
        ] {
            rows.push(native_row(
                FemWorkload::native_cuda_matrix_free(scalar, action),
                if scalar == FemScalar::F64 && action == FemAction::Transpose {
                    FemCapabilityStatus::NativeRealEquivalent
                } else {
                    FemCapabilityStatus::Native
                },
                "production element assembly plus native CUDA batched local contraction",
            ));
        }
    }
    for order in [FemOrder::P1, FemOrder::P2] {
        for action in [
            FemAction::Forward,
            FemAction::Transpose,
            FemAction::Hermitian,
        ] {
            rows.push(native_row(
                FemWorkload::native_cuda_tet(order, action),
                FemCapabilityStatus::Native,
                "host tetrahedral Helmholtz assembly and constraints; retained CUDA CSR solve with caller-prepared preconditioning",
            ));
        }
    }
    for order in [FemOrder::P1, FemOrder::P2, FemOrder::P3] {
        for action in [
            FemAction::Forward,
            FemAction::Transpose,
            FemAction::Hermitian,
        ] {
            rows.push(native_row(
                FemWorkload::native_cuda_hex(order, action),
                FemCapabilityStatus::Native,
                "host hexahedral Helmholtz assembly and constraints; retained CUDA CSR solve with LOR auxiliary AMG",
            ));
        }
    }
    for (id, reason) in [
        (
            "h1.quad.p1.helmholtz.assembled",
            "CUDA element assembly kernel is not implemented",
        ),
        (
            "h1.tetrahedron.p1.helmholtz.assembled.f64",
            "generic tetrahedral Helmholtz CUDA is bounded to Complex64",
        ),
        (
            "h1.hexahedron.p1.helmholtz.assembled.f64",
            "generic hexahedral Helmholtz CUDA is bounded to Complex64",
        ),
        (
            "h1.tri3.p2-or-higher.helmholtz.assembled",
            "higher-order CUDA element assembly kernel is not implemented",
        ),
        (
            "h1.tri3.p1.other-form.assembled",
            "only the production Helmholtz form is wired to CUDA",
        ),
        (
            "h1.tri3.p1.helmholtz.partial-assembly",
            "partial-assembly CUDA kernel is not implemented",
        ),
        (
            "hcurl.tri3.p1.helmholtz.assembled",
            "H(curl) CUDA FEM dispatcher is not implemented",
        ),
        (
            "hdiv.tri3.p1.helmholtz.assembled",
            "H(div) CUDA FEM dispatcher is not implemented",
        ),
        (
            "dg.tri3.p1.helmholtz.assembled",
            "DG CUDA FEM dispatcher is not implemented",
        ),
        (
            "h1.tri3.p1.helmholtz.assembled.f32",
            "f32 CUDA FEM scalar path is not implemented",
        ),
        (
            "h1.tri3.p1.helmholtz.assembled.Complex32",
            "Complex32 CUDA FEM scalar path is not implemented",
        ),
    ] {
        rows.push(unsupported_row(id, reason));
    }
    rows
}

/// Return bounded linear scalar magnetostatic CUDA solve coverage.
pub fn fem_cuda_magnetic_scalar_capability_rows() -> Vec<FemCapabilityRow> {
    let mut rows = Vec::with_capacity(12);
    for element in [FemElement::Tri3, FemElement::Tetrahedron] {
        for order in [FemOrder::P1, FemOrder::P2] {
            for action in [
                FemAction::Forward,
                FemAction::Transpose,
                FemAction::Hermitian,
            ] {
                rows.push(native_row(
                    FemWorkload::native_cuda_magnetic_scalar_potential(element, order, action),
                    FemCapabilityStatus::Native,
                    "host permeability/remanence assembly and constraints; retained CUDA PCG with symmetric AMG; layered potential/flux and repeated-source tests",
                ));
            }
        }
    }
    rows
}

/// Return the steady heat-conduction CUDA solve coverage separately from acoustics.
pub fn fem_cuda_thermal_capability_rows() -> Vec<FemCapabilityRow> {
    let mut rows = Vec::with_capacity(12);
    for element in [FemElement::Tri3, FemElement::Tetrahedron] {
        for order in [FemOrder::P1, FemOrder::P2] {
            for action in [
                FemAction::Forward,
                FemAction::Transpose,
                FemAction::Hermitian,
            ] {
                rows.push(native_row(
                    FemWorkload::native_cuda_heat_conduction(element, order, action),
                    FemCapabilityStatus::Native,
                    "host conduction assembly and constraints; retained CUDA PCG with symmetric AMG; analytical temperature/source and numeric-refresh tests",
                ));
            }
        }
    }
    rows
}

/// Return the separate bounded CUDA matrix for the native 3-D structural
/// action cell.  Keeping this matrix separate prevents an acoustic producer
/// from accidentally claiming that it exercised structural physics.
pub fn fem_cuda_structural_capability_rows() -> Vec<FemCapabilityRow> {
    let mut rows = Vec::with_capacity(10);
    rows.push(native_row(
        FemWorkload::native_cuda_structural(FemScalar::F64, FemAction::Forward),
        FemCapabilityStatus::Native,
        "production ElasticityForm assembly plus DeviceCsr forward action",
    ));
    rows.push(native_row(
        FemWorkload::native_cuda_structural(FemScalar::F64, FemAction::Hermitian),
        FemCapabilityStatus::Native,
        "production ElasticityForm assembly plus AdjointDeviceCsr Hermitian action",
    ));
    rows.push(native_row(
        FemWorkload::native_cuda_structural(FemScalar::F64, FemAction::Transpose),
        FemCapabilityStatus::NativeRealEquivalent,
        "real f64 Hermitian action is transpose-equivalent",
    ));
    for (id, reason) in [
        (
            "h1.tetrahedron.p1.linear-elasticity.assembled.Complex64.forward",
            "linear-elasticity CUDA cell is bounded to f64",
        ),
        (
            "h1.tetrahedron.p1.linear-elasticity.assembled.Complex64.hermitian",
            "linear-elasticity CUDA cell is bounded to f64",
        ),
        (
            "h1.tetrahedron.p2-or-higher.linear-elasticity.assembled",
            "higher-order linear-elasticity CUDA element assembly is not implemented",
        ),
        (
            "h1.tri3.p1.linear-elasticity.assembled",
            "linear-elasticity CUDA cell is bounded to tetrahedron P1",
        ),
        (
            "h1.tetrahedron.p1.linear-elasticity.partial-assembly",
            "partial-assembly linear-elasticity CUDA kernel is not implemented",
        ),
        (
            "h1.tetrahedron.p1.linear-elasticity.matrix-free",
            "matrix-free linear-elasticity CUDA kernel is not implemented",
        ),
        (
            "hcurl.tetrahedron.p1.linear-elasticity.assembled",
            "linear-elasticity CUDA cell uses H1 displacement space",
        ),
    ] {
        rows.push(unsupported_row(id, reason));
    }
    rows
}

/// Return the bounded CUDA matrix for the native H(curl) Maxwell action cell.
/// The cells advertise Complex64 Tri3/Tet4 lowest-order assembled actions;
/// geometry/quadrature kernels and other Maxwell element families
/// remain fail-closed before assembly.
pub fn fem_cuda_maxwell_capability_rows() -> Vec<FemCapabilityRow> {
    let mut rows = Vec::with_capacity(13);
    for action in [
        FemAction::Forward,
        FemAction::Transpose,
        FemAction::Hermitian,
    ] {
        for element in [FemElement::Tri3, FemElement::Tetrahedron] {
            rows.push(native_row(
                FemWorkload {
                    element,
                    ..FemWorkload::native_cuda_maxwell(FemScalar::Complex64, action)
                },
                FemCapabilityStatus::Native,
                "host MaxwellForm assembly plus retained CUDA CSR and auxiliary-space solve",
            ));
        }
    }
    for (workload, reason) in [
        (
            FemWorkload {
                space: FemSpace::H1,
                ..FemWorkload::native_cuda_maxwell(FemScalar::Complex64, FemAction::Forward)
            },
            "Maxwell CUDA cell is bounded to H(curl) space",
        ),
        (
            FemWorkload {
                element: FemElement::Quad,
                ..FemWorkload::native_cuda_maxwell(FemScalar::Complex64, FemAction::Forward)
            },
            "Maxwell CUDA cells are bounded to Tri3 and tetrahedron",
        ),
        (
            FemWorkload {
                order: FemOrder::P2OrHigher,
                ..FemWorkload::native_cuda_maxwell(FemScalar::Complex64, FemAction::Forward)
            },
            "higher-order Maxwell CUDA element assembly is not implemented",
        ),
        (
            FemWorkload {
                scalar: FemScalar::F64,
                ..FemWorkload::native_cuda_maxwell(FemScalar::F64, FemAction::Forward)
            },
            "Maxwell CUDA cell is bounded to Complex64",
        ),
        (
            FemWorkload {
                scalar: FemScalar::Complex32,
                ..FemWorkload::native_cuda_maxwell(FemScalar::Complex32, FemAction::Forward)
            },
            "Complex32 Maxwell CUDA scalar path is not implemented",
        ),
        (
            FemWorkload {
                mode: FemAssemblyMode::PartialAssembly,
                ..FemWorkload::native_cuda_maxwell(FemScalar::Complex64, FemAction::Forward)
            },
            "partial-assembly Maxwell CUDA kernel is not implemented",
        ),
        (
            FemWorkload {
                mode: FemAssemblyMode::MatrixFree,
                ..FemWorkload::native_cuda_maxwell(FemScalar::Complex64, FemAction::Forward)
            },
            "matrix-free Maxwell CUDA kernel is not implemented",
        ),
    ] {
        rows.push(unsupported_workload_row(workload, reason));
    }
    rows
}

/// Return the bounded CUDA matrix for the native Taylor--Hood Stokes action
/// cell.  The matrix is intentionally separate from the scalar acoustics
/// rows: this cell proves a production mixed CFD assembly flattened to CSR,
/// not a general FEM physics dispatcher or nonlinear solve.
pub fn fem_cuda_stokes_capability_rows() -> Vec<FemCapabilityRow> {
    let mut rows = Vec::with_capacity(10);
    for action in [
        FemAction::Forward,
        FemAction::Transpose,
        FemAction::Hermitian,
    ] {
        rows.push(native_row(
            FemWorkload::native_cuda_stokes(action),
            if action == FemAction::Transpose {
                FemCapabilityStatus::NativeRealEquivalent
            } else {
                FemCapabilityStatus::Native
            },
            "production Taylor-Hood Stokes mixed assembly plus native CUDA CSR action",
        ));
    }
    for (workload, reason) in [
        (
            FemWorkload {
                order: FemOrder::P1,
                ..FemWorkload::native_cuda_stokes(FemAction::Forward)
            },
            "Taylor-Hood Stokes CUDA cell is bounded to P2/P1",
        ),
        (
            FemWorkload {
                order: FemOrder::P2OrHigher,
                ..FemWorkload::native_cuda_stokes(FemAction::Forward)
            },
            "higher-order Stokes CUDA element assembly is not implemented",
        ),
        (
            FemWorkload {
                scalar: FemScalar::Complex64,
                ..FemWorkload::native_cuda_stokes(FemAction::Forward)
            },
            "Stokes CUDA cell is bounded to f64",
        ),
        (
            FemWorkload {
                mode: FemAssemblyMode::PartialAssembly,
                ..FemWorkload::native_cuda_stokes(FemAction::Forward)
            },
            "partial-assembly Stokes CUDA kernel is not implemented",
        ),
        (
            FemWorkload {
                mode: FemAssemblyMode::MatrixFree,
                ..FemWorkload::native_cuda_stokes(FemAction::Forward)
            },
            "matrix-free Stokes CUDA kernel is not implemented",
        ),
        (
            FemWorkload {
                element: FemElement::Quad,
                ..FemWorkload::native_cuda_stokes(FemAction::Forward)
            },
            "Stokes CUDA cell is bounded to Tri3",
        ),
        (
            FemWorkload {
                physics: FemPhysics::Other,
                ..FemWorkload::native_cuda_stokes(FemAction::Forward)
            },
            "requested FEM physics form is not wired to CUDA",
        ),
    ] {
        rows.push(unsupported_workload_row(workload, reason));
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_cell_is_admitted_without_fallback() {
        let admission = admit_fem_cuda(FemWorkload::native_cuda(
            FemScalar::Complex64,
            FemAction::Hermitian,
        ))
        .expect("native CUDA cell should be admitted");
        assert_eq!(admission.provenance.actual_backend, FemActualBackend::Cuda);
        assert_eq!(admission.provenance.fallback, "none");
        assert_eq!(admission.provenance.fallback_reason, "not_applicable");
    }

    #[test]
    fn native_p2_cell_is_admitted_without_fallback() {
        let admission = admit_fem_cuda(FemWorkload::native_cuda_p2(
            FemScalar::Complex64,
            FemAction::Hermitian,
        ))
        .expect("native P2 CUDA cell should be admitted");
        assert_eq!(admission.provenance.actual_backend, FemActualBackend::Cuda);
        assert_eq!(admission.provenance.fallback, "none");
        assert_eq!(admission.workload.order, FemOrder::P2);
    }

    #[test]
    fn native_structural_cell_is_admitted_without_fallback() {
        let admission = admit_fem_cuda(FemWorkload::native_cuda_structural(
            FemScalar::F64,
            FemAction::Hermitian,
        ))
        .expect("native structural CUDA cell should be admitted");
        assert_eq!(admission.provenance.actual_backend, FemActualBackend::Cuda);
        assert_eq!(admission.provenance.fallback, "none");
    }

    #[test]
    fn native_stokes_cell_is_admitted_without_fallback() {
        let admission = admit_fem_cuda(FemWorkload::native_cuda_stokes(FemAction::Hermitian))
            .expect("native Stokes CUDA cell should be admitted");
        assert_eq!(admission.provenance.actual_backend, FemActualBackend::Cuda);
        assert_eq!(admission.provenance.fallback, "none");
        assert_eq!(admission.workload.physics, FemPhysics::Stokes);
    }

    #[test]
    fn native_matrix_free_cell_is_admitted_without_fallback() {
        let admission = admit_fem_cuda(FemWorkload::native_cuda_matrix_free(
            FemScalar::Complex64,
            FemAction::Transpose,
        ))
        .expect("native matrix-free CUDA cell should be admitted");
        assert_eq!(admission.provenance.actual_backend, FemActualBackend::Cuda);
        assert_eq!(admission.provenance.fallback, "none");
        assert_eq!(admission.workload.mode, FemAssemblyMode::MatrixFree);
    }

    #[test]
    fn unsupported_cell_is_rejected_before_assembly_with_provenance() {
        let workload = FemWorkload {
            scalar: FemScalar::F32,
            ..FemWorkload::native_cuda(FemScalar::F64, FemAction::Forward)
        };
        let error = admit_fem_cuda(workload).expect_err("f32 CUDA must fail closed");
        match error {
            FemAdmissionError::Unsupported {
                requested_backend,
                actual_backend,
                fallback,
                fallback_reason,
                ..
            } => {
                assert_eq!(requested_backend, FemBackend::Cuda);
                assert_eq!(actual_backend, FemActualBackend::NotExecuted);
                assert_eq!(fallback, "none");
                assert!(fallback_reason.contains("scalar"));
                assert!(fallback_reason.contains("no fallback"));
            }
            other => panic!("unexpected admission result: {other:?}"),
        }
    }

    #[test]
    fn explicit_cpu_fallback_is_visible_and_does_not_claim_cuda_execution() {
        let workload = FemWorkload {
            element: FemElement::Quad,
            ..FemWorkload::native_cuda(FemScalar::F64, FemAction::Forward)
        };
        let admission = admit_fem_workload(workload, FemBackendRequest::with_cpu_fallback())
            .expect("explicit CPU fallback should be admitted");
        assert_eq!(admission.provenance.actual_backend, FemActualBackend::Cpu);
        assert_eq!(admission.provenance.fallback, "cpu");
        assert!(admission.provenance.fallback_reason.contains("unsupported"));
    }

    #[test]
    fn capability_rows_carry_requested_actual_and_fallback_fields() {
        let rows = fem_cuda_capability_rows();
        assert_eq!(rows.len(), 44);
        assert!(
            rows.iter()
                .all(|row| row.requested_backend == FemBackend::Cuda)
        );
        assert!(rows.iter().any(|row| {
            row.status == FemCapabilityStatus::UnsupportedBeforeAssembly
                && row.actual_backend == FemActualBackend::NotExecuted
                && !row.dispatch
                && row.fallback == "none"
                && row.fallback_reason.contains("before assembly")
        }));
        let unsupported = rows
            .iter()
            .find(|row| row.status == FemCapabilityStatus::UnsupportedBeforeAssembly)
            .expect("matrix has unsupported rows");
        let json = serde_json::to_value(unsupported).expect("row serializes");
        assert_eq!(json["requested_backend"], "cuda");
        assert_eq!(json["actual_backend"], "none");
        assert_eq!(json["fallback"], "none");
        assert!(
            json["fallback_reason"]
                .as_str()
                .is_some_and(|reason| reason.contains("before assembly"))
        );
    }

    #[test]
    fn structural_capability_rows_are_bounded_and_fail_closed() {
        let rows = fem_cuda_structural_capability_rows();
        assert_eq!(rows.len(), 10);
        assert_eq!(
            rows.iter().filter(|row| row.dispatch).count(),
            3,
            "only f64 forward/transpose/Hermitian are native"
        );
        assert!(rows.iter().all(|row| {
            row.requested_backend == FemBackend::Cuda
                && row.fallback == "none"
                && (row.dispatch || row.fallback_reason.contains("before assembly"))
        }));
    }

    #[test]
    fn maxwell_capability_rows_are_bounded_and_fail_closed() {
        let rows = fem_cuda_maxwell_capability_rows();
        assert_eq!(rows.len(), 13);
        assert_eq!(rows.iter().filter(|row| row.dispatch).count(), 6);
        assert!(rows.iter().all(|row| {
            row.requested_backend == FemBackend::Cuda
                && row.fallback == "none"
                && (row.dispatch || row.fallback_reason.contains("before assembly"))
        }));
        for action in [
            FemAction::Forward,
            FemAction::Transpose,
            FemAction::Hermitian,
        ] {
            for element in [FemElement::Tri3, FemElement::Tetrahedron] {
                let workload = FemWorkload {
                    element,
                    ..FemWorkload::native_cuda_maxwell(FemScalar::Complex64, action)
                };
                let admission =
                    admit_fem_cuda(workload).expect("bounded Maxwell action should be admitted");
                assert_eq!(admission.provenance.actual_backend, FemActualBackend::Cuda);
                assert_eq!(admission.provenance.fallback, "none");
                for rejected in [
                    FemWorkload {
                        mode: FemAssemblyMode::MatrixFree,
                        ..workload
                    },
                    FemWorkload {
                        order: FemOrder::P2OrHigher,
                        ..workload
                    },
                    FemWorkload {
                        scalar: FemScalar::F64,
                        ..workload
                    },
                ] {
                    assert!(admit_fem_cuda(rejected).is_err());
                }
            }
        }
    }

    #[test]
    fn stokes_capability_rows_are_bounded_and_fail_closed() {
        let rows = fem_cuda_stokes_capability_rows();
        assert_eq!(rows.len(), 10);
        assert_eq!(rows.iter().filter(|row| row.dispatch).count(), 3);
        assert!(rows.iter().all(|row| {
            row.requested_backend == FemBackend::Cuda
                && row.fallback == "none"
                && (row.dispatch || row.fallback_reason.contains("before assembly"))
        }));
    }
    #[test]
    fn thermal_capability_rows_match_bounded_admission() {
        let rows = fem_cuda_thermal_capability_rows();
        assert_eq!(rows.len(), 12);
        let unique: std::collections::HashSet<_> = rows.iter().map(|r| &r.id).collect();
        assert_eq!(unique.len(), rows.len());
        for element in [FemElement::Tri3, FemElement::Tetrahedron] {
            for order in [FemOrder::P1, FemOrder::P2] {
                for action in [
                    FemAction::Forward,
                    FemAction::Transpose,
                    FemAction::Hermitian,
                ] {
                    let workload = FemWorkload::native_cuda_heat_conduction(element, order, action);
                    assert!(admit_fem_cuda(workload).is_ok());
                    assert!(rows.iter().any(|r| r.id == workload.id() && r.dispatch));
                }
            }
        }
        assert!(
            rows.iter()
                .all(|r| r.fallback == "none" && r.evidence.is_some())
        );
    }

    #[test]
    fn magnetic_scalar_capability_rows_match_bounded_admission() {
        let rows = fem_cuda_magnetic_scalar_capability_rows();
        assert_eq!(rows.len(), 12);
        let unique: std::collections::HashSet<_> = rows.iter().map(|r| &r.id).collect();
        assert_eq!(unique.len(), rows.len());
        for element in [FemElement::Tri3, FemElement::Tetrahedron] {
            for order in [FemOrder::P1, FemOrder::P2] {
                for action in [
                    FemAction::Forward,
                    FemAction::Transpose,
                    FemAction::Hermitian,
                ] {
                    let workload =
                        FemWorkload::native_cuda_magnetic_scalar_potential(element, order, action);
                    assert!(admit_fem_cuda(workload).is_ok());
                    assert!(rows.iter().any(|r| r.id == workload.id() && r.dispatch));
                }
            }
        }
        assert!(
            rows.iter()
                .all(|r| r.fallback == "none" && r.evidence.is_some())
        );
    }
}
