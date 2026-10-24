//! Retained CUDA Maxwell solves with additive nodal auxiliary-space corrections.
//!
//! This route supports affine Tri3/Tet4, lowest-order edge elements, constant
//! coefficients and homogeneous PEC on the entire exterior boundary. Assembly
//! and AMG hierarchy setup run on the host. Edge smoothing, restriction, AMG,
//! prolongation and FGMRES run on CUDA with retained storage.
//!
//! The positive auxiliary operator is `alpha curl curl + beta I`, beta > 0.
//! Its additive inverse approximation uses edge Jacobi, a scalar nodal solve
//! for `beta Laplace`, and vector nodal solves for `alpha Laplace + beta I`.
//! The physical complex operator may be indefinite; use FGMRES, not PCG.
//! This is not a general AMS implementation or a mesh-independent convergence
//! guarantee for resonant, multiply connected, heterogeneous or PML problems.

// Rust guideline compliant 2026-02-21
use super::{MaxwellCurrentSourceForm, MaxwellForm};
use crate::assembly::{TripletMatrix, assemble_bilinear_checked, assemble_linear_checked};
use crate::backend::{FemAction, FemElement, FemScalar, FemWorkload};
use crate::basis::PolynomialDegree;
use crate::boundary::generic::{FemSystem, try_apply_homogeneous_dirichlet_complex_dofs};
use crate::forms::{LaplacianForm, MassForm};
use crate::mesh::{ElementType, Mesh, topology::MeshTopology};
use crate::solver::device::{CudaFemSolveSession, FemDeviceMethod, FemDeviceSolveError};
use crate::spaces::{H1Space, HcurlSpace};
use num_complex::Complex64;
use sonium_compute::preconditioner::PointJacobiBackend;
use sonium_compute::{
    Backend, BackendError, BackendResult, CudaBackend, CudaBuffer, DeviceBufferLike,
    DevicePreconditioner, MemoryPool, PersistentDeviceCsr,
};
use sonium_solvers::{AmgConfig, CsrMatrix, preconditioners::BackendAmgPreconditioner};
use std::sync::Mutex;

type C64 = Complex64;

/// Parameters for normalized Maxwell assembly and its positive auxiliary operator.
#[derive(Debug, Clone, Copy)]
pub struct MaxwellCudaConfig {
    /// Positive coefficient of the physical and auxiliary curl-curl term.
    pub curl_coefficient: f64,
    /// Signed complex coefficient of the physical edge mass term.
    pub mass_coefficient: C64,
    /// Positive mass coefficient used only for preconditioning.
    pub auxiliary_mass: f64,
}

/// A retained Maxwell solve with homogeneous exterior PEC constraints.
pub type MaxwellCudaSession<'a> = CudaFemSolveSession<'a, C64, MaxwellAuxiliaryCuda>;

fn invalid(reason: impl Into<String>) -> BackendError {
    BackendError::InvalidArgument {
        operation: "maxwell_cuda",
        reason: reason.into(),
    }
}

fn setup_error(error: impl std::fmt::Display) -> BackendError {
    invalid(error.to_string())
}

/// Assemble a PEC Maxwell problem and prepare its retained CUDA solver.
///
/// `current` supplies J in the normalized equation
/// `alpha curl curl E + mass E = -i J` (omega = 1). All exterior edges receive
/// homogeneous tangential data, independently of boundary marker labels.
/// The returned session accepts device RHS updates and explicit preconditioner
/// replacement. Rebuild for changed geometry, constraints or coefficients;
/// generic lagged updates are available only as an explicit caller decision.
///
/// # Errors
/// Rejects non-affine/non-simplex meshes, missing interior nodes, invalid positive
/// coefficients, non-finite data, invalid connectivity and backend failures.
pub fn try_prepare_cuda<'a>(
    pool: &'a MemoryPool<CudaBackend>,
    mesh: &Mesh,
    config: MaxwellCudaConfig,
    current: impl Fn(f64, f64, f64) -> [C64; 3] + Send + Sync,
) -> Result<MaxwellCudaSession<'a>, FemDeviceSolveError> {
    try_prepare_cuda_action(pool, mesh, config, FemAction::Forward, current)
}

/// Prepare a forward, transpose or Hermitian PEC Maxwell solve.
///
/// `current` supplies the load for the requested action: the assembled RHS is
/// always `-i J`. For a Hermitian problem the caller supplies the adjoint load;
/// the matrix is conjugate-transposed by the shared session. The positive
/// auxiliary operator is unchanged. Geometry and coefficients are frozen at setup.
///
/// # Errors
/// Rejects unsupported meshes, invalid coefficients, assembly and backend failures.
pub fn try_prepare_cuda_action<'a>(
    pool: &'a MemoryPool<CudaBackend>,
    mesh: &Mesh,
    config: MaxwellCudaConfig,
    action: FemAction,
    current: impl Fn(f64, f64, f64) -> [C64; 3] + Send + Sync,
) -> Result<MaxwellCudaSession<'a>, FemDeviceSolveError> {
    let prepared = prepare(mesh, config)?;
    let space = HcurlSpace::new();
    let matrix = assemble_bilinear_checked(
        &MaxwellForm::from_coefficients(config.curl_coefficient, config.mass_coefficient),
        &space,
        &space,
        mesh,
        4,
    )?;
    let rhs = assemble_linear_checked(
        &MaxwellCurrentSourceForm::new(1.0, current),
        &space,
        mesh,
        6,
    )?;
    let mut system = FemSystem::new(matrix, rhs);
    try_apply_homogeneous_dirichlet_complex_dofs(&mut system, &prepared.pec)?;
    CudaFemSolveSession::try_assemble(
        pool,
        FemWorkload {
            element: if mesh.dimension == 3 {
                FemElement::Tetrahedron
            } else {
                FemElement::Tri3
            },
            ..FemWorkload::native_cuda_maxwell(FemScalar::Complex64, action)
        },
        FemDeviceMethod::Fgmres,
        || Ok(system),
        |_| MaxwellAuxiliaryCuda::upload(pool.backend(), prepared),
    )
}

struct Prepared {
    pec: Vec<usize>,
    diagonal: Vec<C64>,
    gradient: CsrMatrix<C64>,
    interpolation: CsrMatrix<C64>,
    scalar: CsrMatrix<C64>,
    vector: CsrMatrix<C64>,
}

fn prepare(mesh: &Mesh, config: MaxwellCudaConfig) -> Result<Prepared, FemDeviceSolveError> {
    if !matches!(mesh.dimension, 2 | 3)
        || mesh.elements.is_empty()
        || mesh.elements.iter().any(|e| match mesh.dimension {
            2 => e.element_type != ElementType::Triangle || e.nodes.len() != 3,
            3 => e.element_type != ElementType::Tetrahedron || e.nodes.len() != 4,
            _ => true,
        })
        || !config.curl_coefficient.is_finite()
        || config.curl_coefficient <= 0.0
        || !config.auxiliary_mass.is_finite()
        || config.auxiliary_mass <= 0.0
        || !config.mass_coefficient.re.is_finite()
        || !config.mass_coefficient.im.is_finite()
    {
        return Err(invalid(
            "requires affine 2D Tri3 or 3D Tet4, finite mass and positive curl/auxiliary coefficients",
        )
        .into());
    }
    let topology =
        MeshTopology::try_build(mesh).map_err(|e| invalid(format!("invalid topology: {e:?}")))?;
    let n = topology.edges.len();
    let pec = exterior_edges(mesh.dimension, &topology)?;
    let mut constrained = vec![false; n];
    let mut boundary = vec![false; mesh.nodes.len()];
    for &edge in &pec {
        constrained[edge] = true;
        let (a, b) = topology.edges[edge];
        boundary[a] = true;
        boundary[b] = true;
    }
    let mut nodes = vec![None; mesh.nodes.len()];
    let mut free = 0;
    for (i, &is_boundary) in boundary.iter().enumerate() {
        if !is_boundary {
            nodes[i] = Some(free);
            free += 1;
        }
    }
    if free == 0 || pec.is_empty() {
        return Err(invalid("requires exterior PEC edges and interior auxiliary nodes").into());
    }
    let mut gradient = TripletMatrix::new(n, free);
    let dim = mesh.dimension;
    let mut interpolation = TripletMatrix::new(n, dim * free);
    for (edge, &(a, b)) in topology.edges.iter().enumerate() {
        if constrained[edge] {
            continue;
        }
        let delta = [
            mesh.nodes[b].x - mesh.nodes[a].x,
            mesh.nodes[b].y - mesh.nodes[a].y,
            mesh.nodes[b].z - mesh.nodes[a].z,
        ];
        for (node, sign) in [(a, -1.0), (b, 1.0)] {
            if let Some(i) = nodes[node] {
                gradient.push(edge, i, C64::new(sign, 0.0));
                // Integrate a linear nodal vector along the globally oriented edge.
                for (d, &length) in delta.iter().take(dim).enumerate() {
                    interpolation.push(edge, dim * i + d, C64::new(0.5 * length, 0.0));
                }
            }
        }
    }
    let nodal = H1Space::new(PolynomialDegree::P1);
    let stiffness = assemble_bilinear_checked(&LaplacianForm, &nodal, &nodal, mesh, 4)?;
    let mass = assemble_bilinear_checked(&MassForm, &nodal, &nodal, mesh, 4)?;
    let mut scalar = TripletMatrix::new(free, free);
    let mut vector = TripletMatrix::new(dim * free, dim * free);
    for (matrix, coefficient, is_stiffness) in [
        (&stiffness, config.curl_coefficient, true),
        (&mass, config.auxiliary_mass, false),
    ] {
        for ((&r, &c), &value) in matrix.rows.iter().zip(&matrix.cols).zip(&matrix.values) {
            if let (Some(i), Some(j)) = (nodes[r], nodes[c]) {
                if is_stiffness {
                    scalar.push(i, j, C64::new(config.auxiliary_mass * value, 0.0));
                }
                for d in 0..dim {
                    vector.push(dim * i + d, dim * j + d, C64::new(coefficient * value, 0.0));
                }
            }
        }
    }
    let edge = HcurlSpace::new();
    let mut auxiliary = FemSystem::new(
        assemble_bilinear_checked(
            &MaxwellForm::from_coefficients(
                config.curl_coefficient,
                C64::new(config.auxiliary_mass, 0.0),
            ),
            &edge,
            &edge,
            mesh,
            4,
        )?,
        vec![C64::default(); n],
    );
    try_apply_homogeneous_dirichlet_complex_dofs(&mut auxiliary, &pec)?;
    let mut diagonal = vec![C64::default(); n];
    for ((&r, &c), &value) in auxiliary
        .matrix
        .rows
        .iter()
        .zip(&auxiliary.matrix.cols)
        .zip(&auxiliary.matrix.values)
    {
        if r == c {
            diagonal[r] += value;
        }
    }
    Ok(Prepared {
        pec,
        diagonal,
        gradient: gradient.try_to_csr().map_err(setup_error)?,
        interpolation: interpolation.try_to_csr().map_err(setup_error)?,
        scalar: scalar.try_to_csr().map_err(setup_error)?,
        vector: vector.try_to_csr().map_err(setup_error)?,
    })
}

// In 3D, edge incidence cannot identify the surface: a boundary edge may
// belong to many tetrahedra. First identify exterior faces, then their edges.
fn exterior_edges(dim: usize, topology: &MeshTopology) -> BackendResult<Vec<usize>> {
    let entities = if dim == 2 {
        &topology.element_edges
    } else {
        &topology.element_faces
    };
    let count = if dim == 2 {
        topology.edges.len()
    } else {
        topology.faces.len()
    };
    let mut incidence = vec![0usize; count];
    for entity in entities {
        for &index in entity {
            incidence[index] += 1;
        }
    }
    if incidence.iter().any(|&count| count == 0 || count > 2) {
        return Err(invalid("requires manifold simplex facet connectivity"));
    }
    if dim == 2 {
        return Ok((0..count).filter(|&i| incidence[i] == 1).collect());
    }
    let mut boundary = vec![false; topology.edges.len()];
    for (face, &count) in topology.faces.iter().zip(&incidence) {
        if count != 1 {
            continue;
        }
        let crate::mesh::topology::Face::Triangle(nodes) = face else {
            return Err(invalid("requires triangular tetrahedron faces"));
        };
        for (a, b) in [
            (nodes[0], nodes[1]),
            (nodes[1], nodes[2]),
            (nodes[2], nodes[0]),
        ] {
            let edge = topology
                .edge_map
                .get(&(a.min(b), a.max(b)))
                .ok_or_else(|| invalid("face edge missing from topology"))?;
            boundary[*edge] = true;
        }
    }
    Ok(boundary
        .iter()
        .enumerate()
        .filter_map(|(i, &surface)| surface.then_some(i))
        .collect())
}

struct Correction {
    prolongation: PersistentDeviceCsr<CudaBackend, C64>,
    restriction: PersistentDeviceCsr<CudaBackend, C64>,
    amg: BackendAmgPreconditioner<CudaBackend, C64>,
    rhs: CudaBuffer<C64>,
    solution: CudaBuffer<C64>,
}

struct Workspace {
    corrections: [Correction; 2],
    result: CudaBuffer<C64>,
    correction: CudaBuffer<C64>,
}

/// Retained edge smoothing and scalar/vector nodal AMG corrections.
///
/// Only the checked forward action is exposed. It is intended for FGMRES;
/// this API does not certify a symmetric positive-definite AMG cycle.
pub struct MaxwellAuxiliaryCuda {
    diagonal: PointJacobiBackend<CudaBackend, C64>,
    workspace: Mutex<Workspace>,
    n: usize,
}

impl std::fmt::Debug for MaxwellAuxiliaryCuda {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaxwellAuxiliaryCuda")
            .field("edge_dofs", &self.n)
            .finish_non_exhaustive()
    }
}

impl MaxwellAuxiliaryCuda {
    /// Build a replacement auxiliary action for the same PEC discretization.
    ///
    /// Use with the retained session's explicit preconditioner replacement when
    /// changing the positive auxiliary coefficients. Host setup is repeated;
    /// each subsequent application reuses the uploaded hierarchy and scratch.
    ///
    /// # Errors
    /// Rejects unsupported meshes, invalid coefficients and assembly/backend failures.
    pub fn try_from_mesh(
        backend: &CudaBackend,
        mesh: &Mesh,
        config: MaxwellCudaConfig,
    ) -> Result<Self, FemDeviceSolveError> {
        Ok(Self::upload(backend, prepare(mesh, config)?)?)
    }

    fn upload(backend: &CudaBackend, prepared: Prepared) -> BackendResult<Self> {
        let correction = |p: CsrMatrix<C64>, a: CsrMatrix<C64>| -> BackendResult<Correction> {
            let mut transpose = TripletMatrix::new(p.num_cols, p.num_rows);
            for r in 0..p.num_rows {
                for j in p.row_ptrs[r]..p.row_ptrs[r + 1] {
                    transpose.push(p.col_indices[j], r, p.values[j].conj());
                }
            }
            let transpose = transpose.try_to_csr().map_err(setup_error)?;
            Ok(Correction {
                prolongation: PersistentDeviceCsr::try_from_host_csr_ref(
                    backend,
                    p.as_host_csr_ref(),
                )?,
                restriction: PersistentDeviceCsr::try_from_host_csr_ref(
                    backend,
                    transpose.as_host_csr_ref(),
                )?,
                amg:
                    BackendAmgPreconditioner::<CudaBackend, C64>::try_from_csr_with_cuda_coarse_lu(
                        backend,
                        &a,
                        AmgConfig {
                            coarse_size: 8,
                            ..AmgConfig::for_backend_pcg()
                        },
                    )?,
                rhs: backend.raw_alloc(a.num_rows)?,
                solution: backend.raw_alloc(a.num_rows)?,
            })
        };
        let n = prepared.diagonal.len();
        Ok(Self {
            diagonal: PointJacobiBackend::try_from_diagonal(backend, &prepared.diagonal)?,
            workspace: Mutex::new(Workspace {
                corrections: [
                    correction(prepared.gradient, prepared.scalar)?,
                    correction(prepared.interpolation, prepared.vector)?,
                ],
                result: backend.raw_alloc(n)?,
                correction: backend.raw_alloc(n)?,
            }),
            n,
        })
    }
}

impl DevicePreconditioner<CudaBackend, C64> for MaxwellAuxiliaryCuda {
    fn apply(&self, backend: &CudaBackend, r: &CudaBuffer<C64>, z: &mut CudaBuffer<C64>) {
        if let Err(error) = self.try_apply(backend, r, z) {
            log::error!("Maxwell auxiliary application failed: {error}");
        }
    }

    fn try_apply(
        &self,
        backend: &CudaBackend,
        r: &CudaBuffer<C64>,
        z: &mut CudaBuffer<C64>,
    ) -> BackendResult<()> {
        if r.len() != self.n || z.len() != self.n {
            return Err(invalid("incorrect Maxwell vector length"));
        }
        let mut workspace = self
            .workspace
            .lock()
            .map_err(|_| invalid("poisoned Maxwell workspace"))?;
        let Workspace {
            corrections,
            result,
            correction,
        } = &mut *workspace;
        self.diagonal.try_apply(backend, r, result)?;
        for auxiliary in corrections {
            auxiliary
                .restriction
                .try_apply(backend, r, &mut auxiliary.rhs)?;
            auxiliary
                .amg
                .try_apply(backend, &auxiliary.rhs, &mut auxiliary.solution)?;
            auxiliary
                .prolongation
                .try_apply(backend, &auxiliary.solution, correction)?;
            backend.try_axpy(C64::new(1.0, 0.0), correction, result)?;
        }
        // Publish only after every subspace action succeeds.
        backend.try_copy_d2d(result, z)
    }
}

#[cfg(test)]
mod tests;
