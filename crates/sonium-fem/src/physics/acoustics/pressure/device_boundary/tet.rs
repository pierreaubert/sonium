//! Affine P2 tetrahedral traces in the volume space's global numbering.

// Rust guideline compliant 2026-02-21
use super::{PreparedRobinC64, constrain_boundary, invalid};
use crate::assembly::TripletMatrix;
use crate::boundary::{FemSystem, try_apply_robin_complex_p2};
use crate::mesh::Mesh;
use crate::physics::acoustics::pressure::{HelmholtzForm, device_tet::prepare_tet_p2_helmholtz};
use crate::spaces::{FunctionSpace, H1Space};
use num_complex::Complex64;
use sonium_compute::BackendResult;

/// Prepare affine P2 Robin/Neumann facets with the volume H1 DOF numbering.
///
/// Host setup promotes triangular traces and remaps their vertex/edge nodes to
/// the H1 space before checked quadratic quadrature. Prescribed edge DOFs must
/// be included in `dirichlet`. Coefficients remain fixed until rebuilt.
///
/// # Errors
/// Rejects invalid or non-affine tetrahedra, non-P2 spaces, inconsistent trace
/// numbering, invalid constraints and non-finite boundary coefficients.
pub fn prepare_tet_p2_robin(
    mesh: &Mesh,
    space: &H1Space,
    marker: i32,
    alpha: &dyn Fn(f64, f64, f64) -> Complex64,
    source: &dyn Fn(f64, f64, f64) -> Complex64,
    dirichlet: &[usize],
) -> BackendResult<PreparedRobinC64> {
    mesh.validate_connectivity()
        .map_err(|e| invalid(format!("invalid mesh: {e:?}")))?;
    let volume = prepare_tet_p2_helmholtz(mesh, space, &HelmholtzForm::from_real(0.0))?;
    let n = volume.dimension();
    if dirichlet.iter().any(|&d| d >= n) {
        return Err(invalid("Dirichlet DOF out of bounds"));
    }
    let trace = prepare_p2_trace(mesh, space)?;
    let mut system = FemSystem::new(TripletMatrix::new(n, n), vec![Complex64::default(); n]);
    try_apply_robin_complex_p2(&mut system, &trace, marker, alpha, source)
        .map_err(|e| invalid(e.to_string()))?;
    constrain_boundary(system, dirichlet)
}

// Validate before invoking the mesher's infallible P2 promotion API.
pub(super) fn prepare_p2_trace(mesh: &Mesh, space: &H1Space) -> BackendResult<Mesh> {
    mesh.validate_connectivity()
        .map_err(|e| invalid(format!("invalid mesh: {e:?}")))?;
    let n = prepare_tet_p2_helmholtz(mesh, space, &HelmholtzForm::from_real(0.0))?.dimension();
    // The volume validator proves four-node affine tetrahedra before promotion.
    // Promotion and H1 may enumerate shared edges differently: map IDs explicitly.
    let mut trace = mesh.to_p2();
    if trace.nodes.len() != n {
        return Err(invalid("promoted trace and H1 dimensions differ"));
    }
    let mut remap = vec![None; n];
    for (vertex, target) in remap.iter_mut().enumerate().take(mesh.nodes.len()) {
        *target = Some(vertex);
    }
    for (index, element) in trace.elements.iter().enumerate() {
        let dofs = space.local_to_global(mesh, index);
        if dofs.len() != element.nodes.len() {
            return Err(invalid("promoted trace and H1 element layouts differ"));
        }
        for (&old, new) in element.nodes.iter().zip(dofs) {
            if new >= n || remap[old].is_some_and(|previous| previous != new) {
                return Err(invalid("inconsistent shared P2 trace DOF"));
            }
            remap[old] = Some(new);
        }
    }
    let remap: Vec<_> = remap
        .into_iter()
        .collect::<Option<_>>()
        .ok_or_else(|| invalid("unmapped P2 trace node"))?;
    let mut seen = vec![false; n];
    let mut nodes = trace.nodes.clone();
    for (old, &new) in remap.iter().enumerate() {
        if std::mem::replace(&mut seen[new], true) {
            return Err(invalid("non-bijective P2 trace numbering"));
        }
        nodes[new] = trace.nodes[old];
    }
    trace.nodes = nodes;
    for element in &mut trace.elements {
        for node in &mut element.nodes {
            *node = remap[*node];
        }
    }
    for face in &mut trace.boundaries {
        for node in &mut face.nodes {
            *node = remap[*node];
        }
    }
    Ok(trace)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{basis::PolynomialDegree, mesh::unit_cube_tetrahedra};

    #[test]
    fn p2_trace_reproduces_surface_polynomials_and_prescribed_edge_coupling() {
        let mut mesh = unit_cube_tetrahedra(2);
        for face in &mut mesh.boundaries {
            face.marker = 1;
        }
        let space = H1Space::new(PolynomialDegree::P2);
        let one = |_: f64, _: f64, _: f64| Complex64::new(1.0, 0.0);
        let whole = prepare_tet_p2_robin(&mesh, &space, 1, &one, &one, &[]).unwrap();
        let xyz = space.dof_coordinates(&mesh).unwrap();
        let x2: Vec<_> = xyz.iter().map(|p| p[0] * p[0]).collect();
        let a = &whole.matrix;
        let energy: Complex64 = (0..x2.len())
            .map(|r| {
                (a.row_ptrs[r]..a.row_ptrs[r + 1])
                    .map(|j| a.values[j] * x2[r] * x2[a.col_indices[j]])
                    .sum::<Complex64>()
            })
            .sum();
        assert!((energy.re - 1.8).abs() < 1e-11);
        let load: Complex64 = whole.rhs.iter().zip(&x2).map(|(b, x)| *b * x).sum();
        assert!((load.re - 7.0 / 3.0).abs() < 1e-11);
        assert!((whole.matrix.values.iter().sum::<Complex64>().re - 6.0).abs() < 1e-11);
        let prescribed: Vec<_> = xyz
            .iter()
            .enumerate()
            .filter_map(|(i, p)| (p[1] == 0.0).then_some(i))
            .collect();
        assert!(prescribed.iter().any(|&i| i >= mesh.nodes.len()));
        let split = prepare_tet_p2_robin(&mesh, &space, 1, &one, &one, &prescribed).unwrap();
        for r in 0..xyz.len() {
            let constrained = prescribed.contains(&r);
            assert_eq!(
                split.rhs[r],
                if constrained {
                    Complex64::default()
                } else {
                    whole.rhs[r]
                }
            );
            for j in a.row_ptrs[r]..a.row_ptrs[r + 1] {
                let c = a.col_indices[j];
                let matrix = if prescribed.contains(&c) {
                    &split.dirichlet_coupling
                } else {
                    &split.matrix
                };
                let actual: Complex64 = (matrix.row_ptrs[r]..matrix.row_ptrs[r + 1])
                    .filter(|&k| matrix.col_indices[k] == c)
                    .map(|k| matrix.values[k])
                    .sum();
                assert!(
                    (actual
                        - if constrained {
                            Complex64::default()
                        } else {
                            a.values[j]
                        })
                    .norm()
                        < 1e-13
                );
            }
        }
        assert!(
            prepare_tet_p2_robin(
                &mesh,
                &space,
                1,
                &|_, _, _| Complex64::new(f64::NAN, 0.0),
                &one,
                &[]
            )
            .is_err()
        );
    }
}
