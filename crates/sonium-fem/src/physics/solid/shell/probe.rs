//! Linear shell resultant probes for bench equivalence checks.
//!
//! Evaluates membrane resultants `[N11, N22, N12]` in N/m and bending
//! resultants `[M11, M22, M12]` in N m/m from a global displacement
//! state on the existing CST/DKT triangle kernels. Each triangle
//! contributes its constant membrane strain (via
//! [`ShellElement::cst_b_matrix`]) and its centroid DKT curvature
//! (via [`dkt_b_matrix_at`]); quad probes average their two children
//! as area-weighted global tensors projected onto the quad frame
//! (local X from node 1 to node 2, matching the Radioss shell frame).
//! Resultants stay in physical units here; any engine output scaling
//! belongs to the bench caller, not this module.
//!
//! Only homogeneous linear kinematics are modeled: callers pass the
//! global displacement vector, and constrained DOFs read as stored.

// Rust guideline compliant 2026-02-21

use super::ShellElement;
use super::dkt::dkt_b_matrix_at;
use super::explicit::ShellExplicitError;
use super::section::ShellSection;

/// Local frame of a shell quad in global coordinates.
#[derive(Debug, Clone, Copy)]
pub struct QuadFrame {
    /// Unit local X axis (node 1 toward node 2).
    pub ex: [f64; 3],
    /// Unit local Y axis (in-plane, completing the frame).
    pub ey: [f64; 3],
}

/// Builds a quad frame from four global corner points.
///
/// The normal comes from the diagonal cross product, so mildly
/// warped quads get the mean plane; local X runs from the first
/// toward the second corner.
///
/// # Errors
///
/// Returns [`ShellExplicitError`] for coincident leading corners or
/// a degenerate (zero-area) quad.
pub fn quad_frame(points: &[[f64; 3]; 4]) -> Result<QuadFrame, ShellExplicitError> {
    let edge = [
        points[1][0] - points[0][0],
        points[1][1] - points[0][1],
        points[1][2] - points[0][2],
    ];
    let edge_len = (edge[0] * edge[0] + edge[1] * edge[1] + edge[2] * edge[2]).sqrt();
    if !edge_len.is_finite() || edge_len <= 0.0 {
        return Err(ShellExplicitError::new(
            "quad frame needs distinct first corners",
        ));
    }
    let ex = [edge[0] / edge_len, edge[1] / edge_len, edge[2] / edge_len];
    let diag_a = [
        points[2][0] - points[0][0],
        points[2][1] - points[0][1],
        points[2][2] - points[0][2],
    ];
    let diag_b = [
        points[3][0] - points[1][0],
        points[3][1] - points[1][1],
        points[3][2] - points[1][2],
    ];
    let normal = [
        diag_a[1] * diag_b[2] - diag_a[2] * diag_b[1],
        diag_a[2] * diag_b[0] - diag_a[0] * diag_b[2],
        diag_a[0] * diag_b[1] - diag_a[1] * diag_b[0],
    ];
    let normal_len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if !normal_len.is_finite() || normal_len <= 0.0 {
        return Err(ShellExplicitError::new("quad frame needs nonzero area"));
    }
    let ez = [
        normal[0] / normal_len,
        normal[1] / normal_len,
        normal[2] / normal_len,
    ];
    let ey = [
        ez[1] * ex[2] - ez[2] * ex[1],
        ez[2] * ex[0] - ez[0] * ex[2],
        ez[0] * ex[1] - ez[1] * ex[0],
    ];
    Ok(QuadFrame { ex, ey })
}

/// Gathers one triangle's global DOFs into the local frame.
///
/// Applies the element rotation transpose to each node's
/// translation and rotation triple; rotations use the same map
/// because the frame is a proper rotation.
fn gather_local(element: &ShellElement, u_global: &[f64]) -> Result<[f64; 18], ShellExplicitError> {
    let dofs = element.global_dofs();
    let top = dofs.iter().copied().max().unwrap_or(0);
    if u_global.len() <= top {
        return Err(ShellExplicitError::new(
            "displacement state is shorter than the element DOFs",
        ));
    }
    let rot = element.rotation;
    let mut local = [0.0; 18];
    for node in 0..3 {
        for triple in 0..2 {
            let g = [
                u_global[dofs[6 * node + 3 * triple]],
                u_global[dofs[6 * node + 3 * triple + 1]],
                u_global[dofs[6 * node + 3 * triple + 2]],
            ];
            if !g.iter().all(|v| v.is_finite()) {
                return Err(ShellExplicitError::new(
                    "displacement state holds non-finite entries",
                ));
            }
            for axis in 0..3 {
                local[6 * node + 3 * triple + axis] =
                    rot[axis] * g[0] + rot[3 + axis] * g[1] + rot[6 + axis] * g[2];
            }
        }
    }
    Ok(local)
}

/// Unsigned triangle area from local corner coordinates.
fn local_area(coords: &[(f64, f64); 3]) -> f64 {
    let [(x1, y1), (x2, y2), (x3, y3)] = *coords;
    0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)).abs()
}

/// Membrane strain `[exx, eyy, gxy]` of one triangle.
fn tri_strain(element: &ShellElement, local: &[f64; 18]) -> [f64; 3] {
    let b = element.cst_b_matrix();
    let membrane = [local[0], local[1], local[6], local[7], local[12], local[13]];
    let mut strain = [0.0; 3];
    for row in 0..3 {
        for col in 0..6 {
            strain[row] += b[row * 6 + col] * membrane[col];
        }
    }
    strain
}

/// Curvature `[kxx, kyy, 2kxy]` of one triangle at its centroid.
fn tri_curvature(element: &ShellElement, local: &[f64; 18]) -> [f64; 3] {
    let area = local_area(&element.local_coords);
    let b = dkt_b_matrix_at(&element.local_coords, area, 1.0 / 3.0, 1.0 / 3.0);
    let bending = [
        local[2], local[3], local[4], local[8], local[9], local[10], local[14], local[15],
        local[16],
    ];
    let mut curvature = [0.0; 3];
    for row in 0..3 {
        for col in 0..9 {
            curvature[row] += b[row * 9 + col] * bending[col];
        }
    }
    curvature
}

/// Lifts a Voigt triple to a symmetric 3x3 tensor (plane block).
fn voigt_to_tensor(voigt: &[f64; 3]) -> [f64; 9] {
    [
        voigt[0],
        voigt[2] / 2.0,
        0.0,
        voigt[2] / 2.0,
        voigt[1],
        0.0,
        0.0,
        0.0,
        0.0,
    ]
}

/// Rotates a 3x3 tensor by a local-to-global frame: `R T R^T`.
fn rotate_tensor(frame: &[f64; 9], tensor: &[f64; 9]) -> [f64; 9] {
    let mut out = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = 0.0;
            for a in 0..3 {
                for b in 0..3 {
                    sum += frame[i * 3 + a] * tensor[a * 3 + b] * frame[j * 3 + b];
                }
            }
            out[i * 3 + j] = sum;
        }
    }
    out
}

/// Projects a global tensor onto a frame direction pair (Voigt out).
fn project_voigt(tensor: &[f64; 9], dx: &[f64; 3], dy: &[f64; 3]) -> [f64; 3] {
    let form = |a: &[f64; 3], b: &[f64; 3]| {
        let mut sum = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                sum += a[i] * tensor[i * 3 + j] * b[j];
            }
        }
        sum
    };
    [form(dx, dx), form(dy, dy), 2.0 * form(dx, dy)]
}

/// Applies a row-major 3x3 constitutive matrix to a Voigt triple.
fn apply_constitutive(matrix: &[f64; 9], voigt: &[f64; 3]) -> [f64; 3] {
    let mut out = [0.0; 3];
    for row in 0..3 {
        out[row] = matrix[row * 3] * voigt[0]
            + matrix[row * 3 + 1] * voigt[1]
            + matrix[row * 3 + 2] * voigt[2];
    }
    out
}

/// Membrane resultants `[N11, N22, N12]` in N/m of one triangle.
///
/// Strains are constant over the CST triangle, so no sample point
/// is needed; the frame is the element's own local frame.
///
/// # Errors
///
/// Returns [`ShellExplicitError`] for a short or non-finite state.
pub fn tri_membrane(
    element: &ShellElement,
    section: &ShellSection,
    u_global: &[f64],
) -> Result<[f64; 3], ShellExplicitError> {
    let local = gather_local(element, u_global)?;
    Ok(apply_constitutive(
        &section.membrane_constitutive(),
        &tri_strain(element, &local),
    ))
}

/// Bending resultants `[M11, M22, M12]` in N m/m of one triangle.
///
/// Curvatures are sampled at the centroid; the frame is the
/// element's own local frame.
///
/// # Errors
///
/// Returns [`ShellExplicitError`] for a short or non-finite state.
pub fn tri_bending(
    element: &ShellElement,
    section: &ShellSection,
    u_global: &[f64],
) -> Result<[f64; 3], ShellExplicitError> {
    let local = gather_local(element, u_global)?;
    Ok(apply_constitutive(
        &section.bending_constitutive(),
        &tri_curvature(element, &local),
    ))
}

/// Quad resultants in the quad frame from two child triangles.
///
/// Each child's strain and curvature tensors rotate to global,
/// average area-weighted, and project onto `frame`; membrane and
/// bending constitutives then give `([N11, N22, N12], [M11, M22,
/// M12])`. Averaging in the global frame keeps warped-child pairs
/// consistent without extracting per-child angles.
///
/// # Errors
///
/// Returns [`ShellExplicitError`] for a short or non-finite state,
/// or a degenerate child triangle.
pub fn quad_resultants(
    section: &ShellSection,
    frame: &QuadFrame,
    children: [&ShellElement; 2],
    u_global: &[f64],
) -> Result<([f64; 3], [f64; 3]), ShellExplicitError> {
    let mut strain = [0.0; 9];
    let mut curvature = [0.0; 9];
    let mut area_sum = 0.0;
    for child in children {
        let area = local_area(&child.local_coords);
        if !area.is_finite() || area <= 0.0 {
            return Err(ShellExplicitError::new(
                "resultant quad needs nonzero child areas",
            ));
        }
        let local = gather_local(child, u_global)?;
        let child_strain = rotate_tensor(
            &child.rotation,
            &voigt_to_tensor(&tri_strain(child, &local)),
        );
        let child_curvature = rotate_tensor(
            &child.rotation,
            &voigt_to_tensor(&tri_curvature(child, &local)),
        );
        for i in 0..9 {
            strain[i] += area * child_strain[i];
            curvature[i] += area * child_curvature[i];
        }
        area_sum += area;
    }
    for i in 0..9 {
        strain[i] /= area_sum;
        curvature[i] /= area_sum;
    }
    if !strain.iter().chain(curvature.iter()).all(|v| v.is_finite()) {
        return Err(ShellExplicitError::new(
            "resultant average holds non-finite entries",
        ));
    }
    Ok((
        apply_constitutive(
            &section.membrane_constitutive(),
            &project_voigt(&strain, &frame.ex, &frame.ey),
        ),
        apply_constitutive(
            &section.bending_constitutive(),
            &project_voigt(&curvature, &frame.ex, &frame.ey),
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Point;

    fn steel(thickness: f64) -> ShellSection {
        ShellSection::new(200.0e9, 0.3, thickness, 7800.0)
    }

    fn quad_pair() -> (ShellElement, ShellElement) {
        let section = steel(0.02);
        let p0 = Point::new_3d(0.0, 0.0, 0.0);
        let p1 = Point::new_3d(1.0, 0.0, 0.0);
        let p2 = Point::new_3d(1.0, 1.0, 0.0);
        let p3 = Point::new_3d(0.0, 1.0, 0.0);
        let first = ShellElement::from_points([0, 1, 2], &p0, &p1, &p2, section);
        let second = ShellElement::from_points([0, 2, 3], &p0, &p2, &p3, section);
        (first, second)
    }

    #[test]
    fn rigid_motion_gives_zero_resultants() {
        let (first, second) = quad_pair();
        let frame = quad_frame(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        .unwrap();
        // Uniform translation plus a small rigid rotation about z:
        // u = [0.5 - 0.01 y, -0.25 + 0.01 x, 0.125], θz = 0.01.
        let mut u = vec![0.0; 24];
        for (node, (x, y)) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
            .iter()
            .enumerate()
        {
            u[6 * node] = 0.5 - 0.01 * y;
            u[6 * node + 1] = -0.25 + 0.01 * x;
            u[6 * node + 2] = 0.125;
            u[6 * node + 5] = 0.01;
        }
        let (membrane, bending) =
            quad_resultants(&steel(0.02), &frame, [&first, &second], &u).unwrap();
        for value in membrane.iter().chain(bending.iter()) {
            assert!(value.abs() < 1.0e-6, "rigid residue {value:.3e}");
        }
    }

    #[test]
    fn uniform_stretch_matches_analytic_membrane() {
        let (first, second) = quad_pair();
        let section = steel(0.02);
        let frame = quad_frame(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        .unwrap();
        // Uniaxial stretch exx = 1e-3 with free contraction: the CST
        // triangles reproduce the linear field exactly.
        let mut u = vec![0.0; 24];
        for (node, (x, y)) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
            .iter()
            .enumerate()
        {
            u[6 * node] = 1.0e-3 * x;
            u[6 * node + 1] = -0.3e-3 * y;
        }
        let (membrane, bending) = quad_resultants(&section, &frame, [&first, &second], &u).unwrap();
        // Free contraction cancels the 1-nu^2 denominator: N11 = E t exx.
        let expected = 200.0e9 * 0.02 * 1.0e-3;
        assert!((membrane[0] - expected).abs() < 1.0e-6 * expected);
        assert!(membrane[1].abs() < 1.0e-6 * expected);
        assert!(membrane[2].abs() < 1.0e-6 * expected);
        for value in &bending {
            assert!(value.abs() < 1.0e-9);
        }
    }

    #[test]
    fn quad_frame_rejects_degenerate_corners() {
        assert!(quad_frame(&[[0.0; 3]; 4]).is_err());
        assert!(
            quad_frame(&[
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ])
            .is_err()
        );
    }
}
