//! Website demo drivers: run small showcase simulations and dump VTU
//! frame series (plus a JSON manifest) for the website movie pipeline.
//!
//! Each subcommand writes `frame_XXXX.vtu` files and `manifest.json`
//! into the output directory given as the second argument. The Python
//! pipeline (`scripts/website_demos.py`) renders and encodes them.
//!
//! These are demonstrations, not QA gates: physics parameters favour
//! short, legible movies. Where a demo mirrors a QA case (Taylor-Green),
//! the setup follows the gated configuration.

use std::path::{Path, PathBuf};

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::locate_dofs_geometrical;
use sonium_fem::io::vtk::{CellKind, MeshView, VtuWriter};
use sonium_fem::mesh::Mesh;
use sonium_fem::mesh::generators::unit_square_triangles;
use sonium_fem::physics::fluid::stokes::navier_stokes::{
    LinearSolverCpuPreconditionerKind, NavierStokesConfig, solve_bdf_step_cached,
};
use sonium_fem::physics::fluid::stokes::time_stepping::{BdfConfig, BdfState};
use sonium_fem::physics::fluid::stokes::workspace::BdfWorkspace;
use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};
use sonium_qa::analytical::ns_taylor_green_vortex;

/// In-memory mesh exposed to the VTU writer.
struct SimpleMesh {
    points: Vec<[f64; 3]>,
    cells: Vec<(CellKind, Vec<u64>)>,
}

impl MeshView for SimpleMesh {
    fn num_nodes(&self) -> usize {
        self.points.len()
    }

    fn num_cells(&self) -> usize {
        self.cells.len()
    }

    fn nodes(&self) -> Box<dyn Iterator<Item = [f64; 3]> + '_> {
        Box::new(self.points.iter().copied())
    }

    fn cells(&self) -> Box<dyn Iterator<Item = (CellKind, Vec<u64>)> + '_> {
        Box::new(self.cells.iter().cloned())
    }
}

/// Scalar/vector point fields for one frame.
struct FrameFields {
    point_scalars: Vec<(&'static str, Vec<f64>)>,
    point_vectors: Vec<(&'static str, Vec<[f64; 3]>)>,
    cell_scalars: Vec<(&'static str, Vec<f64>)>,
}

/// Write one VTU frame and return its file name.
fn write_frame(
    out_dir: &Path,
    stem: &str,
    mesh: &SimpleMesh,
    fields: &FrameFields,
) -> Result<String, String> {
    let mut writer = VtuWriter::new(mesh);
    for (name, values) in &fields.point_scalars {
        writer
            .try_add_point_scalar(name, values)
            .map_err(|e| format!("frame {stem}: point scalar {name}: {e}"))?;
    }
    for (name, values) in &fields.point_vectors {
        let mut vx = Vec::with_capacity(values.len());
        let mut vy = Vec::with_capacity(values.len());
        let mut vz = Vec::with_capacity(values.len());
        for value in values {
            vx.push(value[0]);
            vy.push(value[1]);
            vz.push(value[2]);
        }
        writer
            .try_add_point_vector(name, &vx, &vy, &vz)
            .map_err(|e| format!("frame {stem}: point vector {name}: {e}"))?;
    }
    for (name, values) in &fields.cell_scalars {
        writer
            .try_add_cell_scalar(name, values)
            .map_err(|e| format!("frame {stem}: cell scalar {name}: {e}"))?;
    }
    let file_name = format!("{stem}.vtu");
    writer
        .write(out_dir.join(&file_name))
        .map_err(|e| format!("frame {stem}: write: {e}"))?;
    Ok(file_name)
}

/// Escape a JSON string literal (frame/demo metadata is ASCII-only by
/// construction, but keep the writer honest).
fn json_escape(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Write the demo manifest consumed by the render pipeline.
fn write_manifest(
    out_dir: &Path,
    demo: &str,
    title: &str,
    frames: &[String],
    times: &[f64],
    field: &str,
    field_range: (f64, f64),
) -> Result<(), String> {
    let frame_list: Vec<String> = frames
        .iter()
        .map(|name| format!("\"{}\"", json_escape(name)))
        .collect();
    let time_list: Vec<String> = times.iter().map(|t| format!("{t:.6}")).collect();
    let body = format!(
        "{{\n  \"demo\": \"{}\",\n  \"title\": \"{}\",\n  \"frames\": [{}],\n  \"times\": [{}],\n  \"field\": \"{}\",\n  \"field_range\": [{:.6}, {:.6}]\n}}\n",
        json_escape(demo),
        json_escape(title),
        frame_list.join(", "),
        time_list.join(", "),
        json_escape(field),
        field_range.0,
        field_range.1,
    );
    std::fs::write(out_dir.join("manifest.json"), body).map_err(|e| format!("manifest write: {e}"))
}

/// Build a `SimpleMesh` from a FEM mesh (triangles and tetrahedra only).
fn simple_mesh_from_fem(mesh: &Mesh) -> Result<SimpleMesh, String> {
    let points: Vec<[f64; 3]> = mesh
        .nodes
        .iter()
        .map(|node| [node.x, node.y, node.z])
        .collect();
    let mut cells = Vec::with_capacity(mesh.elements.len());
    for element in &mesh.elements {
        let kind = match element.nodes.len() {
            3 => CellKind::Triangle,
            4 => CellKind::Tetra,
            other => return Err(format!("unsupported demo cell size {other}")),
        };
        let indices: Vec<u64> = element
            .nodes
            .iter()
            .map(|&index| u64::try_from(index).map_err(|_| "node index overflow".to_string()))
            .collect::<Result<_, _>>()?;
        cells.push((kind, indices));
    }
    Ok(SimpleMesh { points, cells })
}

/// Taylor-Green vortex decay (mirrors the gated NS benchmark setup).
fn taylor_green(out_dir: &Path) -> Result<(), String> {
    // Demo parameters: coarser than the QA gate, larger viscosity so the
    // decay is visible inside a short movie.
    let mesh_n = 80usize;
    let nu = 0.04f64;
    let dt = 0.08f64;
    let steps = 75usize;
    let period = std::f64::consts::TAU;

    let mut mesh = unit_square_triangles(mesh_n);
    for node in mesh.nodes.iter_mut() {
        node.x *= period;
        node.y *= period;
    }
    let v_space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let p_space = H1Space::new(PolynomialDegree::P1);
    let n_v = v_space.num_global_dofs(&mesh);
    let n_p = p_space.num_global_dofs(&mesh);
    let geom_tol = 1e-9;
    let bc_dofs = locate_dofs_geometrical(&mesh, &v_space, |x, y, _z| {
        x < geom_tol || (period - x) < geom_tol || y < geom_tol || (period - y) < geom_tol
    });
    let config = NavierStokesConfig {
        mu: nu,
        rho: 1.0,
        use_direct_solver: false,
        linear_solver_max_iterations: 3000,
        linear_solver_tolerance: 1e-8,
        linear_solver_cpu_preconditioner: LinearSolverCpuPreconditionerKind::IlutSchur,
        ..Default::default()
    };
    let bdf = BdfConfig::bdf2(dt, steps);
    let mut workspace = BdfWorkspace::new(&mesh, &v_space, &p_space);
    let gauge = mesh
        .nodes
        .iter()
        .position(|node| node.x < geom_tol && node.y < geom_tol)
        .ok_or("Taylor-Green mesh has no origin corner node")?;
    workspace.set_pressure_gauge(gauge, 0.0);
    let velocity_coords = v_space
        .dof_coordinates(&mesh)
        .ok_or("P1 velocity space did not expose nodal coordinates")?;

    let simple = simple_mesh_from_fem(&mesh)?;
    let n_nodes = mesh.nodes.len();
    // Velocity DOFs are nodal pairs (u then v) but the pair order follows
    // the space's DOF numbering, not the mesh node order: map each pair
    // back to its node by coordinates.
    if velocity_coords.len() != 2 * n_nodes {
        return Err("velocity dof layout is not nodal pairs".to_string());
    }
    let mut dof_pair_node = vec![usize::MAX; n_nodes];
    for pair in 0..n_nodes {
        let point = velocity_coords[2 * pair];
        let node = mesh
            .nodes
            .iter()
            .position(|candidate| {
                (candidate.x - point[0]).abs() < 1e-9 && (candidate.y - point[1]).abs() < 1e-9
            })
            .ok_or("velocity dof pair has no matching mesh node")?;
        dof_pair_node[pair] = node;
    }

    let bc = |x: f64, y: f64, _z: f64| {
        let (u, v, _) = ns_taylor_green_vortex(x, y, 0.0, nu);
        vec![u, v]
    };
    let mut state = BdfState::from_rest(n_v, n_p);
    for (dof, point) in velocity_coords.iter().enumerate() {
        let (u, v, _) = ns_taylor_green_vortex(point[0], point[1], 0.0, nu);
        state.velocity[dof] = if dof % 2 == 0 { u } else { v };
        state.velocity_prev[dof] = state.velocity[dof];
    }

    let mut frames = Vec::new();
    let mut times = Vec::new();
    let mut max_speed: f64 = 0.0;
    let dump = |state: &BdfState, step: usize, _time: f64, max_speed: &mut f64| {
        let mut velocity = vec![[0.0f64; 3]; n_nodes];
        let mut speed = vec![0.0f64; n_nodes];
        for pair in 0..n_nodes {
            let node = dof_pair_node[pair];
            let u = state.velocity[2 * pair];
            let v = state.velocity[2 * pair + 1];
            velocity[node] = [u, v, 0.0];
            let magnitude = (u * u + v * v).sqrt();
            *max_speed = max_speed.max(magnitude);
            speed[node] = magnitude;
        }
        let fields = FrameFields {
            point_scalars: vec![("speed", speed)],
            point_vectors: vec![("velocity", velocity)],
            cell_scalars: Vec::new(),
        };
        write_frame(out_dir, &format!("frame_{step:04}"), &simple, &fields)
    };

    frames.push(dump(&state, 0, 0.0, &mut max_speed)?);
    times.push(0.0);
    for step in 0..steps {
        if !solve_bdf_step_cached(
            &mesh,
            &v_space,
            &p_space,
            &config,
            &bdf,
            &mut state,
            &bc,
            &bc_dofs,
            &[],
            &workspace,
        ) {
            return Err(format!("Taylor-Green BDF solve failed at step {step}"));
        }
        let time = dt * (step + 1) as f64;
        frames.push(dump(&state, step + 1, time, &mut max_speed)?);
        times.push(time);
    }
    write_manifest(
        out_dir,
        "taylor-green",
        "Taylor-Green vortex decay",
        &frames,
        &times,
        "speed",
        (0.0, max_speed),
    )
}

/// Per-element membrane strain intensity (constant-strain triangle in
/// the element local frame) averaged onto nodes for colouring.
fn shell_strain_field(
    elements: &[sonium_fem::physics::solid::shell::ShellElement],
    u: &[f64],
    total_nodes: usize,
) -> Vec<f64> {
    let mut nodal = vec![0.0f64; total_nodes];
    let mut weight = vec![0.0f64; total_nodes];
    for element in elements {
        let nodes = element.nodes;
        let mut local_u = [(0.0f64, 0.0f64); 3];
        for (k, &node) in nodes.iter().enumerate() {
            let g = [u[node * 6], u[node * 6 + 1], u[node * 6 + 2]];
            let r = &element.rotation;
            local_u[k] = (
                r[0] * g[0] + r[3] * g[1] + r[6] * g[2],
                r[1] * g[0] + r[4] * g[1] + r[7] * g[2],
            );
        }
        let [(x1, y1), (x2, y2), (x3, y3)] = element.local_coords;
        let two_a = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
        if two_a.abs() < 1e-30 {
            continue;
        }
        let (u1, v1) = local_u[0];
        let (u2, v2) = local_u[1];
        let (u3, v3) = local_u[2];
        let ex = ((y2 - y3) * u1 + (y3 - y1) * u2 + (y1 - y2) * u3) / two_a;
        let ey = ((x3 - x2) * v1 + (x1 - x3) * v2 + (x2 - x1) * v3) / two_a;
        let gam = ((x3 - x2) * u1
            + (x1 - x3) * u2
            + (x2 - x1) * u3
            + (y2 - y3) * v1
            + (y3 - y1) * v2
            + (y1 - y2) * v3)
            / two_a;
        let vm = (ex * ex - ex * ey + ey * ey + 0.75 * gam * gam).sqrt();
        for &node in &nodes {
            nodal[node] += vm * element.area;
            weight[node] += element.area;
        }
    }
    for node in 0..total_nodes {
        if weight[node] > 0.0 {
            nodal[node] /= weight[node];
        }
    }
    nodal
}

/// Bookkeeping for one contact projection pass.
#[derive(Default)]
struct ContactStats {
    /// Total normal impulse applied in N s.
    impulse: f64,
    /// Kinetic energy removed by velocity projection in J.
    ke_removed: f64,
    /// Work done on the system by position projection in J
    /// (estimated as -force . displacement with the step force;
    /// positive when the projection injects energy).
    work: f64,
    /// Deepest penetration corrected in m.
    max_penetration: f64,
    /// Number of node projections applied.
    projections: u64,
}

/// Closest point on triangle `abc` to `p` (Ericson's barycentric
/// region tests, *Real-Time Collision Detection* 5.1.5).
fn closest_point_triangle(p: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> [f64; 3] {
    let sub = |u: [f64; 3], v: [f64; 3]| [u[0] - v[0], u[1] - v[1], u[2] - v[2]];
    let dot = |u: [f64; 3], v: [f64; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let t = d1 / (d1 - d3);
        return [a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2]];
    }
    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let t = d2 / (d2 - d6);
        return [a[0] + t * ac[0], a[1] + t * ac[1], a[2] + t * ac[2]];
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return [
            b[0] + t * (c[0] - b[0]),
            b[1] + t * (c[1] - b[1]),
            b[2] + t * (c[2] - b[2]),
        ];
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    [
        a[0] + ab[0] * v + ac[0] * w,
        a[1] + ab[1] * v + ac[1] * w,
        a[2] + ab[2] * v + ac[2] * w,
    ]
}

/// Barycentric weights of `p` with respect to triangle `abc`
/// (clamped to the triangle; used to distribute contact corrections
/// over the opposing surface nodes).
fn barycentric_weights(p: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> [f64; 3] {
    let sub = |u: [f64; 3], v: [f64; 3]| [u[0] - v[0], u[1] - v[1], u[2] - v[2]];
    let dot = |u: [f64; 3], v: [f64; 3]| u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let v0 = sub(b, a);
    let v1 = sub(c, a);
    let v2 = sub(p, a);
    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);
    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() <= 1.0e-300 {
        return [1.0, 0.0, 0.0];
    }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    [
        (1.0 - v - w).clamp(0.0, 1.0),
        v.clamp(0.0, 1.0),
        w.clamp(0.0, 1.0),
    ]
}

/// Thickness-aware tube self-contact: shell midsurfaces may not
/// approach closer than one thickness (the two solid faces then
/// touch). The position correction is split between the node and
/// the opposing surface nodes by inverse mass (barycentric
/// weights), so the projection does not act on one side only; the
/// approaching relative normal velocity is removed (perfectly
/// inelastic, like the wall projection). Work done by the
/// projection is estimated with the step force and accounted in
/// `stats.work`: one-sided projection measurably injected energy
/// at 30 x 18 (plastic diagnostic 3.1x the platen work), which the
/// split plus this account keep visible and bounded.
/// Topologically near quads (two-ring neighbourhood) are excluded:
/// their midsurface distance is a membrane measure, not contact.
/// Broadphase is a uniform grid over triangle centroids.
#[allow(clippy::too_many_arguments)]
fn crush_self_contact(
    points: &mut [[f64; 3]],
    velocity: &mut [[f64; 6]],
    force: &[[f64; 6]],
    masses: &[[f64; 6]],
    quads: &[[usize; 4]],
    excluded: &[Vec<u32>],
    fixed: &[bool],
    tube_nodes: usize,
    thickness: f64,
    stats: &mut ContactStats,
) {
    use std::collections::HashMap;
    const CELL: f64 = 0.008;
    let cell_of = |p: [f64; 3]| {
        (
            (p[0] / CELL).floor() as i32,
            (p[1] / CELL).floor() as i32,
            (p[2] / CELL).floor() as i32,
        )
    };
    // Triangle list: quad split (a, b, c) and (a, c, d), matching
    // the render/colouring split.
    let mut grid: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
    for (qi, quad) in quads.iter().enumerate() {
        for tri in 0..2 {
            let idx = [
                quad[0],
                if tri == 0 { quad[1] } else { quad[2] },
                if tri == 0 { quad[2] } else { quad[3] },
            ];
            let centroid = [
                (points[idx[0]][0] + points[idx[1]][0] + points[idx[2]][0]) / 3.0,
                (points[idx[0]][1] + points[idx[1]][1] + points[idx[2]][1]) / 3.0,
                (points[idx[0]][2] + points[idx[1]][2] + points[idx[2]][2]) / 3.0,
            ];
            grid.entry(cell_of(centroid))
                .or_default()
                .push(2 * qi as u32 + tri as u32);
        }
    }
    for node in 0..tube_nodes {
        if fixed[node] {
            continue;
        }
        let p = points[node];
        let (cx, cy, cz) = cell_of(p);
        let mut best: Option<(f64, [f64; 3], [usize; 3])> = None;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let Some(tris) = grid.get(&(cx + dx, cy + dy, cz + dz)) else {
                        continue;
                    };
                    for &t in tris {
                        let qi = (t / 2) as usize;
                        if excluded[node].contains(&(qi as u32)) {
                            continue;
                        }
                        let quad = quads[qi];
                        let idx = if t % 2 == 0 {
                            [quad[0], quad[1], quad[2]]
                        } else {
                            [quad[0], quad[2], quad[3]]
                        };
                        let closest = closest_point_triangle(
                            p,
                            points[idx[0]],
                            points[idx[1]],
                            points[idx[2]],
                        );
                        let d = [p[0] - closest[0], p[1] - closest[1], p[2] - closest[2]];
                        let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                        if dist < thickness && best.is_none_or(|(b, ..)| dist < b) {
                            best = Some((dist, closest, idx));
                        }
                    }
                }
            }
        }
        let Some((dist, closest, surf)) = best else {
            continue;
        };
        if dist <= 1.0e-12 {
            continue; // Degenerate: node on the surface, no direction.
        }
        let penetration = thickness - dist;
        let n = [
            (p[0] - closest[0]) / dist,
            (p[1] - closest[1]) / dist,
            (p[2] - closest[2]) / dist,
        ];
        stats.max_penetration = stats.max_penetration.max(penetration);
        stats.projections += 1;
        // Inverse-mass split of the correction. Fixed (driven or
        // clamped) surface nodes carry infinite mass and do not move.
        let weights =
            barycentric_weights(closest, points[surf[0]], points[surf[1]], points[surf[2]]);
        let inv_node = 1.0 / masses[node][0];
        let mut inv_surf = [0.0; 3];
        let mut denom = inv_node;
        for i in 0..3 {
            if !fixed[surf[i]] {
                inv_surf[i] = weights[i] * weights[i] / masses[surf[i]][0];
                denom += inv_surf[i];
            }
        }
        let correction = [n[0] * penetration, n[1] * penetration, n[2] * penetration];
        let apply =
            |points: &mut [[f64; 3]], stats: &mut ContactStats, target: usize, delta: [f64; 3]| {
                for c in 0..3 {
                    points[target][c] += delta[c];
                    stats.work -= force[target][c] * delta[c];
                }
            };
        let share = inv_node / denom;
        apply(
            points,
            stats,
            node,
            [
                correction[0] * share,
                correction[1] * share,
                correction[2] * share,
            ],
        );
        for i in 0..3 {
            if inv_surf[i] > 0.0 {
                let share_i = inv_surf[i] / denom;
                apply(
                    points,
                    stats,
                    surf[i],
                    [
                        -correction[0] * share_i,
                        -correction[1] * share_i,
                        -correction[2] * share_i,
                    ],
                );
            }
        }
        // Remove the approaching relative normal velocity with the
        // same inverse-mass weighting; account the exact kinetic
        // energy change of every moved node.
        let mut involved: Vec<usize> = vec![node];
        for i in 0..3 {
            if inv_surf[i] > 0.0 && !involved.contains(&surf[i]) {
                involved.push(surf[i]);
            }
        }
        let ke_before: f64 = involved
            .iter()
            .map(|&nd| {
                0.5 * masses[nd][0]
                    * (velocity[nd][0].powi(2) + velocity[nd][1].powi(2) + velocity[nd][2].powi(2))
            })
            .sum();
        let v_surf = [
            weights[0] * velocity[surf[0]][0]
                + weights[1] * velocity[surf[1]][0]
                + weights[2] * velocity[surf[2]][0],
            weights[0] * velocity[surf[0]][1]
                + weights[1] * velocity[surf[1]][1]
                + weights[2] * velocity[surf[2]][1],
            weights[0] * velocity[surf[0]][2]
                + weights[1] * velocity[surf[1]][2]
                + weights[2] * velocity[surf[2]][2],
        ];
        let vn_rel = (velocity[node][0] - v_surf[0]) * n[0]
            + (velocity[node][1] - v_surf[1]) * n[1]
            + (velocity[node][2] - v_surf[2]) * n[2];
        if vn_rel < 0.0 {
            let impulse = -vn_rel / denom;
            stats.impulse += impulse;
            for c in 0..3 {
                velocity[node][c] += impulse * inv_node * n[c];
            }
            for i in 0..3 {
                if inv_surf[i] > 0.0 {
                    for c in 0..3 {
                        velocity[surf[i]][c] -= impulse * weights[i] / masses[surf[i]][0] * n[c];
                    }
                }
            }
            let ke_after: f64 = involved
                .iter()
                .map(|&nd| {
                    0.5 * masses[nd][0]
                        * (velocity[nd][0].powi(2)
                            + velocity[nd][1].powi(2)
                            + velocity[nd][2].powi(2))
                })
                .sum();
            stats.ke_removed += ke_before - ke_after;
        }
    }
}

/// Thin-walled tube crushed by prescribed end-shortening against a
/// rigid wall (Q4 crash shells, J2 plasticity; driven top ring).
fn crush(out_dir: &Path) -> Result<(), String> {
    use sonium_fem::mesh::Point;
    use sonium_fem::physics::solid::plasticity::{HardeningLaw, J2Material};
    use sonium_fem::physics::solid::shell::crash::{
        DefShell, ElasticShearHook, HourglassParams, LayerStates, Q4CrashGeom, q4_crash_forces,
    };
    use sonium_fem::physics::solid::shell::layered::{J2LayerState, SmallStrainJ2Model};
    use sonium_fem::physics::solid::shell::{ShellElement, ShellSection};

    // Aluminium tube: 90 mm diameter, 160 mm tall, 1.2 mm wall.
    let radius = 0.045f64;
    let height = 0.16f64;
    let z_base = 0.002f64;
    // Resolution and formulation variants for the convergence and
    // sensitivity studies behind `fem.website-crush-regression.v1`.
    // Defaults are the shipped demo configuration; overrides exist so
    // the study matrix runs the identical driver, not a copy.
    let env_usize = |key: &str, default: usize| -> usize {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let env_f64 = |key: &str, default: f64| -> f64 {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let n_around = env_usize("CRUSH_N_AROUND", 40);
    let n_axial = env_usize("CRUSH_N_AXIAL", 24);
    let dt_scale = env_f64("CRUSH_DT_SCALE", 1.0);
    // `gauss5`: 5-point Gauss layers (shipped). `linear3`: 3
    // surface-sampled piecewise-linear layers, the TYPE1 rule of the
    // verified crash-box card.
    let linear_layers = std::env::var("CRUSH_LAYERS").is_ok_and(|v| v == "linear3");
    // `clamp`: bottom ring fixed in all DOF (shipped). `contact`:
    // bottom ring rests on the wall through the contact model below
    // (normal projection plus friction), free to slide and rotate.
    // Sensitivity result (2026-10-18): the contact variant lowers
    // the mean force ~15% (24.2 vs 28.6 kN) but its sustained
    // whole-ring projection contact is not energy-consistent
    // (closure -58%: each step's position reset re-stretches the
    // loaded ring elements), so `clamp` remains the shipped
    // configuration and `contact` stays a study variant.
    let bottom_contact = std::env::var("CRUSH_BOTTOM").is_ok_and(|v| v == "contact");
    // Self-contact can be disabled to isolate its energy account
    // (`CRUSH_SELF=0`); the shipped configuration keeps it on.
    let self_contact = !std::env::var("CRUSH_SELF").is_ok_and(|v| v == "0");
    let section = ShellSection::new(70.0e9, 0.33, 0.0012, 2700.0);
    let node_id = |level: usize, around: usize| level * n_around + (around % n_around);
    let point = |level: usize, around: usize| {
        let theta = std::f64::consts::TAU * around as f64 / n_around as f64;
        let z = z_base + height * level as f64 / n_axial as f64;
        // Slight six-lobed imperfection (0.4%) to seed diamond folds.
        let r = radius
            * (1.0
                + 0.004
                    * (6.0 * theta).cos()
                    * (std::f64::consts::PI * (z - z_base) / height).sin());
        Point::new_3d(r * theta.cos(), r * theta.sin(), z)
    };

    let tube_nodes = (n_axial + 1) * n_around;
    let wall_start = tube_nodes;
    let total_nodes = tube_nodes + 4;

    // Solver mesh: one Q4 crash quad per cell. Colouring mesh: the
    // same cells split into triangles (shell_strain_field input).
    let mut quads: Vec<[usize; 4]> = Vec::new();
    let mut elements: Vec<ShellElement> = Vec::new();
    let mut tube_faces: Vec<[usize; 3]> = Vec::new();
    for level in 0..n_axial {
        for around in 0..n_around {
            let a = node_id(level, around);
            let b = node_id(level, around + 1);
            let c = node_id(level + 1, around + 1);
            let d = node_id(level + 1, around);
            quads.push([a, b, c, d]);
            for (nodes, pa, pb, pc) in [
                (
                    [a, b, c],
                    point(level, around),
                    point(level, around + 1),
                    point(level + 1, around + 1),
                ),
                (
                    [a, c, d],
                    point(level, around),
                    point(level + 1, around + 1),
                    point(level + 1, around),
                ),
            ] {
                elements.push(ShellElement::from_points(nodes, &pa, &pb, &pc, section));
                tube_faces.push(nodes);
            }
        }
    }
    // Deformed coordinates of every node (tube + wall corners).
    let initial: Vec<[f64; 3]> = (0..tube_nodes)
        .map(|id| {
            let p = point(id / n_around, id % n_around);
            [p.x, p.y, p.z]
        })
        .chain([
            [-0.10, -0.10, 0.0],
            [0.10, -0.10, 0.0],
            [0.10, 0.10, 0.0],
            [-0.10, 0.10, 0.0],
        ])
        .collect();
    let wall_faces: Vec<[usize; 3]> = vec![
        [wall_start, wall_start + 1, wall_start + 2],
        [wall_start, wall_start + 2, wall_start + 3],
    ];

    // Q4 crash integrator (Belytschko-Tsay one-point quads,
    // hourglass control, velocity-increment J2 layers), driven by
    // prescribed end-shortening: the bottom ring sits on the wall
    // (fixed) and the top ring is driven down at a constant rate.
    // The corotational ShellExplicitModel path tried first diverged
    // for this tube in every regime (free flight and driven,
    // dt = 0.08-0.85 x wave_dt, two mesh densities, elastic and
    // plastic, damped and not; steps 10191-41091): its strain
    // measure breaks once folds rotate elements, while the crash
    // kernel's rate form is the gated production path for this
    // load case (see the fem_crash_shells QA driver).
    let density = 2700.0f64;
    let thickness = 0.0012f64;
    let material = SmallStrainJ2Model {
        material: J2Material::new(
            70.0e9,
            0.33,
            250.0e6,
            HardeningLaw::LinearIsotropic { modulus: 600.0e6 },
        ),
    };
    let def_shell = DefShell::explicit_defaults();
    let shear_modulus = 70.0e9 / (2.0 * (1.0 + 0.33));
    let shear_hook = ElasticShearHook::new(shear_modulus);
    let sound = (70.0e9 / density).sqrt();
    let mut layers: Vec<LayerStates<J2LayerState>> = quads
        .iter()
        .map(|_| {
            if linear_layers {
                LayerStates::new_piecewise_linear(J2LayerState::new(), 3, thickness)
            } else {
                LayerStates::new(J2LayerState::new(), 5, thickness)
            }
            .map_err(|e| e.message)
        })
        .collect::<Result<_, _>>()?;
    let bottom_ring: Vec<usize> = (0..n_around).map(|a| node_id(0, a)).collect();
    let top_ring: Vec<usize> = (0..n_around).map(|a| node_id(n_axial, a)).collect();
    let mut fixed = vec![false; total_nodes];
    for &node in top_ring.iter() {
        fixed[node] = true;
    }
    if !bottom_contact {
        for &node in &bottom_ring {
            fixed[node] = true;
        }
    }
    for wall_node in 0..4 {
        fixed[wall_start + wall_node] = true;
    }
    // Lumped masses plus the original explicit-shell rotary
    // stabilization (m_node * A / 9, as in the crash-box QA driver).
    let mut masses = vec![[0.0f64; 6]; total_nodes];
    let mut dt_min = f64::INFINITY;
    for quad in &quads {
        let geom = Q4CrashGeom::from_corners(quad.map(|n| initial[n]), thickness)
            .map_err(|e| e.message)?;
        dt_min = dt_min.min(geom.critical_step(sound));
        let mut lumped = geom.lumped_masses(density);
        for node in 0..4 {
            for d in 3..6 {
                lumped[6 * node + d] += lumped[6 * node] * geom.area / 9.0;
            }
        }
        for (slot, &node) in quad.iter().enumerate() {
            for d in 0..6 {
                masses[node][d] += lumped[6 * slot + d];
            }
        }
    }
    let dt = 0.5 * dt_min * dt_scale;
    let end_time = 0.010f64;
    // Total prescribed end-shortening in m (34% of the tube height).
    let crush_distance = 0.055f64;
    let drive_rate = crush_distance / end_time;
    let total_steps = (end_time / dt).ceil() as usize;
    let frame_count = 72usize;
    let frame_every = (total_steps / frame_count).max(1);

    // Render mesh: tube triangles + wall, deformed positions per frame.
    let mut render_cells: Vec<(CellKind, Vec<u64>)> = tube_faces
        .iter()
        .map(|face| {
            (
                CellKind::Triangle,
                face.iter().map(|&n| n as u64).collect::<Vec<u64>>(),
            )
        })
        .collect();
    for face in &wall_faces {
        render_cells.push((
            CellKind::Triangle,
            face.iter().map(|&n| n as u64).collect::<Vec<u64>>(),
        ));
    }

    // Per-element membrane strain intensity (constant-strain triangle in
    // the element local frame) averaged onto nodes for colouring.
    let element_strain = |u: &[f64]| -> Vec<f64> { shell_strain_field(&elements, u, total_nodes) };

    let mut frames = Vec::new();
    let mut times = Vec::new();
    let mut max_strain: f64 = 0.0;
    let dump =
        |points: &[[f64; 3]], frame_index: usize, max_strain: &mut f64| -> Result<String, String> {
            // Colouring strain from the triangle split of the quads.
            let mut u6 = vec![0.0f64; total_nodes * 6];
            for node in 0..total_nodes {
                for c in 0..3 {
                    u6[node * 6 + c] = points[node][c] - initial[node][c];
                }
            }
            let strain = element_strain(&u6);
            *max_strain = (*max_strain).max(strain.iter().copied().fold(0.0, f64::max));
            let mesh = SimpleMesh {
                points: points.to_vec(),
                cells: render_cells.clone(),
            };
            let fields = FrameFields {
                point_scalars: vec![("strain", strain)],
                point_vectors: Vec::new(),
                cell_scalars: Vec::new(),
            };
            write_frame(out_dir, &format!("frame_{frame_index:04}"), &mesh, &fields)
        };

    // Self-contact exclusion: quads in the two-ring neighbourhood of
    // each node (their distance is membrane strain, not contact).
    let excluded: Vec<Vec<u32>> = {
        let mut node_quads: Vec<Vec<u32>> = vec![Vec::new(); tube_nodes];
        for (qi, quad) in quads.iter().enumerate() {
            for &node in quad {
                node_quads[node].push(qi as u32);
            }
        }
        (0..tube_nodes)
            .map(|node| {
                let mut set: Vec<u32> = Vec::new();
                for &qi in &node_quads[node] {
                    for &other in &quads[qi as usize] {
                        for &qj in &node_quads[other] {
                            if !set.contains(&qj) {
                                set.push(qj);
                            }
                        }
                    }
                }
                set
            })
            .collect()
    };

    // Reliability bookkeeping (gate `fem.website-crush-regression.v1`):
    // energy, momentum and contact accounts for the whole run.
    let mut plastic_work = 0.0f64;
    let mut hourglass_energy = 0.0f64;
    let mut external_work = 0.0f64;
    let mut peak_reaction = 0.0f64;
    let mut min_z = f64::INFINITY;
    let mut wall_stats = ContactStats::default();
    let mut self_stats = ContactStats::default();
    let mut force_curve: Vec<(f64, f64)> = Vec::new();
    let mut reaction_window = (0.0f64, 0usize);

    let mut points = initial.clone();
    let mut velocity = vec![[0.0f64; 6]; total_nodes];
    frames.push(dump(&points, 0, &mut max_strain)?);
    times.push(0.0);
    let mut frame_index = 1usize;
    for step in 1..=total_steps {
        // Prescribed top-ring end-shortening at a constant rate.
        let drive = crush_distance * (step as f64 * dt / end_time).min(1.0);
        for &node in &top_ring {
            points[node][2] = initial[node][2] - drive;
            velocity[node] = [0.0, 0.0, -drive_rate, 0.0, 0.0, 0.0];
        }
        let mut force = vec![[0.0f64; 6]; total_nodes];
        for (qi, quad) in quads.iter().enumerate() {
            let geom = Q4CrashGeom::from_corners(quad.map(|n| points[n]), thickness)
                .map_err(|e| e.message)?;
            let hg = HourglassParams::for_shell(
                def_shell.formulation,
                density,
                sound,
                shear_modulus,
                geom.area,
                thickness,
                geom.char_length,
                false,
            )
            .map_err(|e| e.message)?;
            let to_local = |w: [f64; 3]| {
                [
                    geom.ex[0] * w[0] + geom.ex[1] * w[1] + geom.ex[2] * w[2],
                    geom.ey[0] * w[0] + geom.ey[1] * w[1] + geom.ey[2] * w[2],
                    geom.ez[0] * w[0] + geom.ez[1] * w[1] + geom.ez[2] * w[2],
                ]
            };
            let from_local = |w: [f64; 3]| {
                [
                    geom.ex[0] * w[0] + geom.ey[0] * w[1] + geom.ez[0] * w[2],
                    geom.ex[1] * w[0] + geom.ey[1] * w[1] + geom.ez[1] * w[2],
                    geom.ex[2] * w[0] + geom.ey[2] * w[1] + geom.ez[2] * w[2],
                ]
            };
            let mut u = [0.0f64; 24];
            let mut v = [0.0f64; 24];
            for (slot, &node) in quad.iter().enumerate() {
                let disp = [
                    points[node][0] - initial[node][0],
                    points[node][1] - initial[node][1],
                    points[node][2] - initial[node][2],
                ];
                let lu = to_local(disp);
                let lv = to_local([velocity[node][0], velocity[node][1], velocity[node][2]]);
                let lw = to_local([velocity[node][3], velocity[node][4], velocity[node][5]]);
                for c in 0..3 {
                    u[6 * slot + c] = lu[c];
                    v[6 * slot + c] = lv[c];
                    v[6 * slot + 3 + c] = lw[c];
                }
            }
            let out = q4_crash_forces(
                &geom,
                &mut layers[qi],
                &material,
                &def_shell,
                &u,
                &v,
                dt,
                &hg,
                &shear_hook,
            )
            .map_err(|e| e.message)?;
            plastic_work += out.plastic_work_rate * dt;
            hourglass_energy += out.hg_power * dt;
            for (slot, &node) in quad.iter().enumerate() {
                let ft = from_local([
                    out.forces[6 * slot],
                    out.forces[6 * slot + 1],
                    out.forces[6 * slot + 2],
                ]);
                let fr = from_local([
                    out.forces[6 * slot + 3],
                    out.forces[6 * slot + 4],
                    out.forces[6 * slot + 5],
                ]);
                // q4_crash_forces returns internal forces; the
                // applied nodal force is their negation (as in the
                // crash-box QA driver).
                for c in 0..3 {
                    force[node][c] -= ft[c];
                    force[node][3 + c] -= fr[c];
                }
            }
        }
        // Driven-ring reaction and platen work for this step. The
        // assembled `force` is the element force on the ring (it
        // pushes the ring up under compression); the platen cancels
        // it, so the compressive reaction magnitude is its sum and
        // the platen power is reaction x drive rate.
        let reaction: f64 = top_ring.iter().map(|&node| force[node][2]).sum();
        external_work += reaction * drive_rate * dt;
        peak_reaction = peak_reaction.max(reaction.abs());
        reaction_window.0 += reaction;
        reaction_window.1 += 1;
        for node in 0..tube_nodes {
            if fixed[node] {
                continue;
            }
            for d in 0..6 {
                velocity[node][d] += dt * force[node][d] / masses[node][d];
            }
            for c in 0..3 {
                points[node][c] += dt * velocity[node][c];
            }
        }
        // Rigid-wall contact: the wall face is z = 0 and the shell
        // midsurface may not pass z = thickness / 2. The pinned
        // bottom ring alone does not stop free nodes beside it from
        // crossing once a fold forms at the base, which the refined
        // 40 x 24 mesh does (the coarse mesh folded mid-tube and
        // never reached the wall). Project penetrating nodes back
        // and cancel their inward velocity, mirroring the wall
        // projection of the verified crash-box driver
        // (`bin/support/crash_box.rs`), and account the impulse and
        // removed kinetic energy (the fixed wall does no work; its
        // impulse work is exactly that removed kinetic energy). The
        // `contact` bottom variant also
        // applies Coulomb friction capped by the normal impulse.
        let floor = thickness / 2.0;
        for node in 0..tube_nodes {
            if points[node][2] < floor {
                wall_stats.max_penetration =
                    wall_stats.max_penetration.max(floor - points[node][2]);
                wall_stats.projections += 1;
                points[node][2] = floor;
                if velocity[node][2] < 0.0 {
                    let m = masses[node][0];
                    let normal_impulse = m * (-velocity[node][2]);
                    wall_stats.impulse += normal_impulse;
                    wall_stats.ke_removed += 0.5 * m * velocity[node][2].powi(2);
                    velocity[node][2] = 0.0;
                    if bottom_contact && bottom_ring.contains(&node) {
                        // Friction: tangential impulse at most
                        // mu * normal impulse (mu = 0.3).
                        let vt = [velocity[node][0], velocity[node][1]];
                        let speed = (vt[0] * vt[0] + vt[1] * vt[1]).sqrt();
                        if speed > 0.0 {
                            let dv = (0.3 * normal_impulse / m).min(speed);
                            velocity[node][0] -= dv * vt[0] / speed;
                            velocity[node][1] -= dv * vt[1] / speed;
                        }
                    }
                }
            }
        }
        // Fold self-contact every fourth step: nodal travel per
        // step (~3 um) is far below the thickness it resolves, and
        // every-step projection was measured to drain kinetic
        // energy continuously on persistent fold contacts.
        if self_contact && (step % 4 == 0 || step == total_steps) {
            crush_self_contact(
                &mut points,
                &mut velocity,
                &force,
                &masses,
                &quads,
                &excluded,
                &fixed,
                tube_nodes,
                thickness,
                &mut self_stats,
            );
        }
        for node in 0..tube_nodes {
            min_z = min_z.min(points[node][2]);
        }
        if !points.iter().flatten().all(|value| value.is_finite()) {
            return Err(format!("crush diverged at step {step}"));
        }
        if step % frame_every == 0 || step == total_steps {
            if reaction_window.1 > 0 {
                force_curve.push((drive, reaction_window.0 / reaction_window.1 as f64));
                reaction_window = (0.0, 0);
            }
            frames.push(dump(&points, frame_index, &mut max_strain)?);
            times.push(step as f64 * dt);
            frame_index += 1;
        }
    }
    // Final kinetic energy and the derived gate metrics.
    let mut kinetic_final = 0.0f64;
    for node in 0..tube_nodes {
        for d in 0..6 {
            kinetic_final += 0.5 * masses[node][d] * velocity[node][d].powi(2);
        }
    }
    let mean_crush_force = external_work / crush_distance;
    let hourglass_fraction = hourglass_energy / plastic_work.max(1.0e-30);
    let metrics = format!(
        "{{\n  \"n_around\": {n_around}, \"n_axial\": {n_axial},\n  \"layers\": \"{}\", \"bottom\": \"{}\",\n  \"dt\": {dt:.6e}, \"steps\": {total_steps}, \"dt_scale\": {dt_scale},\n  \"crush_distance_m\": {crush_distance}, \"external_work_J\": {external_work:.6e},\n  \"plastic_work_J\": {plastic_work:.6e}, \"hourglass_energy_J\": {hourglass_energy:.6e},\n  \"hourglass_fraction\": {hourglass_fraction:.6e}, \"kinetic_final_J\": {kinetic_final:.6e},\n  \"mean_crush_force_N\": {mean_crush_force:.6e}, \"peak_reaction_N\": {peak_reaction:.6e},\n  \"wall_impulse_Ns\": {:.6e}, \"wall_ke_removed_J\": {:.6e}, \"wall_projections\": {},\n  \"self_impulse_Ns\": {:.6e}, \"self_ke_removed_J\": {:.6e}, \"self_work_J\": {:.6e}, \"self_projections\": {},\n  \"max_wall_penetration_m\": {:.6e}, \"max_self_penetration_m\": {:.6e},\n  \"min_z_m\": {min_z:.6e}, \"max_strain\": {max_strain:.6e},\n  \"force_curve\": [{}]\n}}\n",
        if linear_layers { "linear3" } else { "gauss5" },
        if bottom_contact { "contact" } else { "clamp" },
        wall_stats.impulse,
        wall_stats.ke_removed,
        wall_stats.projections,
        self_stats.impulse,
        self_stats.ke_removed,
        self_stats.work,
        self_stats.projections,
        wall_stats.max_penetration,
        self_stats.max_penetration,
        force_curve
            .iter()
            .map(|(d, f)| format!("[{d:.6e}, {f:.6e}]"))
            .collect::<Vec<_>>()
            .join(", "),
    );
    std::fs::write(out_dir.join("crush_metrics.json"), &metrics)
        .map_err(|e| format!("write crush metrics: {e}"))?;
    println!("CRUSH_METRICS: {}", metrics.replace('\n', " "));
    println!(
        "crush: {total_steps} steps, dt={dt:.3e}, max strain {max_strain:.3}, final time {:.4}",
        total_steps as f64 * dt
    );
    write_manifest(
        out_dir,
        "crush",
        "Thin-walled tube crush",
        &frames,
        &times,
        "strain",
        (0.0, max_strain),
    )
}

/// Necking bar: finite-strain J2 slender bar (same formulation as the
/// gated `fem.necking-bar-spatial` case), rendered as a surface of
/// revolution so the diffuse neck localises in 3D.
fn necking(out_dir: &Path) -> Result<(), String> {
    use sonium_fem::physics::solid::plasticity::{
        FiniteStrainJ2Material, FiniteStrainState, HardeningLaw, StressUpdateModel,
    };

    const E_MOD: f64 = 200.0e9;
    const NU: f64 = 0.3;
    const YIELD: f64 = 250.0e6;
    const Q_HARD: f64 = 300.0e6;
    const B_HARD: f64 = 5.0;
    const AREA: f64 = 1.0e-4;
    const GAUSS: [f64; 2] = [-0.577_350_269_189_625_8, 0.577_350_269_189_625_8];

    #[derive(Clone)]
    struct PointState {
        material: FiniteStrainState,
        axial: f64,
        lateral: f64,
    }
    impl Default for PointState {
        fn default() -> Self {
            Self {
                material: FiniteStrainState::new(),
                axial: 1.0,
                lateral: 1.0,
            }
        }
    }
    struct PointTrial {
        state: PointState,
        force: f64,
        tangent: f64,
    }
    fn initial_area(x: f64, imperfection: f64) -> f64 {
        AREA * (1.0 - imperfection * (-0.5 * ((x - 0.5) / 0.1).powi(2)).exp())
    }
    fn point_trial(
        old: &PointState,
        axial: f64,
        a0: f64,
        mat: &FiniteStrainJ2Material,
    ) -> Result<PointTrial, String> {
        if axial <= 0.0 || !axial.is_finite() {
            return Err("bar inverted during Newton trial".into());
        }
        let dx = (axial / old.axial).ln();
        let mut dt = -0.5 * dx;
        for _ in 0..30 {
            let increment = [dx.exp(), 0.0, 0.0, 0.0, dt.exp(), 0.0, 0.0, 0.0, dt.exp()];
            let update = mat.update(&old.material, &increment);
            let residual = 0.5 * (update.stress[1] + update.stress[2]);
            let c = &update.tangent;
            let transverse = c[7] + c[8];
            if !transverse.is_finite() || transverse <= 0.0 {
                return Err("invalid transverse tangent".into());
            }
            if residual.abs() / YIELD < 1e-12 {
                let lateral = old.lateral * dt.exp();
                let slope = -c[6] / transverse;
                let condensed = c[0] + (c[1] + c[2]) * slope;
                let area = a0 * lateral * lateral;
                let mut state = old.material.clone();
                mat.commit(&mut state, &update);
                return Ok(PointTrial {
                    state: PointState {
                        material: state,
                        axial,
                        lateral,
                    },
                    force: area * update.stress[0],
                    tangent: area / axial * (condensed + 2.0 * update.stress[0] * slope),
                });
            }
            dt -= residual / transverse;
        }
        Err("bar traction-free transverse solve failed".into())
    }
    struct Assembly {
        residual: Vec<f64>,
        diagonal: Vec<f64>,
        upper: Vec<f64>,
        states: Vec<PointState>,
    }
    fn assemble(
        u: &[f64],
        old: &[PointState],
        imperfection: f64,
        mat: &FiniteStrainJ2Material,
    ) -> Result<Assembly, String> {
        let n = u.len() - 1;
        let h = 1.0 / n as f64;
        let mut out = Assembly {
            residual: vec![0.0; n + 1],
            diagonal: vec![0.0; n + 1],
            upper: vec![0.0; n],
            states: Vec::with_capacity(2 * n),
        };
        for e in 0..n {
            let axial = 1.0 + (u[e + 1] - u[e]) / h;
            for (q, xi) in GAUSS.into_iter().enumerate() {
                let x = (e as f64 + 0.5 * (1.0 + xi)) * h;
                let p = point_trial(&old[2 * e + q], axial, initial_area(x, imperfection), mat)?;
                let force = 0.5 * p.force;
                let stiffness = 0.5 * p.tangent / h;
                out.residual[e] -= force;
                out.residual[e + 1] += force;
                out.diagonal[e] += stiffness;
                out.diagonal[e + 1] += stiffness;
                out.upper[e] -= stiffness;
                out.states.push(p.state);
            }
        }
        Ok(out)
    }
    fn solve_tridiagonal(diagonal: &[f64], upper: &[f64], rhs: &[f64]) -> Result<Vec<f64>, String> {
        let mut d = diagonal.to_vec();
        let mut r = rhs.to_vec();
        for i in 1..d.len() {
            if d[i - 1].abs() < 1e-20 || !d[i - 1].is_finite() {
                return Err("singular bar tangent".into());
            }
            let factor = upper[i - 1] / d[i - 1];
            d[i] -= factor * upper[i - 1];
            r[i] -= factor * r[i - 1];
        }
        let mut x = vec![0.0; d.len()];
        for i in (0..d.len()).rev() {
            x[i] = (r[i]
                - if i + 1 < d.len() {
                    upper[i] * x[i + 1]
                } else {
                    0.0
                })
                / d[i];
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite bar correction".into());
        }
        Ok(x)
    }

    let n = 96usize;
    let steps = 200usize;
    let imperfection = 0.01f64;
    let end_extension = 0.32f64;
    let mat = FiniteStrainJ2Material::new(
        E_MOD,
        NU,
        YIELD,
        HardeningLaw::VoceSaturation {
            q: Q_HARD,
            b: B_HARD,
        },
    );
    let mut u = vec![0.0f64; n + 1];
    let mut states = vec![PointState::default(); 2 * n];

    // Surface of revolution: one ring per bar node.
    let n_around = 20usize;
    let radius0 = (AREA / std::f64::consts::PI).sqrt() * 6.0; // exaggerated for legibility
    let mut cells: Vec<(CellKind, Vec<u64>)> = Vec::new();
    for e in 0..n {
        for k in 0..n_around {
            let a = (e * n_around + k) as u64;
            let b = (e * n_around + (k + 1) % n_around) as u64;
            let c = ((e + 1) * n_around + (k + 1) % n_around) as u64;
            let d = ((e + 1) * n_around + k) as u64;
            cells.push((CellKind::Quad, vec![a, b, c, d]));
        }
    }

    let mut frames = Vec::new();
    let mut times = Vec::new();
    let mut max_strain: f64 = 0.0;
    let dump = |u: &[f64],
                states: &[PointState],
                frame_index: usize,
                max_strain: &mut f64|
     -> Result<String, String> {
        // Nodal lateral stretch and axial strain from adjacent points.
        let mut points = Vec::with_capacity((n + 1) * n_around);
        let mut strain = Vec::with_capacity((n + 1) * n_around);
        for node in 0..=n {
            let (lateral, axial_strain) = if node == 0 {
                (states[0].lateral, states[0].axial.ln())
            } else if node == n {
                (states[2 * n - 1].lateral, states[2 * n - 1].axial.ln())
            } else {
                let left = &states[2 * node - 1];
                let right = &states[2 * node];
                (
                    0.5 * (left.lateral + right.lateral),
                    0.5 * (left.axial.ln() + right.axial.ln()),
                )
            };
            *max_strain = (*max_strain).max(axial_strain);
            let x = node as f64 / n as f64 + u[node];
            let r = radius0 * lateral;
            for k in 0..n_around {
                let theta = std::f64::consts::TAU * k as f64 / n_around as f64;
                points.push([x, r * theta.cos(), r * theta.sin()]);
                strain.push(axial_strain);
            }
        }
        let mesh = SimpleMesh {
            points,
            cells: cells.clone(),
        };
        let fields = FrameFields {
            point_scalars: vec![("strain", strain)],
            point_vectors: Vec::new(),
            cell_scalars: Vec::new(),
        };
        write_frame(out_dir, &format!("frame_{frame_index:04}"), &mesh, &fields)
    };

    frames.push(dump(&u, &states, 0, &mut max_strain)?);
    times.push(0.0);
    // Adaptive march: near localization the equilibrium path stiffens,
    // so halve the increment on a stalled Newton and stop gracefully at
    // the smallest useful increment (the movie ends at the neck).
    let mut extension_done = 0.0f64;
    let mut step_size = end_extension / steps as f64;
    let mut frame_index = 0usize;
    while extension_done < end_extension - 1e-12 && frame_index < 400 {
        let target = (extension_done + step_size).min(end_extension);
        let saved_u = u.clone();
        let saved_states = states.clone();
        let increment = target - u[n];
        for (i, value) in u.iter_mut().enumerate() {
            *value += increment * i as f64 / n as f64;
        }
        u[0] = 0.0;
        u[n] = target;
        let mut converged = None;
        let mut stalled = false;
        for _ in 0..200 {
            let trial = assemble(&u, &states, imperfection, &mat)?;
            let norm = trial.residual[1..n]
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt()
                / (AREA * YIELD);
            if norm < 1e-9 {
                converged = Some(trial);
                break;
            }
            let rhs: Vec<_> = trial.residual[1..n].iter().map(|v| -v).collect();
            let correction =
                solve_tridiagonal(&trial.diagonal[1..n], &trial.upper[1..n - 1], &rhs)?;
            let mut accepted = false;
            let mut alpha = 1.0;
            for _ in 0..16 {
                let mut candidate = u.clone();
                for (v, du) in candidate[1..n].iter_mut().zip(&correction) {
                    *v += alpha * du;
                }
                if let Ok(next) = assemble(&candidate, &states, imperfection, &mat) {
                    let next_norm = next.residual[1..n]
                        .iter()
                        .map(|v| v * v)
                        .sum::<f64>()
                        .sqrt()
                        / (AREA * YIELD);
                    if next_norm < norm {
                        u = candidate;
                        accepted = true;
                        break;
                    }
                }
                alpha *= 0.5;
            }
            if !accepted {
                stalled = true;
                break;
            }
        }
        if stalled || converged.is_none() {
            u = saved_u;
            states = saved_states;
            step_size *= 0.5;
            if step_size < 1.0e-5 {
                println!("necking: stopping at extension {extension_done:.3} (localization limit)");
                break;
            }
            continue;
        }
        let trial = converged.expect("converged Newton trial");
        states = trial.states;
        extension_done = target;
        frame_index += 1;
        frames.push(dump(&u, &states, frame_index, &mut max_strain)?);
        times.push(extension_done);
    }
    // Explicit-dynamics continuation through localization: the
    // quasi-static Newton stalls at the turning point, so march the
    // same residual with lumped masses (mass-scaled, still
    // quasi-static rates) until the neck is fully developed.
    if extension_done > 0.15 {
        let rho_scaled = 7800.0 * 25.0;
        let nodal_mass = AREA * rho_scaled / n as f64;
        let wave_speed = (E_MOD / rho_scaled).sqrt();
        let dyn_dt = 0.7 * (1.0 / n as f64) / wave_speed;
        let end_velocity = 0.45f64;
        let target_extension = 0.42f64;
        let mut velocity = vec![0.0f64; n + 1];
        let mut dyn_time = 0.0f64;
        let mut dyn_step = 0usize;
        while u[n] < target_extension && dyn_step < 200_000 {
            let trial = assemble(&u, &states, imperfection, &mat)?;
            states = trial.states;
            let ramp = (dyn_time / 0.02).min(1.0);
            for i in 1..n {
                let accel = -trial.residual[i] / nodal_mass - 30.0 * velocity[i];
                velocity[i] += accel * dyn_dt;
            }
            for i in 1..n {
                u[i] += velocity[i] * dyn_dt;
            }
            u[0] = 0.0;
            u[n] += end_velocity * ramp * dyn_dt;
            dyn_time += dyn_dt;
            dyn_step += 1;
            if dyn_step % 60 == 0 {
                frame_index += 1;
                frames.push(dump(&u, &states, frame_index, &mut max_strain)?);
                times.push(u[n]);
            }
        }
        println!(
            "necking: dynamic continuation {dyn_step} steps to extension {:.3}",
            u[n]
        );
    }
    println!("necking: max axial strain {max_strain:.3}");
    write_manifest(
        out_dir,
        "necking",
        "Necking bar",
        &frames,
        &times,
        "strain",
        (0.0, max_strain),
    )
}

/// Vibrating plate: first shell eigenmodes (generalized eigenproblem
/// on the assembled stiffness/lumped-mass pair, LOBPCG), animated over
/// one period per mode.
fn plate_modes(out_dir: &Path) -> Result<(), String> {
    use sonium_fem::mesh::Point;
    use sonium_fem::physics::solid::shell::explicit::lumped_masses;
    use sonium_fem::physics::solid::shell::{ShellElement, ShellSection};
    use sonium_solvers::{LobpcgConfig, lobpcg_generalized_preconditioned};
    use sonium_solvers_core::traits::{LinearOperator, Preconditioner};

    struct DenseOp {
        rows: usize,
        cols: usize,
        data: Vec<f64>,
    }
    impl LinearOperator<f64> for DenseOp {
        fn num_rows(&self) -> usize {
            self.rows
        }
        fn num_cols(&self) -> usize {
            self.cols
        }
        fn apply_into(&self, x: &[f64], y: &mut [f64]) {
            for (row, out) in y.iter_mut().enumerate() {
                *out = self.data[row * self.cols..(row + 1) * self.cols]
                    .iter()
                    .zip(x)
                    .map(|(a, b)| a * b)
                    .sum();
            }
        }
        fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
            // The assembled plate stiffness is symmetric.
            LinearOperator::apply_into(self, x, y);
        }
    }
    struct DiagOp {
        diag: Vec<f64>,
    }
    impl LinearOperator<f64> for DiagOp {
        fn num_rows(&self) -> usize {
            self.diag.len()
        }
        fn num_cols(&self) -> usize {
            self.diag.len()
        }
        fn apply_into(&self, x: &[f64], y: &mut [f64]) {
            for (out, (d, value)) in y.iter_mut().zip(self.diag.iter().zip(x)) {
                *out = d * value;
            }
        }
        fn apply_transpose_into(&self, x: &[f64], y: &mut [f64]) {
            LinearOperator::apply_into(self, x, y);
        }
    }
    impl Preconditioner<f64> for DiagOp {
        fn apply(&self, r: &[f64]) -> Vec<f64> {
            r.iter()
                .zip(&self.diag)
                .map(|(value, d)| value * d)
                .collect()
        }
        fn apply_into(&self, r: &[f64], y: &mut [f64]) {
            LinearOperator::apply_into(self, r, y);
        }
    }

    // Clamped square steel plate, 260 mm across, 3 mm thick.
    let side = 0.26f64;
    let grid = 24usize; // elements per side
    let nodes_per_side = grid + 1;
    let n_nodes = nodes_per_side * nodes_per_side;
    let n_dofs = n_nodes * 6;
    let section = ShellSection::new(200.0e9, 0.3, 0.003, 7800.0);
    let node_point = |i: usize, j: usize| {
        Point::new_3d(
            side * i as f64 / grid as f64,
            side * j as f64 / grid as f64,
            0.0,
        )
    };
    let node_id = |i: usize, j: usize| j * nodes_per_side + i;
    let mut elements: Vec<ShellElement> = Vec::new();
    let mut cells: Vec<(CellKind, Vec<u64>)> = Vec::new();
    for j in 0..grid {
        for i in 0..grid {
            let a = node_id(i, j);
            let b = node_id(i + 1, j);
            let c = node_id(i + 1, j + 1);
            let d = node_id(i, j + 1);
            for (nodes, pa, pb, pc) in [
                (
                    [a, b, c],
                    node_point(i, j),
                    node_point(i + 1, j),
                    node_point(i + 1, j + 1),
                ),
                (
                    [a, c, d],
                    node_point(i, j),
                    node_point(i + 1, j + 1),
                    node_point(i, j + 1),
                ),
            ] {
                elements.push(ShellElement::from_points(nodes, &pa, &pb, &pc, section));
                cells.push((
                    CellKind::Triangle,
                    nodes.iter().map(|&n| n as u64).collect(),
                ));
            }
        }
    }
    let masses = lumped_masses(&elements, n_dofs).map_err(|e| e.message)?;
    let base_points: Vec<[f64; 3]> = (0..n_nodes)
        .map(|id| {
            let p = node_point(id % nodes_per_side, id / nodes_per_side);
            [p.x, p.y, p.z]
        })
        .collect();

    // Free DOFs: everything except boundary nodes (clamped edges).
    let is_boundary = |node: usize| {
        let (i, j) = (node % nodes_per_side, node / nodes_per_side);
        i == 0 || j == 0 || i == grid || j == grid
    };
    let mut free_index = vec![usize::MAX; n_dofs];
    let mut n_free = 0usize;
    for dof in 0..n_dofs {
        if !is_boundary(dof / 6) {
            free_index[dof] = n_free;
            n_free += 1;
        }
    }
    let mut k_data = vec![0.0f64; n_free * n_free];
    for element in &elements {
        let ke = element.stiffness_global();
        let mut gdofs = [0usize; 18];
        for (slot, &node) in element.nodes.iter().enumerate() {
            for c in 0..6 {
                gdofs[slot * 6 + c] = node * 6 + c;
            }
        }
        for li in 0..18 {
            let fi = free_index[gdofs[li]];
            if fi == usize::MAX {
                continue;
            }
            for lj in 0..18 {
                let fj = free_index[gdofs[lj]];
                if fj == usize::MAX {
                    continue;
                }
                k_data[fi * n_free + fj] += ke[li * 18 + lj];
            }
        }
    }
    let m_free: Vec<f64> = (0..n_dofs)
        .filter(|&dof| free_index[dof] != usize::MAX)
        .map(|dof| masses[dof])
        .collect();
    if m_free.iter().any(|&m| m <= 0.0 || !m.is_finite()) {
        return Err("plate mass matrix has a non-positive free mass".into());
    }
    let k_op = DenseOp {
        rows: n_free,
        cols: n_free,
        data: k_data,
    };
    let m_op = DiagOp {
        diag: m_free.clone(),
    };
    let precond = DiagOp {
        diag: (0..n_free)
            .map(|i| {
                let d = k_op.data[i * n_free + i];
                if d > 0.0 { 1.0 / d } else { 0.0 }
            })
            .collect(),
    };
    let config = LobpcgConfig {
        num_eigenvalues: 4,
        // 2400 iterations: the doubled grid (24 per side) needs
        // roughly 3-4x the iterations of the original 12-grid solve
        // to reach the same 1e-8 residual.
        max_iterations: 2400,
        tolerance: 1e-8,
        block_padding: 4,
    };
    let result = lobpcg_generalized_preconditioned(&k_op, &m_op, &precond, &config)
        .map_err(|e| format!("plate LOBPCG failed: {e}"))?;
    if !result.converged {
        return Err(format!(
            "plate LOBPCG did not converge (residuals {:?})",
            result.residual_norms
        ));
    }
    let frequencies: Vec<f64> = result
        .eigenvalues
        .iter()
        .map(|lambda| lambda.max(0.0).sqrt() / std::f64::consts::TAU)
        .collect();
    println!("plate modes (Hz): {frequencies:.1?}");

    // Animate modes 0..3 sequentially, one period each.
    let frames_per_mode = 18usize;
    let amplitude = 0.02f64;
    let mut frames = Vec::new();
    let mut times = Vec::new();
    let mut frame_index = 0usize;
    for (mode, &lambda) in result.eigenvalues.iter().enumerate() {
        // Eigenvector column -> full DOF displacement shape.
        let mut shape = vec![0.0f64; n_dofs];
        for dof in 0..n_dofs {
            let fi = free_index[dof];
            if fi != usize::MAX {
                shape[dof] = result.eigenvectors[[fi, mode]];
            }
        }
        let max_z = (0..n_nodes)
            .map(|node| shape[node * 6 + 2].abs())
            .fold(0.0, f64::max);
        if max_z <= 0.0 {
            return Err(format!("plate mode {mode} has no transverse shape"));
        }
        let scale = amplitude / max_z;
        for frame in 0..frames_per_mode {
            let phase = std::f64::consts::TAU * frame as f64 / frames_per_mode as f64;
            let sway = phase.sin();
            let points: Vec<[f64; 3]> = (0..n_nodes)
                .map(|node| {
                    [
                        base_points[node][0] + scale * sway * shape[node * 6],
                        base_points[node][1] + scale * sway * shape[node * 6 + 1],
                        base_points[node][2] + scale * sway * shape[node * 6 + 2],
                    ]
                })
                .collect();
            let disp: Vec<f64> = (0..n_nodes)
                .map(|node| scale * sway * shape[node * 6 + 2])
                .collect();
            let mesh = SimpleMesh {
                points,
                cells: cells.clone(),
            };
            let fields = FrameFields {
                point_scalars: vec![("disp", disp)],
                point_vectors: Vec::new(),
                cell_scalars: Vec::new(),
            };
            frames.push(write_frame(
                out_dir,
                &format!("frame_{frame_index:04}"),
                &mesh,
                &fields,
            )?);
            times.push(mode as f64 + frame as f64 / frames_per_mode as f64);
            frame_index += 1;
        }
        let _ = lambda;
    }
    write_manifest(
        out_dir,
        "plate-modes",
        "Plate vibration modes",
        &frames,
        &times,
        "disp",
        (-amplitude, amplitude),
    )
}

/// Room impulse: acoustic wave equation in a 3D room (P1 tetrahedra,
/// lumped mass, central differences; rigid-wall Neumann boundaries).
fn room_wave(out_dir: &Path) -> Result<(), String> {
    use sonium_fem::mesh::generators::box_mesh_tetrahedra;

    let (lx, ly, lz) = (3.6f64, 2.8f64, 2.4f64);
    let mesh = box_mesh_tetrahedra(0.0, lx, 0.0, ly, 0.0, lz, 48, 36, 32);
    let n = mesh.nodes.len();
    let c = 343.0f64;
    let source = [1.8, 1.4, 1.2];
    let sigma = 0.22f64;

    // P1 tet element stiffness (unit coefficient) and lumped masses.
    struct Tet {
        nodes: [usize; 4],
        ke: [[f64; 4]; 4],
    }
    let mut tets: Vec<Tet> = Vec::new();
    let mut lumped = vec![0.0f64; n];
    for element in &mesh.elements {
        if element.nodes.len() != 4 {
            return Err("room mesh is not tetrahedral".into());
        }
        let idx = [
            element.nodes[0],
            element.nodes[1],
            element.nodes[2],
            element.nodes[3],
        ];
        let p: [[f64; 3]; 4] = [
            [
                mesh.nodes[idx[0]].x,
                mesh.nodes[idx[0]].y,
                mesh.nodes[idx[0]].z,
            ],
            [
                mesh.nodes[idx[1]].x,
                mesh.nodes[idx[1]].y,
                mesh.nodes[idx[1]].z,
            ],
            [
                mesh.nodes[idx[2]].x,
                mesh.nodes[idx[2]].y,
                mesh.nodes[idx[2]].z,
            ],
            [
                mesh.nodes[idx[3]].x,
                mesh.nodes[idx[3]].y,
                mesh.nodes[idx[3]].z,
            ],
        ];
        // Jacobian of the affine map from the reference tet.
        let j = [
            [p[1][0] - p[0][0], p[2][0] - p[0][0], p[3][0] - p[0][0]],
            [p[1][1] - p[0][1], p[2][1] - p[0][1], p[3][1] - p[0][1]],
            [p[1][2] - p[0][2], p[2][2] - p[0][2], p[3][2] - p[0][2]],
        ];
        let det = j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
            - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
            + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);
        if det.abs() < 1e-300 {
            return Err("degenerate tetrahedron in room mesh".into());
        }
        let volume = det.abs() / 6.0;
        // Inverse Jacobian (rows), for reference gradients.
        let inv = [
            [
                (j[1][1] * j[2][2] - j[1][2] * j[2][1]) / det,
                (j[0][2] * j[2][1] - j[0][1] * j[2][2]) / det,
                (j[0][1] * j[1][2] - j[0][2] * j[1][1]) / det,
            ],
            [
                (j[1][2] * j[2][0] - j[1][0] * j[2][2]) / det,
                (j[0][0] * j[2][2] - j[0][2] * j[2][0]) / det,
                (j[0][2] * j[1][0] - j[0][0] * j[1][2]) / det,
            ],
            [
                (j[1][0] * j[2][1] - j[1][1] * j[2][0]) / det,
                (j[0][1] * j[2][0] - j[0][0] * j[2][1]) / det,
                (j[0][0] * j[1][1] - j[0][1] * j[1][0]) / det,
            ],
        ];
        // Reference gradients: N0=(-1,-1,-1), N1=(1,0,0), N2=(0,1,0), N3=(0,0,1).
        let ref_grads = [
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        // Physical gradients: grad N = J^{-T} grad_ref N.
        let mut grads = [[0.0f64; 3]; 4];
        for a in 0..4 {
            for component in 0..3 {
                grads[a][component] = inv[0][component] * ref_grads[a][0]
                    + inv[1][component] * ref_grads[a][1]
                    + inv[2][component] * ref_grads[a][2];
            }
        }
        let mut ke = [[0.0f64; 4]; 4];
        for a in 0..4 {
            for b in 0..4 {
                ke[a][b] = volume
                    * (grads[a][0] * grads[b][0]
                        + grads[a][1] * grads[b][1]
                        + grads[a][2] * grads[b][2]);
            }
        }
        for &node in &idx {
            lumped[node] += volume / 4.0;
        }
        tets.push(Tet { nodes: idx, ke });
    }
    if lumped.iter().any(|&m| m <= 0.0) {
        return Err("room mesh has a node without mass".into());
    }

    let simple = simple_mesh_from_fem(&mesh)?;
    let mut pressure: Vec<f64> = mesh
        .nodes
        .iter()
        .map(|node| {
            let dx = node.x - source[0];
            let dy = node.y - source[1];
            let dz = node.z - source[2];
            (-(dx * dx + dy * dy + dz * dz) / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let mut previous = pressure.clone();
    let h = lx / 48.0;
    let dt = 0.45 * h / (c * 3.0f64.sqrt());
    let total_time = 0.012f64;
    let steps = (total_time / dt) as usize;
    let frame_every = (steps / 72).max(1);

    let mut frames = Vec::new();
    let mut times = Vec::new();
    let mut max_pressure: f64 = 0.0;
    let dump =
        |pressure: &[f64], frame_index: usize, max_pressure: &mut f64| -> Result<String, String> {
            *max_pressure =
                (*max_pressure).max(pressure.iter().fold(0.0, |acc: f64, v| acc.max(v.abs())));
            let fields = FrameFields {
                point_scalars: vec![("pressure", pressure.to_vec())],
                point_vectors: Vec::new(),
                cell_scalars: Vec::new(),
            };
            write_frame(
                out_dir,
                &format!("frame_{frame_index:04}"),
                &simple,
                &fields,
            )
        };

    frames.push(dump(&pressure, 0, &mut max_pressure)?);
    times.push(0.0);
    let mut frame_index = 1usize;
    let mut kp = vec![0.0f64; n];
    for step in 1..=steps {
        kp.iter_mut().for_each(|v| *v = 0.0);
        for tet in &tets {
            for (a, &na) in tet.nodes.iter().enumerate() {
                let mut sum = 0.0;
                for (b, &nb) in tet.nodes.iter().enumerate() {
                    sum += tet.ke[a][b] * pressure[nb];
                }
                kp[na] += sum;
            }
        }
        let mut next = vec![0.0f64; n];
        for i in 0..n {
            next[i] = 2.0 * pressure[i] - previous[i] - dt * dt * c * c * kp[i] / lumped[i];
        }
        previous = std::mem::replace(&mut pressure, next);
        if step % frame_every == 0 || step == steps {
            frames.push(dump(&pressure, frame_index, &mut max_pressure)?);
            times.push(step as f64 * dt);
            frame_index += 1;
        }
    }
    println!("room: {steps} steps, max pressure {max_pressure:.3}");
    write_manifest(
        out_dir,
        "room-wave",
        "Room impulse response",
        &frames,
        &times,
        "pressure",
        (-max_pressure, max_pressure),
    )
}

/// Shallow arch snap-through: a curved steel strip whose crown is
/// driven down quasi-statically (explicit corotational shells with a
/// near-elastic J2 law) until it snaps to the inverted equilibrium.
fn arch(out_dir: &Path) -> Result<(), String> {
    use sonium_fem::mesh::Point;
    use sonium_fem::physics::solid::plasticity::{HardeningLaw, J2Material};
    use sonium_fem::physics::solid::shell::explicit::{ShellExplicitModel, ShellExplicitState};
    use sonium_fem::physics::solid::shell::{ShellElement, ShellSection};

    let span = 1.0f64;
    let rise = 0.09f64;
    let width = 0.14f64;
    let n_along = 48usize;
    let n_across = 6usize;
    let radius = ((span / 2.0).powi(2) + rise.powi(2)) / (2.0 * rise);
    let phi_max = (span / 2.0 / radius).asin();
    let section = ShellSection::new(200.0e9, 0.3, 0.004, 7800.0);
    let node_id = |i: usize, j: usize| i * (n_across + 1) + j;
    let point = |i: usize, j: usize| {
        let phi = -phi_max + 2.0 * phi_max * i as f64 / n_along as f64;
        Point::new_3d(
            radius * phi.sin(),
            -width / 2.0 + width * j as f64 / n_across as f64,
            radius * phi.cos() - (radius - rise),
        )
    };
    let total_nodes = (n_along + 1) * (n_across + 1);
    let n_dofs = total_nodes * 6;
    let mut elements: Vec<ShellElement> = Vec::new();
    let mut cells: Vec<(CellKind, Vec<u64>)> = Vec::new();
    for i in 0..n_along {
        for j in 0..n_across {
            let a = node_id(i, j);
            let b = node_id(i + 1, j);
            let c = node_id(i + 1, j + 1);
            let d = node_id(i, j + 1);
            for (nodes, pa, pb, pc) in [
                ([a, b, c], point(i, j), point(i + 1, j), point(i + 1, j + 1)),
                ([a, c, d], point(i, j), point(i + 1, j + 1), point(i, j + 1)),
            ] {
                elements.push(ShellElement::from_points(nodes, &pa, &pb, &pc, section));
                cells.push((
                    CellKind::Triangle,
                    nodes.iter().map(|&n| n as u64).collect(),
                ));
            }
        }
    }
    // Near-elastic J2 law on the corotational path (geometric
    // nonlinearity without plasticity): the snap is elastic.
    let material = J2Material::new(
        200.0e9,
        0.3,
        250.0e6,
        HardeningLaw::LinearIsotropic { modulus: 0.0 },
    );
    let mut model = ShellExplicitModel::elastoplastic(elements.clone(), n_dofs, material)
        .map_err(|e| e.message)?;
    for j in 0..=n_across {
        model.fix_node(node_id(0, j));
        model.fix_node(node_id(n_along, j));
    }
    let initial: Vec<[f64; 3]> = (0..total_nodes)
        .map(|id| {
            let p = point(id / (n_across + 1), id % (n_across + 1));
            [p.x, p.y, p.z]
        })
        .collect();
    let crown_nodes: Vec<usize> = (0..=n_across).map(|j| node_id(n_along / 2, j)).collect();

    let mut state = ShellExplicitState::new(n_dofs);
    let dt = 0.85 * model.wave_dt();
    let total_time = 1.15f64;
    let total_steps = (total_time / dt).ceil() as usize;
    let frame_every = (total_steps / 80).max(1);
    let drive_time = 0.85f64;
    let drive_depth = 0.15f64;

    let mut frames = Vec::new();
    let mut times = Vec::new();
    let mut max_strain: f64 = 0.0;
    let dump = |state: &ShellExplicitState,
                frame_index: usize,
                max_strain: &mut f64|
     -> Result<String, String> {
        let points: Vec<[f64; 3]> = (0..total_nodes)
            .map(|node| {
                [
                    initial[node][0] + state.u[node * 6],
                    initial[node][1] + state.u[node * 6 + 1],
                    initial[node][2] + state.u[node * 6 + 2],
                ]
            })
            .collect();
        let strain = shell_strain_field(&elements, &state.u, total_nodes);
        *max_strain = (*max_strain).max(strain.iter().copied().fold(0.0, f64::max));
        let mesh = SimpleMesh {
            points,
            cells: cells.clone(),
        };
        let fields = FrameFields {
            point_scalars: vec![("strain", strain)],
            point_vectors: Vec::new(),
            cell_scalars: Vec::new(),
        };
        write_frame(out_dir, &format!("frame_{frame_index:04}"), &mesh, &fields)
    };

    frames.push(dump(&state, 0, &mut max_strain)?);
    times.push(0.0);
    let zero_force = vec![0.0f64; n_dofs];
    let mut frame_index = 1usize;
    for step in 1..=total_steps {
        model
            .step(&mut state, &zero_force, dt)
            .map_err(|e| format!("arch step {step}: {}", e.message))?;
        // Prescribed crown displacement (smooth ramp) + light damping.
        let t = step as f64 * dt;
        let driven = drive_depth * (t / drive_time).min(1.0);
        for &node in &crown_nodes {
            state.u[node * 6 + 2] = -driven;
            state.v[node * 6 + 2] = 0.0;
        }
        let damp = (-1.2 * dt).exp();
        for value in state.v.iter_mut() {
            *value *= damp;
        }
        if step % frame_every == 0 || step == total_steps {
            frames.push(dump(&state, frame_index, &mut max_strain)?);
            times.push(t);
            frame_index += 1;
        }
    }
    let crown_z = initial[crown_nodes[0]][2] + state.u[crown_nodes[0] * 6 + 2];
    println!(
        "arch: {total_steps} steps, crown final z {crown_z:.3} (started {rise:.3}), max strain {max_strain:.4}"
    );
    write_manifest(
        out_dir,
        "arch",
        "Arch snap-through",
        &frames,
        &times,
        "strain",
        (0.0, max_strain),
    )
}

/// One tetrahedron's P1 stiffness block plus the mesh lumped masses, for
/// the hand-rolled explicit wave/conduction updates used by the acoustic
/// demos (same construction as `room_wave`, factored out for reuse).
struct WaveTet {
    nodes: [usize; 4],
    ke: [[f64; 4]; 4],
}

fn lumped_wave_tets(mesh: &Mesh) -> Result<(Vec<WaveTet>, Vec<f64>), String> {
    let n = mesh.nodes.len();
    let mut tets: Vec<WaveTet> = Vec::new();
    let mut lumped = vec![0.0f64; n];
    for element in &mesh.elements {
        if element.nodes.len() != 4 {
            return Err("wave mesh is not tetrahedral".into());
        }
        let idx = [
            element.nodes[0],
            element.nodes[1],
            element.nodes[2],
            element.nodes[3],
        ];
        let p: [[f64; 3]; 4] = [
            [mesh.nodes[idx[0]].x, mesh.nodes[idx[0]].y, mesh.nodes[idx[0]].z],
            [mesh.nodes[idx[1]].x, mesh.nodes[idx[1]].y, mesh.nodes[idx[1]].z],
            [mesh.nodes[idx[2]].x, mesh.nodes[idx[2]].y, mesh.nodes[idx[2]].z],
            [mesh.nodes[idx[3]].x, mesh.nodes[idx[3]].y, mesh.nodes[idx[3]].z],
        ];
        // Jacobian of the affine map from the reference tet.
        let j = [
            [p[1][0] - p[0][0], p[2][0] - p[0][0], p[3][0] - p[0][0]],
            [p[1][1] - p[0][1], p[2][1] - p[0][1], p[3][1] - p[0][1]],
            [p[1][2] - p[0][2], p[2][2] - p[0][2], p[3][2] - p[0][2]],
        ];
        let det = j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
            - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
            + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);
        if det.abs() < 1e-300 {
            return Err("degenerate tetrahedron in wave mesh".into());
        }
        let volume = det.abs() / 6.0;
        // Inverse Jacobian (rows), for reference gradients.
        let inv = [
            [
                (j[1][1] * j[2][2] - j[1][2] * j[2][1]) / det,
                (j[0][2] * j[2][1] - j[0][1] * j[2][2]) / det,
                (j[0][1] * j[1][2] - j[0][2] * j[1][1]) / det,
            ],
            [
                (j[1][2] * j[2][0] - j[1][0] * j[2][2]) / det,
                (j[0][0] * j[2][2] - j[0][2] * j[2][0]) / det,
                (j[0][2] * j[1][0] - j[0][0] * j[1][2]) / det,
            ],
            [
                (j[1][0] * j[2][1] - j[1][1] * j[2][0]) / det,
                (j[0][1] * j[2][0] - j[0][0] * j[1][2]) / det,
                (j[0][0] * j[1][1] - j[0][1] * j[2][0]) / det,
            ],
        ];
        // Reference gradients: N0=(-1,-1,-1), N1=(1,0,0), N2=(0,1,0), N3=(0,0,1).
        let ref_grads = [
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        // Physical gradients: grad N = J^{-T} grad_ref N.
        let mut grads = [[0.0f64; 3]; 4];
        for a in 0..4 {
            for component in 0..3 {
                grads[a][component] = inv[0][component] * ref_grads[a][0]
                    + inv[1][component] * ref_grads[a][1]
                    + inv[2][component] * ref_grads[a][2];
            }
        }
        let mut ke = [[0.0f64; 4]; 4];
        for a in 0..4 {
            for b in 0..4 {
                ke[a][b] = volume
                    * (grads[a][0] * grads[b][0]
                        + grads[a][1] * grads[b][1]
                        + grads[a][2] * grads[b][2]);
            }
        }
        for &node in &idx {
            lumped[node] += volume / 4.0;
        }
        tets.push(WaveTet { nodes: idx, ke });
    }
    if lumped.iter().any(|&m| m <= 0.0) {
        return Err("wave mesh has a node without mass".into());
    }
    Ok((tets, lumped))
}

/// Propeller tonal noise: three rotating axial loading dipoles radiate
/// into a sponge-terminated free field (linear wave equation, P1 FEM).
///
/// Source model: each blade is a compact dipole (node pair straddling the
/// rotor plane, axis along the thrust direction) whose strength pulsates
/// at blade-passage frequency as the blade sweeps. Dipole order matches
/// loading noise; the film shows the radiating tonal field, not SPL.
/// Tip Mach is exaggerated for wavelength legibility, as with the other
/// demos' "short, legible movie" parameters.
fn propeller_noise(out_dir: &Path) -> Result<(), String> {
    use sonium_fem::mesh::generators::box_mesh_tetrahedra;

    // Domain and discretisation. Box edge 4 m holds ~2 wavelengths at the
    // blade-passage frequency below; 20 cells/edge keeps h <= lambda/10.
    let (lx, ly, lz) = (4.0f64, 4.0f64, 4.0f64);
    let mesh = box_mesh_tetrahedra(0.0, lx, 0.0, ly, 0.0, lz, 40, 40, 40);
    let n = mesh.nodes.len();
    let c = 343.0f64;
    let center = [lx / 2.0, ly / 2.0, lz / 2.0];

    // Rotor: 3 blades at 60 rev/s (blade-passage 180 Hz, lambda 1.9 m),
    // radius 0.55 m, tip speed 207 m/s (Mach 0.60).
    let blades = 3usize;
    let radius = 0.55f64;
    let omega = 2.0 * std::f64::consts::PI * 60.0;
    // Dipole half-separation: pair nodes straddle the rotor plane so the
    // pair acts as one axial dipole of moment ~ 2 * delta * strength.
    let delta = 0.15f64;
    // Source strength scale: tuned so far pressure peaks are O(1 Pa) with
    // the dt^2 / lumped-mass weighting of the explicit update.
    let strength = 2.0e5f64;

    let (tets, lumped) = lumped_wave_tets(&mesh)?;
    let simple = simple_mesh_from_fem(&mesh)?;

    // Sponge layer: per-node damping factor, unity inside 1.4 m of the
    // centre, ramping to 0.90 at the walls so outgoing waves die at the
    // boundary instead of reflecting (free-field surrogate).
    let damping: Vec<f64> = mesh
        .nodes
        .iter()
        .map(|node| {
            let d = ((node.x - center[0]).powi(2)
                + (node.y - center[1]).powi(2)
                + (node.z - center[2]).powi(2))
            .sqrt();
            let t = ((d - 1.4) / 0.6).clamp(0.0, 1.0);
            1.0 - 0.10 * t * t
        })
        .collect();

    let nearest = |target: [f64; 3]| -> usize {
        mesh.nodes
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let da = (a.x - target[0]).powi(2)
                    + (a.y - target[1]).powi(2)
                    + (a.z - target[2]).powi(2);
                let db = (b.x - target[0]).powi(2)
                    + (b.y - target[1]).powi(2)
                    + (b.z - target[2]).powi(2);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(index, _)| index)
            .unwrap_or(0)
    };

    let h = lx / 40.0;
    let dt = 0.45 * h / (c * 3.0f64.sqrt());
    // Four revolutions at ~220 steps each; dump every 12 steps (~73 frames).
    let steps = 880usize;
    let frame_every = 12usize;

    let mut pressure = vec![0.0f64; n];
    let mut previous = vec![0.0f64; n];
    let mut frames = Vec::new();
    let mut times = Vec::new();
    let mut max_pressure: f64 = 0.0;
    let dump =
        |pressure: &[f64], frame_index: usize, max_pressure: &mut f64| -> Result<String, String> {
            *max_pressure =
                (*max_pressure).max(pressure.iter().fold(0.0, |acc: f64, v| acc.max(v.abs())));
            let fields = FrameFields {
                point_scalars: vec![("pressure", pressure.to_vec())],
                point_vectors: Vec::new(),
                cell_scalars: Vec::new(),
            };
            write_frame(out_dir, &format!("frame_{frame_index:04}"), &simple, &fields)
        };

    frames.push(dump(&pressure, 0, &mut max_pressure)?);
    times.push(0.0);
    let mut frame_index = 1usize;
    let mut kp = vec![0.0f64; n];
    for step in 1..=steps {
        kp.iter_mut().for_each(|v| *v = 0.0);
        for tet in &tets {
            for (a, &na) in tet.nodes.iter().enumerate() {
                let mut sum = 0.0;
                for (b, &nb) in tet.nodes.iter().enumerate() {
                    sum += tet.ke[a][b] * pressure[nb];
                }
                kp[na] += sum;
            }
        }
        // Rotating dipole forcing at this step's blade positions.
        let time = step as f64 * dt;
        let mut force = vec![0.0f64; n];
        for blade in 0..blades {
            let theta = omega * time + 2.0 * std::f64::consts::PI * blade as f64 / blades as f64;
            let (x, y) = (
                center[0] + radius * theta.cos(),
                center[1] + radius * theta.sin(),
            );
            // Loading pulsation as the blade sweeps past a fixed observer:
            // strength peaks once per blade passage (factor blades*theta).
            let load = 1.0 + 0.35 * (blades as f64 * theta).sin();
            let plus = nearest([x, y, center[2] + delta]);
            let minus = nearest([x, y, center[2] - delta]);
            force[plus] += strength * load;
            force[minus] -= strength * load;
        }
        let mut next = vec![0.0f64; n];
        for i in 0..n {
            next[i] = 2.0 * pressure[i] - previous[i] - dt * dt * c * c * kp[i] / lumped[i]
                + dt * dt * force[i] / lumped[i];
            next[i] *= damping[i];
        }
        // The sponge must damp the history term too, else the boundary
        // keeps ringing from the undamped previous state.
        for i in 0..n {
            previous[i] *= damping[i];
        }
        previous = std::mem::replace(&mut pressure, next);
        if step % frame_every == 0 || step == steps {
            frames.push(dump(&pressure, frame_index, &mut max_pressure)?);
            times.push(time);
            frame_index += 1;
        }
    }
    println!("propeller: {steps} steps, max pressure {max_pressure:.3}");
    write_manifest(
        out_dir,
        "propeller-noise",
        "Propeller tonal noise",
        &frames,
        &times,
        "pressure",
        (-max_pressure, max_pressure),
    )
}

/// Zero a CSR row and pin the diagonal, for Dirichlet enforcement
/// (mirrors the QA thermal bin's helper).
fn apply_dirichlet_csr(csr: &mut sonium_solvers::CsrMatrix<f64>, dof: usize) {
    let range = csr.row_range(dof);
    for i in range {
        csr.values[i] = 0.0;
    }
    let diag_pos = csr.row_entries(dof).position(|(j, _)| j == dof);
    if let Some(idx) = diag_pos {
        let global_idx = csr.row_ptrs[dof] + idx;
        csr.values[global_idx] = 1.0;
    }
}

/// Finned heatsink transient: a chip on the base underside heats an
/// aluminium finned plate while every exposed surface loses heat by
/// natural convection (h = 8 W/m^2K) and by radiation (eps = 0.85,
/// eps*sigma*T^4 via the thermal forms' secant coefficient). P1 FEM
/// heat conduction with backward-Euler stepping.
fn thermal_spreading(out_dir: &Path) -> Result<(), String> {
    use sonium_fem::assembly::assemble_bilinear;
    use sonium_fem::mesh::generators::box_mesh_tetrahedra;
    use sonium_fem::physics::thermal::heat_conduction::{
        ConductionForm, HeatCapacityForm, RadiationResidual, ThermalMaterial, ThermalTimeStepper,
    };
    use sonium_solvers::{LinearSolveConfig, LinearSolverType, solve_real};

    // Aluminium: k = 167 W/(m K), rho = 2700 kg/m^3, cp = 900 J/(kg K),
    // alpha ~ 6.9e-5 m^2/s. Geometry: 10 mm base plate with five 10 mm
    // fins (35 mm tall) on a 45 mm pitch, cut from a structured tet grid
    // by dropping air-gap elements. Nodes left without elements (in the
    // gaps) are pinned to ambient below; they carry no geometry.
    let (lx, ly, lz) = (0.30f64, 0.22f64, 0.045f64);
    let mut mesh = box_mesh_tetrahedra(0.0, lx, 0.0, ly, 0.0, lz, 60, 88, 18);
    let fin_bands = [
        (0.010f64, 0.020f64),
        (0.055, 0.065),
        (0.100, 0.110),
        (0.145, 0.155),
        (0.190, 0.200),
    ];
    mesh.elements.retain(|element| {
        let (mut cy, mut cz) = (0.0, 0.0);
        for &node in &element.nodes {
            cy += mesh.nodes[node].y;
            cz += mesh.nodes[node].z;
        }
        let count = element.nodes.len() as f64;
        let (cy, cz) = (cy / count, cz / count);
        cz <= 0.010 || fin_bands.iter().any(|&(lo, hi)| cy >= lo && cy <= hi)
    });
    let space = H1Space::new(PolynomialDegree::P1);
    let n_dofs = space.num_global_dofs(&mesh);

    // Map DOFs back to mesh nodes by coordinates (P1 DOF order is not
    // guaranteed to follow node order).
    let dof_coords = space
        .dof_coordinates(&mesh)
        .ok_or("P1 space did not expose nodal coordinates")?;
    let mut dof_node = vec![usize::MAX; n_dofs];
    for (dof, point) in dof_coords.iter().enumerate() {
        let node = mesh
            .nodes
            .iter()
            .position(|candidate| {
                (candidate.x - point[0]).abs() < 1e-9
                    && (candidate.y - point[1]).abs() < 1e-9
                    && (candidate.z - point[2]).abs() < 1e-9
            })
            .ok_or("thermal dof has no matching mesh node")?;
        dof_node[dof] = node;
    }

    let material = ThermalMaterial::new(167.0, 2700.0, 900.0);
    let conduction = ConductionForm::from_conductivity(167.0);
    let capacity = HeatCapacityForm::new(&material);
    let stiffness = assemble_bilinear(&conduction, &space, &space, &mesh, 1).to_csr();
    let mass = assemble_bilinear(&capacity, &space, &space, &mesh, 1).to_csr();

    // Lumped boundary area per node from exterior tet faces (faces owned
    // by exactly one element), for the convection/radiation Robin terms.
    let mut face_use: std::collections::HashMap<[usize; 3], usize> =
        std::collections::HashMap::new();
    let mut face_nodes: std::collections::HashMap<[usize; 3], [usize; 3]> =
        std::collections::HashMap::new();
    for element in &mesh.elements {
        let idx = [
            element.nodes[0],
            element.nodes[1],
            element.nodes[2],
            element.nodes[3],
        ];
        for face in [
            [idx[0], idx[1], idx[2]],
            [idx[0], idx[1], idx[3]],
            [idx[0], idx[2], idx[3]],
            [idx[1], idx[2], idx[3]],
        ] {
            let mut key = face;
            key.sort_unstable();
            *face_use.entry(key).or_insert(0) += 1;
            face_nodes.insert(key, face);
        }
    }
    let mut node_area = vec![0.0f64; mesh.nodes.len()];
    for (key, &count) in &face_use {
        if count == 1 {
            let face = face_nodes[key];
            let point = |i: usize| {
                let node = &mesh.nodes[face[i]];
                [node.x, node.y, node.z]
            };
            let (a, b, c) = (point(0), point(1), point(2));
            let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let cross = [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ];
            let area =
                0.5 * (cross[0].powi(2) + cross[1].powi(2) + cross[2].powi(2)).sqrt();
            for &node in &face {
                node_area[node] += area / 3.0;
            }
        }
    }
    let dof_area: Vec<f64> = dof_node.iter().map(|&node| node_area[node]).collect();
    // Orphaned nodes (inside air gaps) carry no elements: pin to ambient.
    let mut node_used = vec![false; mesh.nodes.len()];
    for element in &mesh.elements {
        for &node in &element.nodes {
            node_used[node] = true;
        }
    }
    // Solve on solid DOFs only: air-gap nodes have no elements (zero
    // matrix rows, no diagonal to pin), so eliminate them from the
    // system and hold them at ambient in the output field.
    let mut reduced_index = vec![usize::MAX; n_dofs];
    let mut active_dofs: Vec<usize> = Vec::new();
    for dof in 0..n_dofs {
        if node_used[dof_node[dof]] {
            reduced_index[dof] = active_dofs.len();
            active_dofs.push(dof);
        }
    }
    let reduce = |matrix: &sonium_solvers::CsrMatrix<f64>| {
        let mut triplets = Vec::new();
        for (ri, &dof) in active_dofs.iter().enumerate() {
            for (col, value) in matrix.row_entries(dof) {
                let ci = reduced_index[col];
                if ci != usize::MAX {
                    triplets.push((ri, ci, value));
                }
            }
        }
        sonium_solvers::CsrMatrix::from_triplets(
            active_dofs.len(),
            active_dofs.len(),
            triplets,
        )
    };
    let stiffness = reduce(&stiffness);
    let mass = reduce(&mass);

    // Chip footprint: nodes on the bottom face within 45 mm of the centre.
    let hot_dofs: Vec<usize> = (0..n_dofs)
        .filter(|&dof| {
            let node = &mesh.nodes[dof_node[dof]];
            node.z.abs() < 1e-9
                && (node.x - lx / 2.0).powi(2) + (node.y - ly / 2.0).powi(2) < 0.045f64.powi(2)
        })
        .collect();
    let hot_reduced: Vec<usize> = hot_dofs.iter().map(|&dof| reduced_index[dof]).collect();
    let area_reduced: Vec<f64> = active_dofs.iter().map(|&dof| dof_area[dof]).collect();
    if hot_dofs.is_empty() {
        return Err("thermal demo found no hot-spot nodes".into());
    }
    let t_ambient = 25.0f64;
    let t_chip = 95.0f64;

    let dt = 1.5f64;
    let steps = 80usize;
    let stepper = ThermalTimeStepper::backward_euler(dt);
    // Constant conduction part; the Robin (convection + radiation)
    // diagonal is temperature-dependent and rebuilt every step.
    let base_effective = stepper.effective_matrix(&mass, &stiffness);
    // Natural convection to air plus radiation (emissivity 0.85,
    // anodised aluminium) from every exposed surface node.
    let h_conv = 8.0f64;
    let radiation = RadiationResidual::new(0.85, t_ambient + 273.15);
    let add_diagonal = |csr: &mut sonium_solvers::CsrMatrix<f64>, dof: usize, value: f64| {
        let diag_pos = csr.row_entries(dof).position(|(j, _)| j == dof);
        if let Some(idx) = diag_pos {
            let global_idx = csr.row_ptrs[dof] + idx;
            csr.values[global_idx] += value;
        }
    };
    let solve_config = LinearSolveConfig {
        solver: LinearSolverType::GmresIlu,
        max_iterations: 1000,
        tolerance: 1e-10,
        restart: 50,
        exact_residual_check_interval: 0,
        verbosity: 0,
    };

    let simple = simple_mesh_from_fem(&mesh)?;
    let mut temperature = vec![t_ambient; active_dofs.len()];
    for &ri in &hot_reduced {
        temperature[ri] = t_chip;
    }
    let zero_source = vec![0.0f64; active_dofs.len()];

    let mut frames = Vec::new();
    let mut times = Vec::new();
    let mut max_temp: f64 = t_ambient;
    let dump = |temperature: &[f64],
                frame_index: usize,
                max_temp: &mut f64|
     -> Result<String, String> {
        *max_temp = (*max_temp).max(temperature.iter().fold(0.0, |acc: f64, v| acc.max(*v)));
        // VTU point order follows mesh nodes, not DOFs; eliminated air
        // nodes stay at ambient.
        let mut nodal = vec![t_ambient; mesh.nodes.len()];
        for (ri, &value) in temperature.iter().enumerate() {
            nodal[dof_node[active_dofs[ri]]] = value;
        }
        let fields = FrameFields {
            point_scalars: vec![("temperature", nodal)],
            point_vectors: Vec::new(),
            cell_scalars: Vec::new(),
        };
        write_frame(out_dir, &format!("frame_{frame_index:04}"), &simple, &fields)
    };

    frames.push(dump(&temperature, 0, &mut max_temp)?);
    times.push(0.0);
    let mut frame_index = 1usize;
    for step in 1..=steps {
        let mut rhs = stepper.form_rhs(&temperature, &zero_source, None, &mass, &stiffness);
        let mut effective = base_effective.clone();
        // Robin loss at the current surface temperature: convection plus
        // radiation in secant form, so eps*sigma*(T^4 - T_inf^4) is exact
        // at this step's temperature (linearised_h at ambient itself).
        for ri in 0..active_dofs.len() {
            let area = area_reduced[ri];
            if area > 0.0 {
                let t_k = temperature[ri] + 273.15;
                let h_rad = if (t_k - (t_ambient + 273.15)).abs() > 1e-6 {
                    radiation.heat_flux(t_k) / (t_k - (t_ambient + 273.15))
                } else {
                    radiation.linearized_h(t_k)
                };
                let h = h_conv + h_rad.max(0.0);
                add_diagonal(&mut effective, ri, h * area);
                rhs[ri] += h * area * t_ambient;
            }
        }
        for &ri in &hot_reduced {
            apply_dirichlet_csr(&mut effective, ri);
            rhs[ri] = t_chip;
        }
        let result = solve_real(&effective, &rhs, &solve_config)
            .map_err(|e| format!("thermal solve failed at step {step}: {e}"))?;
        if !result.converged {
            return Err(format!("thermal solver did not converge at step {step}"));
        }
        temperature = result.x;
        if step % 2 == 0 || step == steps {
            frames.push(dump(&temperature, frame_index, &mut max_temp)?);
            times.push(step as f64 * dt);
            frame_index += 1;
        }
    }
    println!("thermal: {steps} steps, max temperature {max_temp:.1}");
    write_manifest(
        out_dir,
        "thermal-spreading",
        "Transient heat spreading",
        &frames,
        &times,
        "temperature",
        (t_ambient, max_temp),
    )
}

/// Square metal plate with a central circular hole, heated on the hole
/// rim until a melt front grows outward into the plate.
///
/// Phase change uses the apparent-heat-capacity method: the volumetric
/// enthalpy `H(T) = rho * (cp * T + L * f(T))`, with liquid fraction `f`
/// ramping from 0 at the solidus to 1 at the liquidus, is stepped with
/// backward Euler and a per-node secant capacity inside a Picard loop.
/// This is a thermal-only model: the molten region is tracked through
/// `f`, but the melt does not flow, slump, or lose mass.
fn melting_plate(out_dir: &Path) -> Result<(), String> {
    use sonium_fem::assembly::assemble_bilinear;
    use sonium_fem::mesh::generators::box_mesh_tetrahedra;
    use sonium_fem::physics::thermal::heat_conduction::{
        ConductionForm, HeatCapacityForm, RadiationResidual, ThermalMaterial,
    };
    use sonium_solvers::{LinearSolveConfig, LinearSolverType, solve_real};

    // Plate and hole: 300 mm square, 12 mm thick, 90 mm central hole.
    // The structured tet grid cuts the hole by dropping elements whose
    // centroid falls inside the radius, as in the heatsink demo.
    let (lx, ly, lz) = (0.30f64, 0.30f64, 0.012f64);
    let hole_radius = 0.045f64;
    let (nx, ny, nz) = (120usize, 120usize, 4usize);
    let mut mesh = box_mesh_tetrahedra(0.0, lx, 0.0, ly, 0.0, lz, nx, ny, nz);
    let (cx, cy) = (lx / 2.0, ly / 2.0);
    let radius_of = |x: f64, y: f64| ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
    mesh.elements.retain(|element| {
        let (mut ex, mut ey) = (0.0, 0.0);
        for &node in &element.nodes {
            ex += mesh.nodes[node].x;
            ey += mesh.nodes[node].y;
        }
        let count = element.nodes.len() as f64;
        radius_of(ex / count, ey / count) > hole_radius
    });

    // Aluminium alloy, matching the heatsink demo (k = 167 W/(m K)).
    // The 580-650 C melting range and 380 kJ/kg latent heat are
    // representative alloy values (pure aluminium melts sharply at
    // 660 C); the range gives a visible mushy zone for the film.
    let material = ThermalMaterial::new(167.0, 2700.0, 900.0);
    let t_solidus = 580.0f64;
    let t_liquidus = 650.0f64;
    let latent_heat = 380_000.0f64;
    // A hot mandrel in the hole holds the rim well above the liquidus
    // so the front keeps moving for the whole film.
    let t_rim = 950.0f64;
    let t_ambient = 25.0f64;
    let h_conv = 8.0f64;
    let radiation = RadiationResidual::new(0.85, t_ambient + 273.15);

    let liquid_fraction = |t: f64| ((t - t_solidus) / (t_liquidus - t_solidus)).clamp(0.0, 1.0);
    // Volumetric enthalpy up to an arbitrary reference (only
    // differences enter the time stepping).
    let enthalpy = |t: f64| {
        material.density * (material.specific_heat * t + latent_heat * liquid_fraction(t))
    };
    let enthalpy_derivative = |t: f64| {
        let mushy = if t > t_solidus && t < t_liquidus {
            latent_heat / (t_liquidus - t_solidus)
        } else {
            0.0
        };
        material.density * (material.specific_heat + mushy)
    };

    let space = H1Space::new(PolynomialDegree::P1);
    let n_dofs = space.num_global_dofs(&mesh);
    let dof_coords = space
        .dof_coordinates(&mesh)
        .ok_or("P1 space did not expose nodal coordinates")?;
    let mut dof_node = vec![usize::MAX; n_dofs];
    for (dof, point) in dof_coords.iter().enumerate() {
        let node = mesh
            .nodes
            .iter()
            .position(|candidate| {
                (candidate.x - point[0]).abs() < 1e-9
                    && (candidate.y - point[1]).abs() < 1e-9
                    && (candidate.z - point[2]).abs() < 1e-9
            })
            .ok_or("melting-plate dof has no matching mesh node")?;
        dof_node[dof] = node;
    }

    let conduction = ConductionForm::from_conductivity(material.conductivity);
    let capacity = HeatCapacityForm::new(&material);
    let stiffness_full = assemble_bilinear(&conduction, &space, &space, &mesh, 1).to_csr();
    let mass_full = assemble_bilinear(&capacity, &space, &space, &mesh, 1).to_csr();
    // Lumped nodal volumes from the consistent mass row sums
    // (row sum / (rho * cp)); the Picard capacity scales these.
    let rho_cp = material.volumetric_heat_capacity();
    let dof_volume: Vec<f64> = (0..n_dofs)
        .map(|dof| mass_full.row_entries(dof).map(|(_, value)| value).sum::<f64>() / rho_cp)
        .collect();

    // Lumped boundary area per node from exterior tet faces, for the
    // convection/radiation Robin terms (same scheme as the heatsink).
    let mut face_use: std::collections::HashMap<[usize; 3], usize> = std::collections::HashMap::new();
    let mut face_nodes: std::collections::HashMap<[usize; 3], [usize; 3]> =
        std::collections::HashMap::new();
    for element in &mesh.elements {
        let idx = [
            element.nodes[0],
            element.nodes[1],
            element.nodes[2],
            element.nodes[3],
        ];
        for face in [
            [idx[0], idx[1], idx[2]],
            [idx[0], idx[1], idx[3]],
            [idx[0], idx[2], idx[3]],
            [idx[1], idx[2], idx[3]],
        ] {
            let mut key = face;
            key.sort_unstable();
            *face_use.entry(key).or_insert(0) += 1;
            face_nodes.insert(key, face);
        }
    }
    let mut node_area = vec![0.0f64; mesh.nodes.len()];
    for (key, &count) in &face_use {
        if count == 1 {
            let face = face_nodes[key];
            let point = |i: usize| {
                let node = &mesh.nodes[face[i]];
                [node.x, node.y, node.z]
            };
            let (a, b, c) = (point(0), point(1), point(2));
            let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let cross = [
                u[1] * v[2] - u[2] * v[1],
                u[0] * v[2] - u[2] * v[0],
                u[1] * v[0] - u[0] * v[1],
            ];
            let area = 0.5 * (cross[0].powi(2) + cross[1].powi(2) + cross[2].powi(2)).sqrt();
            for &node in &face {
                node_area[node] += area / 3.0;
            }
        }
    }

    // Solve on solid DOFs only: nodes left inside the hole carry no
    // elements and are held at ambient in the output field.
    let mut node_used = vec![false; mesh.nodes.len()];
    for element in &mesh.elements {
        for &node in &element.nodes {
            node_used[node] = true;
        }
    }
    let mut reduced_index = vec![usize::MAX; n_dofs];
    let mut active_dofs: Vec<usize> = Vec::new();
    for dof in 0..n_dofs {
        if node_used[dof_node[dof]] {
            reduced_index[dof] = active_dofs.len();
            active_dofs.push(dof);
        }
    }
    let reduce = |matrix: &sonium_solvers::CsrMatrix<f64>| {
        let mut triplets = Vec::new();
        for (ri, &dof) in active_dofs.iter().enumerate() {
            for (col, value) in matrix.row_entries(dof) {
                let ci = reduced_index[col];
                if ci != usize::MAX {
                    triplets.push((ri, ci, value));
                }
            }
        }
        sonium_solvers::CsrMatrix::from_triplets(active_dofs.len(), active_dofs.len(), triplets)
    };
    let stiffness = reduce(&stiffness_full);
    let n_active = active_dofs.len();
    let volume_reduced: Vec<f64> = active_dofs.iter().map(|&dof| dof_volume[dof]).collect();
    let area_reduced: Vec<f64> = active_dofs
        .iter()
        .map(|&dof| node_area[dof_node[dof]])
        .collect();
    // Rim DOFs: the first solid ring outside the cut hole. The 0.75 *
    // dx band reaches the ring that bounds the retained elements.
    let dx = lx / nx as f64;
    let rim_reduced: Vec<usize> = active_dofs
        .iter()
        .enumerate()
        .filter(|&(_, &dof)| {
            let node = &mesh.nodes[dof_node[dof]];
            radius_of(node.x, node.y) < hole_radius + 0.75 * dx
        })
        .map(|(ri, _)| ri)
        .collect();
    if rim_reduced.is_empty() {
        return Err("melting plate found no hole-rim nodes".into());
    }
    let is_rim = {
        let mut flags = vec![false; n_active];
        for &ri in &rim_reduced {
            flags[ri] = true;
        }
        flags
    };

    let add_diagonal = |csr: &mut sonium_solvers::CsrMatrix<f64>, dof: usize, value: f64| {
        let diag_pos = csr.row_entries(dof).position(|(j, _)| j == dof);
        if let Some(idx) = diag_pos {
            let global_idx = csr.row_ptrs[dof] + idx;
            csr.values[global_idx] += value;
        }
    };
    let solve_config = || LinearSolveConfig {
        solver: LinearSolverType::GmresIlu,
        max_iterations: 1000,
        tolerance: 1e-10,
        restart: 50,
        exact_residual_check_interval: 0,
        verbosity: 0,
    };

    // 300 s of heating at dt = 2 s: the conduction length
    // sqrt(alpha * t) reaches ~0.14 m, so the front crosses much of
    // the plate half-width inside the film.
    let dt = 2.0f64;
    let steps = 150usize;
    let max_picard = 10usize;
    let picard_tol = 0.5f64;

    let simple = simple_mesh_from_fem(&mesh)?;
    let mut temperature = vec![t_ambient; n_active];
    for &ri in &rim_reduced {
        temperature[ri] = t_rim;
    }

    let mut frames = Vec::new();
    let mut times = Vec::new();
    let dump = |temperature: &[f64], frame_index: usize| -> Result<String, String> {
        let mut nodal_t = vec![t_ambient; mesh.nodes.len()];
        let mut nodal_f = vec![0.0f64; mesh.nodes.len()];
        for (ri, &value) in temperature.iter().enumerate() {
            nodal_t[dof_node[active_dofs[ri]]] = value;
            nodal_f[dof_node[active_dofs[ri]]] = liquid_fraction(value);
        }
        let fields = FrameFields {
            point_scalars: vec![
                ("temperature", nodal_t),
                ("liquid_fraction", nodal_f),
            ],
            point_vectors: Vec::new(),
            cell_scalars: Vec::new(),
        };
        write_frame(out_dir, &format!("frame_{frame_index:04}"), &simple, &fields)
    };

    frames.push(dump(&temperature, 0)?);
    times.push(0.0);
    let mut frame_index = 1usize;
    let mut total_picard = 0usize;
    for step in 1..=steps {
        let previous = temperature.clone();
        let mut iterate = previous.clone();
        for _ in 0..max_picard {
            total_picard += 1;
            // Secant apparent capacity per node between the previous
            // step and the current iterate (derivative on the first
            // pass, where the two states coincide).
            let mut effective = stiffness.clone();
            let mut rhs = vec![0.0f64; n_active];
            for ri in 0..n_active {
                if is_rim[ri] {
                    continue;
                }
                let (t_new, t_old) = (iterate[ri], previous[ri]);
                let capacity = if (t_new - t_old).abs() > 1e-9 {
                    (enthalpy(t_new) - enthalpy(t_old)) / (t_new - t_old)
                } else {
                    enthalpy_derivative(t_old)
                };
                let cap_rate = capacity * volume_reduced[ri] / dt;
                add_diagonal(&mut effective, ri, cap_rate);
                rhs[ri] = cap_rate * t_old;
                // Robin loss at the iterate temperature, convection
                // plus exact-secant T^4 radiation, as in the heatsink.
                let area = area_reduced[ri];
                if area > 0.0 {
                    let t_k = iterate[ri] + 273.15;
                    let h_rad = if (t_k - (t_ambient + 273.15)).abs() > 1e-6 {
                        radiation.heat_flux(t_k) / (t_k - (t_ambient + 273.15))
                    } else {
                        radiation.linearized_h(t_k)
                    };
                    let h = h_conv + h_rad.max(0.0);
                    add_diagonal(&mut effective, ri, h * area);
                    rhs[ri] += h * area * t_ambient;
                }
            }
            for &ri in &rim_reduced {
                apply_dirichlet_csr(&mut effective, ri);
                rhs[ri] = t_rim;
            }
            let result = solve_real(&effective, &rhs, &solve_config())
                .map_err(|e| format!("melting-plate solve failed at step {step}: {e}"))?;
            if !result.converged {
                return Err(format!("melting-plate solver did not converge at step {step}"));
            }
            let delta = result
                .x
                .iter()
                .zip(iterate.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            iterate = result.x;
            if delta < picard_tol {
                break;
            }
        }
        temperature = iterate;
        if step % 2 == 0 || step == steps {
            frames.push(dump(&temperature, frame_index)?);
            times.push(step as f64 * dt);
            frame_index += 1;
        }
    }
    let max_temp = temperature.iter().fold(0.0f64, |acc: f64, v| acc.max(*v));
    let melted = temperature
        .iter()
        .filter(|&&t| liquid_fraction(t) > 0.99)
        .count();
    println!(
        "melting-plate: {steps} steps, {} Picard solves, max temperature {max_temp:.1}, fully molten nodes {melted}/{n_active}",
        total_picard
    );
    write_manifest(
        out_dir,
        "melting-plate",
        "Melting plate",
        &frames,
        &times,
        "liquid_fraction",
        (0.0, 1.0),
    )
}

/// Measured outcome of one sloshing run (movie or sweep point).
struct SloshStats {
    /// Largest surface-elevation deviation from the fill depth at
    /// either wall probe, over the whole run (m).
    max_eta_deviation: f64,
    /// Largest nodal speed over the whole run (m/s).
    max_speed: f64,
    /// Water volume fraction at the end (conservation check).
    final_water_fraction: f64,
}

/// Sloshing water tank core: a 2 m x 0.5 m tank filled to 0.20 m is
/// shaken horizontally at its first sloshing resonance (~0.34 Hz) in
/// the tank frame, i.e. fixed walls plus an oscillating horizontal
/// gravity component. Two-phase model: BDF2 Navier-Stokes with
/// element-local density/viscosity interpolated from the phase field
/// (solver `element_density` + transported per-element viscosity),
/// segregated with an implicit Cahn-Hilliard phase-field step
/// advected by the fresh velocity. Time stepping is adaptive:
/// a CFL controller sizes each step from the fresh peak speed, and
/// the BDF2 step uses the true variable-step coefficients.
///
/// Documented approximations: properties are lagged one step behind
/// the phase field; capillary force on the momentum is neglected,
/// which the Bond number (~5e5 for this tank) justifies; surface
/// tension enters only through the phase-field interface profile;
/// stabilisation keeps reference (water) properties; the light
/// phase carries a bulk Rayleigh drag (25 1/s, weighted by the air
/// fraction) that keeps the air quasi-steady -- with true air
/// inertia the unstabilised coupling otherwise grows a spurious
/// air-layer gyre that destroys the phase field over a few hundred
/// steps (see the drag site below).
fn sloshing_core(
    drive_amplitude: f64,
    nx: usize,
    ny: usize,
    steps: usize,
    out_dir: Option<&Path>,
) -> Result<SloshStats, String> {
    use sonium_fem::assembly::{TripletMatrix, assemble_bilinear};
    use sonium_fem::mesh::generators::rectangular_mesh_triangles;
    use sonium_fem::physics::fluid::phase_field_flow::{
        PhaseFieldFluid, PhaseMassForm, assemble_linearized_cahn_hilliard, planar_interface_phase,
    };
    use sonium_fem::physics::fluid::stokes::navier_stokes::solve_bdf_step_cached_with_force;
    use sonium_fem::physics::fluid::stokes::time_stepping::AdaptiveStepController;
    use sonium_solvers::SparseLuFactorization;

    // Tank and fill (metres): quasi-2D section of the 0.1 m wide box.
    let (tank_l, tank_h, fill_depth) = (2.0f64, 0.5f64, 0.20f64);
    let (dx, dy) = (tank_l / nx as f64, tank_h / ny as f64);
    let gravity = 9.81f64;
    let (rho_water, rho_air) = (998.0f64, 1.2f64);
    // Viscosity is a named study parameter; the film uses water's
    // 1.0e-3 Pa s so the resonant growth is barely damped.
    let viscosity = 0.001f64;

    // First sloshing mode of the fill: omega^2 = g k tanh(k h).
    let wave_k = std::f64::consts::PI / tank_l;
    let omega_res = (gravity * wave_k * (wave_k * fill_depth).tanh()).sqrt();
    let drive_frequency = omega_res / std::f64::consts::TAU;
    // Tank displacement amplitude arrives as a parameter;
    // pseudo-acceleration in the tank frame peaks at
    // amplitude * omega^2.
    let drive_omega = std::f64::consts::TAU * drive_frequency;

    let material = PhaseFieldFluid {
        density_minus: rho_air,
        density_plus: rho_water,
        viscosity_minus: 1.8e-5,
        viscosity_plus: 1.0e-3,
        surface_tension: 0.072,
        interface_thickness: 0.03,
        mobility: 5.0e-6,
    }
    .validate()
    .map_err(|e| format!("phase material: {e:?}"))?;
    let beta = material.double_well_coefficient();

    let mesh = rectangular_mesh_triangles(0.0, tank_l, 0.0, tank_h, nx, ny);
    let v_space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let p_space = H1Space::new(PolynomialDegree::P1);
    let phase_space = H1Space::new(PolynomialDegree::P1);
    let n_v = v_space.num_global_dofs(&mesh);
    let n_p = p_space.num_global_dofs(&mesh);
    let n_c = phase_space.num_global_dofs(&mesh);
    let n_nodes = mesh.nodes.len();

    // Grid index helpers: the structured mesh lets every closure
    // (body force, advection velocity) evaluate nodal fields by
    // bilinear interpolation in O(1).
    let grid_size = (nx + 1) * (ny + 1);
    let node_grid: Vec<usize> = mesh
        .nodes
        .iter()
        .map(|node| {
            let i = (node.x / dx).round() as usize;
            let j = (node.y / dy).round() as usize;
            j * (nx + 1) + i
        })
        .collect();
    let phase_coords = phase_space
        .dof_coordinates(&mesh)
        .ok_or("P1 phase space did not expose nodal coordinates")?;
    let mut grid_phase_dof = vec![usize::MAX; grid_size];
    for (dof, point) in phase_coords.iter().enumerate() {
        let node = mesh
            .nodes
            .iter()
            .position(|candidate| {
                (candidate.x - point[0]).abs() < 1e-9 && (candidate.y - point[1]).abs() < 1e-9
            })
            .ok_or("phase dof has no matching mesh node")?;
        grid_phase_dof[node_grid[node]] = dof;
    }
    if grid_phase_dof.contains(&usize::MAX) {
        return Err("phase grid mapping is incomplete".into());
    }
    let interp = move |grid: &[f64], x: f64, y: f64| -> f64 {
        let fx = (x / dx).clamp(0.0, nx as f64 - 1e-9);
        let fy = (y / dy).clamp(0.0, ny as f64 - 1e-9);
        let (i, j) = (fx.floor() as usize, fy.floor() as usize);
        let (tx, ty) = (fx - i as f64, fy - j as f64);
        let idx = |i: usize, j: usize| j * (nx + 1) + i;
        grid[idx(i, j)] * (1.0 - tx) * (1.0 - ty)
            + grid[idx(i + 1, j)] * tx * (1.0 - ty)
            + grid[idx(i, j + 1)] * (1.0 - tx) * ty
            + grid[idx(i + 1, j + 1)] * tx * ty
    };

    // Velocity DOF pairs mapped back to mesh nodes (cylinder pattern).
    let velocity_coords = v_space
        .dof_coordinates(&mesh)
        .ok_or("P1 velocity space did not expose nodal coordinates")?;
    if velocity_coords.len() != 2 * n_nodes {
        return Err("velocity dof layout is not nodal pairs".to_string());
    }
    let mut dof_pair_node = vec![usize::MAX; n_nodes];
    for pair in 0..n_nodes {
        let point = velocity_coords[2 * pair];
        let node = mesh
            .nodes
            .iter()
            .position(|candidate| {
                (candidate.x - point[0]).abs() < 1e-9 && (candidate.y - point[1]).abs() < 1e-9
            })
            .ok_or("velocity dof pair has no matching mesh node")?;
        dof_pair_node[pair] = node;
    }
    let nodal_velocity = |state: &BdfState| -> Vec<[f64; 2]> {
        let mut vel = vec![[0.0f64; 2]; n_nodes];
        for pair in 0..n_nodes {
            vel[dof_pair_node[pair]] = [state.velocity[2 * pair], state.velocity[2 * pair + 1]];
        }
        vel
    };

    // Navier-Stokes setup: fixed walls (tank frame), pressure pinned
    // at the top-left corner node.
    let geom_tol = 1e-9;
    let bc_dofs = locate_dofs_geometrical(&mesh, &v_space, |x, y, _z| {
        x < geom_tol || (tank_l - x) < geom_tol || y < geom_tol || (tank_h - y) < geom_tol
    });
    // Reference properties are water; the element-local fields set
    // per step below carry the actual phase-interpolated values.
    // Base molecular viscosity is air's, so the transported field
    // (mu(c) - mu_air) / rho_water stays nonnegative everywhere.
    let mut config = NavierStokesConfig {
        mu: 1.8e-5,
        rho: rho_water,
        use_direct_solver: false,
        linear_solver_max_iterations: 3000,
        linear_solver_tolerance: 1e-8,
        // Schur-complement preconditioning: plain ILU stagnates on
        // the variable-density saddle system (the BDF mass block
        // jumps by the 830:1 air/water contrast across the
        // interface), which is also why the solver's element-density
        // unit tests use this preconditioner.
        linear_solver_cpu_preconditioner: LinearSolverCpuPreconditionerKind::IlutSchur,
        ..Default::default()
    };
    // Nominal dt = 0.0125 s (~232 steps per drive period); an
    // adaptive CFL controller (target 0.5 on the grid spacing, dt in
    // [dt/20, 2 dt], growth at most 1.25x per step) shrinks the step
    // as the slosh speeds up, so fast motion stays resolved in time
    // without paying the small step at rest. The run advances to the
    // same total time `steps * dt_nominal` as the fixed-step version;
    // `steps` now only sets that horizon, and `max_steps` bounds the
    // adaptive step count (reached only if dt sits at its floor).
    let dt_nominal = 0.0125f64;
    let mut stepper =
        AdaptiveStepController::new(dt_nominal, 0.5, dt_nominal / 20.0, dt_nominal * 2.0, 1.25);
    let t_end = dt_nominal * steps as f64;
    let max_steps = steps * 24 + 16;
    let mut bdf = BdfConfig::bdf2(dt_nominal, steps);
    let mut workspace = BdfWorkspace::new(&mesh, &v_space, &p_space);
    let gauge = mesh
        .nodes
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let da = a.x.powi(2) + (a.y - tank_h).powi(2);
            let db = b.x.powi(2) + (b.y - tank_h).powi(2);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(index, _)| index)
        .ok_or("tank mesh has no nodes")?;
    workspace.set_pressure_gauge(gauge, 0.0);
    let bc_zero = |_x: f64, _y: f64, _z: f64| vec![0.0, 0.0];
    let mut state = BdfState::from_rest(n_v, n_p);

    // Phase field: flat interface at the fill depth (c = +1 water).
    let mut phase: Vec<f64> = phase_coords
        .iter()
        .map(|point| planar_interface_phase(fill_depth - point[1], material.interface_thickness))
        .collect();
    let mass = assemble_bilinear(
        &PhaseMassForm { coefficient: 1.0 },
        &phase_space,
        &phase_space,
        &mesh,
        2,
    )
    .to_csr();
    // The mixed Cahn-Hilliard system is indefinite and defeats ILU
    // preconditioners (zero pivots on the doubled mesh), so each
    // step factorizes it directly once and reuses the factorization
    // across the Picard passes (the matrix is step-constant).
    let phase_grid = |phase: &[f64]| -> Vec<f64> {
        let mut grid = vec![0.0f64; grid_size];
        for (grid_index, &dof) in grid_phase_dof.iter().enumerate() {
            grid[grid_index] = phase[dof];
        }
        grid
    };
    let water_volume = |phase: &[f64]| -> f64 {
        let fraction: Vec<f64> = phase.iter().map(|c| (c + 1.0) / 2.0).collect();
        let weighted = mass.matvec(&fraction);
        let total = mass.matvec(&vec![1.0f64; n_c]);
        weighted.iter().sum::<f64>() / total.iter().sum::<f64>()
    };
    let initial_water = water_volume(&phase);

    // Phase DOFs are nodal (P1): precompute the dof -> mesh-node map.
    let phase_dof_node: Vec<usize> = phase_coords
        .iter()
        .map(|point| {
            mesh.nodes
                .iter()
                .position(|candidate| {
                    (candidate.x - point[0]).abs() < 1e-9 && (candidate.y - point[1]).abs() < 1e-9
                })
                .unwrap_or(usize::MAX)
        })
        .collect();

    let simple = simple_mesh_from_fem(&mesh)?;
    let mut frames = Vec::new();
    let mut times = Vec::new();
    let dump = |state: &BdfState,
                phase: &[f64],
                frame_index: usize,
                out: &Path|
     -> Result<String, String> {
        let vel = nodal_velocity(state);
        let mut velocity = vec![[0.0f64; 3]; n_nodes];
        let mut speed = vec![0.0f64; n_nodes];
        let mut water = vec![0.0f64; n_nodes];
        let mut phase_node = vec![0.0f64; n_nodes];
        for (dof, &node) in phase_dof_node.iter().enumerate() {
            phase_node[node] = phase[dof];
        }
        for node in 0..n_nodes {
            velocity[node] = [vel[node][0], vel[node][1], 0.0];
            speed[node] = (vel[node][0].powi(2) + vel[node][1].powi(2)).sqrt();
            water[node] = ((phase_node[node] + 1.0) / 2.0).clamp(0.0, 1.0);
        }
        let fields = FrameFields {
            point_scalars: vec![("water_fraction", water), ("speed", speed)],
            point_vectors: vec![("velocity", velocity)],
            cell_scalars: Vec::new(),
        };
        write_frame(out, &format!("frame_{frame_index:04}"), &simple, &fields)
    };

    if let Some(out) = out_dir {
        frames.push(dump(&state, &phase, 0, out)?);
        times.push(0.0);
    }
    let mut frame_index = 1usize;
    let mut stats = SloshStats {
        max_eta_deviation: 0.0,
        max_speed: 0.0,
        final_water_fraction: initial_water,
    };
    // Surface elevation at a wall-probe column: first 0.5-fraction
    // crossing scanning up the phase grid.
    let eta_at = |grid: &[f64], i: usize| -> f64 {
        for j in 0..ny {
            let f0 = (grid[j * (nx + 1) + i] + 1.0) / 2.0;
            let f1 = (grid[(j + 1) * (nx + 1) + i] + 1.0) / 2.0;
            if (f0 - 0.5) * (f1 - 0.5) <= 0.0 && f0 != f1 {
                return (j as f64 + (0.5 - f0) / (f1 - f0)) * dy;
            }
        }
        fill_depth
    };
    let mut previous_dt: Option<f64> = None;
    let mut step = 0usize;
    while state.time < t_end - 1.0e-12 {
        if step >= max_steps {
            return Err(format!(
                "sloshing exceeded {max_steps} adaptive steps before t = {t_end:.3} s"
            ));
        }
        // Final partial step lands exactly on the time horizon.
        let dt = stepper.current_dt().min(t_end - state.time);
        bdf.dt = dt;
        bdf.previous_dt = previous_dt;
        let t_next = state.time + dt;
        // Momentum step with phase-weighted gravity at t_next: the
        // horizontal component is the tank-frame pseudo-acceleration
        // for displacement amplitude * sin(omega t).
        let grid_snapshot = phase_grid(&phase);
        // Momentum properties sample a mollified phase (two 3x3 box
        // passes over the structured grid): a sharp elementwise
        // density jump at the interface excites parasitic jets that
        // kill the momentum iteration. The gravity body force keeps
        // the sharp field.
        let grid_momentum = {
            let mut current = grid_snapshot.clone();
            for _ in 0..2 {
                let mut next = current.clone();
                for j in 0..=ny {
                    for i in 0..=nx {
                        let mut sum = 0.0;
                        let mut count = 0.0;
                        for dj in -1i32..=1 {
                            for di in -1i32..=1 {
                                let (ii, jj) = (i as i32 + di, j as i32 + dj);
                                if ii >= 0 && ii <= nx as i32 && jj >= 0 && jj <= ny as i32 {
                                    sum += current[jj as usize * (nx + 1) + ii as usize];
                                    count += 1.0;
                                }
                            }
                        }
                        next[j * (nx + 1) + i] = sum / count;
                    }
                }
                current = next;
            }
            current
        };
        // Element-local viscosity and density from the lagged phase
        // field, through the transported per-element channels. Both
        // sample the same mollified fraction at the element centroid,
        // and gravity below uses the same density law, so inertia,
        // convection, and body force see one density field.
        let mut eddy_elements = vec![0.0f64; mesh.num_elements()];
        let mut density_elements = vec![0.0f64; mesh.num_elements()];
        for (index, element) in mesh.elements.iter().enumerate() {
            let (mut cx_e, mut cy_e) = (0.0, 0.0);
            for &node in &element.nodes {
                cx_e += mesh.nodes[node].x;
                cy_e += mesh.nodes[node].y;
            }
            let count = element.nodes.len() as f64;
            let fraction =
                ((interp(&grid_momentum, cx_e / count, cy_e / count) + 1.0) / 2.0).clamp(0.0, 1.0);
            // Momentum viscosity on the air side is floored at
            // 2e-4 Pa s (~11x physical air): true air viscosity
            // leaves the light phase's velocity almost undamped and
            // spurious air jets kill the iteration. Water keeps the
            // physical value via the interpolation endpoint.
            let mu_e = 2.0e-4 + (viscosity - 2.0e-4) * fraction;
            // Interface-band numerical viscosity (peak at fraction
            // 0.5, zero in bulk): damps the parasitic currents that
            // otherwise grow at the density jump and kill the
            // momentum iteration. Scales with mesh size, so it
            // vanishes under refinement.
            let band = 4.0 * fraction * (1.0 - fraction);
            let mu_total = mu_e + band * rho_water * dx * 0.02;
            eddy_elements[index] = ((mu_total - 1.8e-5) / rho_water).max(0.0);
            density_elements[index] = rho_air + (rho_water - rho_air) * fraction;
        }
        config.element_density = Some(density_elements);
        config.transported_eddy_viscosity_per_element = Some(eddy_elements);
        let gx = drive_amplitude * drive_omega * drive_omega * (drive_omega * t_next).sin();
        // Gravity samples the same mollified phase density as the
        // element inertia above (no separate floor), so the body
        // force has no sharp jump at the interface either and the
        // hydrostatic balance sees one density field.
        let force_grid = grid_momentum.clone();
        let force = move |coords: [f64; 3]| -> Vec<f64> {
            let fraction =
                ((interp(&force_grid, coords[0], coords[1]) + 1.0) / 2.0).clamp(0.0, 1.0);
            let rho = rho_air + (rho_water - rho_air) * fraction;
            vec![rho * gx, -rho * gravity, 0.0]
        };
        // Bulk Rayleigh drag on the light phase: with true air
        // inertia the unstabilised P1-P1 coupling excites a spurious
        // air-layer gyre (discrete vorticity generated at the
        // density jump, amplified ~rho_water/rho_air over the
        // physical air response) that advects the phase field into
        // garbage over a few hundred steps; viscosity, density
        // floors, and smaller steps do not reach it. The air's
        // physical role here is only to transmit pressure to the
        // interface, so its velocity is relaxed toward the tank
        // frame at 25 1/s (~11x the drive frequency, well above the
        // gyre growth rate): through the body-force Jacobian
        // channel (which assembles -J) J = -sigma*I adds an implicit
        // sigma*u term with sigma = rho * 25 * (1 - fraction) --
        // zero in water, full in air, and zero at rest, so the
        // hydrostatic balance and the water dynamics are untouched.
        let drag_grid = grid_momentum;
        let drag_jac = move |coords: [f64; 3]| -> [[f64; 3]; 3] {
            let fraction = ((interp(&drag_grid, coords[0], coords[1]) + 1.0) / 2.0).clamp(0.0, 1.0);
            let rho = rho_air + (rho_water - rho_air) * fraction;
            let sigma = rho * 25.0 * (1.0 - fraction);
            [[-sigma, 0.0, 0.0], [0.0, -sigma, 0.0], [0.0, 0.0, -sigma]]
        };
        if !solve_bdf_step_cached_with_force(
            &mesh,
            &v_space,
            &p_space,
            &config,
            &bdf,
            &mut state,
            &bc_zero,
            &bc_dofs,
            &[],
            force,
            drag_jac,
            &workspace,
        ) {
            return Err(format!("sloshing BDF solve failed at step {step}"));
        }

        // Phase step: three Picard passes over the linearized
        // Cahn-Hilliard system advected by the fresh velocity.
        let vel_node = nodal_velocity(&state);
        let mut vel_grid = vec![[0.0f64; 2]; grid_size];
        for node in 0..n_nodes {
            vel_grid[node_grid[node]] = vel_node[node];
        }
        let previous = phase.clone();
        let mut iterate = previous.clone();
        // Assemble and factorize the step-constant Cahn-Hilliard
        // system once; the Picard passes below only rebuild the RHS.
        let factor = {
            let vel_snapshot = vel_grid.clone();
            let velocity = move |x: f64, y: f64, _z: f64| -> [f64; 3] {
                let fx = (x / dx).clamp(0.0, nx as f64 - 1e-9);
                let fy = (y / dy).clamp(0.0, ny as f64 - 1e-9);
                let (i, j) = (fx.floor() as usize, fy.floor() as usize);
                let (tx, ty) = (fx - i as f64, fy - j as f64);
                let idx = |i: usize, j: usize| j * (nx + 1) + i;
                let mut out = [0.0f64; 3];
                for (component, out_c) in out.iter_mut().enumerate().take(2) {
                    *out_c = vel_snapshot[idx(i, j)][component] * (1.0 - tx) * (1.0 - ty)
                        + vel_snapshot[idx(i + 1, j)][component] * tx * (1.0 - ty)
                        + vel_snapshot[idx(i, j + 1)][component] * (1.0 - tx) * ty
                        + vel_snapshot[idx(i + 1, j + 1)][component] * tx * ty;
                }
                out
            };
            let system = assemble_linearized_cahn_hilliard(
                &mesh,
                PolynomialDegree::P1,
                material,
                dt,
                0.0,
                velocity,
            );
            let mut combined = TripletMatrix::new(2 * n_c, 2 * n_c);
            let push_block = |target: &mut TripletMatrix<f64>,
                              block: &TripletMatrix<f64>,
                              row_offset: usize,
                              col_offset: usize| {
                for index in 0..block.nnz() {
                    target.push(
                        block.rows[index] + row_offset,
                        block.cols[index] + col_offset,
                        block.values[index],
                    );
                }
            };
            push_block(&mut combined, &system.block_00, 0, 0);
            push_block(&mut combined, &system.block_01, 0, n_c);
            push_block(&mut combined, &system.block_10, n_c, 0);
            push_block(&mut combined, &system.block_11, n_c, n_c);
            let matrix = combined.to_csr();
            let factor = SparseLuFactorization::factorize(&matrix).map_err(|e| {
                format!("sloshing Cahn-Hilliard factorization failed at step {step}: {e}")
            })?;
            factor
        };
        for _ in 0..3 {
            // Row A: (M/dt) c_old. Row B (Picard about the iterate,
            // tangent at c = 0): -M g(c_iter) - beta M c_iter with
            // g(c) = beta (c^3 - c) in nodal-interpolated form.
            let mass_previous = mass.matvec(&previous);
            let g_nodal: Vec<f64> = iterate
                .iter()
                .map(|c| beta * (c.powi(3) - c))
                .collect();
            let mass_g = mass.matvec(&g_nodal);
            let mass_iterate = mass.matvec(&iterate);
            let mut rhs = Vec::with_capacity(2 * n_c);
            rhs.extend(mass_previous.iter().map(|value| value / dt));
            rhs.extend(
                mass_g
                    .iter()
                    .zip(mass_iterate.iter())
                    .map(|(g, mc)| -g - beta * mc),
            );
            let solution = factor.solve(&rhs).map_err(|e| {
                format!("sloshing Cahn-Hilliard solve failed at step {step}: {e}")
            })?;
            iterate = solution[..n_c].to_vec();
        }
        phase = iterate;

        // Adapt the next step to the fresh peak speed (vel_node is
        // this step's post-solve field, already used by the phase
        // step above), and record the accepted step size so the next
        // BDF step uses the true variable-step coefficients.
        let peak_speed = vel_node
            .iter()
            .map(|v| (v[0].powi(2) + v[1].powi(2)).sqrt())
            .fold(0.0f64, f64::max);
        stepper.dt_for_speed(peak_speed, dx.min(dy));
        previous_dt = Some(dt);

        if (step + 1).is_multiple_of(10) {
            let vel = nodal_velocity(&state);
            let (max_node, max_speed) = vel
                .iter()
                .enumerate()
                .map(|(node, v)| (node, (v[0].powi(2) + v[1].powi(2)).sqrt()))
                .fold((0usize, 0.0f64), |a, b| if b.1 > a.1 { b } else { a });
            println!(
                "sloshing: step {}, t {t_next:.2}/{t_end:.2} s, dt {dt:.5} s, max speed {max_speed:.3} at ({:.3}, {:.3}), water fraction {:.4} (initial {initial_water:.4}), max |phase| {:.2}",
                step + 1,
                mesh.nodes[max_node].x,
                mesh.nodes[max_node].y,
                water_volume(&phase),
                phase.iter().map(|c| c.abs()).fold(0.0f64, f64::max),
            );
        }
        if (step + 1).is_multiple_of(10) || state.time >= t_end - 1.0e-9 {
            let grid_now = phase_grid(&phase);
            for i in [3usize, nx - 3] {
                stats.max_eta_deviation = stats
                    .max_eta_deviation
                    .max((eta_at(&grid_now, i) - fill_depth).abs());
            }
            let vel = nodal_velocity(&state);
            stats.max_speed = stats.max_speed.max(
                vel.iter()
                    .map(|v| (v[0].powi(2) + v[1].powi(2)).sqrt())
                    .fold(0.0f64, f64::max),
            );
            if let Some(out) = out_dir {
                frames.push(dump(&state, &phase, frame_index, out)?);
                times.push(t_next);
                frame_index += 1;
            }
        }
        step += 1;
    }
    stats.final_water_fraction = water_volume(&phase);
    println!(
        "sloshing: {step} adaptive steps to {t_end:.2} s at {drive_frequency:.3} Hz, amplitude {drive_amplitude:.3} m, max eta deviation {:.3} m, max speed {:.3} m/s, water {:.4} (initial {initial_water:.4})",
        stats.max_eta_deviation, stats.max_speed, stats.final_water_fraction,
    );
    if let Some(out) = out_dir {
        write_manifest(
            out,
            "sloshing-tank",
            "Sloshing water tank",
            &frames,
            &times,
            "water_fraction",
            (0.0, 1.0),
        )?;
    }
    Ok(stats)
}

/// Movie wrapper: 100 mm amplitude at the doubled demo resolution
/// (200 x 50), five drive periods.
fn sloshing_tank(out_dir: &Path) -> Result<(), String> {
    sloshing_core(0.10, 200, 50, 1160, Some(out_dir))?;
    Ok(())
}

/// Driven-amplitude sweep at the base resolution: three drive
/// periods per amplitude, writing sweep.csv (no frames). The
/// 0.15/0.25/0.30 m points probe past the 0.20 m case toward
/// breaking onset; the run aborts at the first failing amplitude,
/// whose step log is the failure record.
fn sloshing_sweep(out_dir: &Path) -> Result<(), String> {
    let mut csv = String::from(
        "amplitude_m,drive_hz,max_eta_deviation_m,max_speed_m_s,final_water_fraction\n",
    );
    // Same drive frequency as the movie (first-mode resonance).
    let wave_k = std::f64::consts::PI / 2.0;
    let drive_hz = (9.81 * wave_k * (wave_k * 0.20f64).tanh()).sqrt() / std::f64::consts::TAU;
    for amplitude in [0.025f64, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30] {
        let stats = sloshing_core(amplitude, 100, 25, 696, None)?;
        csv.push_str(&format!(
            "{amplitude:?},{drive_hz:.6},{:?},{:?},{:?}\n",
            stats.max_eta_deviation, stats.max_speed, stats.final_water_fraction,
        ));
    }
    std::fs::write(out_dir.join("sweep.csv"), csv)
        .map_err(|e| format!("sweep write: {e}"))?;
    println!("sloshing sweep written to {}", out_dir.display());
    Ok(())
}

/// Flow past a cylinder at Re = 100: a Karman vortex street develops
/// behind an immersed (penalised) cylinder. Same BDF2 Navier-Stokes path
/// as the gated Taylor-Green demo; colour field is vorticity (derived by
/// the render pipeline from the velocity frames).
fn cylinder_wake(out_dir: &Path) -> Result<(), String> {
    use sonium_fem::mesh::generators::rectangular_mesh_triangles;

    // Channel and cylinder (Schäfer-Turek proportions, scaled): the
    // unsteady lift on the cylinder is the classic Aeolian dipole source.
    let (lx, ly) = (11.0f64, 4.1f64);
    let center = (2.2f64, 2.05f64);
    let radius = 0.5f64;
    // Mean inflow 1.0, diameter 1.0, nu = 0.01 gives Re = 100.
    let nu = 0.01f64;
    let u_mean = 1.0f64;
    let dt = 0.05f64;
    let steps = 500usize;

    let mesh = rectangular_mesh_triangles(0.0, lx, 0.0, ly, 440, 164);
    let v_space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let p_space = H1Space::new(PolynomialDegree::P1);
    let n_v = v_space.num_global_dofs(&mesh);
    let n_p = p_space.num_global_dofs(&mesh);
    let geom_tol = 1e-9;
    let inside = |x: f64, y: f64| {
        (x - center.0).powi(2) + (y - center.1).powi(2) < (radius * 0.99).powi(2)
    };
    let bc_dofs = locate_dofs_geometrical(&mesh, &v_space, |x, y, _z| {
        x < geom_tol || y < geom_tol || (ly - y) < geom_tol || inside(x, y)
    });
    let config = NavierStokesConfig {
        mu: nu,
        rho: 1.0,
        use_direct_solver: false,
        linear_solver_max_iterations: 8000,
        linear_solver_tolerance: 1e-6,
        linear_solver_cpu_preconditioner: LinearSolverCpuPreconditionerKind::IlutSchur,
        ..Default::default()
    };
    let bdf = BdfConfig::bdf2(dt, steps);
    let mut workspace = BdfWorkspace::new(&mesh, &v_space, &p_space);
    // Pin pressure at an outlet node (natural outflow elsewhere).
    let gauge = mesh
        .nodes
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let da = (a.x - (lx - 0.1)).powi(2) + (a.y - center.1).powi(2);
            let db = (b.x - (lx - 0.1)).powi(2) + (b.y - center.1).powi(2);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(index, _)| index)
        .ok_or("cylinder mesh has no nodes")?;
    workspace.set_pressure_gauge(gauge, 0.0);
    let velocity_coords = v_space
        .dof_coordinates(&mesh)
        .ok_or("P1 velocity space did not expose nodal coordinates")?;

    let simple = simple_mesh_from_fem(&mesh)?;
    let n_nodes = mesh.nodes.len();
    if velocity_coords.len() != 2 * n_nodes {
        return Err("velocity dof layout is not nodal pairs".to_string());
    }
    let mut dof_pair_node = vec![usize::MAX; n_nodes];
    for pair in 0..n_nodes {
        let point = velocity_coords[2 * pair];
        let node = mesh
            .nodes
            .iter()
            .position(|candidate| {
                (candidate.x - point[0]).abs() < 1e-9 && (candidate.y - point[1]).abs() < 1e-9
            })
            .ok_or("velocity dof pair has no matching mesh node")?;
        dof_pair_node[pair] = node;
    }

    // Parabolic inflow profile (peak 1.5 * mean) used for both the inlet
    // Dirichlet data and the initial condition.
    let parabola = |y: f64| 1.5 * u_mean * 4.0 * y * (ly - y) / (ly * ly);
    let bc = |x: f64, y: f64, _z: f64| {
        if inside(x, y) {
            vec![0.0, 0.0]
        } else if x < geom_tol {
            vec![parabola(y), 0.0]
        } else {
            vec![0.0, 0.0]
        }
    };
    let mut state = BdfState::from_rest(n_v, n_p);
    for (dof, point) in velocity_coords.iter().enumerate() {
        state.velocity[dof] = if inside(point[0], point[1]) {
            0.0
        } else if dof % 2 == 0 {
            parabola(point[1])
        } else {
            0.0
        };
        state.velocity_prev[dof] = state.velocity[dof];
    }

    // Optional restart from a saved per-node velocity field (x, y, u, v
    // little-endian f64 per mesh node, in mesh-node order), used to resume
    // a long run whose linear solve failed mid-way without discarding the
    // frames already on disk. The first resumed step runs BDF1 because
    // the true previous time level is not in the restart file.
    let restart_step = std::env::var("SONIUM_CYLINDER_RESTART_BIN")
        .ok()
        .map(|bin_path| -> Result<usize, String> {
            let step: usize = std::env::var("SONIUM_CYLINDER_RESTART_STEP")
                .map_err(|_| "SONIUM_CYLINDER_RESTART_STEP not set".to_string())?
                .parse()
                .map_err(|_| "bad SONIUM_CYLINDER_RESTART_STEP".to_string())?;
            let raw = std::fs::read(&bin_path)
                .map_err(|e| format!("restart file {bin_path}: {e}"))?;
            if raw.len() != 32 * n_nodes {
                return Err(format!(
                    "restart file has {} bytes, expected {}",
                    raw.len(),
                    32 * n_nodes
                ));
            }
            let mut values = vec![0.0f64; 4 * n_nodes];
            for (i, chunk) in raw.chunks_exact(8).enumerate() {
                values[i] = f64::from_le_bytes(chunk.try_into().unwrap());
            }
            for pair in 0..n_nodes {
                let node = dof_pair_node[pair];
                let (x, y) = (values[4 * node], values[4 * node + 1]);
                let point = mesh.nodes[node];
                if (point.x - x).abs() > 1e-9 || (point.y - y).abs() > 1e-9 {
                    return Err(format!("restart point {node} does not match mesh node"));
                }
                state.velocity[2 * pair] = values[4 * node + 2];
                state.velocity[2 * pair + 1] = values[4 * node + 3];
            }
            state.velocity_prev = state.velocity.clone();
            state.pressure = vec![0.0; n_p];
            state.time = dt * step as f64;
            state.step = step;
            Ok(step)
        })
        .transpose()?
        .unwrap_or(0);

    let mut frames = Vec::new();
    let mut times = Vec::new();
    let mut max_speed: f64 = 0.0;
    let dump = |state: &BdfState, step: usize, max_speed: &mut f64| {
        let mut velocity = vec![[0.0f64; 3]; n_nodes];
        let mut speed = vec![0.0f64; n_nodes];
        for pair in 0..n_nodes {
            let node = dof_pair_node[pair];
            let u = state.velocity[2 * pair];
            let v = state.velocity[2 * pair + 1];
            velocity[node] = [u, v, 0.0];
            let magnitude = (u * u + v * v).sqrt();
            *max_speed = max_speed.max(magnitude);
            speed[node] = magnitude;
        }
        let fields = FrameFields {
            point_scalars: vec![("speed", speed)],
            point_vectors: vec![("velocity", velocity)],
            cell_scalars: Vec::new(),
        };
        write_frame(out_dir, &format!("frame_{step:04}"), &simple, &fields)
    };

    if restart_step == 0 {
        frames.push(dump(&state, 0, &mut max_speed)?);
        times.push(0.0);
    } else {
        for s in (0..=restart_step).step_by(10) {
            frames.push(format!("frame_{s:04}.vtu"));
            times.push(dt * s as f64);
        }
        max_speed = state
            .velocity
            .chunks(2)
            .map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt())
            .fold(0.0f64, f64::max);
    }
    let bdf_first = BdfConfig::bdf1(dt, steps);
    for step in restart_step..steps {
        let step_config = if step == restart_step && restart_step > 0 {
            &bdf_first
        } else {
            &bdf
        };
        if !solve_bdf_step_cached(
            &mesh,
            &v_space,
            &p_space,
            &config,
            step_config,
            &mut state,
            &bc,
            &bc_dofs,
            &[],
            &workspace,
        ) {
            return Err(format!("cylinder BDF solve failed at step {step}"));
        }
        let time = dt * (step + 1) as f64;
        if (step + 1) % 10 == 0 || step + 1 == steps {
            frames.push(dump(&state, step + 1, &mut max_speed)?);
            times.push(time);
        }
    }
    println!("cylinder: {steps} steps, max speed {max_speed:.3}");
    write_manifest(
        out_dir,
        "cylinder-wake",
        "Cylinder vortex shedding",
        &frames,
        &times,
        "velocity",
        (0.0, max_speed),
    )
}

/// Compressible flow past a cylinder at freestream Mach 1: explicit P1-DG
/// compressible Navier--Stokes (inviscid limit, slip wall) on a body-fitted
/// mesh. Colour field is the local Mach number, averaged from DG cell
/// averages onto mesh nodes for the render pipeline. Mirrors the gated QA
/// case `qa-fem-compressible-cylinder-mach1` at demo resolution.
fn cylinder_mach1(out_dir: &Path) -> Result<(), String> {
    use sonium_fem::mesh::generators::channel_with_cylinder_triangles;
    use sonium_fem::physics::fluid::compressible_flow::{
        CompressibleBoundaryCondition, CompressibleDgConfig, CompressibleDgSolver,
        CompressibleDgState, ConservativeState, IdealGas, ViscosityModel,
    };
    use std::collections::HashMap;

    // Nondimensional freestream: rho = 1, c = 1, U = Mach = 1, p = 1/gamma.
    let gas = IdealGas {
        gamma: 1.4,
        gas_constant: 1.0,
    };
    let free = ConservativeState::from_primitive(1.0, [1.0, 0.0, 0.0], 1.0 / gas.gamma, gas);
    let mesh = channel_with_cylinder_triangles(-5.0, 12.0, -4.0, 4.0, 68, 32, 0.0, 0.0, 0.5);
    let simple = simple_mesh_from_fem(&mesh)?;
    let mut boundaries: HashMap<i32, CompressibleBoundaryCondition> = HashMap::new();
    boundaries.insert(1, CompressibleBoundaryCondition::FarField(free));
    boundaries.insert(2, CompressibleBoundaryCondition::FarField(free));
    boundaries.insert(3, CompressibleBoundaryCondition::FarField(free));
    boundaries.insert(4, CompressibleBoundaryCondition::SlipWall);
    let config = CompressibleDgConfig {
        gas,
        viscosity: ViscosityModel::Constant(1.0e-10),
        boundaries,
        degree: sonium_fem::basis::PolynomialDegree::P2,
        flux: sonium_fem::physics::fluid::compressible_flow::CompressibleFluxKind::Hllc,
        troubled_cell_relaxation: 0.5,
        troubled_cell_threshold: 0.03,
        persson_threshold: -3.0,
        ..CompressibleDgConfig::default()
    };
    let solver = CompressibleDgSolver::new(&mesh, config)
        .map_err(|error| format!("cylinder-mach1 solver: {error:?}"))?;
    let mut state = CompressibleDgState::from_function_degree(
        &mesh,
        sonium_fem::basis::PolynomialDegree::P2,
        |_| free,
    )
        .map_err(|error| format!("cylinder-mach1 initial state: {error:?}"))?;

    let dump = |state: &CompressibleDgState, step: usize| -> Result<String, String> {
        let mut mach = vec![0.0f64; mesh.num_nodes()];
        let mut density = vec![0.0f64; mesh.num_nodes()];
        let mut schlieren = vec![0.0f64; mesh.num_nodes()];
        let mut counts = vec![0.0f64; mesh.num_nodes()];
        for element in 0..mesh.num_elements() {
            let average = state.cell_average(element);
            let velocity = average.velocity();
            let speed = (velocity[0].powi(2) + velocity[1].powi(2)).sqrt();
            let local_mach =
                speed / gas.speed_of_sound(average.pressure(gas), average.density);
            for &node in mesh.elements[element].vertices() {
                mach[node] += local_mach;
                density[node] += average.density;
                counts[node] += 1.0;
            }
        }
        for field in [&mut mach, &mut density] {
            for (value, count) in field.iter_mut().zip(&counts) {
                if *count > 0.0 {
                    *value /= *count;
                }
            }
        }
        // Numerical schlieren: |grad rho| of the nodal density field over
        // each P1 element, averaged back to nodes. Gradients of the DG
        // vertex values would under-read: the troubled-cell limiter
        // flattens element slopes in shocked regions.
        for element in 0..mesh.num_elements() {
            let nodes = mesh.elements[element].vertices();
            let (p0, p1, p2) = (
                &mesh.nodes[nodes[0]],
                &mesh.nodes[nodes[1]],
                &mesh.nodes[nodes[2]],
            );
            let det = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
            if det.abs() <= 1.0e-300 {
                continue;
            }
            let (d0, d1, d2) = (density[nodes[0]], density[nodes[1]], density[nodes[2]]);
            let gx = ((p2.y - p0.y) * (d1 - d0) - (p1.y - p0.y) * (d2 - d0)) / det;
            let gy = ((p1.x - p0.x) * (d2 - d0) - (p2.x - p0.x) * (d1 - d0)) / det;
            let local_schlieren = (gx * gx + gy * gy).sqrt();
            for &node in mesh.elements[element].vertices() {
                schlieren[node] += local_schlieren;
            }
        }
        for (value, count) in schlieren.iter_mut().zip(&counts) {
            if *count > 0.0 {
                *value /= *count;
            }
        }
        let fields = FrameFields {
            point_scalars: vec![
                ("mach", mach),
                ("density", density),
                ("schlieren", schlieren),
            ],
            point_vectors: Vec::new(),
            cell_scalars: Vec::new(),
        };
        write_frame(out_dir, &format!("frame_{step:04}"), &simple, &fields)
    };

    let t_end = 9.0f64;
    let dump_interval = 0.3f64;
    let mut frames = Vec::new();
    let mut times = Vec::new();
    let mut max_mach: f64 = 0.0;
    frames.push(dump(&state, 0)?);
    times.push(0.0);
    let mut time = 0.0f64;
    let mut next_dump = dump_interval;
    let mut frame_index = 0usize;
    while time < t_end {
        let stable = solver
            .operator
            .stable_time_step(&state)
            .map_err(|error| format!("cylinder-mach1 CFL at t={time:.2}: {error:?}"))?;
        let report = solver
            .step(&mut state, stable)
            .map_err(|error| format!("cylinder-mach1 step at t={time:.2}: {error:?}"))?;
        time += report.accepted_time_step;
        if time >= next_dump || time >= t_end {
            frame_index += 1;
            frames.push(dump(&state, frame_index)?);
            times.push(time);
            next_dump += dump_interval;
            for element in 0..mesh.num_elements() {
                let average = state.cell_average(element);
                let velocity = average.velocity();
                let speed = (velocity[0].powi(2) + velocity[1].powi(2)).sqrt();
                max_mach = max_mach.max(
                    speed / gas.speed_of_sound(average.pressure(gas), average.density),
                );
            }
        }
    }
    println!("cylinder-mach1: t={time:.2}, max Mach {max_mach:.3}");
    write_manifest(
        out_dir,
        "cylinder-mach1",
        "Cylinder at Mach 1 (compressible DG)",
        &frames,
        &times,
        "mach",
        (0.0, max_mach.max(1.0)),
    )
}

/// NACA 0012 airfoil at Mach 0.80, alpha = 1.25 degrees (AGARD inviscid
/// benchmark): explicit P2-DG compressible solver on a body-fitted mesh.
/// Mirrors `qa-fem-compressible-airfoil-naca0012` at demo resolution.
fn naca0012(out_dir: &Path) -> Result<(), String> {
    use sonium_fem::mesh::generators::channel_with_airfoil_triangles;
    use sonium_fem::physics::fluid::compressible_flow::{
        CompressibleBoundaryCondition, CompressibleDgConfig, CompressibleDgSolver,
        CompressibleDgState, ConservativeState, IdealGas, ViscosityModel,
    };
    use std::collections::HashMap;

    let gas = IdealGas {
        gamma: 1.4,
        gas_constant: 1.0,
    };
    let (mach, alpha_deg) = (0.80f64, 1.25f64);
    let alpha = alpha_deg.to_radians();
    let free = ConservativeState::from_primitive(
        1.0,
        [mach * alpha.cos(), mach * alpha.sin(), 0.0],
        1.0 / gas.gamma,
        gas,
    );
    let mesh = channel_with_airfoil_triangles(-6.0, 12.0, -5.0, 5.0, 90, 50, 0.0, 0.0, 1.0);
    let simple = simple_mesh_from_fem(&mesh)?;
    let mut boundaries: HashMap<i32, CompressibleBoundaryCondition> = HashMap::new();
    boundaries.insert(1, CompressibleBoundaryCondition::FarField(free));
    boundaries.insert(2, CompressibleBoundaryCondition::FarField(free));
    boundaries.insert(3, CompressibleBoundaryCondition::FarField(free));
    boundaries.insert(4, CompressibleBoundaryCondition::SlipWall);
    let config = CompressibleDgConfig {
        gas,
        viscosity: ViscosityModel::Constant(1.0e-10),
        boundaries,
        degree: sonium_fem::basis::PolynomialDegree::P2,
        flux: sonium_fem::physics::fluid::compressible_flow::CompressibleFluxKind::Hllc,
        troubled_cell_relaxation: 0.5,
        troubled_cell_threshold: 0.03,
        persson_threshold: -3.0,
        ..CompressibleDgConfig::default()
    };
    let solver = CompressibleDgSolver::new(&mesh, config)
        .map_err(|error| format!("naca0012 solver: {error:?}"))?;
    let mut state = CompressibleDgState::from_function_degree(
        &mesh,
        sonium_fem::basis::PolynomialDegree::P2,
        |_| free,
    )
    .map_err(|error| format!("naca0012 initial state: {error:?}"))?;

    let dump = |state: &CompressibleDgState, step: usize| -> Result<String, String> {
        let mut mach_field = vec![0.0f64; mesh.num_nodes()];
        let mut density = vec![0.0f64; mesh.num_nodes()];
        let mut schlieren = vec![0.0f64; mesh.num_nodes()];
        let mut counts = vec![0.0f64; mesh.num_nodes()];
        for element in 0..mesh.num_elements() {
            let average = state.cell_average(element);
            let velocity = average.velocity();
            let speed = (velocity[0].powi(2) + velocity[1].powi(2)).sqrt();
            let local_mach =
                speed / gas.speed_of_sound(average.pressure(gas), average.density);
            for &node in mesh.elements[element].vertices() {
                mach_field[node] += local_mach;
                density[node] += average.density;
                counts[node] += 1.0;
            }
        }
        for field in [&mut mach_field, &mut density] {
            for (value, count) in field.iter_mut().zip(&counts) {
                if *count > 0.0 {
                    *value /= *count;
                }
            }
        }
        for element in 0..mesh.num_elements() {
            let nodes = mesh.elements[element].vertices();
            let (p0, p1, p2) = (
                &mesh.nodes[nodes[0]],
                &mesh.nodes[nodes[1]],
                &mesh.nodes[nodes[2]],
            );
            let det = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
            if det.abs() <= 1.0e-300 {
                continue;
            }
            let (d0, d1, d2) = (density[nodes[0]], density[nodes[1]], density[nodes[2]]);
            let gx = ((p2.y - p0.y) * (d1 - d0) - (p1.y - p0.y) * (d2 - d0)) / det;
            let gy = ((p1.x - p0.x) * (d2 - d0) - (p2.x - p0.x) * (d1 - d0)) / det;
            let local_schlieren = (gx * gx + gy * gy).sqrt();
            for &node in mesh.elements[element].vertices() {
                schlieren[node] += local_schlieren;
            }
        }
        for (value, count) in schlieren.iter_mut().zip(&counts) {
            if *count > 0.0 {
                *value /= *count;
            }
        }
        let fields = FrameFields {
            point_scalars: vec![
                ("mach", mach_field),
                ("density", density),
                ("schlieren", schlieren),
            ],
            point_vectors: Vec::new(),
            cell_scalars: Vec::new(),
        };
        write_frame(out_dir, &format!("frame_{step:04}"), &simple, &fields)
    };

    let t_end = 8.0f64;
    let dump_interval = 0.3f64;
    let mut frames = Vec::new();
    let mut times = Vec::new();
    let mut max_mach: f64 = 0.0;
    frames.push(dump(&state, 0)?);
    times.push(0.0);
    let mut time = 0.0f64;
    let mut next_dump = dump_interval;
    let mut frame_index = 0usize;
    while time < t_end {
        let stable = solver
            .operator
            .stable_time_step(&state)
            .map_err(|error| format!("naca0012 CFL at t={time:.2}: {error:?}"))?;
        let report = solver
            .step(&mut state, stable)
            .map_err(|error| format!("naca0012 step at t={time:.2}: {error:?}"))?;
        time += report.accepted_time_step;
        if time >= next_dump || time >= t_end {
            frame_index += 1;
            frames.push(dump(&state, frame_index)?);
            times.push(time);
            next_dump += dump_interval;
            for element in 0..mesh.num_elements() {
                let average = state.cell_average(element);
                let velocity = average.velocity();
                let speed = (velocity[0].powi(2) + velocity[1].powi(2)).sqrt();
                max_mach = max_mach.max(
                    speed / gas.speed_of_sound(average.pressure(gas), average.density),
                );
            }
        }
    }
    println!("naca0012: t={time:.2}, max Mach {max_mach:.3}");
    write_manifest(
        out_dir,
        "naca0012",
        "NACA 0012 at Mach 0.8 (compressible DG)",
        &frames,
        &times,
        "mach",
        (0.0, max_mach.max(1.0)),
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: qa-website-demos <demo> <out-dir>");
        std::process::exit(2);
    }
    let out_dir = PathBuf::from(&args[2]);
    if let Err(error) = std::fs::create_dir_all(&out_dir) {
        eprintln!("cannot create {}: {error}", out_dir.display());
        std::process::exit(1);
    }
    let result = match args[1].as_str() {
        "taylor-green" => taylor_green(&out_dir),
        "crush" => crush(&out_dir),
        "necking" => necking(&out_dir),
        "plate-modes" => plate_modes(&out_dir),
        "room-wave" => room_wave(&out_dir),
        "arch" => arch(&out_dir),
        "propeller-noise" => propeller_noise(&out_dir),
        "thermal-spreading" => thermal_spreading(&out_dir),
        "melting-plate" => melting_plate(&out_dir),
        "sloshing-tank" => sloshing_tank(&out_dir),
        "sloshing-sweep" => sloshing_sweep(&out_dir),
        "cylinder-wake" => cylinder_wake(&out_dir),
        "cylinder-mach1" => cylinder_mach1(&out_dir),
        "naca0012" => naca0012(&out_dir),
        other => Err(format!("unknown demo {other:?}")),
    };
    match result {
        Ok(()) => println!("SUCCESS: qa-website-demos {}", args[1]),
        Err(error) => {
            eprintln!("FAILED: qa-website-demos {}: {error}", args[1]);
            std::process::exit(1);
        }
    }
}
