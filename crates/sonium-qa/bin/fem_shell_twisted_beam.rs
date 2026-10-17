//! Twisted beam benchmark (MacNeal-Harder) with MITC4 quad shells.
//!
//! Cantilever beam (L = 12, width 1.1, t = 0.32) with 90-degree twist about
//! the beam axis, E = 29e6, nu = 0.22. Unit tip loads in the width (in-plane)
//! and thickness (out-of-plane) directions. Targets: in-plane tip deflection
//! 0.005424, out-of-plane tip deflection 0.001754.
//!
//! Reference: MacNeal, R. H. and Harder, R. L. (1985), "A proposed standard
//! set of problems to test finite element accuracy", FEAD 1.
//!
//! The mesh is assembled directly with warp-aware MITC4 quads (covariant
//! membrane on the true bilinear surface plus Allman edge bubbles and
//! Hughes-Brezzi drilling). An earlier flat-facet membrane diverged on the
//! twisted (warped-quad) geometry and the QA used CST+DKT triangles as a
//! workaround; the warp-aware membrane converges monotonically from below
//! and lands within 1% of both targets at 32x8 (see the Phase-B Worker-3
//! report for the root-cause analysis).

// Rust guideline compliant 2026-02-21

use sonium_fem::mesh::Point;
use sonium_fem::physics::solid::shell::{
    QuadShellElement, ShellSection, add_quad_shell_point_load, assemble_quad_shell_stiffness,
};
use sonium_solvers::direct::SparseLuFactorization;

const LENGTH: f64 = 12.0;
const WIDTH: f64 = 1.1;
const THICK: f64 = 0.32;
const E_MOD: f64 = 29.0e6;
const NU: f64 = 0.22;
const LOAD: f64 = 1.0;
const TARGET_IN_PLANE: f64 = 0.005424;
const TARGET_OUT_OF_PLANE: f64 = 0.001754;

fn build_beam(n_x: usize, n_y: usize) -> (Vec<Point>, Vec<[usize; 4]>) {
    // Root section spans Y (width), twisted 90 degrees about X at the tip.
    let total = std::f64::consts::FRAC_PI_2;
    let mut points = Vec::with_capacity((n_x + 1) * (n_y + 1));
    for ix in 0..=n_x {
        let x = LENGTH * ix as f64 / n_x as f64;
        let twist = total * ix as f64 / n_x as f64;
        for iy in 0..=n_y {
            let s = WIDTH * (iy as f64 / n_y as f64 - 0.5);
            points.push(Point::new_3d(x, s * twist.cos(), s * twist.sin()));
        }
    }
    let node = |ix: usize, iy: usize| -> usize { ix * (n_y + 1) + iy };
    let mut quads = Vec::with_capacity(n_x * n_y);
    for ix in 0..n_x {
        for iy in 0..n_y {
            quads.push([
                node(ix, iy),
                node(ix + 1, iy),
                node(ix + 1, iy + 1),
                node(ix, iy + 1),
            ]);
        }
    }
    (points, quads)
}

fn solve_tip(n_x: usize, n_y: usize, load: [f64; 6]) -> [f64; 6] {
    let (points, quads) = build_beam(n_x, n_y);
    let section = ShellSection::new(E_MOD, NU, THICK, 0.0);
    let mut elements = Vec::with_capacity(quads.len());
    for q in &quads {
        elements.push(QuadShellElement::from_points_with_section(
            *q,
            &points[q[0]],
            &points[q[1]],
            &points[q[2]],
            &points[q[3]],
            section,
        ));
    }
    let n_dofs = points.len() * 6;
    let mut csr = assemble_quad_shell_stiffness(&elements, n_dofs).to_csr();
    let mut rhs = vec![0.0; n_dofs];
    // Tip center node (n_y even) carries the transverse tip load.
    let tip = n_x * (n_y + 1) + n_y / 2;
    add_quad_shell_point_load(&mut rhs, tip, load);
    // Clamped root: all six DOFs on every root node.
    let mut fixed = Vec::new();
    for iy in 0..=n_y {
        for d in 0..6 {
            fixed.push(iy * 6 + d);
        }
    }
    apply_dirichlet(&mut csr, &mut rhs, &fixed);
    let lu = SparseLuFactorization::factorize(&csr).expect("sparse LU must factorize");
    let x = lu.solve(&rhs).expect("sparse LU must solve");
    [
        x[tip * 6],
        x[tip * 6 + 1],
        x[tip * 6 + 2],
        x[tip * 6 + 3],
        x[tip * 6 + 4],
        x[tip * 6 + 5],
    ]
}

fn apply_dirichlet(csr: &mut sonium_solvers::CsrMatrix<f64>, rhs: &mut [f64], fixed: &[usize]) {
    for &dof in fixed {
        let range = csr.row_range(dof);
        for i in range {
            csr.values[i] = 0.0;
        }
        let diag = csr
            .row_entries(dof)
            .position(|(j, _)| j == dof)
            .expect("shell CSR rows must carry a diagonal");
        csr.values[csr.row_ptrs[dof] + diag] = 1.0;
        rhs[dof] = 0.0;
    }
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        // At the tip the width has twisted into Z: in-plane load is +Z,
        // out-of-plane load is +Y.
        for (nx, ny) in [(12, 4), (24, 6), (32, 8)] {
            let tip_in = solve_tip(nx, ny, [0.0, 0.0, LOAD, 0.0, 0.0, 0.0]);
            let tip_out = solve_tip(nx, ny, [0.0, LOAD, 0.0, 0.0, 0.0, 0.0]);
            println!(
                "  mesh {nx}x{ny}: in-plane z = {:.6e}, out-of-plane y = {:.6e}",
                tip_in[2], tip_out[1],
            );
        }
        let tip_in = solve_tip(32, 8, [0.0, 0.0, LOAD, 0.0, 0.0, 0.0]);
        let tip_out = solve_tip(32, 8, [0.0, LOAD, 0.0, 0.0, 0.0, 0.0]);
        let u_in = tip_in[2];
        let u_out = tip_out[1];
        let err_in = (u_in - TARGET_IN_PLANE).abs() / TARGET_IN_PLANE;
        let err_out = (u_out - TARGET_OUT_OF_PLANE).abs() / TARGET_OUT_OF_PLANE;
        println!(
            "Twisted beam 32x8: in-plane tip = {u_in:.6e} (err {err_in:.3e}, \
             target {TARGET_IN_PLANE}), out-of-plane tip = {u_out:.6e} \
             (err {err_out:.3e}, target {TARGET_OUT_OF_PLANE})"
        );
        // Measured errors: in-plane 6.33e-3, out-of-plane 7.56e-3; gate at ~2.5x.
        assert!(err_in < 0.02, "in-plane deflection misses target");
        assert!(err_out < 0.02, "out-of-plane deflection misses target");
    });
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
