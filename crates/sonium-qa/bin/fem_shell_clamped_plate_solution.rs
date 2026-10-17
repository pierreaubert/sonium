//! Clamped circular plate under uniform pressure (DKT triangles).
//!
//! Radius a = 1, t = 0.02, E = 200e9, nu = 0.3, uniform pressure q = 1000.
//! Clamped rim, solved center deflection vs the Kirchhoff closed form
//! w0 = q a^4 / (64 D).
//!
//! Reference: Timoshenko, S. and Woinowsky-Krieger, S., "Theory of Plates
//! and Shells", 2nd ed., Article 24 (axisymmetrical bending of circular
//! plates).
//!
//! The C3 OpenCourant case uses the shared `sonium_qa::rad` extended API
//! (`shell_clamped_plate_case`); engine-gated runs live in
//! `qa-fem-opencourant-shell-clamped-plate`.

// Rust guideline compliant 2026-02-21

use sonium_fem::mesh::{ElementType, circular_shell_mesh_triangles};
use sonium_fem::physics::solid::shell::{
    ShellElement, ShellSection, assemble_shell_load, assemble_shell_stiffness,
};
use sonium_qa::rad::{assert_probe_close, shell_clamped_plate_case, write_extended_starter};
use sonium_solvers::direct::SparseLuFactorization;

const RADIUS: f64 = 1.0;
const THICK: f64 = 0.02;
const E_MOD: f64 = 200e9;
const NU: f64 = 0.3;
const PRESSURE: f64 = 1000.0;

fn clamped_center_deflection(n_radial: usize, n_theta: usize) -> f64 {
    let mesh = circular_shell_mesh_triangles(RADIUS, n_radial, n_theta);
    let section = ShellSection::new(E_MOD, NU, THICK, 7800.0);
    let elements: Vec<ShellElement> = mesh
        .elements
        .iter()
        .filter(|e| e.element_type == ElementType::Triangle)
        .map(|e| ShellElement::from_mesh([e.nodes[0], e.nodes[1], e.nodes[2]], &mesh, section))
        .collect();
    let n_dofs = mesh.nodes.len() * 6;
    let mut csr = assemble_shell_stiffness(&elements, n_dofs).to_csr();
    let mut rhs = assemble_shell_load(&elements, n_dofs, |_| -PRESSURE);
    // Clamp the rim: all six DOFs on nodes at r = a.
    let mut fixed = Vec::new();
    for (i, p) in mesh.nodes.iter().enumerate() {
        if (p.x.hypot(p.y) - RADIUS).abs() < 1e-9 {
            for d in 0..6 {
                fixed.push(i * 6 + d);
            }
        }
    }
    assert!(!fixed.is_empty(), "rim nodes must be found");
    for &dof in &fixed {
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
    let lu = SparseLuFactorization::factorize(&csr).expect("sparse LU must factorize");
    let x = lu.solve(&rhs).expect("sparse LU must solve");
    // Center node at r = 0.
    let center = mesh
        .nodes
        .iter()
        .position(|p| p.x.hypot(p.y) < 1e-12)
        .expect("center node must exist");
    x[center * 6 + 2]
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        let d = E_MOD * THICK.powi(3) / (12.0 * (1.0 - NU * NU));
        let exact = -PRESSURE * RADIUS.powi(4) / (64.0 * d);
        let w_coarse = clamped_center_deflection(6, 24);
        let w_fine = clamped_center_deflection(12, 48);
        let err_fine = (w_fine - exact).abs() / exact.abs();
        println!(
            "Clamped circular plate DKT: 6x24 w = {w_coarse:.6e}, 12x48 w = \
             {w_fine:.6e} (err {err_fine:.3e}), closed form {exact:.6e}"
        );
        // Measured fine-grid error = 3.42e-3 vs the closed form; gate at ~3x.
        assert!(err_fine < 0.01, "center deflection misses closed form");
        assert!(
            (w_fine - exact).abs() <= (w_coarse - exact).abs(),
            "refinement must not increase the error"
        );
        // C3: static deck verification through the real API (no file writes;
        // engine-gated runs live in qa-fem-opencourant-shell-clamped-plate).
        let case = shell_clamped_plate_case();
        case.validate().expect("shell case must validate");
        let starter = write_extended_starter(&case);
        assert!(
            starter.contains("/PROP/SHELL/1/1") && starter.contains("/CLOAD/1/1"),
            "starter deck must carry shell section and pressure loads"
        );
        println!("C3 shell_clamped_plate: static deck check passes");
        let probe = case
            .base
            .probes
            .iter()
            .find(|probe| probe.label == "center DZ")
            .expect("displacement probe must exist");
        assert_probe_close(
            w_fine,
            probe.expected,
            probe.abs_tol,
            probe.rel_tol,
            &probe.label,
        )
        .expect("FEM deflection must match the C3 probe");
    });
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
