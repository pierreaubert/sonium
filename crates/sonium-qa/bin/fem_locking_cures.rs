//! QA: F-bar and EAS locking cures for nonlinear bricks.
//!
//! A sinusoidal manufactured solution shows optimal bending rates, a tip-load
//! cantilever contrasts EAS against locked Q1, a near-incompressible Cook
//! membrane contrasts F-bar against locked Q1, and distorted patch tests
//! verify exact linear-field reproduction.

// Rust guideline compliant 2026-02-21

use sonium_fem::forms::NonlinearElementForm;
use sonium_fem::physics::solid::nonlinear_solid::{
    EasHexForm, FBarHexForm, HexMesh, NeoHookean, dense_solve, hex_jacobian, hex_quadrature,
    hex_shape, hex_shape_derivatives, q1_element_residual,
};

/// Newton solve over closure-defined residual/tangent element callbacks.
fn newton_solve(
    n: usize,
    num_elements: usize,
    map: &dyn Fn(usize) -> Vec<usize>,
    residual_fn: &dyn Fn(usize, &[f64]) -> Vec<f64>,
    tangent_fn: &dyn Fn(usize, &[f64]) -> Vec<f64>,
    external: &[f64],
    fixed: &[usize],
    initial: &[f64],
    tolerance: f64,
) -> Vec<f64> {
    let mut mask = vec![false; n];
    for &dof in fixed {
        mask[dof] = true;
    }
    let free: Vec<usize> = (0..n).filter(|dof| !mask[*dof]).collect();
    let mut u = initial.to_vec();
    let mut scale = 0.0;
    for _ in 0..15 {
        let mut residual = vec![0.0; n];
        let mut tangent = vec![0.0; n * n];
        for elem in 0..num_elements {
            let m = map(elem);
            let local_r = residual_fn(elem, &u);
            let local_k = tangent_fn(elem, &u);
            for (i, &gi) in m.iter().enumerate() {
                residual[gi] += local_r[i];
                for (j, &gj) in m.iter().enumerate() {
                    tangent[gi * n + gj] += local_k[i * m.len() + j];
                }
            }
        }
        for i in 0..n {
            residual[i] -= external[i];
        }
        let free_norm: f64 = free
            .iter()
            .map(|&d| residual[d] * residual[d])
            .sum::<f64>()
            .sqrt();
        if scale == 0.0 {
            // Load scaling, falling back to the initial imbalance for
            // zero-external-load cases such as the patch test.
            let load_norm: f64 = external
                .iter()
                .map(|f| f * f)
                .sum::<f64>()
                .sqrt()
                .max(1.0e-30);
            scale = free_norm.max(load_norm);
        }
        if free_norm / scale < tolerance {
            return u;
        }
        let m = free.len();
        let mut matrix = vec![0.0; m * m];
        let mut rhs = vec![0.0; m];
        for (i, &gi) in free.iter().enumerate() {
            rhs[i] = -residual[gi];
            for (j, &gj) in free.iter().enumerate() {
                matrix[i * m + j] = tangent[gi * n + gj];
            }
        }
        dense_solve(&mut matrix, &mut rhs, m);
        for (i, &gi) in free.iter().enumerate() {
            u[gi] += rhs[i];
        }
    }
    panic!("locking Newton failed to converge");
}

/// Q1 element tangent via central differences of the Q1 residual.
fn q1_tangent(mesh: &HexMesh, material: &NeoHookean, elem: usize, u: &[f64]) -> Vec<f64> {
    let mut dofs = Vec::with_capacity(24);
    for node in mesh.conn[elem] {
        dofs.extend_from_slice(&[node * 3, node * 3 + 1, node * 3 + 2]);
    }
    let mut out = vec![0.0; 576];
    let mut perturbed = u.to_vec();
    for (col, &dof) in dofs.iter().enumerate() {
        let base = u[dof];
        let h = 1.0e-8 * (1.0 + base.abs());
        perturbed[dof] = base + h;
        let plus = q1_element_residual(material, mesh, elem, &perturbed);
        perturbed[dof] = base - h;
        let minus = q1_element_residual(material, mesh, elem, &perturbed);
        perturbed[dof] = base;
        for row in 0..24 {
            out[row * 24 + col] = (plus[row] - minus[row]) / (2.0 * h);
        }
    }
    out
}

// === Manufactured bending solution ===
//
// ux = A sin(kx x) cos(ky y), uy = B cos(kx x) sin(ky y) on a thin domain.
// Body force from linear elasticity (small-strain regime):
// fx = (P kx + Q ky) sin(kx x) cos(ky y),
// fy = (Q kx + R ky) cos(kx x) sin(ky y).

const MMS_A: f64 = 3.0e-5;
const MMS_B: f64 = 5.0e-5;
const MMS_KX: f64 = std::f64::consts::FRAC_PI_4;
const MMS_KY: f64 = std::f64::consts::FRAC_PI_2;

fn mms_field(x: f64, y: f64) -> [f64; 3] {
    [
        MMS_A * (MMS_KX * x).sin() * (MMS_KY * y).cos(),
        MMS_B * (MMS_KX * x).cos() * (MMS_KY * y).sin(),
        0.0,
    ]
}

fn mms_body_force(x: f64, y: f64) -> [f64; 3] {
    let e = 1.0e6;
    let nu = 0.3;
    let lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = e / (2.0 * (1.0 + nu));
    let p = lam * (MMS_A * MMS_KX + MMS_B * MMS_KY) + 2.0 * mu * MMS_A * MMS_KX;
    let q = mu * (MMS_A * MMS_KY + MMS_B * MMS_KX);
    let r = lam * (MMS_A * MMS_KX + MMS_B * MMS_KY) + 2.0 * mu * MMS_B * MMS_KY;
    let sx = (MMS_KX * x).sin();
    let cx = (MMS_KX * x).cos();
    let sy = (MMS_KY * y).sin();
    let cy = (MMS_KY * y).cos();
    [
        (p * MMS_KX + q * MMS_KY) * sx * cy,
        (q * MMS_KX + r * MMS_KY) * cx * sy,
        0.0,
    ]
}

fn body_force_vector(mesh: &HexMesh) -> Vec<f64> {
    let mut rhs = vec![0.0; 3 * mesh.num_nodes()];
    for elem in 0..mesh.conn.len() {
        let coords = mesh.element_coords(elem);
        for (natural, weight) in hex_quadrature() {
            let d = hex_shape_derivatives(&natural);
            let (_, det, _) = hex_jacobian(&coords, &d);
            let n = hex_shape(&natural);
            let mut point = [0.0; 3];
            for a in 0..8 {
                for i in 0..3 {
                    point[i] += n[a] * coords[a][i];
                }
            }
            let force = mms_body_force(point[0], point[1]);
            let w = weight * det;
            for a in 0..8 {
                let node = mesh.conn[elem][a];
                for i in 0..3 {
                    rhs[node * 3 + i] += n[a] * force[i] * w;
                }
            }
        }
    }
    rhs
}

fn mms_error(nx: usize, ny: usize, nz: usize, use_eas: bool) -> f64 {
    let material = NeoHookean::from_young(1.0e6, 0.3, 3);
    let mesh = HexMesh::brick(nx, ny, nz, [0.0, 0.0, 0.0], [4.0, 1.0, 1.0]);
    let total = 3 * mesh.num_nodes();
    let external = body_force_vector(&mesh);
    let mut prescribed = vec![0.0; total];
    let mut fixed = Vec::new();
    for (node, coords) in mesh.coords.iter().enumerate() {
        let value = mms_field(coords[0], coords[1]);
        for i in 0..3 {
            prescribed[node * 3 + i] = value[i];
        }
        if coords[0] == 0.0
            || coords[0] == 4.0
            || coords[1] == 0.0
            || coords[1] == 1.0
            || coords[2] == 0.0
            || coords[2] == 1.0
        {
            fixed.extend_from_slice(&[node * 3, node * 3 + 1, node * 3 + 2]);
        }
    }
    let u = if use_eas {
        let form = EasHexForm::new(material, mesh);
        let map = |elem: usize| -> Vec<usize> {
            let mut dofs = Vec::with_capacity(24);
            for node in form.mesh.conn[elem] {
                dofs.extend_from_slice(&[node * 3, node * 3 + 1, node * 3 + 2]);
            }
            dofs
        };
        newton_solve(
            total,
            form.num_elements(),
            &map,
            &|elem, u| form.element_residual(elem, u),
            &|elem, u| form.element_tangent(elem, u),
            &external,
            &fixed,
            &prescribed,
            1.0e-9,
        )
    } else {
        let map = |elem: usize| -> Vec<usize> {
            let mut dofs = Vec::with_capacity(24);
            for node in mesh.conn[elem] {
                dofs.extend_from_slice(&[node * 3, node * 3 + 1, node * 3 + 2]);
            }
            dofs
        };
        newton_solve(
            total,
            mesh.conn.len(),
            &map,
            &|elem, u| q1_element_residual(&material, &mesh, elem, u).to_vec(),
            &|elem, u| q1_tangent(&mesh, &material, elem, u),
            &external,
            &fixed,
            &prescribed,
            1.0e-9,
        )
    };
    // Max nodal error over the mesh coordinates.
    let coords = HexMesh::brick(nx, ny, nz, [0.0, 0.0, 0.0], [4.0, 1.0, 1.0]).coords;
    let mut worst: f64 = 0.0;
    for (node, point) in coords.iter().enumerate() {
        let exact = mms_field(point[0], point[1]);
        for i in 0..3 {
            worst = worst.max((u[node * 3 + i] - exact[i]).abs());
        }
    }
    worst
}

fn test_mms_bending() {
    let mut eas = Vec::new();
    let mut q1 = Vec::new();
    for (nx, ny, nz) in [(4, 2, 2), (8, 4, 4), (16, 8, 8)] {
        let eas_err = mms_error(nx, ny, nz, true);
        let q1_err = mms_error(nx, ny, nz, false);
        println!("mms {nx}x{ny}x{nz}: eas={eas_err:.4e} q1={q1_err:.4e}");
        eas.push(eas_err);
        q1.push(q1_err);
    }
    let eas_rate_1 = (eas[0] / eas[1]).log2();
    let eas_rate_2 = (eas[1] / eas[2]).log2();
    let q1_rate_2 = (q1[1] / q1[2]).log2();
    println!("mms rates: eas {eas_rate_1:.2}/{eas_rate_2:.2}, q1 .../{q1_rate_2:.2}");
    assert!(
        eas_rate_1 > 1.5,
        "EAS MMS coarse rate too low: {eas_rate_1:.2}"
    );
    assert!(
        eas_rate_2 > 1.5,
        "EAS MMS fine rate too low: {eas_rate_2:.2}"
    );
}

// === Tip-load bending contrast ===

fn cantilever_tip(nx: usize, use_eas: bool) -> f64 {
    let material = NeoHookean::from_young(1.0e6, 0.3, 3);
    let mesh = HexMesh::brick(nx, 1, 1, [0.0, 0.0, 0.0], [10.0, 1.0, 1.0]);
    let total = 3 * mesh.num_nodes();
    let mut external = vec![0.0; total];
    let mut fixed = Vec::new();
    for (node, coords) in mesh.coords.iter().enumerate() {
        if coords[0] == 0.0 {
            fixed.extend_from_slice(&[node * 3, node * 3 + 1, node * 3 + 2]);
        }
        if coords[0] == 10.0 {
            external[node * 3 + 1] += 25.0 / 4.0;
        }
    }
    let zero = vec![0.0; total];
    let u = if use_eas {
        let form = EasHexForm::new(material, mesh);
        let map = |elem: usize| -> Vec<usize> {
            let mut dofs = Vec::with_capacity(24);
            for node in form.mesh.conn[elem] {
                dofs.extend_from_slice(&[node * 3, node * 3 + 1, node * 3 + 2]);
            }
            dofs
        };
        newton_solve(
            total,
            form.num_elements(),
            &map,
            &|elem, u| form.element_residual(elem, u),
            &|elem, u| form.element_tangent(elem, u),
            &external,
            &fixed,
            &zero,
            1.0e-9,
        )
    } else {
        let map = |elem: usize| -> Vec<usize> {
            let mut dofs = Vec::with_capacity(24);
            for node in mesh.conn[elem] {
                dofs.extend_from_slice(&[node * 3, node * 3 + 1, node * 3 + 2]);
            }
            dofs
        };
        newton_solve(
            total,
            mesh.conn.len(),
            &map,
            &|elem, u| q1_element_residual(&material, &mesh, elem, u).to_vec(),
            &|elem, u| q1_tangent(&mesh, &material, elem, u),
            &external,
            &fixed,
            &zero,
            1.0e-9,
        )
    };
    let mut tip = 0.0;
    let mut count = 0;
    for (dof, force) in external.iter().enumerate() {
        if *force != 0.0 {
            tip += u[dof];
            count += 1;
        }
    }
    tip / count as f64
}

fn test_bending_contrast() {
    let e = 1.0e6;
    let g = e / 2.6;
    let inertia = 1.0 / 12.0;
    let exact = 25.0 * 1000.0 / (3.0 * e * inertia) + 25.0 * 10.0 / ((5.0 / 6.0) * g);
    let eas = cantilever_tip(4, true);
    let q1 = cantilever_tip(4, false);
    let eas_err = (eas - exact).abs() / exact;
    let q1_err = (q1 - exact).abs() / exact;
    println!("bend nx=4: eas err={eas_err:.3e} q1 err={q1_err:.3e} (ref={exact:.6e})");
    assert!(eas_err < 0.06, "EAS bending error too large: {eas_err:.3e}");
    assert!(
        q1_err > 0.40,
        "Q1 should lock on the coarse bending mesh: {q1_err:.3e}"
    );
}

// === Cook-style membrane ===

fn cook_mesh(n: usize) -> HexMesh {
    let corners = [[0.0, 0.0], [48.0, 44.0], [48.0, 60.0], [0.0, 44.0]];
    let mut coords = Vec::new();
    for k in 0..=1 {
        for j in 0..=n {
            for i in 0..=n {
                let xi = i as f64 / n as f64;
                let eta = j as f64 / n as f64;
                let x = (1.0 - xi) * (1.0 - eta) * corners[0][0]
                    + xi * (1.0 - eta) * corners[1][0]
                    + xi * eta * corners[2][0]
                    + (1.0 - xi) * eta * corners[3][0];
                let y = (1.0 - xi) * (1.0 - eta) * corners[0][1]
                    + xi * (1.0 - eta) * corners[1][1]
                    + xi * eta * corners[2][1]
                    + (1.0 - xi) * eta * corners[3][1];
                coords.push([x, y, k as f64]);
            }
        }
    }
    let node = |i: usize, j: usize, k: usize| k * (n + 1) * (n + 1) + j * (n + 1) + i;
    let mut conn = Vec::new();
    for j in 0..n {
        for i in 0..n {
            conn.push([
                node(i, j, 0),
                node(i + 1, j, 0),
                node(i + 1, j + 1, 0),
                node(i, j + 1, 0),
                node(i, j, 1),
                node(i + 1, j, 1),
                node(i + 1, j + 1, 1),
                node(i, j + 1, 1),
            ]);
        }
    }
    HexMesh::new(coords, conn)
}

fn cook_tip(n: usize, nu: f64, use_fbar: bool) -> f64 {
    let material = NeoHookean::from_young(70.0, nu, 3);
    let mesh = cook_mesh(n);
    let total = 3 * mesh.num_nodes();
    let mut external = vec![0.0; total];
    let mut fixed = Vec::new();
    let mut loaded = 0;
    for (node, coords) in mesh.coords.iter().enumerate() {
        if coords[0] == 0.0 {
            fixed.extend_from_slice(&[node * 3, node * 3 + 1, node * 3 + 2]);
        }
        if (coords[0] - 48.0).abs() < 1.0e-12 {
            external[node * 3 + 1] += 1.0;
            loaded += 1;
        }
    }
    for value in external.iter_mut() {
        *value /= loaded as f64;
    }
    let mut u = vec![0.0; total];
    let steps = 5;
    if use_fbar {
        let form = FBarHexForm::new(material, mesh);
        let map = |elem: usize| -> Vec<usize> {
            let mut dofs = Vec::with_capacity(24);
            for node in form.mesh.conn[elem] {
                dofs.extend_from_slice(&[node * 3, node * 3 + 1, node * 3 + 2]);
            }
            dofs
        };
        for step in 1..=steps {
            let scaled: Vec<f64> = external
                .iter()
                .map(|f| f * step as f64 / steps as f64)
                .collect();
            u = newton_solve(
                total,
                form.num_elements(),
                &map,
                &|elem, u| form.element_residual(elem, u),
                &|elem, u| form.element_tangent(elem, u),
                &scaled,
                &fixed,
                &u,
                1.0e-6,
            );
        }
    } else {
        let map = |elem: usize| -> Vec<usize> {
            let mut dofs = Vec::with_capacity(24);
            for node in mesh.conn[elem] {
                dofs.extend_from_slice(&[node * 3, node * 3 + 1, node * 3 + 2]);
            }
            dofs
        };
        for step in 1..=steps {
            let scaled: Vec<f64> = external
                .iter()
                .map(|f| f * step as f64 / steps as f64)
                .collect();
            u = newton_solve(
                total,
                mesh.conn.len(),
                &map,
                &|elem, u| q1_element_residual(&material, &mesh, elem, u).to_vec(),
                &|elem, u| q1_tangent(&mesh, &material, elem, u),
                &scaled,
                &fixed,
                &u,
                1.0e-6,
            );
        }
    }
    let stride = (n + 1) * (n + 1);
    let tip0 = (n * (n + 1) + n) * 3 + 1;
    let tip1 = (stride + n * (n + 1) + n) * 3 + 1;
    (u[tip0] + u[tip1]) / 2.0
}

fn test_cook_membrane() {
    // Classical E=1/F=1 reference 23.96 scales to 23.96/70 at E=70.
    let classical = 23.96 / 70.0;
    let compressible = cook_tip(8, 1.0 / 3.0, true);
    let compressible_err = (compressible - classical).abs() / classical;
    println!("cook compressible n=8: {compressible:.6e} vs {classical:.6e}");
    // Measured compressible Cook error = 2.58e-2 vs the classical tip; gate at ~1.5x.
    assert!(
        compressible_err < 0.04,
        "compressible Cook error {compressible_err:.3e}"
    );
    // Near-incompressible locking demo with a fine-mesh self reference.
    let reference = cook_tip(12, 0.4999, true);
    let fbar = cook_tip(4, 0.4999, true);
    let q1 = cook_tip(4, 0.4999, false);
    println!("cook nu=0.4999: ref={reference:.6e} fbar={fbar:.6e} q1={q1:.6e}");
    assert!(
        (fbar - reference).abs() / reference < 0.20,
        "F-bar Cook error too large"
    );
    assert!(q1 / reference < 0.40, "Q1 should lock: {}", q1 / reference);
}

// === Patch tests ===

fn patch_error(nu: f64, use_fbar: bool) -> f64 {
    let material = NeoHookean::from_young(1.0e6, nu, 3);
    let fresh = HexMesh::brick(2, 2, 2, [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
    let mut mesh = fresh.clone();
    for coords in mesh.coords.iter_mut() {
        let interior = f64::from(coords[0] > 0.0 && coords[0] < 2.0)
            + f64::from(coords[1] > 0.0 && coords[1] < 2.0)
            + f64::from(coords[2] > 0.0 && coords[2] < 2.0);
        if interior >= 2.0 {
            coords[0] += 0.07;
            coords[1] -= 0.05;
        }
    }
    let total = 3 * mesh.num_nodes();
    let field = |x: f64, y: f64, z: f64| -> [f64; 3] {
        [
            0.01 + 0.02 * x - 0.015 * y + 0.01 * z,
            -0.005 + 0.01 * x + 0.025 * y - 0.02 * z,
            0.008 - 0.012 * x + 0.018 * y + 0.022 * z,
        ]
    };
    let mut prescribed = vec![0.0; total];
    for (node, coords) in mesh.coords.iter().enumerate() {
        let value = field(coords[0], coords[1], coords[2]);
        for i in 0..3 {
            prescribed[node * 3 + i] = value[i];
        }
    }
    let mut fixed = Vec::new();
    let mut interior = Vec::new();
    for (node, coords) in fresh.coords.iter().enumerate() {
        let boundary = coords[0] == 0.0
            || coords[0] == 2.0
            || coords[1] == 0.0
            || coords[1] == 2.0
            || coords[2] == 0.0
            || coords[2] == 2.0;
        if boundary {
            fixed.extend_from_slice(&[node * 3, node * 3 + 1, node * 3 + 2]);
        } else {
            interior.extend_from_slice(&[node * 3, node * 3 + 1, node * 3 + 2]);
        }
    }
    let mut initial = prescribed.clone();
    for &dof in &interior {
        initial[dof] += 1.0e-3;
    }
    let external = vec![0.0; total];
    let u = if use_fbar {
        let form = FBarHexForm::new(material, mesh);
        let map = |elem: usize| -> Vec<usize> {
            let mut dofs = Vec::with_capacity(24);
            for node in form.mesh.conn[elem] {
                dofs.extend_from_slice(&[node * 3, node * 3 + 1, node * 3 + 2]);
            }
            dofs
        };
        newton_solve(
            total,
            form.num_elements(),
            &map,
            &|elem, u| form.element_residual(elem, u),
            &|elem, u| form.element_tangent(elem, u),
            &external,
            &fixed,
            &initial,
            1.0e-10,
        )
    } else {
        let form = EasHexForm::new(material, mesh);
        let map = |elem: usize| -> Vec<usize> {
            let mut dofs = Vec::with_capacity(24);
            for node in form.mesh.conn[elem] {
                dofs.extend_from_slice(&[node * 3, node * 3 + 1, node * 3 + 2]);
            }
            dofs
        };
        newton_solve(
            total,
            form.num_elements(),
            &map,
            &|elem, u| form.element_residual(elem, u),
            &|elem, u| form.element_tangent(elem, u),
            &external,
            &fixed,
            &initial,
            1.0e-10,
        )
    };
    interior
        .iter()
        .map(|&dof| (u[dof] - prescribed[dof]).abs())
        .fold(0.0, f64::max)
}

fn test_patch() {
    let fbar = patch_error(0.45, true);
    let eas = patch_error(0.3, false);
    println!("patch: fbar={fbar:.3e} eas={eas:.3e}");
    assert!(fbar < 1.0e-9, "F-bar patch failed: {fbar:.3e}");
    assert!(eas < 1.0e-9, "EAS patch failed: {eas:.3e}");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_mms_bending();
        test_bending_contrast();
        test_cook_membrane();
        test_patch();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
