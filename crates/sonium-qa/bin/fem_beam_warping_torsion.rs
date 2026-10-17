//! QA: warping torsion of an I-beam cantilever vs Vlasov theory.
//!
//! Covers the Saint-Venant limit (zero warping constant), free warping at
//! both ends, and the warping-restrained cantilever (tip/midspan twist and
//! root bimoment). Closed forms (Vlasov thin-walled beam theory; see also
//! Timoshenko & Gere, Mechanics of Materials): tip twist
//! theta = (T/GJ)(L - tanh(mu L)/mu) with mu = sqrt(GJ/E Iw), and root
//! bimoment magnitude (T/mu) tanh(mu L).

// Rust guideline compliant 2026-02-21

use sonium_fem::physics::solid::beam::{Section3D, WarpingBeamElement3D};

/// Solve a dense system in place with partial pivoting.
fn dense_solve(matrix: &mut [f64], rhs: &mut [f64], n: usize) {
    for col in 0..n {
        let mut pivot = col;
        let mut best = matrix[col * n + col].abs();
        for row in col + 1..n {
            let candidate = matrix[row * n + col].abs();
            if candidate > best {
                best = candidate;
                pivot = row;
            }
        }
        assert!(best > 1.0e-300, "singular QA system");
        if pivot != col {
            for k in col..n {
                matrix.swap(col * n + k, pivot * n + k);
            }
            rhs.swap(col, pivot);
        }
        let diag = matrix[col * n + col];
        for row in col + 1..n {
            let factor = matrix[row * n + col] / diag;
            matrix[row * n + col] = 0.0;
            for k in col + 1..n {
                matrix[row * n + k] -= factor * matrix[col * n + k];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    for row in (0..n).rev() {
        let mut sum = rhs[row];
        for k in row + 1..n {
            sum -= matrix[row * n + k] * rhs[k];
        }
        rhs[row] = sum / matrix[row * n + row];
    }
}

fn i_section() -> Section3D {
    Section3D::i_beam(200.0e9, 0.3, 0.1, 0.2, 0.01, 7800.0, false)
}

/// Solve the warping cantilever and return the full 7-DOF/node solution.
///
/// `fix_root_warping` selects the built-in (psi = 0) or free warping root.
fn solve_cantilever(n: usize, torque: f64, fix_root_warping: bool) -> Vec<f64> {
    let section = i_section();
    let l = 2.0;
    let dx = l / n as f64;
    let elements: Vec<WarpingBeamElement3D> = (0..n)
        .map(|i| {
            WarpingBeamElement3D::new(
                [i, i + 1],
                [i as f64 * dx, 0.0, 0.0],
                [(i + 1) as f64 * dx, 0.0, 0.0],
                section,
                [0.0, 1.0, 0.0],
            )
        })
        .collect();
    let n_dofs = (n + 1) * 7;
    let mut k = vec![0.0; n_dofs * n_dofs];
    for elem in &elements {
        let ke = elem.stiffness_global();
        let dofs = elem.global_dofs();
        for (i, &gi) in dofs.iter().enumerate() {
            for (j, &gj) in dofs.iter().enumerate() {
                k[gi * n_dofs + gj] += ke[i * 14 + j];
            }
        }
    }
    let mut fixed = vec![false; n_dofs];
    for c in 0..6 {
        fixed[c] = true;
    }
    if fix_root_warping {
        fixed[6] = true;
    }
    let free: Vec<usize> = (0..n_dofs).filter(|d| !fixed[*d]).collect();
    let m = free.len();
    let mut matrix = vec![0.0; m * m];
    let mut rhs = vec![0.0; m];
    for (i, &gi) in free.iter().enumerate() {
        for (j, &gj) in free.iter().enumerate() {
            matrix[i * m + j] = k[gi * n_dofs + gj];
        }
    }
    rhs[free.iter().position(|&d| d == n_dofs - 4).unwrap()] = torque;
    dense_solve(&mut matrix, &mut rhs, m);
    let mut u = vec![0.0; n_dofs];
    for (i, &gi) in free.iter().enumerate() {
        u[gi] = rhs[i];
    }
    u
}

/// Vlasov twist field of the restrained cantilever at position x.
fn vlasov_twist(section: &Section3D, l: f64, torque: f64, x: f64) -> f64 {
    let mu = (section.gj() / section.warping_stiffness()).sqrt();
    torque / section.gj()
        * (x - (mu * x).sinh() / mu + (mu * l).tanh() * ((mu * x).cosh() - 1.0) / mu)
}

fn test_saint_venant_limit() {
    // Zero warping constant with warping unrestrained (psi free at both
    // ends): one element must give TL/GJ exactly. Fixing psi would restrain
    // the twist rate and stiffen the response even without Iw.
    let section = Section3D::rectangular(200.0e9, 0.3, 0.05, 0.05, 7800.0, false);
    let elem = WarpingBeamElement3D::new(
        [0, 1],
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        section,
        [0.0, 1.0, 0.0],
    );
    let k = elem.stiffness_local();
    // Node 1 structural DOFs fixed, psi1 (index 6) free.
    let free: Vec<usize> = [6].into_iter().chain(7..14).collect();
    let n = free.len();
    let mut matrix = [0.0; 64];
    for (i, &gi) in free.iter().enumerate() {
        for (j, &gj) in free.iter().enumerate() {
            matrix[i * n + j] = k[gi * 14 + gj];
        }
    }
    let mut rhs = [0.0; 8];
    rhs[free.iter().position(|&d| d == 10).unwrap()] = 1000.0;
    dense_solve(&mut matrix, &mut rhs, n);
    let tip = rhs[free.iter().position(|&d| d == 10).unwrap()];
    let expected = 1000.0 * 2.0 / section.gj();
    let error = (tip - expected).abs() / expected;
    println!("saint-venant: FEM={tip:.12e} exact={expected:.12e} rel={error:.2e}");
    assert!(error < 1.0e-12, "saint-venant error {error:.3e}");
}

fn test_free_warping() {
    // Warping stiffness present but psi free at both ends: uniform torsion.
    let u = solve_cantilever(1, 1000.0, false);
    let expected = 1000.0 * 2.0 / i_section().gj();
    let error = (u[10] - expected).abs() / expected;
    println!(
        "free warping: FEM={:.12e} exact={expected:.12e} rel={error:.2e}",
        u[10]
    );
    assert!(error < 1.0e-9, "free warping error {error:.3e}");
}

fn test_restrained_cantilever() {
    let section = i_section();
    let l = 2.0;
    let t = 1000.0;
    let mu = (section.gj() / section.warping_stiffness()).sqrt();
    let expected_tip = t / section.gj() * (l - (mu * l).tanh() / mu);
    let expected_mid = vlasov_twist(&section, l, t, l / 2.0);
    let u = solve_cantilever(12, t, true);
    let n_dofs = 13 * 7;
    let tip = u[n_dofs - 4];
    let mid = u[6 * 7 + 3];
    let err_tip = (tip - expected_tip).abs() / expected_tip;
    let err_mid = (mid - expected_mid).abs() / expected_mid;
    println!("restrained tip: FEM={tip:.12e} exact={expected_tip:.12e} rel={err_tip:.2e}");
    println!("restrained mid: FEM={mid:.12e} exact={expected_mid:.12e} rel={err_mid:.2e}");
    assert!(err_tip < 1.0e-3, "tip twist error {err_tip:.3e}");
    assert!(err_mid < 1.0e-3, "midspan error {err_mid:.3e}");
    println!("GOLDEN beam_warping_tip fem={tip:.17e} hand={expected_tip:.17e}");
}

fn test_root_bimoment() {
    let section = i_section();
    let l = 2.0;
    let t = 1000.0;
    let mu = (section.gj() / section.warping_stiffness()).sqrt();
    let expected = -(t / mu) * (mu * l).tanh();
    let n = 32;
    let u = solve_cantilever(n, t, true);
    // Bimoment from the root element twist DOFs via the Vlasov relation.
    let root = WarpingBeamElement3D::new(
        [0, 1],
        [0.0, 0.0, 0.0],
        [l / n as f64, 0.0, 0.0],
        section,
        [0.0, 1.0, 0.0],
    );
    let twist = [u[3], u[6], u[10], u[13]];
    let fem = root.bimoment_at(&twist, 0.0);
    let error = (fem - expected).abs() / expected.abs();
    println!("root bimoment: FEM={fem:.12e} exact={expected:.12e} rel={error:.2e}");
    // Measured rel = 8.63e-4 against the exact bimoment; gate at ~6x.
    assert!(error < 5.0e-3, "bimoment error {error:.3e}");
}

fn test_convergence() {
    let section = i_section();
    let l = 2.0;
    let t = 1000.0;
    let mu = (section.gj() / section.warping_stiffness()).sqrt();
    let expected = t / section.gj() * (l - (mu * l).tanh() / mu);
    let tip6 = solve_cantilever(6, t, true)[7 * 7 - 4];
    let tip24 = solve_cantilever(24, t, true)[25 * 7 - 4];
    let err6 = (tip6 - expected).abs() / expected;
    let err24 = (tip24 - expected).abs() / expected;
    println!(
        "convergence: err6={err6:.2e} err24={err24:.2e} ratio={:.1}",
        err6 / err24
    );
    assert!(err24 < err6 / 64.0, "no fourth-order convergence");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_saint_venant_limit();
        test_free_warping();
        test_restrained_cantilever();
        test_root_bimoment();
        test_convergence();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
