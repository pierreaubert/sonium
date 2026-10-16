//! Verification ladder for the D3Q19 core, independent of the JLR case.
//!
//! Poiseuille flow against the analytic parabola, Taylor–Green vortex
//! decay against the lattice viscosity, lid-driven cavity sanity against
//! the qualitative Ghia profile shape, voxelised annulus area against
//! the analytic annulus, and refinement-interface continuity.

// Rust guideline compliant 2026-02-21

use sonium_sim_jlr::geometry::JlrGeometry;
use sonium_sim_jlr::lbm::{Cell, Lattice, LatticeConfig, tau_for_viscosity, viscosity_lattice};
use sonium_sim_jlr::refinement::{RefinedSolver, channel_cells};

fn channel(nx: usize, ny: usize, nz: usize, tau: f64, force_x: f64) -> Lattice {
    let mut cfg = LatticeConfig::new(nx, ny, nz, tau);
    cfg.periodic = [true, true, false];
    cfg.force = [force_x, 0.0, 0.0];
    Lattice::new(cfg, channel_cells(nx, ny, nz))
}

#[test]
fn poiseuille_centre_velocity_matches_analytic() {
    // Fluid layers z = 1..=48 (nz = 50 with two solid layers), halfway
    // walls at z = 0.5 and z = 48.5, so H = 48, half-height a = 24.
    let (nx, ny, nz) = (4, 4, 50);
    let nu = 0.1;
    let tau = tau_for_viscosity(nu);
    let force = 1.0e-5;
    let mut lat = channel(nx, ny, nz, tau, force);
    for _ in 0..150_000 {
        lat.step(1.0);
    }
    let a = 24.0f64;
    let u_max = force * a * a / (2.0 * nu); // rho = 1
    // Node z = 24 sits 0.5 below the centre plane.
    let centre = lat.macroscopic(lat.index(2, 2, 24)).1[0];
    let centre_expected = u_max * (1.0 - (0.5 / a).powi(2));
    let rel = ((centre - centre_expected) / centre_expected).abs();
    // BGK halfway bounce-back has a tau-dependent effective wall shift;
    // measured deviation at H = 48 is 1.0%, so the gate is 1.5% here and
    // the Taylor-Green test below carries the tighter viscosity check.
    assert!(rel < 0.015, "centre {centre:.6} vs analytic {centre_expected:.6} (rel {rel:.4})");
    // Node z = 12 is 11.5 above the wall plane, 12.5 from the centre.
    let off = lat.macroscopic(lat.index(2, 2, 12)).1[0];
    let expected = u_max * (1.0 - (12.5 / a).powi(2));
    let rel2 = ((off - expected) / expected).abs();
    assert!(rel2 < 0.02, "off-centre {off:.6} vs {expected:.6} (rel {rel2:.4})");
}

#[test]
fn poiseuille_central_moment_matches_analytic() {
    let (nx, ny, nz) = (4, 4, 34);
    let nu = 0.1;
    let force = 1.0e-5;
    let mut cfg = LatticeConfig::new(nx, ny, nz, tau_for_viscosity(nu));
    cfg.periodic = [true, true, false];
    cfg.force = [force, 0.0, 0.0];
    cfg.central_moment = true;
    let mut lat = Lattice::new(cfg, channel_cells(nx, ny, nz));
    for _ in 0..80_000 {
        lat.step(1.0);
    }
    let a = 16.0f64;
    let u_max = force * a * a / (2.0 * nu);
    let centre = lat.macroscopic(lat.index(2, 2, 16)).1[0];
    let expected = u_max * (1.0 - (0.5 / a).powi(2));
    let rel = ((centre - expected) / expected).abs();
    assert!(rel < 0.015, "central-moment centre {centre:.6} vs {expected:.6} (rel {rel:.4})");
}

#[test]
fn taylor_green_central_moment_recovers_viscosity() {
    let n = 32usize;
    let nu = 0.05;
    let mut cfg = LatticeConfig::new(n, n, n, tau_for_viscosity(nu));
    cfg.periodic = [true, true, true];
    cfg.central_moment = true;
    let mut lat = Lattice::new(cfg, vec![Cell::Fluid; n * n * n]);
    let k = 2.0 * std::f64::consts::PI / n as f64;
    let u0 = 0.05f64;
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let (xf, yf, zf) = (x as f64 * k, y as f64 * k, z as f64 * k);
                let u = [
                    u0 * xf.sin() * yf.cos() * zf.cos(),
                    -u0 * xf.cos() * yf.sin() * zf.cos(),
                    0.0,
                ];
                lat.set_equilibrium(lat.index(x, y, z), 1.0, u);
            }
        }
    }
    let energy = |lat: &Lattice| -> f64 {
        let mut e = 0.0;
        for idx in 0..n * n * n {
            let (_, u) = lat.macroscopic(idx);
            e += u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
        }
        e / (n * n * n) as f64
    };
    let e0 = energy(&lat);
    for _ in 0..400 {
        lat.step(1.0);
    }
    let e1 = energy(&lat);
    let nu_meas = -(e1 / e0).ln() / (2.0 * 3.0 * k * k * 400.0);
    let rel = ((nu_meas - nu) / nu).abs();
    assert!(rel < 0.05, "central-moment nu {nu_meas:.5} vs {nu} (rel {rel:.4})");
}

#[test]
fn taylor_green_decay_recovers_viscosity() {
    let n = 32usize;
    let nu = 0.05;
    let tau = tau_for_viscosity(nu);
    let mut cfg = LatticeConfig::new(n, n, n, tau);
    cfg.periodic = [true, true, true];
    let mut lat = Lattice::new(cfg, vec![Cell::Fluid; n * n * n]);
    let k = 2.0 * std::f64::consts::PI / n as f64;
    let u0 = 0.05f64;
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let (xf, yf, zf) = (x as f64 * k, y as f64 * k, z as f64 * k);
                let u = [
                    u0 * xf.sin() * yf.cos() * zf.cos(),
                    -u0 * xf.cos() * yf.sin() * zf.cos(),
                    0.0,
                ];
                let idx = lat.index(x, y, z);
                lat.set_equilibrium(idx, 1.0, u);
            }
        }
    }
    let energy = |lat: &Lattice| -> f64 {
        let mut e = 0.0;
        for idx in 0..n * n * n {
            let (_, u) = lat.macroscopic(idx);
            e += u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
        }
        e / (n * n * n) as f64
    };
    let e0 = energy(&lat);
    let steps = 400;
    for _ in 0..steps {
        lat.step(1.0);
    }
    let e1 = energy(&lat);
    // E(t) ~ E0 * exp(-2 nu |k|^2 t) with |k|^2 = 3k^2 for this mode.
    let nu_meas = -(e1 / e0).ln() / (2.0 * 3.0 * k * k * steps as f64);
    let rel = ((nu_meas - nu) / nu).abs();
    assert!(rel < 0.05, "measured nu {nu_meas:.5} vs {nu} (rel {rel:.4}); e0={e0:.3e} e1={e1:.3e}");
}

#[test]
fn lid_driven_cavity_develops_primary_vortex() {
    let n = 24usize;
    let nu = 0.05; // Re = U*L/nu = 0.1*24/0.05 = 48
    let tau = tau_for_viscosity(nu);
    let mut cells = vec![Cell::Fluid; n * n * n];
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                if x == 0 || y == 0 || z == 0 || x == n - 1 || y == n - 1 || z == n - 1 {
                    cells[x + n * (y + n * z)] = Cell::Solid;
                }
            }
        }
    }
    let mut lat = Lattice::new(LatticeConfig::new(n, n, n, tau), cells);
    // Moving lid: top layer (z = n-1) solid cells carry +x wall velocity.
    for y in 0..n {
        for x in 0..n {
            let idx = lat.index(x, y, n - 1);
            lat.set_wall_velocity(idx, [0.1, 0.0, 0.0]);
        }
    }
    for _ in 0..8_000 {
        lat.step(1.0);
    }
    // Below the lid the flow follows +x; near the bottom it returns -x.
    let upper = lat.macroscopic(lat.index(n / 2, n / 2, n - 3)).1[0];
    let lower = lat.macroscopic(lat.index(n / 2, n / 2, 2)).1[0];
    assert!(upper > 0.01, "upper flow should follow the lid, got {upper:.4}");
    assert!(lower < 0.0, "lower flow should return against the lid, got {lower:.4}");
}

#[test]
fn voxelised_annulus_area_within_bounce_back_band() {
    let geom = JlrGeometry::default();
    let dx = 1.5;
    let (cells, nx, ny, _nz) = geom.voxelise(dx);
    let voxel = geom.voxel_annulus_area_mm2(&cells, nx, ny, dx);
    let analytic = geom.annulus_area_mm2();
    let rel = (voxel - analytic) / analytic;
    // The paper's artefact is a slightly narrowed effective annulus; the
    // voxel count itself must stay within a few percent at 1.5 mm.
    assert!(rel.abs() < 0.04, "voxel annulus {voxel:.0} vs analytic {analytic:.0} (rel {rel:.4})");
}

#[test]
fn jlr_voxelisation_has_inlet_outlet_and_draft_tube() {
    let geom = JlrGeometry::default();
    let (cells, _nx, _ny, _nz) = geom.voxelise(4.5);
    let count = |kind: Cell| cells.iter().filter(|c| **c == kind).count();
    assert!(count(Cell::Inlet) > 20, "feed-pipe inlet cells missing");
    assert!(count(Cell::Outlet) > 100, "bottom outlet cells missing");
    assert!(count(Cell::Fluid) > 100_000, "fluid volume implausibly small");
    // Axis point inside the draft tube at mid height is fluid; the
    // draft tube wall ring is solid.
    assert_eq!(geom.classify(0.0, 0.0, 900.0), Cell::Fluid);
    assert_eq!(geom.classify(52.5, 0.0, 900.0), Cell::Solid);
    assert_eq!(geom.classify(70.0, 0.0, 900.0), Cell::Fluid);
}

#[test]
fn d3q19_tables_are_self_consistent() {
    use sonium_sim_jlr::lbm::{CX, CY, CZ, OPP, W};
    // Every direction's opposite must be the negated velocity, and the
    // weights must be normalized and isotropic. A mismatched OPP table
    // corrupts every diagonal bounce-back (mass leak at walls and
    // spurious wall blow-up were the measured symptoms).
    for i in 0..19 {
        let o = OPP[i];
        assert_eq!(
            (CX[o], CY[o], CZ[o]),
            (-CX[i], -CY[i], -CZ[i]),
            "OPP[{i}]={o} is not the opposite direction"
        );
        assert_eq!(OPP[o], i, "OPP must be an involution");
    }
    let sum: f64 = W.iter().sum();
    assert!((sum - 1.0).abs() < 1e-15, "weights sum to {sum}");
    for a in 0..3 {
        for b in 0..3 {
            let c = [CX, CY, CZ];
            let mut s = 0.0;
            for i in 0..19 {
                s += W[i] * c[a][i] as f64 * c[b][i] as f64;
            }
            let expected = if a == b { 1.0 / 3.0 } else { 0.0 };
            assert!((s - expected).abs() < 1e-15, "second moment [{a}][{b}] = {s}");
        }
    }
}

#[test]
fn moment_basis_roundtrip() {
    use sonium_sim_jlr::lbm::{moment_inverse, moment_matrix};
    let m = moment_matrix();
    let inv = moment_inverse();
    // f -> moments -> f must be the identity on arbitrary populations.
    let f: [f64; 19] = std::array::from_fn(|i| 0.05 + 0.01 * (i as f64).sin());
    let mut mom = [0.0; 19];
    for (k, mk) in mom.iter_mut().enumerate() {
        for i in 0..19 {
            *mk += m[k][i] * f[i];
        }
    }
    for i in 0..19 {
        let mut back = 0.0;
        for k in 0..19 {
            back += inv[i][k] * mom[k];
        }
        assert!((back - f[i]).abs() < 1e-10, "roundtrip {i}: {back} vs {}", f[i]);
    }
}

#[test]
fn viscosity_roundtrip() {
    for tau in [0.51, 0.8, 1.0, 1.9] {
        let nu = viscosity_lattice(tau);
        assert!((tau_for_viscosity(nu) - tau).abs() < 1e-12);
    }
}

#[test]
fn refined_channel_keeps_poiseuille_profile() {
    // Coarse channel with a fine patch across the middle; the centre
    // velocity after coupling must stay near the single-grid value.
    let (nx, ny, nz) = (12, 6, 16);
    let nu = 0.1;
    let force = 1.0e-5;
    let coarse = channel(nx, ny, nz, tau_for_viscosity(nu), force);
    let origin = [0, 0, 4];
    let extent = [12, 6, 8];
    let (fx, fy, fz) = (2 * extent[0], 2 * extent[1], 2 * extent[2]);
    // Fine lattice: half spacing => dt/4 diffusive scaling keeps the same
    // physical viscosity with tau_f from nu_lb_f = nu_lb * (dt_f/dt) /
    // (dx_f/dx)^2 = nu_lb * 0.5 / 0.25 = 2 * nu_lb... use tau directly:
    let nu_f = nu * 2.0;
    let mut fcfg = LatticeConfig::new(fx, fy, fz, tau_for_viscosity(nu_f));
    fcfg.periodic = [true, true, false];
    fcfg.force = [force * 0.5, 0.0, 0.0]; // Guo force density scales as dt^2/dx: fine/coarse = 1/2
    // Fine patch z walls are interior coarse fluid; only its z extent is
    // fluid, bounded by halo coupling, so no solid layers of its own.
    let fine = Lattice::new(fcfg, vec![Cell::Fluid; fx * fy * fz]);
    let mut solver = RefinedSolver::new(coarse, fine, origin, extent);
    for _ in 0..20_000 {
        solver.step(1.0);
    }
    let centre = solver.patch_macroscopic(6, 3, 7).1[0];
    let coarse_edge = solver.coarse().macroscopic(solver.coarse().index(2, 3, 7)).1[0];
    let a = 7.0f64;
    let u_max = force * a * a / (2.0 * nu);
    assert!(coarse_edge > 0.5 * u_max, "coarse edge {coarse_edge:.5} vs analytic {u_max:.5}");
    assert!(centre > 0.5 * u_max, "refined centre {centre:.5} vs analytic {u_max:.5}");
    assert!(centre < 1.6 * u_max, "refined centre {centre:.5} overshoots analytic {u_max:.5}");
}

#[test]
fn rest_state_is_stable() {
    let mut lat = Lattice::new(
        LatticeConfig::new(8, 8, 8, 1.0),
        vec![Cell::Fluid; 512],
    );
    for _ in 0..100 {
        lat.step(1.0);
    }
    let (rho, u) = lat.macroscopic(100);
    assert!((rho - 1.0).abs() < 1e-12 && u.iter().all(|v| v.abs() < 1e-12));
}
