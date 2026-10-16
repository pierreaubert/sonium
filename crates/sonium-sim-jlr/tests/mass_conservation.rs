//! Mass conservation of the central-moment collision and BCs.

// Rust guideline compliant 2026-02-21

use sonium_sim_jlr::lbm::{Cell, Lattice, LatticeConfig, tau_for_viscosity};

fn total_mass(lat: &Lattice, n: usize) -> f64 {
    (0..n).map(|idx| lat.macroscopic(idx).0).sum()
}

fn split_mass(lat: &Lattice, n: usize) -> (f64, f64) {
    let mut fluid = 0.0;
    let mut solid = 0.0;
    for idx in 0..n {
        if lat.cell(idx) == Cell::Solid {
            solid += lat.macroscopic(idx).0;
        } else {
            fluid += lat.macroscopic(idx).0;
        }
    }
    (fluid, solid)
}

#[test]
fn central_moment_closed_box_conserves_mass() {
    let n = 12usize;
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
    let mut cfg = LatticeConfig::new(n, n, n, tau_for_viscosity(0.0033333));
    cfg.central_moment = true;
    cfg.smagorinsky = Some(0.1);
    let mut lat = Lattice::new(cfg, cells);
    // Perturb: a jet-like blob in the middle.
    for z in 4..8 {
        for y in 4..8 {
            for x in 4..8 {
                let idx = lat.index(x, y, z);
                lat.set_equilibrium(idx, 1.0, [0.0, 0.0, -0.05]);
            }
        }
    }
    let m0 = total_mass(&lat, n * n * n);
    for _ in 0..2000 {
        lat.step(1.0);
    }
    let m1 = total_mass(&lat, n * n * n);
    let drift = ((m1 - m0) / m0).abs();
    assert!(drift < 1e-9, "closed-box mass drift {drift:e} (m0={m0}, m1={m1})");
}

#[test]
fn bgk_closed_box_first_steps() {
    let n = 12usize;
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
    let mut cfg = LatticeConfig::new(n, n, n, tau_for_viscosity(0.0033333));
    cfg.smagorinsky = Some(0.1);
    let mut lat = Lattice::new(cfg, cells);
    for z in 4..8 {
        for y in 4..8 {
            for x in 4..8 {
                let idx = lat.index(x, y, z);
                lat.set_equilibrium(idx, 1.0, [0.0, 0.0, -0.05]);
            }
        }
    }
    // Mass must be conserved from the very first steps, including the
    // steps where blob populations first reach the walls (a mismatched
    // opposite-direction table leaked exactly there).
    for _ in 0..12 {
        lat.step(1.0);
        let (fl, _so) = split_mass(&lat, n * n * n);
        assert!((fl - 1000.0).abs() < 1e-6, "fluid mass {fl} after step");
    }
}

#[test]
fn bgk_closed_box_conserves_mass() {
    let n = 12usize;
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
    let mut cfg = LatticeConfig::new(n, n, n, tau_for_viscosity(0.0033333));
    cfg.smagorinsky = Some(0.1);
    let mut lat = Lattice::new(cfg, cells);
    for z in 4..8 {
        for y in 4..8 {
            for x in 4..8 {
                let idx = lat.index(x, y, z);
                lat.set_equilibrium(idx, 1.0, [0.0, 0.0, -0.05]);
            }
        }
    }
    let m0 = total_mass(&lat, n * n * n);
    for _ in 0..2000 {
        lat.step(1.0);
    }
    let m1 = total_mass(&lat, n * n * n);
    let drift = ((m1 - m0) / m0).abs();
    assert!(drift < 1e-9, "BGK closed-box mass drift {drift:e} (m0={m0}, m1={m1})");
}

#[test]
fn central_moment_inlet_outlet_box_mass_balance() {
    // Box with a top inlet layer and bottom outlet layer, otherwise
    // walls: at steady state the density must settle, not drain.
    let (nx, ny, nz) = (10usize, 10usize, 20usize);
    let mut cells = vec![Cell::Solid; nx * ny * nz];
    for z in 0..nz {
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let idx = x + nx * (y + ny * z);
                cells[idx] = if z == nz - 1 {
                    Cell::Inlet
                } else if z == 0 {
                    Cell::Outlet
                } else {
                    Cell::Fluid
                };
            }
        }
    }
    let mut cfg = LatticeConfig::new(nx, ny, nz, tau_for_viscosity(0.0033333));
    cfg.central_moment = true;
    cfg.inlet_velocity = [0.0, 0.0, -0.01];
    cfg.outlet_velocity = [0.0, 0.0, -0.01];
    cfg.outlet_normal = [0, 0, 1];
    let mut lat = Lattice::new(cfg, cells);
    let m0 = total_mass(&lat, nx * ny * nz);
    for _ in 0..4000 {
        lat.step(1.0);
    }
    let m1 = total_mass(&lat, nx * ny * nz);
    let drift = (m1 - m0) / m0;
    assert!(drift.abs() < 0.02, "inlet/outlet box mass drift {drift:.4} (m0={m0:.1}, m1={m1:.1})");
}
