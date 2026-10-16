//! Coarse JLR shakedown regression: the circulation loop must form
//! and stay finite at a coarse grid before any production run.

// Rust guideline compliant 2026-02-21

use sonium_sim_jlr::case::{CaseConfig, OPERATING_POINTS};
use sonium_sim_jlr::geometry::JlrGeometry;
use sonium_sim_jlr::lbm::{Cell, Lattice, LatticeConfig, tau_for_viscosity};

#[test]
fn coarse_jlr_paper_tau_regularized_stays_finite() {
    // Paper operating regime: tau = 0.500005 with Smagorinsky LES and
    // the regularised collision. SRT-BGK without regularisation blows
    // up from round-off in this geometry (plain-cylinder probe), so
    // this test pins the stabilised configuration.
    let geom = JlrGeometry::default();
    let dx_mm = 4.5;
    let dt_s = 3.6e-5;
    let dx_m = dx_mm * 1.0e-3;
    let nu_lb = 1.0e-6 * dt_s / (dx_m * dx_m);
    let (cells, nx, ny, nz) = geom.voxelise(dx_mm);
    let op = OPERATING_POINTS[1];
    let a_feed = std::f64::consts::PI * (geom.nozzle_inlet_diameter_mm * 1.0e-3 / 2.0).powi(2);
    let v_feed_lb = (op.external_flow_m3_s / a_feed) * dt_s / dx_m;
    let mut cfg = LatticeConfig::new(nx, ny, nz, tau_for_viscosity(nu_lb));
    cfg.smagorinsky = Some(0.1);
    cfg.central_moment = true;
    cfg.inlet_velocity = [0.0, 0.0, -v_feed_lb];
    cfg.outlet_normal = [0, 0, 1];
    let mut lat = Lattice::new(cfg, cells);
    let mut max_u = 0.0f64;
    for step in 0..11111 {
        lat.step((step as f64 / 500.0).min(1.0));
        if step % 100 == 0 {
            for idx in 0..nx * ny * nz {
                let (_, u) = lat.macroscopic(idx);
                for v in u {
                    assert!(v.is_finite(), "non-finite velocity at step {step}");
                    max_u = max_u.max(v.abs());
                }
            }
        }
    }
    assert!(max_u > 0.005, "jet should be flowing, max |u_lb| = {max_u:.4}");
    assert!(max_u < 0.35, "velocity should stay low-Mach, max |u_lb| = {max_u:.4}");
}

#[test]
fn coarse_jlr_paper_tau_parabolic_inlet_stays_finite() {
    // Same regime as the plug-inlet paper-tau test, but with a
    // parabolic feed-pipe profile: the plug's one-cell wall shear
    // layer is where the plug variant first blows up (feed-pipe wall
    // cell, measured), so this pins the stabilised configuration.
    let geom = JlrGeometry::default();
    let dx_mm = 4.5;
    let dt_s = 3.6e-5;
    let dx_m = dx_mm * 1.0e-3;
    let nu_lb = 1.0e-6 * dt_s / (dx_m * dx_m);
    let (cells, nx, ny, nz) = geom.voxelise(dx_mm);
    let op = OPERATING_POINTS[1];
    let a_feed = std::f64::consts::PI * (geom.nozzle_inlet_diameter_mm * 1.0e-3 / 2.0).powi(2);
    let v_feed_lb = (op.external_flow_m3_s / a_feed) * dt_s / dx_m;
    let mut cfg = LatticeConfig::new(nx, ny, nz, tau_for_viscosity(nu_lb));
    cfg.smagorinsky = Some(0.1);
    cfg.central_moment = true;
    cfg.inlet_velocity = [0.0, 0.0, -v_feed_lb];
    cfg.outlet_normal = [0, 0, 1];
    let mut lat = Lattice::new(cfg, cells);
    let r_feed_mm = geom.nozzle_inlet_diameter_mm / 2.0;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let idx = lat.index(x, y, z);
                if lat.cell(idx) != Cell::Inlet {
                    continue;
                }
                let xm = (x as f64 + 0.5) * dx_mm - nx as f64 * dx_mm / 2.0;
                let ym = (y as f64 + 0.5) * dx_mm - ny as f64 * dx_mm / 2.0;
                let r = (xm * xm + ym * ym).sqrt();
                let shape = 2.0 * (1.0 - (r / r_feed_mm).powi(2)).max(0.0);
                lat.set_inlet_velocity(idx, [0.0, 0.0, -v_feed_lb * shape]);
            }
        }
    }
    let mut max_u = 0.0f64;
    for step in 0..11111 {
        lat.step((step as f64 / 500.0).min(1.0));
        if step % 100 == 0 {
            for idx in 0..nx * ny * nz {
                let (_, u) = lat.macroscopic(idx);
                for v in u {
                    assert!(v.is_finite(), "non-finite velocity at step {step}");
                    max_u = max_u.max(v.abs());
                }
            }
        }
    }
    assert!(max_u > 0.005, "jet should be flowing, max |u_lb| = {max_u:.4}");
    assert!(max_u < 0.35, "velocity should stay low-Mach, max |u_lb| = {max_u:.4}");
}

#[test]
fn coarse_jlr_short_run_stays_finite() {
    let geom = JlrGeometry::default();
    let dx_mm = 4.5;
    let dt_s = 3.6e-5;
    let dx_m = dx_mm * 1.0e-3;
    // Coarse-grid shakedown uses a viscous surrogate at tau = 0.8
    // (nu_lb = 0.1): at 4.5 mm the 12 mm nozzle is 2.7 cells wide and
    // SRT-BGK at the paper's tau = 0.500005 is under-resolved and
    // unstable (verified: plain-cylinder and JLR probes both blow up
    // at tau ~ 0.5005 with or without LES, with inlet or body force).
    // This test validates geometry and boundary plumbing, not
    // turbulent stability at the paper's operating point.
    let nu_lb = 0.1;
    let (cells, nx, ny, nz) = geom.voxelise(dx_mm);
    let op = OPERATING_POINTS[1];
    let a_feed = std::f64::consts::PI * (geom.nozzle_inlet_diameter_mm * 1.0e-3 / 2.0).powi(2);
    let v_feed_lb = (op.external_flow_m3_s / a_feed) * dt_s / dx_m;
    let mut cfg = LatticeConfig::new(nx, ny, nz, tau_for_viscosity(nu_lb));
    cfg.smagorinsky = Some(0.1);
    cfg.inlet_velocity = [0.0, 0.0, -v_feed_lb];
    cfg.outlet_normal = [0, 0, 1];
    let mut lat = Lattice::new(cfg, cells);
    let mut max_u = 0.0f64;
    for step in 0..2000 {
        lat.step((step as f64 / 500.0).min(1.0));
        if step % 100 == 0 {
            for idx in 0..nx * ny * nz {
                let (_, u) = lat.macroscopic(idx);
                for v in u {
                    assert!(v.is_finite(), "non-finite velocity at step {step}");
                    max_u = max_u.max(v.abs());
                }
            }
        }
    }
    assert!(max_u > 0.01, "jet should be flowing, max |u_lb| = {max_u:.4}");
    assert!(max_u < 0.3, "velocity should stay low-Mach, max |u_lb| = {max_u:.4}");
    // Downward flow inside the draft tube at its mid height.
    let centre = lat.index(nx / 2, ny / 2, (900.0 / dx_mm) as usize);
    let (_, u_dt) = lat.macroscopic(centre);
    assert!(u_dt[2] < 0.0, "draft tube flow should be downward, got {}", u_dt[2]);
    let _ = Cell::Fluid;
    let _ = CaseConfig::default();
}
