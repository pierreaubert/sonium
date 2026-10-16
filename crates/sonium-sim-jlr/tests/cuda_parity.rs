#![cfg(feature = "cuda")]
//! CPU↔GPU parity on a forced channel, plus a JLR-grid smoke step.

// Rust guideline compliant 2026-02-21

use sonium_sim_jlr::cuda::CudaLattice;
use sonium_sim_jlr::lbm::{Lattice, LatticeConfig, tau_for_viscosity};
use sonium_sim_jlr::refinement::channel_cells;

#[test]
fn cuda_channel_regularized_near_half_tau_matches_cpu() {
    let (nx, ny, nz) = (6, 6, 16);
    let tau = tau_for_viscosity(0.0033333); // tau = 0.51
    let force = 1.0e-6;
    let cells = channel_cells(nx, ny, nz);
    let mut cfg = LatticeConfig::new(nx, ny, nz, tau);
    cfg.periodic = [true, true, false];
    cfg.force = [force, 0.0, 0.0];
    cfg.regularized = true;
    cfg.smagorinsky = Some(0.1);
    let mut cpu = Lattice::new(cfg, cells.clone());
    let mut gpu = CudaLattice::with_periodic(
        nx, ny, nz, tau, Some(0.1), true, [force, 0.0, 0.0], [0.0; 3], [0, 0, 1],
        [true, true, false], &cells,
    )
    .expect("CUDA lattice");
    for _ in 0..3_000 {
        cpu.step(1.0);
        gpu.step(1.0, false).expect("GPU step");
    }
    let cpu_u = cpu.macroscopic(cpu.index(3, 3, 7)).1[0];
    assert!(cpu_u.is_finite() && cpu_u > 0.0, "CPU centre {cpu_u}");
    let vel = gpu.download_velocity().expect("download");
    let gpu_u = vel[3 + nx * (3 + ny * 7)][0];
    assert!(gpu_u.is_finite(), "GPU centre not finite: {gpu_u}");
    let rel = ((gpu_u - cpu_u) / cpu_u).abs();
    assert!(rel < 0.05, "GPU {gpu_u:.6} vs CPU {cpu_u:.6} (rel {rel:.4})");
}

#[test]
fn cuda_channel_central_moment_matches_cpu() {
    let (nx, ny, nz) = (6, 6, 16);
    let tau = tau_for_viscosity(0.0033333); // tau = 0.51
    let force = 1.0e-6;
    let cells = channel_cells(nx, ny, nz);
    let mut cfg = LatticeConfig::new(nx, ny, nz, tau);
    cfg.periodic = [true, true, false];
    cfg.force = [force, 0.0, 0.0];
    cfg.central_moment = true;
    cfg.smagorinsky = Some(0.1);
    let mut cpu = Lattice::new(cfg, cells.clone());
    let mut gpu = CudaLattice::with_periodic(
        nx, ny, nz, tau, Some(0.1), false, [force, 0.0, 0.0], [0.0; 3], [0, 0, 1],
        [true, true, false], &cells,
    )
    .expect("CUDA lattice");
    gpu.set_central_moment(true);
    for _ in 0..3_000 {
        cpu.step(1.0);
        gpu.step(1.0, false).expect("GPU step");
    }
    let cpu_u = cpu.macroscopic(cpu.index(3, 3, 7)).1[0];
    assert!(cpu_u.is_finite() && cpu_u > 0.0, "CPU centre {cpu_u}");
    let vel = gpu.download_velocity().expect("download");
    let gpu_u = vel[3 + nx * (3 + ny * 7)][0];
    assert!(gpu_u.is_finite(), "GPU centre not finite: {gpu_u}");
    let rel = ((gpu_u - cpu_u) / cpu_u).abs();
    assert!(rel < 0.05, "GPU {gpu_u:.6} vs CPU {cpu_u:.6} (rel {rel:.4})");
}

#[test]
fn cuda_channel_matches_cpu() {
    let (nx, ny, nz) = (6, 6, 16);
    let tau = tau_for_viscosity(0.1);
    let force = 1.0e-5;
    let cells = channel_cells(nx, ny, nz);
    let mut cfg = LatticeConfig::new(nx, ny, nz, tau);
    cfg.periodic = [true, true, false];
    cfg.force = [force, 0.0, 0.0];
    let mut cpu = Lattice::new(cfg, cells.clone());
    let mut gpu = CudaLattice::with_periodic(
        nx, ny, nz, tau, None, false, [force, 0.0, 0.0], [0.0; 3], [0, 0, 1],
        [true, true, false], &cells,
    )
    .expect("CUDA lattice");
    for _ in 0..3_000 {
        cpu.step(1.0);
        gpu.step(1.0, false).expect("GPU step");
    }
    let cpu_u = cpu.macroscopic(cpu.index(3, 3, 7)).1[0];
    let vel = gpu.download_velocity().expect("download");
    let gpu_u = vel[3 + nx * (3 + ny * 7)][0];
    let rel = ((gpu_u - cpu_u) / cpu_u).abs();
    assert!(rel < 0.05, "GPU {gpu_u:.6} vs CPU {cpu_u:.6} (rel {rel:.4})");
}
