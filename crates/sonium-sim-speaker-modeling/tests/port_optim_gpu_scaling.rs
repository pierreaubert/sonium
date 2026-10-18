//! Tier-2 GPU scaling ladder: CPU direct versus CUDA backend solves.
//!
//! Screening evidence only. Without the `cuda` feature this file is a
//! no-op explanation test; with it, the ignored ladder runs the straight
//! tube at three mesh sizes and two step counts on both devices, checks
//! metric agreement, and prints wall time per BDF step so the scaling of
//! each backend is visible. Run explicitly in release mode:
//!
//! ```sh
//! cargo test -p sonium-sim-speaker-modeling --release --features cuda \
//!   --test port_optim_gpu_scaling -- --ignored --nocapture
//! ```

#[cfg(not(feature = "cuda"))]
#[test]
fn gpu_scaling_requires_cuda_feature() {
    // The ladder only exists in cuda-feature builds; see the file docs.
}

#[cfg(feature = "cuda")]
use sonium_sim_speaker_modeling::port_flow::{
    PortEvalDrive, PortEvalFidelity, PortOptimReferenceCase, ProfileOptions, Tier2EvalContext,
    Tier2SolverDevice, Tier2SolverSettings,
};

#[cfg(feature = "cuda")]
fn run_one(
    mesh_size_m: f64,
    samples_per_cycle: usize,
    device: Tier2SolverDevice,
) -> (f64, usize, f64) {
    let case = PortOptimReferenceCase::default();
    let frequency_hz = case.tier2_drive_frequency_hz().expect("drive frequency");
    let context = Tier2EvalContext::from_reference_case(case)
        .expect("context")
        .with_mesh_size_m(mesh_size_m)
        .expect("mesh override")
        .with_solver_settings(Tier2SolverSettings {
            device,
            ..Tier2SolverSettings::default()
        });
    let profile = context
        .candidate_profile(5, &[0.0, 0.0], ProfileOptions::default())
        .expect("straight profile");
    let fidelity = PortEvalFidelity {
        samples_per_cycle,
        total_cycles: 6,
        tail_cycles: 2,
        snapshot_interval_steps: 0,
        compression_reference: None,
    };
    let drive = PortEvalDrive {
        voltage_peak_v: 5.0,
        frequency_hz,
        second_tone: None,
    };
    let elements = context.baseline_mesh().elements.len();
    let started = std::time::Instant::now();
    let outcome = context
        .evaluate(&profile, &drive, &fidelity)
        .expect("evaluation must succeed on the requested device");
    let wall_s = started.elapsed().as_secs_f64();
    assert!(outcome.metrics.health.passed, "run must be healthy");
    (wall_s, elements, outcome.metrics.port_flow_rms_m3s)
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "GPU scaling ladder; run explicitly in release mode with --features cuda"]
fn gpu_scaling_ladder() {
    // Mesh ladder at a fixed step count, then a step ladder at the
    // production mesh. Each cell runs on both devices.
    let mut cells: Vec<(f64, usize)> = vec![(0.02, 48), (0.01, 48), (0.005, 48)];
    cells.push((0.01, 144));
    for (mesh_size_m, samples_per_cycle) in cells {
        let steps = 6 * samples_per_cycle;
        let (cpu_s, elements, cpu_q) =
            run_one(mesh_size_m, samples_per_cycle, Tier2SolverDevice::Cpu);
        let (gpu_s, _, gpu_q) = run_one(mesh_size_m, samples_per_cycle, Tier2SolverDevice::Cuda);
        let agree_pct = 100.0 * (gpu_q / cpu_q - 1.0);
        eprintln!(
            "mesh={mesh_size_m} elements={elements} steps={steps}: \
             CPU {cpu_s:.2}s ({:.1} ms/step) | CUDA {gpu_s:.2}s ({:.1} ms/step) | \
             speedup {:.2}x | flow agreement {agree_pct:+.2}%",
            1000.0 * cpu_s / steps as f64,
            1000.0 * gpu_s / steps as f64,
            cpu_s / gpu_s,
        );
        assert!(
            agree_pct.abs() < 10.0,
            "CPU/CUDA flow must agree within 10%, got {agree_pct}%"
        );
    }
}
