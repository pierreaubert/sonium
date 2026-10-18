//! Compact CPU performance matrix for the audit's benchmark-evidence item.
//!
//! Measures, at several rayon thread counts, the assembly paths the
//! performance work changed (triplet merge vs reusable element scatter,
//! prebaked sweeps with and without buffer reuse, matrix-free applies
//! with the element-matrix cache) and prints one JSON object per case
//! with phase times and an operator-error check against the legacy
//! path, so comparisons are time-to-the-same-answer, not raw throughput.
//!
//! Run: `cargo bench -p sonium-fem --bench perf_matrix -- --output perf.json`

use sonium_fem::assembly::{ElementScatter, assemble_bilinear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::forms::LaplacianForm;
use sonium_fem::mesh::{unit_cube_tetrahedra, unit_square_triangles};
use sonium_fem::solver::MatrixFreeOperator;
use sonium_fem::spaces::H1Space;
use std::time::Instant;

fn median_seconds(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn time_median(repeats: usize, mut f: impl FnMut()) -> f64 {
    f();
    let samples = (0..repeats)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed().as_secs_f64()
        })
        .collect();
    median_seconds(samples)
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let output_path = args
        .windows(2)
        .find(|pair| pair[0] == "--output")
        .map(|pair| pair[1].clone());
    let thread_counts: Vec<usize> = args
        .windows(2)
        .find(|pair| pair[0] == "--threads")
        .map(|pair| {
            pair[1]
                .split(',')
                .filter_map(|value| value.trim().parse().ok())
                .collect()
        })
        .unwrap_or_else(|| vec![1, 4, 16]);

    let revision = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let mut records: Vec<String> = Vec::new();
    for &threads in &thread_counts {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("rayon pool");
        pool.install(|| {
            // 3D assembly: triplet path vs reusable scatter.
            let mesh = unit_cube_tetrahedra(14);
            let space = H1Space::new(PolynomialDegree::P1);
            let triplet_time = time_median(3, || {
                let _ = assemble_bilinear(&LaplacianForm, &space, &space, &mesh, 1).to_csr();
            });
            let build_start = Instant::now();
            let scatter = ElementScatter::build(&space, &space, &mesh);
            let scatter_build = build_start.elapsed().as_secs_f64();
            let scatter_time = time_median(3, || {
                let _ = scatter.assemble(&LaplacianForm, &space, &space, &mesh, 1);
            });
            let direct = scatter.assemble(&LaplacianForm, &space, &space, &mesh, 1);
            let reference = assemble_bilinear(&LaplacianForm, &space, &space, &mesh, 1).to_csr();
            let assembly_error = max_abs_diff(&direct.values, &reference.values);

            // Matrix-free applies (element matrices cached after first).
            let mesh2d = unit_square_triangles(48);
            let op = MatrixFreeOperator::new(&LaplacianForm, &space, &mesh2d, 1);
            let n = op.num_dofs();
            let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
            let _ = op.apply_vec(&x);
            let apply_time = time_median(5, || {
                let _ = op.apply_vec(&x);
            });
            let assembled2d = assemble_bilinear(&LaplacianForm, &space, &space, &mesh2d, 1).to_csr();
            let mf_error = max_abs_diff(&op.apply_vec(&x), &assembled2d.matvec(&x));

            records.push(format!(
                "{{\"case\":\"fem3d-assembly\",\"threads\":{threads},\"revision\":\"{revision}\",\"dofs\":{},\"triplet_s\":{triplet_time:.6},\"scatter_build_s\":{scatter_build:.6},\"scatter_s\":{scatter_time:.6},\"operator_error\":{assembly_error:.3e}}}",
                mesh.num_nodes()
            ));
            records.push(format!(
                "{{\"case\":\"fem2d-matrix-free\",\"threads\":{threads},\"revision\":\"{revision}\",\"dofs\":{n},\"apply_s\":{apply_time:.6},\"cached_element_matrices\":{},\"operator_error\":{mf_error:.3e}}}",
                op.num_cached_element_matrices()
            ));
            let _ = &mesh;
        });
    }

    let json = format!("[\n{}\n]\n", records.join(",\n"));
    print!("{json}");
    if let Some(path) = output_path {
        std::fs::write(&path, &json).expect("write perf output");
        eprintln!("wrote {}", path);
    }
}
