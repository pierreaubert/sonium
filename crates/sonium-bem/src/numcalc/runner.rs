//! Run a Mesh2HRTF project across all frequencies and capture the surface
//! pressure plus optional evaluation-grid pressure for each one.

// Rust guideline compliant 2026-02-21
use std::fs;
use std::path::Path;

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::core::bem_solver::{AssemblyMethod, BemError};
use crate::core::incident::IncidentField;
use crate::core::postprocess::pressure::compute_total_field;
use crate::core::solver::frequency_sweep::{FrequencySweepConfig, frequency_sweep};
use crate::core::types::{BasisOrder, ElementProperty};

use super::project::NumCalcProject;

/// Explicit execution route used by the NumCalc producer.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SweepExecution {
    /// CPU assembly and GMRES with CPU field evaluation.
    Cpu,
    /// CUDA TBEM, GMRES and Jacobi; exported solutions feed CPU field evaluation.
    CudaJacobiWithCpuFields,
    /// CUDA TBEM, GMRES, Jacobi and off-surface fields; host JSON export only.
    CudaJacobi,
}

struct SolvedSweep {
    solutions: Vec<Vec<Complex64>>,
    fields: Option<Vec<[f64; 2]>>,
}

/// One row per frequency × eval-point complex pressure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SweepOutput {
    /// Execution provenance; absent when reading older output files.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution: Option<SweepExecution>,
    /// Solve frequencies (Hz).
    pub frequencies: Vec<f64>,
    /// Speed of sound (m/s) used for the sweep.
    pub speed_of_sound: f64,
    /// Air density (kg/m³) used for the sweep.
    pub density: f64,
    /// Number of surface DOFs.
    pub num_surface_dofs: usize,
    /// Per-frequency surface pressure (`frequencies.len()` × `num_surface_dofs`),
    /// flattened row-major. Each entry stored as `[re, im]`.
    pub surface_pressure: Vec<[f64; 2]>,
    /// Evaluation-grid coordinates (`n_eval` × 3), flattened row-major.
    pub eval_points: Vec<f64>,
    /// Per-frequency pressure on the evaluation grid (`frequencies.len()` ×
    /// `n_eval`), flattened row-major; empty if there is no evaluation grid.
    pub eval_pressure: Vec<[f64; 2]>,
}

impl SweepOutput {
    /// Write the output as JSON next to the project.
    pub fn write_json(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let s = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        fs::write(path, s)
    }
}

/// Run the BEM frequency sweep described by `project`, optionally restricted
/// to `[freq_start_idx, freq_end_idx)` of the frequency curve. Both indices
/// are inclusive on the lower bound and exclusive on the upper bound; pass
/// `None` to run the full sweep.
pub fn run_project(
    project: &NumCalcProject,
    freq_start_idx: Option<usize>,
    freq_end_idx: Option<usize>,
    max_iterations: usize,
    tolerance: f64,
) -> Result<SweepOutput, BemError> {
    run_project_using(
        project,
        freq_start_idx,
        freq_end_idx,
        max_iterations,
        tolerance,
        SweepExecution::Cpu,
        |elements, incident, config| {
            frequency_sweep(
                elements,
                &project.nodes,
                incident,
                config,
                AssemblyMethod::Tbem,
                project.speed_of_sound,
                project.density,
            )
            .map(|sweep| SolvedSweep {
                solutions: sweep.solutions,
                fields: None,
            })
        },
    )
}

/// Run a NumCalc project using retained native CUDA TBEM and point Jacobi.
///
/// The caller supplies a pool reused throughout the sweep. Prepared boundary
/// data and the CPU producer's frequency/configuration conventions are preserved.
/// Field evaluation consumes device solutions before explicit JSON export.
/// Evaluation grids require off-surface points and velocity/impedance traces.
///
/// # Errors
/// Rejects unsupported geometry, dense systems with 1000 or more DOFs, invalid
/// controls and nonconvergence. Propagates CUDA failures without CPU fallback.
#[cfg(feature = "cuda")]
pub fn run_project_cuda_jacobi(
    project: &NumCalcProject,
    freq_start_idx: Option<usize>,
    freq_end_idx: Option<usize>,
    max_iterations: usize,
    tolerance: f64,
    pool: &sonium_compute::MemoryPool<sonium_compute::CudaBackend>,
) -> Result<SweepOutput, BemError> {
    run_project_using(
        project,
        freq_start_idx,
        freq_end_idx,
        max_iterations,
        tolerance,
        SweepExecution::CudaJacobi,
        |elements, incident, config| {
            use crate::core::bem_solver::pack_incident_sources;
            use crate::core::postprocess::pressure::CudaBemFieldWorkspace;
            use sonium_compute::Backend;
            config.validate()?;
            let physics = project
                .try_physics_at(config.frequencies.first().copied().unwrap_or(1.0))
                .map_err(|e| BemError::InvalidParameters(e.to_string()))?;
            if config.frequencies.is_empty() {
                return Ok(SolvedSweep {
                    solutions: Vec::new(),
                    fields: Some(Vec::new()),
                });
            }
            if elements.len() >= 1000 {
                return Err(BemError::InvalidParameters(
                    "dense CUDA NumCalc requires fewer than 1000 surface DOFs".into(),
                ));
            }
            let points = project.evaluation_centers();
            let failed = |e: String| BemError::SolverFailed(e);
            let mut field = if points.is_empty() {
                None
            } else {
                Some(CudaBemFieldWorkspace::try_new(
                    pool,
                    elements,
                    &project.nodes,
                    &points,
                )?)
            };
            let mut packed = pack_incident_sources(incident);
            let source_count = packed.len() / 6;
            packed.resize(packed.len().max(6), 0.0);
            let mut sources = pool
                .try_alloc::<f64>(packed.len())
                .map_err(|e| failed(e.to_string()))?;
            pool.backend()
                .try_copy_h2d(&packed, sources.raw_mut())
                .map_err(|e| failed(e.to_string()))?;
            let mut solutions = Vec::with_capacity(config.frequencies.len());
            let mut fields = Vec::with_capacity(points.len() * config.frequencies.len());
            crate::core::solver::frequency_sweep::frequency_sweep_cuda_tbem_jacobi_device(
                elements,
                &project.nodes,
                incident,
                config,
                &physics,
                None,
                pool,
                |physics, _, pressure| {
                    if let Some(field) = field.as_mut() {
                        let output = field.try_evaluate(
                            pool.backend(),
                            pressure,
                            physics,
                            sources.raw(),
                            source_count,
                        )?;
                        let mut values = vec![Complex64::default(); points.len()];
                        pool.backend()
                            .try_copy_d2h(output, &mut values)
                            .map_err(|e| failed(e.to_string()))?;
                        fields.extend(values.into_iter().map(|z| [z.re, z.im]));
                    }
                    let mut surface = vec![Complex64::default(); elements.len()];
                    pool.backend()
                        .try_copy_d2h(pressure, &mut surface)
                        .map_err(|e| failed(e.to_string()))?;
                    solutions.push(surface);
                    Ok(())
                },
            )?;
            Ok(SolvedSweep {
                solutions,
                fields: Some(fields),
            })
        },
    )
}

fn run_project_using(
    project: &NumCalcProject,
    freq_start_idx: Option<usize>,
    freq_end_idx: Option<usize>,
    max_iterations: usize,
    tolerance: f64,
    execution: SweepExecution,
    solve: impl FnOnce(
        &[crate::core::types::Element],
        &IncidentField,
        &FrequencySweepConfig,
    ) -> Result<SolvedSweep, BemError>,
) -> Result<SweepOutput, BemError> {
    let total = project.frequencies.len();
    let start = freq_start_idx.unwrap_or(0).min(total);
    let end = freq_end_idx.unwrap_or(total).min(total).max(start);
    let frequencies: Vec<f64> = project.frequencies[start..end].to_vec();

    let surface_elements: Vec<_> = project
        .elements
        .iter()
        .filter(|e| !e.property.is_evaluation())
        .cloned()
        .collect();

    let incident = pick_incident(project);

    let config = FrequencySweepConfig {
        frequencies: frequencies.clone(),
        tolerance,
        max_iterations,
        recycle_solution: true,
        gmres_restart: 30,
        beta_scale: 4.0,
        recycle_subspace_size: 0,
        max_recycle_relative_frequency_delta: 0.1,
    };

    let SolvedSweep { solutions, fields } = solve(&surface_elements, &incident, &config)?;

    let n_dof = solutions.first().map(|s| s.len()).unwrap_or(0);
    let mut surface_flat: Vec<[f64; 2]> = Vec::with_capacity(solutions.len() * n_dof.max(1));
    for sol in &solutions {
        for &c in sol {
            surface_flat.push([c.re, c.im]);
        }
    }

    let eval_points = project.evaluation_centers();
    let eval_flat = flatten_eval_points(&eval_points);
    let eval_pressure = if let Some(fields) = fields {
        fields
    } else if project.evaluation_indices.is_empty() {
        Vec::new()
    } else {
        eval_pressure_per_frequency(
            project,
            &frequencies,
            &incident,
            &surface_elements,
            &solutions,
            &eval_points,
        )
    };

    Ok(SweepOutput {
        execution: Some(execution),
        frequencies,
        speed_of_sound: project.speed_of_sound,
        density: project.density,
        num_surface_dofs: n_dof,
        surface_pressure: surface_flat,
        eval_points: eval_flat,
        eval_pressure,
    })
}

fn flatten_eval_points(pts: &[[f64; 3]]) -> Vec<f64> {
    pts.iter().flat_map(|p| p.iter().copied()).collect()
}

fn eval_pressure_per_frequency(
    project: &NumCalcProject,
    frequencies: &[f64],
    incident: &IncidentField,
    surface_elements: &[crate::core::types::Element],
    solutions: &[Vec<Complex64>],
    eval_points: &[[f64; 3]],
) -> Vec<[f64; 2]> {
    debug_assert_eq!(solutions.len(), frequencies.len());
    let mut out = Vec::with_capacity(solutions.len() * eval_points.len());
    for (i, surface_pressure) in solutions.iter().enumerate() {
        let physics = project.physics_at(frequencies[i]);
        let field = compute_total_field(
            eval_points,
            surface_elements,
            &project.nodes,
            surface_pressure,
            None,
            incident,
            &physics,
            BasisOrder::Constant,
        );
        for fp in field {
            out.push([fp.p_total.re, fp.p_total.im]);
        }
    }
    out
}

fn pick_incident(project: &NumCalcProject) -> IncidentField {
    if let Some(pw) = project.plane_waves.first() {
        IncidentField::PlaneWave {
            direction: pw.direction,
            amplitude: Complex64::new(pw.amplitude_re, pw.amplitude_im),
        }
    } else if let Some(ps) = project.point_sources.first() {
        IncidentField::PointSource {
            position: ps.position,
            strength: Complex64::new(ps.amplitude_re, ps.amplitude_im),
        }
    } else {
        // No external incident field — only velocity BCs drive the system.
        IncidentField::PlaneWave {
            direction: [0.0, 0.0, 1.0],
            amplitude: Complex64::new(0.0, 0.0),
        }
    }
}

// Convenience re-export so downstream callers don't need to know about the
// internal element type when filtering by property.
#[allow(dead_code)]
pub(crate) fn is_surface(p: ElementProperty) -> bool {
    !p.is_evaluation()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numcalc::load_project;
    use std::path::PathBuf;

    fn sample_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/sample_project/NC.inp")
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn native_numcalc_cuda_project_and_fields() {
        use sonium_compute::{CudaBackend, MemoryPool, MemoryPoolConfig};
        let backend = match CudaBackend::new_default() {
            Ok(backend) => backend,
            Err(error) => {
                assert!(
                    std::env::var("SONIUM_REQUIRE_CUDA").as_deref() != Ok("1"),
                    "{error}"
                );
                return;
            }
        };
        let pool = MemoryPool::new(backend, MemoryPoolConfig::default()).unwrap();
        let mut project = load_project(sample_path()).unwrap();
        for center in [[2.0, 0.5, 0.3], [-2.0, 0.3, 0.4]] {
            let mut evaluation = project.elements[0].clone();
            evaluation.property = ElementProperty::Evaluation;
            evaluation.center = center;
            evaluation.dof_addresses.clear();
            project.evaluation_indices.push(project.elements.len());
            project.elements.push(evaluation);
        }
        for (start, end) in [
            (None, None),
            (Some(1), Some(3)),
            (Some(99), Some(200)),
            (Some(2), Some(0)),
        ] {
            let cpu = run_project(&project, start, end, 250, 1e-10).unwrap();
            let gpu = run_project_cuda_jacobi(&project, start, end, 250, 1e-10, &pool).unwrap();
            assert_eq!(gpu.execution, Some(SweepExecution::CudaJacobi));
            assert_eq!(cpu.execution, Some(SweepExecution::Cpu));
            assert_eq!(cpu.frequencies, gpu.frequencies);
            assert_eq!(cpu.eval_points, gpu.eval_points);
            assert_eq!(cpu.num_surface_dofs, gpu.num_surface_dofs);
            for (actual, expected) in [
                (&gpu.surface_pressure, &cpu.surface_pressure),
                (&gpu.eval_pressure, &cpu.eval_pressure),
            ] {
                assert_eq!(actual.len(), expected.len());
                let error = actual
                    .iter()
                    .flatten()
                    .zip(expected.iter().flatten())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>();
                let norm = expected.iter().flatten().map(|x| x * x).sum::<f64>();
                assert!(error.sqrt() <= 1e-7 * norm.sqrt().max(1e-12));
            }
            let mut json = serde_json::to_value(&gpu).unwrap();
            assert_eq!(json["execution"], "cuda_jacobi");
            json["execution"] = serde_json::Value::String("cuda_jacobi_with_cpu_fields".into());
            assert_eq!(
                serde_json::from_value::<SweepOutput>(json.clone())
                    .unwrap()
                    .execution,
                Some(SweepExecution::CudaJacobiWithCpuFields)
            );
            json.as_object_mut().unwrap().remove("execution");
            assert!(
                serde_json::from_value::<SweepOutput>(json)
                    .unwrap()
                    .execution
                    .is_none()
            );
        }
        project.elements[0].boundary_condition =
            crate::core::types::BoundaryCondition::Pressure(vec![Complex64::new(1.0, 0.0)]);
        let before = pool.backend().begin_runtime_stats();
        assert!(run_project_cuda_jacobi(&project, None, None, 250, 1e-10, &pool).is_err());
        assert_eq!(
            pool.backend()
                .runtime_stats_since(before)
                .allocation_operations,
            0
        );
    }

    #[test]
    fn run_project_partial_range_uses_correct_frequencies() {
        let project = load_project(sample_path()).unwrap();
        // Sample has 3 frequencies. Solve only [1, 3) → indices 1 and 2.
        let out = run_project(&project, Some(1), Some(3), 50, 1e-4).unwrap();
        assert_eq!(out.frequencies.len(), 2);
        assert_eq!(out.frequencies, project.frequencies[1..3]);
    }

    #[test]
    fn run_project_clamps_out_of_range_indices() {
        let project = load_project(sample_path()).unwrap();
        // start > total → empty sweep, no panic.
        let out = run_project(&project, Some(99), Some(200), 50, 1e-4).unwrap();
        assert!(out.frequencies.is_empty());
        assert!(out.surface_pressure.is_empty());
    }

    #[test]
    fn run_project_inverted_range_returns_empty() {
        let project = load_project(sample_path()).unwrap();
        // end < start should be clamped, not panic.
        let out = run_project(&project, Some(2), Some(0), 50, 1e-4).unwrap();
        assert!(out.frequencies.is_empty());
    }
}
