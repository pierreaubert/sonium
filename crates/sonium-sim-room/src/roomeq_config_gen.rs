//! Generate sonium-roomeq RoomConfig JSON files from scenarios.
//!
//! Creates the configuration JSON that the `roomeq` optimizer reads.

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use sonium_roomeq::roomeq::{
    CrossoverConfig, MeasurementMultiple, MeasurementRef, MeasurementSource,
    MultiMeasurementConfig, MultiMeasurementStrategy, MultiSubGroup, OptimizerConfig,
    ProcessingMode, RoomConfig, SpeakerConfig, SpeakerGroup, SubwooferStrategy,
    SubwooferSystemConfig, SystemConfig, SystemModel,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::energy_validation::{
    FemBoundaryFluxEvidence, FemOperatorWorkEvidence, FemPhysicalPowerEvidence,
};
use crate::fem_runner::{FemFrequencyStats, FemRunDiagnostics};
use crate::scenarios::Scenario;

/// Serializable solver evidence attached to generated simulation metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FemDiagnosticsMetadata {
    pub backend: String,
    pub preconditioner: String,
    pub shift: String,
    pub modal_damping_ratio: f64,
    /// Explicitly states whether physical-domain energy evidence is present
    /// or unavailable for the selected FEM path.
    pub physical_evidence_status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution: Option<FemExecutionMetadata>,
    pub total_iterations: usize,
    pub total_inner_iterations: usize,
    pub retry_count: usize,
    pub nonconverged_count: usize,
    pub max_residual: f64,
    pub frequency_stats: Vec<FemFrequencyDiagnosticsMetadata>,
}

/// Per-frequency solver evidence retained in [`FemDiagnosticsMetadata`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FemFrequencyDiagnosticsMetadata {
    pub frequency_hz: f64,
    /// CUDA block attempt, distinct from scalar recovery and backend fallback.
    #[serde(default)]
    pub block_solve: crate::fem_runner::FemBlockSolveOutcome,
    pub symmetry_mode: Option<usize>,
    pub matrix_ms: f64,
    pub preconditioner_ms: f64,
    pub rhs_ms: f64,
    pub solve_ms: f64,
    pub iterations: usize,
    pub inner_iterations: usize,
    pub retries: usize,
    pub preconditioner_age: usize,
    pub preconditioner_rebuild_reason: String,
    pub fallback_stage: usize,
    pub max_residual: f64,
    pub converged: bool,
    /// Algebraic FEM work evidence; these values are not watts until the
    /// source normalization contract is independently established.
    pub operator_work: Option<FemOperatorWorkMetadata>,
    /// Cycle-average watts evidence derived from the FEM source contract.
    pub physical_power: Option<FemPhysicalPowerMetadata>,
    /// Per-boundary cycle-average active pressure/velocity flux evidence.
    pub boundary_flux: Option<FemBoundaryFluxMetadata>,
    /// Checked transfer activity for this frequency; absent for host-only
    /// solves and for legacy metadata written before this field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transfer_stats: Option<FemTransferMetadata>,
}

/// Serializable host/device transfer counters from a FEM evidence packet.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FemTransferMetadata {
    pub h2d_operations: usize,
    pub h2d_bytes: usize,
    pub d2h_operations: usize,
    pub d2h_bytes: usize,
    pub d2d_operations: usize,
    pub d2d_bytes: usize,
}

impl From<crate::fem_runner::FemTransferStats> for FemTransferMetadata {
    fn from(stats: crate::fem_runner::FemTransferStats) -> Self {
        Self {
            h2d_operations: stats.h2d_operations,
            h2d_bytes: stats.h2d_bytes,
            d2h_operations: stats.d2h_operations,
            d2h_bytes: stats.d2h_bytes,
            d2d_operations: stats.d2d_operations,
            d2d_bytes: stats.d2d_bytes,
        }
    }
}

/// Serializable backend execution and residency provenance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FemExecutionMetadata {
    pub requested_backend: String,
    pub actual_backend: String,
    pub precision: String,
    pub fallback_status: String,
    pub host_architecture: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_simd_backend: Option<String>,
    pub setup_transfer_stats: FemTransferMetadata,
    pub total_transfer_stats: FemTransferMetadata,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_peak_memory_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_free_memory_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_total_memory_bytes: Option<u64>,
}

impl From<&crate::fem_runner::FemExecutionEvidence> for FemExecutionMetadata {
    fn from(evidence: &crate::fem_runner::FemExecutionEvidence) -> Self {
        Self {
            requested_backend: evidence.requested_backend.clone(),
            actual_backend: evidence.actual_backend.clone(),
            precision: evidence.precision.clone(),
            fallback_status: evidence.fallback_status.clone(),
            host_architecture: evidence.host_architecture.clone(),
            cpu_simd_backend: evidence.cpu_simd_backend.clone(),
            setup_transfer_stats: evidence.setup_transfer_stats.into(),
            total_transfer_stats: evidence.total_transfer_stats.into(),
            estimated_peak_memory_bytes: evidence.estimated_peak_memory_bytes,
            device_free_memory_bytes: evidence.device_free_memory_bytes,
            device_total_memory_bytes: evidence.device_total_memory_bytes,
        }
    }
}

/// Serializable algebraic source/volume/boundary work evidence from one FEM
/// frequency.  The explicit name prevents this metadata from being mistaken
/// for calibrated acoustic power.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FemOperatorWorkMetadata {
    pub source_work: f64,
    pub volume_damping_work: f64,
    pub boundary_work: f64,
    pub balance_residual: f64,
}

impl From<FemOperatorWorkEvidence> for FemOperatorWorkMetadata {
    fn from(evidence: FemOperatorWorkEvidence) -> Self {
        Self {
            source_work: evidence.source_work,
            volume_damping_work: evidence.volume_damping_work,
            boundary_work: evidence.boundary_work,
            balance_residual: evidence.balance_residual,
        }
    }
}

/// Serializable physically normalized source/volume/boundary power evidence
/// from one FEM frequency. Boundary power remains distinct from radiated
/// power because the boundary terms may represent absorptive walls as well as
/// open boundaries.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FemPhysicalPowerMetadata {
    pub source_power_w: f64,
    pub volume_damping_power_w: f64,
    pub boundary_power_w: f64,
    pub balance_residual_w: f64,
}

impl From<FemPhysicalPowerEvidence> for FemPhysicalPowerMetadata {
    fn from(evidence: FemPhysicalPowerEvidence) -> Self {
        Self {
            source_power_w: evidence.source_power_w,
            volume_damping_power_w: evidence.volume_damping_power_w,
            boundary_power_w: evidence.boundary_power_w,
            balance_residual_w: evidence.balance_residual_w,
        }
    }
}

/// Serializable per-boundary active flux evidence from one FEM frequency.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FemBoundaryFluxMetadata {
    pub samples: Vec<FemBoundaryFluxSampleMetadata>,
    pub total_power_w: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FemBoundaryFluxSampleMetadata {
    pub boundary_tag: usize,
    pub active_power_w: f64,
}

impl From<FemBoundaryFluxEvidence> for FemBoundaryFluxMetadata {
    fn from(evidence: FemBoundaryFluxEvidence) -> Self {
        Self {
            samples: evidence
                .samples
                .into_iter()
                .map(|sample| FemBoundaryFluxSampleMetadata {
                    boundary_tag: sample.boundary_tag,
                    active_power_w: sample.active_power_w,
                })
                .collect(),
            total_power_w: evidence.total_power_w,
        }
    }
}

/// Metadata describing how the synthetic measurements were generated. This is
/// intentionally kept outside RoomEQ's optimizer config so consumers that only
/// need `config.json` do not inherit simulation-specific schema churn.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SimulationMetadata {
    pub version: String,
    pub generator: String,
    pub scenario: String,
    pub solver_backend: String,
    pub modal_damping_ratio: f64,
    pub modal_damping_enabled: bool,
    pub source_directivity_convention: String,
    pub atmospheric_absorption_model: String,
    pub frequency_band_model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dense_blr_max_freq_hz: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dense_blr_overlap_hz: Option<f64>,
    pub high_frequency_extension: String,
    pub coherence_model: String,
    pub noise_floor_model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fem_diagnostics: Option<FemDiagnosticsMetadata>,
}

pub fn generate_simulation_metadata(
    scenario: &Scenario,
    modal_damping_ratio: f64,
    dense_blr_max_freq_hz: Option<f64>,
) -> SimulationMetadata {
    generate_simulation_metadata_for_solver(
        scenario,
        modal_damping_ratio,
        dense_blr_max_freq_hz,
        "fem",
    )
}

pub fn generate_simulation_metadata_for_solver(
    scenario: &Scenario,
    modal_damping_ratio: f64,
    dense_blr_max_freq_hz: Option<f64>,
    solver_backend: &str,
) -> SimulationMetadata {
    generate_simulation_metadata_for_solver_with_overlap(
        scenario,
        modal_damping_ratio,
        dense_blr_max_freq_hz,
        None,
        solver_backend,
    )
}

pub fn generate_simulation_metadata_for_solver_with_overlap(
    scenario: &Scenario,
    modal_damping_ratio: f64,
    dense_blr_max_freq_hz: Option<f64>,
    dense_blr_overlap_hz: Option<f64>,
    solver_backend: &str,
) -> SimulationMetadata {
    SimulationMetadata {
        version: "1.1.0".to_string(),
        generator: "sonium-sim-room".to_string(),
        scenario: scenario.name.clone(),
        solver_backend: solver_backend.to_string(),
        modal_damping_ratio,
        modal_damping_enabled: modal_damping_ratio > 0.0,
        source_directivity_convention:
            "receiver_only_low_band: isotropic room excitation, receiver-side directivity"
                .to_string(),
        atmospheric_absorption_model: "ISO 9613-1 fixed 20C/50%RH/101325Pa".to_string(),
        frequency_band_model: if solver_backend == "fast-hybrid" {
            "hybrid: analytical shoebox modes blend into coherent image sources; high band is synthetic extension"
                .to_string()
        } else {
            "hybrid: low band is numerical room solve; high band is synthetic extension".to_string()
        },
        dense_blr_max_freq_hz,
        dense_blr_overlap_hz,
        high_frequency_extension:
            "enabled: deterministic synthetic speaker/room tail anchored at simulated cutoff to 20 kHz"
                .to_string(),
        coherence_model:
            "deterministic noiseless simulation: coherence=1.0 for every frequency bin"
                .to_string(),
        noise_floor_model:
            "synthetic finite sentinel: noise_floor_db=-120; not a measured ambient-noise estimate"
                .to_string(),
        fem_diagnostics: None,
    }
}

/// Attach converged FEM solver evidence to generated metadata.
///
/// Diagnostics are published only after the same convergence contract used by
/// the public FEM runner has passed. This prevents metadata from presenting a
/// partially solved pressure field as valid simulation evidence.
pub fn attach_fem_diagnostics(
    metadata: &mut SimulationMetadata,
    diagnostics: &FemRunDiagnostics,
) -> Result<()> {
    diagnostics
        .require_converged()
        .map_err(anyhow::Error::msg)?;
    metadata.fem_diagnostics = Some(FemDiagnosticsMetadata::from(diagnostics));
    Ok(())
}

impl From<&FemRunDiagnostics> for FemDiagnosticsMetadata {
    fn from(diagnostics: &FemRunDiagnostics) -> Self {
        Self {
            backend: diagnostics.backend.clone(),
            preconditioner: diagnostics.preconditioner.clone(),
            shift: diagnostics.shift.label(),
            modal_damping_ratio: diagnostics.modal_damping_ratio,
            physical_evidence_status: diagnostics.physical_evidence_status.label().to_string(),
            execution: diagnostics.execution.as_ref().map(Into::into),
            total_iterations: diagnostics.total_iterations,
            total_inner_iterations: diagnostics.total_inner_iterations,
            retry_count: diagnostics.retry_count,
            nonconverged_count: diagnostics.nonconverged_count,
            max_residual: diagnostics.max_residual,
            frequency_stats: diagnostics
                .frequency_stats
                .iter()
                .map(FemFrequencyDiagnosticsMetadata::from)
                .collect(),
        }
    }
}

impl From<&FemFrequencyStats> for FemFrequencyDiagnosticsMetadata {
    fn from(stats: &FemFrequencyStats) -> Self {
        Self {
            frequency_hz: stats.frequency_hz,
            block_solve: stats.block_solve.clone(),
            symmetry_mode: stats.symmetry_mode,
            matrix_ms: stats.matrix_ms,
            preconditioner_ms: stats.preconditioner_ms,
            rhs_ms: stats.rhs_ms,
            solve_ms: stats.solve_ms,
            iterations: stats.iterations,
            inner_iterations: stats.inner_iterations,
            retries: stats.retries,
            preconditioner_age: stats.preconditioner_age,
            preconditioner_rebuild_reason: stats.preconditioner_rebuild_reason.clone(),
            fallback_stage: stats.fallback_stage,
            max_residual: stats.max_residual,
            converged: stats.converged,
            operator_work: stats.operator_work.map(Into::into),
            physical_power: stats.physical_power.map(Into::into),
            boundary_flux: stats
                .boundary_flux
                .as_ref()
                .map(|evidence| evidence.clone().into()),
            transfer_stats: stats.transfer_stats.map(Into::into),
        }
    }
}

pub fn write_simulation_metadata(metadata: &SimulationMetadata, output_path: &Path) -> Result<()> {
    let json = serde_json::to_string_pretty(metadata)?;
    std::fs::write(output_path, json)?;
    Ok(())
}

/// Create a plural measurement source with one CSV per listening position.
/// The plural wire contract is also used for a single position so generated
/// configs remain compatible with the production RoomEQ schema.
fn make_csv_source_for_lps(name: &str, num_lps: usize) -> MeasurementSource {
    let measurements: Vec<MeasurementRef> = (0..num_lps.max(1))
        .map(|i| MeasurementRef::Path(PathBuf::from(format!("{name}_lp{i}.csv"))))
        .collect();
    MeasurementSource::Multiple(MeasurementMultiple {
        measurements,
        speaker_name: None,
    })
}

fn make_crossovers() -> HashMap<String, CrossoverConfig> {
    let mut xovers = HashMap::new();
    xovers.insert(
        "first".to_string(),
        CrossoverConfig {
            crossover_type: "LR24".to_string(),
            frequency: Some(80.0),
            frequencies: None,
            frequency_range: None,
        },
    );
    xovers
}

/// Map source name to logical speaker role
fn role_for_name(name: &str) -> &str {
    match name {
        "left" => "L",
        "right" => "R",
        "center" => "C",
        "surround_left" => "SL",
        "surround_right" => "SR",
        "top_front_left" => "TFL",
        "top_front_right" => "TFR",
        "top_rear_left" => "TRL",
        "top_rear_right" => "TRR",
        _ => name,
    }
}

/// Generate a roomeq JSON config for a scenario.
///
/// The config references CSV files by filename only (config.json sits alongside CSVs).
/// `_csv_dir` is accepted for API compatibility but paths are always relative filenames.
///
/// When a scenario has multiple listening positions, each speaker source gets a
/// `MeasurementSource::Multiple` referencing all LP CSVs, and the optimizer is
/// configured with `multi_measurement` strategy `minimax` (optimize worst seat).
pub fn generate_config(scenario: &Scenario, _csv_dir: &Path) -> Result<RoomConfig> {
    let source_names = &scenario.source_names;
    let num_lps = scenario.simulation.listening_positions.len();
    let max_freq = scenario
        .simulation
        .frequencies
        .last()
        .copied()
        .unwrap_or(2000.0);

    // 1. Detect group pattern: sources ending with "_sub" that have matching mains
    let mut group_pairs: Vec<(&str, &str)> = Vec::new(); // (main_name, sub_name)
    let mut group_sub_names: Vec<&str> = Vec::new();

    for name in source_names {
        if let Some(main_prefix) = name.strip_suffix("_sub")
            && source_names.iter().any(|n| n == main_prefix)
        {
            group_pairs.push((main_prefix, name.as_str()));
            group_sub_names.push(name.as_str());
        }
    }
    let is_group = !group_pairs.is_empty();

    // 2. Classify remaining sources into main vs sub
    let mut main_sources: Vec<&str> = Vec::new();
    let mut sub_sources: Vec<&str> = Vec::new();

    for name in source_names {
        if group_sub_names.contains(&name.as_str()) {
            continue; // handled by group logic
        }
        if name.starts_with("sub") {
            sub_sources.push(name);
        } else {
            main_sources.push(name);
        }
    }

    let mut speakers: HashMap<String, SpeakerConfig> = HashMap::new();
    let mut system_speakers: HashMap<String, String> = HashMap::new();

    if is_group {
        // Group mode: each main+sub pair becomes a SpeakerConfig::Group
        for (main_name, sub_name) in &group_pairs {
            let group = SpeakerGroup {
                name: format!("{main_name}_group"),
                speaker_name: None,
                measurements: vec![
                    make_csv_source_for_lps(main_name, num_lps),
                    make_csv_source_for_lps(sub_name, num_lps),
                ],
                crossover: Some("first".to_string()),
            };
            speakers.insert(main_name.to_string(), SpeakerConfig::Group(group));
            let role = role_for_name(main_name);
            system_speakers.insert(role.to_string(), main_name.to_string());
        }

        // Add any non-grouped main speakers (shouldn't happen for 2.2 group, but be robust)
        for &name in &main_sources {
            if group_pairs.iter().any(|(m, _)| *m == name) {
                continue; // already handled
            }
            speakers.insert(
                name.to_string(),
                SpeakerConfig::Single(make_csv_source_for_lps(name, num_lps)),
            );
            let role = role_for_name(name);
            system_speakers.insert(role.to_string(), name.to_string());
        }

        // Group mode: Stereo model, crossovers defined, no subwoofers section
        let model = SystemModel::Stereo;
        let crossovers = Some(make_crossovers());

        let optimizer = make_optimizer(num_lps, max_freq);
        let config = RoomConfig {
            version: "1.2.0".to_string(),
            system: Some(SystemConfig {
                model,
                speakers: system_speakers,
                subwoofers: None,
                bass_management: None,
            }),
            speakers,
            crossovers,
            target_curve: None,
            optimizer,
            recording_config: None,
            cea2034_cache: None,
            ctc: None,
        };
        return Ok(config);
    }

    // Non-group path: add main speakers
    for &name in &main_sources {
        speakers.insert(
            name.to_string(),
            SpeakerConfig::Single(make_csv_source_for_lps(name, num_lps)),
        );
        let role = role_for_name(name);
        system_speakers.insert(role.to_string(), name.to_string());
    }

    // Configure System Model
    let model = if system_speakers.contains_key("C") || system_speakers.contains_key("SL") {
        SystemModel::HomeCinema
    } else {
        SystemModel::Stereo
    };

    // Configure Subwoofers & Crossovers.
    //
    // The previous version detected `sub_bottom + sub_top` and emitted a
    // `CardioidConfig`. That scenario is a *vertically stacked* sub pair —
    // not a cardioid (which needs horizontal front+rear units with delay /
    // inversion). Treat stacked subs the same way as any other multi-sub
    // group and let MSO handle them.
    let (subwoofers_config, crossovers) = if sub_sources.is_empty() {
        (None, None)
    } else {
        // Standard Single/Multi Sub
        let sub_key = "lfe".to_string();
        system_speakers.insert("LFE".to_string(), sub_key.clone());

        let (speaker_config, strategy) = if sub_sources.len() == 1 {
            let name = sub_sources[0];
            (
                SpeakerConfig::Single(make_csv_source_for_lps(name, num_lps)),
                SubwooferStrategy::Single,
            )
        } else {
            let sub_sources_objs: Vec<MeasurementSource> = sub_sources
                .iter()
                .map(|name| make_csv_source_for_lps(name, num_lps))
                .collect();
            (
                SpeakerConfig::MultiSub(MultiSubGroup {
                    name: "subs".to_string(),
                    speaker_name: None,
                    subwoofers: sub_sources_objs,
                    allpass_optimization: false,
                }),
                SubwooferStrategy::Mso,
            )
        };

        speakers.insert(sub_key.clone(), speaker_config);

        let mut mapping = HashMap::new();
        mapping.insert(sub_key, "L".to_string());

        (
            Some(SubwooferSystemConfig {
                config: strategy,
                crossover: Some("first".to_string()),
                mapping,
            }),
            Some(make_crossovers()),
        )
    };

    let optimizer = make_optimizer(num_lps, max_freq);

    let config = RoomConfig {
        version: "1.2.0".to_string(),
        system: Some(SystemConfig {
            model,
            speakers: system_speakers,
            subwoofers: subwoofers_config,
            bass_management: None,
        }),
        speakers,
        crossovers,
        target_curve: None,
        optimizer,
        recording_config: None,
        cea2034_cache: None,
        ctc: None,
    };

    Ok(config)
}

fn make_optimizer(num_lps: usize, max_freq: f64) -> OptimizerConfig {
    let multi_measurement = if num_lps > 1 {
        Some(MultiMeasurementConfig {
            strategy: MultiMeasurementStrategy::Minimax,
            weights: None,
            variance_lambda: 1.0,
            spatial_robustness: None,
            bootstrap_uncertainty: None,
        })
    } else {
        None
    };

    OptimizerConfig {
        algorithm: "autoeq:de".to_string(),
        num_filters: 7,
        max_freq,
        loss_type: "flat".to_string(),
        seed: Some(42),
        max_iter: 5000,
        refine: true,
        asymmetric_loss: true,
        processing_mode: ProcessingMode::LowLatency,
        multi_measurement,
        ..OptimizerConfig::default()
    }
}

/// Write a roomeq config JSON file
pub fn write_config(config: &RoomConfig, output_path: &Path) -> Result<()> {
    config
        .validate()
        .map_err(|error| anyhow!("invalid RoomEQ configuration: {error}"))?;
    let json = serde_json::to_string_pretty(config)?;
    std::fs::write(output_path, json)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fem_runner::{FemExecutionEvidence, FemTransferStats};
    use crate::scenarios;

    fn assert_plural_measurement_contract(value: &serde_json::Value, path: &str) {
        match value {
            serde_json::Value::Object(object) => {
                assert!(
                    !object.contains_key("measurement"),
                    "generated config contains legacy singular measurement at {path}"
                );
                for (key, child) in object {
                    assert_plural_measurement_contract(child, &format!("{path}.{key}"));
                }
            }
            serde_json::Value::Array(array) => {
                for (index, child) in array.iter().enumerate() {
                    assert_plural_measurement_contract(child, &format!("{path}[{index}]"));
                }
            }
            _ => {}
        }
    }

    #[test]
    fn single_position_uses_plural_measurements_contract() {
        let source = make_csv_source_for_lps("left", 1);
        let json = serde_json::to_value(source).unwrap();
        assert_eq!(json["measurements"][0], "left_lp0.csv");
        assert!(json.get("measurement").is_none());
    }

    #[test]
    fn every_scenario_uses_plural_measurements_including_nested_subwoofers() {
        for scenario in scenarios::all_scenarios() {
            let config = generate_config(&scenario, Path::new(".")).unwrap();
            let json = serde_json::to_value(config).unwrap();
            assert_plural_measurement_contract(&json, &scenario.name);
        }
    }

    #[test]
    fn simulation_metadata_records_modal_damping_and_dense_band() {
        let scenario = scenarios::scenario_by_name("small_stereo_2_0").unwrap();
        let metadata = generate_simulation_metadata(&scenario, 0.0125, Some(250.0));

        assert_eq!(metadata.scenario, "small_stereo_2_0");
        assert_eq!(metadata.version, "1.1.0");
        assert_eq!(metadata.modal_damping_ratio, 0.0125);
        assert!(metadata.modal_damping_enabled);
        assert!(
            metadata
                .source_directivity_convention
                .contains("receiver_only_low_band")
        );
        assert!(metadata.frequency_band_model.contains("hybrid"));
        assert!(metadata.high_frequency_extension.contains("synthetic"));
        assert!(metadata.coherence_model.contains("coherence=1.0"));
        assert!(metadata.noise_floor_model.contains("-120"));
        assert_eq!(metadata.dense_blr_max_freq_hz, Some(250.0));

        let json = serde_json::to_string_pretty(&metadata).unwrap();
        assert!(json.contains("\"modal_damping_ratio\": 0.0125"));
        assert!(json.contains("\"frequency_band_model\""));
        assert!(json.contains("high band is synthetic extension"));
        let roundtrip: SimulationMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip, metadata);
    }

    #[test]
    fn simulation_metadata_records_dense_overlap() {
        let scenario = scenarios::scenario_by_name("small_stereo_2_0").unwrap();
        let metadata = generate_simulation_metadata_for_solver_with_overlap(
            &scenario,
            0.01,
            Some(250.0),
            Some(60.0),
            "fast-hybrid",
        );

        assert_eq!(metadata.dense_blr_max_freq_hz, Some(250.0));
        assert_eq!(metadata.dense_blr_overlap_hz, Some(60.0));
        let json = serde_json::to_string(&metadata).unwrap();
        assert!(json.contains("dense_blr_overlap_hz"));
    }

    fn converged_fem_diagnostics() -> FemRunDiagnostics {
        FemRunDiagnostics {
            backend: "cpu".to_string(),
            preconditioner: "ilut".to_string(),
            shift: crate::fem_runner::ShiftedLaplacianShift::K2 {
                alpha: 0.4,
                beta: 0.6,
            },
            modal_damping_ratio: 0.01,
            total_iterations: 7,
            total_inner_iterations: 2,
            retry_count: 1,
            nonconverged_count: 0,
            listener_interpolation_fallbacks: 0,
            max_residual: 1.0e-7,
            physical_evidence_status: crate::fem_runner::FemPhysicalEvidenceStatus::Complete,
            execution: Some(FemExecutionEvidence {
                requested_backend: "cpu".to_string(),
                actual_backend: "cpu".to_string(),
                precision: "f64".to_string(),
                fallback_status: "none".to_string(),
                host_architecture: "x86_64".to_string(),
                cpu_simd_backend: Some("avx2".to_string()),
                setup_transfer_stats: FemTransferStats::default(),
                total_transfer_stats: FemTransferStats::default(),
                estimated_peak_memory_bytes: None,
                device_free_memory_bytes: None,
                device_total_memory_bytes: None,
            }),
            frequency_stats: vec![FemFrequencyStats {
                block_solve: Default::default(),
                frequency_hz: 50.0,
                symmetry_mode: Some(1),
                matrix_ms: 1.0,
                preconditioner_ms: 2.0,
                rhs_ms: 3.0,
                solve_ms: 4.0,
                iterations: 7,
                inner_iterations: 2,
                retries: 1,
                preconditioner_age: 3,
                preconditioner_rebuild_reason: "periodic".to_string(),
                fallback_stage: 2,
                max_residual: 1.0e-7,
                converged: true,
                operator_work: Some(FemOperatorWorkEvidence {
                    source_work: 2.0,
                    volume_damping_work: 0.75,
                    boundary_work: 1.25,
                    balance_residual: 0.0,
                }),
                physical_power: Some(FemPhysicalPowerEvidence {
                    source_power_w: 0.5,
                    volume_damping_power_w: 0.1875,
                    boundary_power_w: 0.3125,
                    balance_residual_w: 0.0,
                }),
                boundary_flux: Some(FemBoundaryFluxEvidence {
                    samples: vec![crate::energy_validation::FemBoundaryFluxSample {
                        boundary_tag: 7,
                        active_power_w: 0.3125,
                    }],
                    total_power_w: 0.3125,
                }),
                transfer_stats: None,
                allocation_operations: 0,
                allocation_high_water_bytes: 0,
                synchronization_calls: 0,
            }],
        }
    }

    #[test]
    fn simulation_metadata_roundtrips_fem_diagnostics() {
        let scenario = scenarios::scenario_by_name("small_stereo_2_0").unwrap();
        let mut metadata = generate_simulation_metadata(&scenario, 0.01, None);
        let diagnostics = converged_fem_diagnostics();

        attach_fem_diagnostics(&mut metadata, &diagnostics).unwrap();
        let json = serde_json::to_string_pretty(&metadata).unwrap();
        assert!(json.contains("\"fem_diagnostics\""));
        assert!(json.contains("\"operator_work\""));
        assert!(json.contains("\"physical_power\""));
        assert!(json.contains("\"boundary_flux\""));
        assert!(json.contains("\"requested_backend\": \"cpu\""));
        assert!(json.contains("\"cpu_simd_backend\": \"avx2\""));
        assert!(json.contains("\"fallback_stage\": 2"));
        assert!(json.contains("\"preconditioner_rebuild_reason\": \"periodic\""));

        let roundtrip: SimulationMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip, metadata);
        let evidence = metadata.fem_diagnostics.as_ref().unwrap();
        assert_eq!(evidence.shift, "k2(alpha=0.400,beta=0.600)");
        assert_eq!(evidence.physical_evidence_status, "complete");
        assert_eq!(evidence.frequency_stats[0].iterations, 7);
        let operator_work = evidence.frequency_stats[0].operator_work.as_ref().unwrap();
        assert_eq!(operator_work.source_work, 2.0);
        let physical_power = evidence.frequency_stats[0].physical_power.as_ref().unwrap();
        assert_eq!(physical_power.source_power_w, 0.5);
        let boundary_flux = evidence.frequency_stats[0].boundary_flux.as_ref().unwrap();
        assert_eq!(boundary_flux.samples[0].boundary_tag, 7);
        assert_eq!(boundary_flux.total_power_w, 0.3125);
    }

    #[test]
    fn block_outcome_metadata_preserves_recovery_and_legacy_records() {
        use crate::fem_runner::FemBlockSolveOutcome;
        let mut diagnostics = converged_fem_diagnostics();
        for outcome in [
            FemBlockSolveOutcome::Accepted { correction_steps: 1 },
            FemBlockSolveOutcome::Rejected { reason: "corrections exhausted".into() },
        ] {
            diagnostics.frequency_stats[0].block_solve = outcome.clone();
            let metadata = FemFrequencyDiagnosticsMetadata::from(&diagnostics.frequency_stats[0]);
            let mut value = serde_json::to_value(&metadata).unwrap();
            let decoded: FemFrequencyDiagnosticsMetadata = serde_json::from_value(value.clone()).unwrap();
            assert_eq!(decoded.block_solve, outcome);
            value.as_object_mut().unwrap().remove("block_solve");
            let legacy: FemFrequencyDiagnosticsMetadata = serde_json::from_value(value).unwrap();
            assert_eq!(legacy.block_solve, FemBlockSolveOutcome::NotAttempted);
        }
    }

    #[test]
    fn simulation_metadata_labels_unavailable_physical_evidence() {
        let scenario = scenarios::scenario_by_name("small_stereo_2_0").unwrap();
        let mut metadata = generate_simulation_metadata(&scenario, 0.01, None);
        let mut diagnostics = converged_fem_diagnostics();
        diagnostics.physical_evidence_status =
            crate::fem_runner::FemPhysicalEvidenceStatus::UnavailableReducedDomain;
        for stats in &mut diagnostics.frequency_stats {
            stats.operator_work = None;
            stats.physical_power = None;
            stats.boundary_flux = None;
        }

        attach_fem_diagnostics(&mut metadata, &diagnostics).unwrap();
        assert_eq!(
            metadata
                .fem_diagnostics
                .as_ref()
                .unwrap()
                .physical_evidence_status,
            "unavailable-reduced-domain"
        );
    }

    #[test]
    fn simulation_metadata_preserves_device_diagnostics_provenance() {
        let scenario = scenarios::scenario_by_name("small_stereo_2_0").unwrap();
        let mut metadata = generate_simulation_metadata(&scenario, 0.01, None);
        let mut diagnostics = converged_fem_diagnostics();
        diagnostics.backend = "metal".to_string();
        diagnostics.physical_evidence_status =
            crate::fem_runner::FemPhysicalEvidenceStatus::UnavailableDeviceBackend;
        for stats in &mut diagnostics.frequency_stats {
            stats.operator_work = None;
            stats.physical_power = None;
            stats.boundary_flux = None;
        }

        attach_fem_diagnostics(&mut metadata, &diagnostics).unwrap();
        let json = serde_json::to_string_pretty(&metadata).unwrap();
        assert!(json.contains("\"backend\": \"metal\""));
        assert!(json.contains("\"physical_evidence_status\": \"unavailable-device-backend\""));

        let evidence = metadata.fem_diagnostics.as_ref().unwrap();
        assert_eq!(evidence.backend, "metal");
        assert_eq!(
            evidence.physical_evidence_status,
            "unavailable-device-backend"
        );
        assert_eq!(evidence.frequency_stats.len(), 1);
        assert_eq!(evidence.frequency_stats[0].frequency_hz, 50.0);
        assert_eq!(evidence.frequency_stats[0].iterations, 7);
        assert_eq!(evidence.frequency_stats[0].preconditioner_age, 3);
        assert_eq!(
            evidence.frequency_stats[0].preconditioner_rebuild_reason,
            "periodic"
        );
        assert_eq!(evidence.frequency_stats[0].fallback_stage, 2);
        assert!(evidence.frequency_stats[0].converged);
        assert!(evidence.frequency_stats[0].operator_work.is_none());
        assert!(evidence.frequency_stats[0].physical_power.is_none());
        assert!(evidence.frequency_stats[0].boundary_flux.is_none());

        let roundtrip: SimulationMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip, metadata);
    }

    #[test]
    fn simulation_metadata_rejects_nonconverged_fem_diagnostics() {
        let scenario = scenarios::scenario_by_name("small_stereo_2_0").unwrap();
        let mut metadata = generate_simulation_metadata(&scenario, 0.01, None);
        let mut diagnostics = converged_fem_diagnostics();
        diagnostics.frequency_stats[0].converged = false;
        diagnostics.nonconverged_count = 1;

        let error = attach_fem_diagnostics(&mut metadata, &diagnostics).unwrap_err();
        assert!(error.to_string().contains("nonconverged"));
        assert!(metadata.fem_diagnostics.is_none());
    }
}
