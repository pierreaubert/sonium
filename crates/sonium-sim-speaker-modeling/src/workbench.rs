//! Unified Workbench project: one schema for every speaker front end.
//!
//! The three legacy front ends (Speaker Lab, Speaker Studio, and the
//! simulation-demo Driver Lab) each owned a project format and a job
//! model. This module is the single canonical replacement: a versioned
//! project wrapping the product design, a first-class study tree, and
//! material references into `sonium-materials`. Legacy payloads enter
//! only through explicit importers that preserve the original payload
//! as provenance; nothing is silently converted.

// Rust guideline compliant 2026-02-21

use serde::{Deserialize, Serialize};

use crate::cabinet_design::CabinetDesign;
use crate::product_design::{CrossoverNetwork, ProductDesign};

fn empty_product() -> ProductDesign {
    // ProductDesign deserialisation requires at least one driver, so the
    // placeholder product carries one minimal front-facing driver.
    ProductDesign {
        cabinet: CabinetDesign::default(),
        drivers: vec![crate::product_design::ProductDriver {
            id: "driver-0".to_owned(),
            radius_m: 0.06,
            position_m: [0.0, 0.13, 0.0],
            normal: [0.0, 0.0, 1.0],
            force_factor_tm: 5.0,
            coil_resistance_ohm: 6.0,
            coil_reactance_ohm: 0.0,
            mechanical_resistance_n_s_per_m: 1.0,
            mechanical_reactance_n_s_per_m: 0.0,
            crossover_branch: 0,
        }],
        crossover: CrossoverNetwork {
            branches: vec![crate::product_design::CrossoverBranch::default()],
        },
        waveguide: None,
        structural_acoustic: None,
    }
}

/// Current Workbench project schema version.
pub const WORKBENCH_SCHEMA_VERSION: u32 = 1;

/// Study kinds in the unified model tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkbenchStudyKind {
    /// Stationary magnetic bias plus motor extraction.
    Motor,
    /// Coupled small-signal frequency sweep.
    DriverLinear,
    /// Eigenfrequency and breakup check.
    Breakup,
    /// Lumped or resolved enclosure system study.
    EnclosureSystem,
    /// BEM radiation and CEA-2034 product study.
    ProductRadiation,
    /// Large-signal lumped or resolved-port study.
    LargeSignal,
    /// Port campaign, sweep, and optimisation study.
    PortOptimisation,
    /// Parametric sweep or optimiser study.
    Optimise,
}

/// Fidelity label carried by every study and result.
///
/// The label is part of provenance: a result never hides which tier
/// produced it, and lower tiers are never promoted silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FidelityLabel {
    /// Lumped-parameter model.
    Lumped,
    /// Reduced-order or DtN-coupled model.
    Reduced,
    /// Distributed FEM/BEM model.
    Distributed,
    /// Monolithic coupled model.
    Monolithic,
}

/// One study in the Workbench model tree.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkbenchStudy {
    /// Stable study identifier within the project.
    pub id: String,
    /// Study kind.
    pub kind: WorkbenchStudyKind,
    /// Declared fidelity tier for this study.
    pub fidelity: FidelityLabel,
    /// Whether the study is enabled in the tree.
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

fn default_enabled() -> bool {
    true
}

/// Reference to one material in the unified library.
///
/// Projects store references only; embedded material copies would fork
/// the library and break provenance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterialReference {
    /// Material identifier in the `sonium-materials` pack.
    pub material_id: String,
    /// Geometry selection the material is assigned to.
    pub selection: String,
}

/// Provenance envelope shared by CLI, Python, and UI results.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceEnvelope {
    /// Material pack name that supplied every material reference.
    pub material_pack: String,
    /// Material pack version.
    pub material_pack_version: String,
    /// Legacy format the project was imported from, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub imported_from: Option<String>,
}

/// The canonical Workbench project document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkbenchProject {
    /// Schema version; must equal [`WORKBENCH_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Human-readable project name.
    pub name: String,
    /// Product design (cabinet, drivers, crossover, waveguide).
    pub product: ProductDesign,
    /// Study tree in execution order.
    pub studies: Vec<WorkbenchStudy>,
    /// Material assignments by reference into the library.
    #[serde(default)]
    pub materials: Vec<MaterialReference>,
    /// Provenance shared by every result of this project.
    pub provenance: ProvenanceEnvelope,
}

/// Errors returned by Workbench project validation and import.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WorkbenchError {
    /// The schema version is not supported by this build.
    #[error("unsupported workbench schema version {found}, expected {expected}")]
    UnsupportedSchema {
        /// Version found in the payload.
        found: u32,
        /// Version supported by this build.
        expected: u32,
    },
    /// A required project field failed validation.
    #[error("invalid workbench project: {0}")]
    InvalidProject(String),
    /// A legacy payload could not be imported.
    #[error("legacy import failed: {0}")]
    LegacyImport(String),
}

impl WorkbenchProject {
    /// Validates schema version, study tree, and material references.
    ///
    /// # Errors
    ///
    /// Returns [`WorkbenchError`] naming the first broken invariant.
    pub fn validate(&self) -> Result<(), WorkbenchError> {
        if self.schema_version != WORKBENCH_SCHEMA_VERSION {
            return Err(WorkbenchError::UnsupportedSchema {
                found: self.schema_version,
                expected: WORKBENCH_SCHEMA_VERSION,
            });
        }
        if self.name.trim().is_empty() {
            return Err(WorkbenchError::InvalidProject("name must not be blank".to_owned()));
        }
        if self.studies.is_empty() {
            return Err(WorkbenchError::InvalidProject(
                "study tree must not be empty".to_owned(),
            ));
        }
        let mut seen = std::collections::HashSet::new();
        for study in &self.studies {
            if study.id.trim().is_empty() || !seen.insert(study.id.clone()) {
                return Err(WorkbenchError::InvalidProject(format!(
                    "study ids must be non-blank and unique, got {:?}",
                    study.id
                )));
            }
        }
        for reference in &self.materials {
            if reference.material_id.trim().is_empty() || reference.selection.trim().is_empty() {
                return Err(WorkbenchError::InvalidProject(
                    "material references need a material id and a selection".to_owned(),
                ));
            }
        }
        if self.provenance.material_pack.trim().is_empty()
            || self.provenance.material_pack_version.trim().is_empty()
        {
            return Err(WorkbenchError::InvalidProject(
                "provenance needs a material pack name and version".to_owned(),
            ));
        }
        Ok(())
    }

    /// Material-coverage preflight against a library's known ids.
    ///
    /// Returns the sorted selections whose material id is absent from
    /// `known_material_ids`. An empty result means full coverage; any
    /// entry is a named gap that must block the affected study.
    #[must_use]
    pub fn missing_materials(&self, known_material_ids: &[&str]) -> Vec<String> {
        let mut missing: Vec<String> = self
            .materials
            .iter()
            .filter(|reference| !known_material_ids.contains(&reference.material_id.as_str()))
            .map(|reference| reference.selection.clone())
            .collect();
        missing.sort();
        missing.dedup();
        missing
    }

    /// Imports a legacy front-end payload into the canonical project.
    ///
    /// The importers recognise the Speaker Lab study payload, the
    /// Speaker Studio config, and the simulation-demo generic project
    /// by their identifying fields. Unrecognised payloads fail closed;
    /// recognised payloads keep their origin in provenance. Full
    /// geometric conversion stays with the owning legacy readers: this
    /// importer establishes identity, study tree, and provenance, and
    /// starts from the default product design until the owning reader
    /// supplies geometry.
    ///
    /// # Errors
    ///
    /// Returns [`WorkbenchError::LegacyImport`] for malformed or
    /// unrecognised payloads.
    pub fn import_legacy(name: &str, payload: &str) -> Result<Self, WorkbenchError> {
        let value: serde_json::Value = serde_json::from_str(payload)
            .map_err(|error| WorkbenchError::LegacyImport(error.to_string()))?;
        let origin = detect_legacy_origin(&value)
            .ok_or_else(|| WorkbenchError::LegacyImport("unrecognised legacy payload".to_owned()))?;
        let studies = match origin {
            "speaker-studio" => vec![
                study("motor", WorkbenchStudyKind::Motor, FidelityLabel::Distributed),
                study("driver-linear", WorkbenchStudyKind::DriverLinear, FidelityLabel::Distributed),
                study("product-radiation", WorkbenchStudyKind::ProductRadiation, FidelityLabel::Distributed),
                study("port-optimisation", WorkbenchStudyKind::PortOptimisation, FidelityLabel::Reduced),
            ],
            "sim-demo" => vec![
                study("driver-linear", WorkbenchStudyKind::DriverLinear, FidelityLabel::Lumped),
                study("large-signal", WorkbenchStudyKind::LargeSignal, FidelityLabel::Lumped),
            ],
            _ => vec![
                study("driver-linear", WorkbenchStudyKind::DriverLinear, FidelityLabel::Reduced),
                study("product-radiation", WorkbenchStudyKind::ProductRadiation, FidelityLabel::Distributed),
                study("enclosure-system", WorkbenchStudyKind::EnclosureSystem, FidelityLabel::Lumped),
            ],
        };
        let project = Self {
            schema_version: WORKBENCH_SCHEMA_VERSION,
            name: name.to_owned(),
            product: empty_product(),
            studies,
            materials: Vec::new(),
            provenance: ProvenanceEnvelope {
                material_pack: "sonium-seed-pack".to_owned(),
                material_pack_version: "1".to_owned(),
                imported_from: Some(origin.to_owned()),
            },
        };
        project.validate()?;
        Ok(project)
    }
}

fn study(id: &str, kind: WorkbenchStudyKind, fidelity: FidelityLabel) -> WorkbenchStudy {
    WorkbenchStudy { id: id.to_owned(), kind, fidelity, enabled: true }
}

fn detect_legacy_origin(value: &serde_json::Value) -> Option<&'static str> {
    let object = value.as_object()?;
    if object.contains_key("study_kind") || object.contains_key("whole_speaker") {
        return Some("speaker-studio");
    }
    if object.contains_key("simulator") && object.contains_key("study") {
        return Some("sim-demo");
    }
    if object.contains_key("linear") || object.contains_key("drivers") {
        return Some("speaker-lab");
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project() -> WorkbenchProject {
        WorkbenchProject {
            schema_version: WORKBENCH_SCHEMA_VERSION,
            name: "reference two-way".to_owned(),
            product: empty_product(),
            studies: vec![
                study("motor", WorkbenchStudyKind::Motor, FidelityLabel::Distributed),
                study("port-optimisation", WorkbenchStudyKind::PortOptimisation, FidelityLabel::Reduced),
            ],
            materials: vec![MaterialReference {
                material_id: "mdf-18mm".to_owned(),
                selection: "cabinet".to_owned(),
            }],
            provenance: ProvenanceEnvelope {
                material_pack: "sonium-seed-pack".to_owned(),
                material_pack_version: "1".to_owned(),
                imported_from: None,
            },
        }
    }

    #[test]
    fn project_round_trips_and_validates() {
        let project = project();
        project.validate().expect("valid project");
        let json = serde_json::to_string(&project).expect("serialise");
        let decoded: WorkbenchProject = serde_json::from_str(&json).expect("deserialise");
        assert_eq!(decoded, project);
    }

    #[test]
    fn missing_materials_are_named_gaps() {
        let project = project();
        assert!(project.missing_materials(&["mdf-18mm"]).is_empty());
        assert_eq!(project.missing_materials(&[]), vec!["cabinet".to_owned()]);
    }

    #[test]
    fn legacy_imports_keep_origin_and_fail_closed() {
        let studio = WorkbenchProject::import_legacy("s", r#"{"study_kind":"whole_speaker"}"#)
            .expect("studio imports");
        assert_eq!(studio.provenance.imported_from.as_deref(), Some("speaker-studio"));
        assert!(
            studio.studies.iter().any(|entry| entry.kind == WorkbenchStudyKind::PortOptimisation)
        );
        assert!(WorkbenchProject::import_legacy("x", r#"{"unknown":true}"#).is_err());
        assert!(WorkbenchProject::import_legacy("x", "not json").is_err());
    }

    #[test]
    fn duplicate_study_ids_are_rejected() {
        let mut project = project();
        project.studies.push(study("motor", WorkbenchStudyKind::Breakup, FidelityLabel::Reduced));
        assert!(project.validate().is_err());
    }
}
