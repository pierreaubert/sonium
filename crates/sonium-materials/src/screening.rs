//! Computed screening drafts (OPTIMADE / Materials Project).
//!
//! These databases hold **computed inorganic** properties only. A record
//! imported from them is a *screening draft*: it must stay
//! [`ProvenanceClass::Computed`], it is never a datasheet, and it can
//! **never auto-promote** to Curated/Measured. Promotion requires a human
//! reviewer to re-source the value from a measurement or handbook and
//! build a new record — this type deliberately offers no promotion path.

use serde::{Deserialize, Serialize};

use crate::{Material, MaterialError, ProvenanceClass};

/// External computed database a draft came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScreeningSource {
    /// OPTIMADE federation (<https://www.optimade.org/>).
    Optimade,
    /// Materials Project (<https://docs.materialsproject.org/>).
    MaterialsProject,
}

impl ScreeningSource {
    /// Human-readable source name.
    pub fn name(&self) -> &'static str {
        match self {
            ScreeningSource::Optimade => "OPTIMADE",
            ScreeningSource::MaterialsProject => "Materials Project",
        }
    }

    /// Canonical documentation URL.
    pub fn docs_url(&self) -> &'static str {
        match self {
            ScreeningSource::Optimade => "https://www.optimade.org/",
            ScreeningSource::MaterialsProject => "https://docs.materialsproject.org/",
        }
    }
}

/// A computed screening draft. Outer tools (network fetchers) build these;
/// the pure-Rust core only validates and stores them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScreeningDraft {
    /// Source database.
    pub source: ScreeningSource,
    /// Identifier inside the source database.
    pub external_id: String,
    /// The computed material record (class must be `Computed`).
    pub material: Material,
    /// Why this draft is only a screening candidate.
    pub screening_note: String,
}

impl ScreeningDraft {
    /// Validate that the draft is honestly labelled as computed screening.
    pub fn validate(&self) -> Result<(), MaterialError> {
        self.material.validate()?;
        if self.external_id.trim().is_empty() {
            return Err(MaterialError::InvalidField {
                material_id: self.material.id.clone(),
                field: "screening.external_id",
                reason: "external_id must not be empty".into(),
            });
        }
        if self.material.provenance.class != ProvenanceClass::Computed {
            return Err(MaterialError::InvalidField {
                material_id: self.material.id.clone(),
                field: "provenance.class",
                reason: format!(
                    "screening drafts from {} must stay Computed, found {:?}",
                    self.source.name(),
                    self.material.provenance.class
                ),
            });
        }
        Ok(())
    }

    /// Attempt to promote this draft to a trusted record. **Always fails.**
    ///
    /// There is intentionally no success path: re-source the values from a
    /// measurement or handbook and author a new `Curated`/`Measured`
    /// record with its own citation instead.
    pub fn auto_promote(&self) -> Result<Material, MaterialError> {
        Err(MaterialError::ScreeningPromotionRefused {
            source_name: self.source.name().to_owned(),
            external_id: self.external_id.clone(),
            detail: "computed screening values are not datasheets; human re-sourcing required".into(),
        })
    }
}
