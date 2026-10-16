//! Preflight report: pack-level checks beyond per-record validation.
//!
//! Preflight is the gate a simulation should run before trusting a pack:
//! checksum/schema (via [`MaterialPack::verify`]), provenance completeness,
//! permissive-licence presence on seed entries, adapter coverage warnings,
//! and screening-draft hygiene.

use serde::{Deserialize, Serialize};

use crate::{MaterialPack, ProvenanceClass};

/// Outcome of [`preflight`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PreflightReport {
    /// Pack name.
    pub pack_name: String,
    /// Schema version found.
    pub schema_version: u32,
    /// Number of materials checked.
    pub material_count: usize,
    /// Hard failures (pack must not be used).
    pub errors: Vec<String>,
    /// Soft warnings (pack usable, but coverage is partial).
    pub warnings: Vec<String>,
    /// Per-material adapter coverage tokens, sorted.
    pub adapter_coverage: Vec<String>,
}

impl PreflightReport {
    /// `true` when there are no hard failures.
    pub fn passed(&self) -> bool {
        self.errors.is_empty()
    }
}

/// Run preflight over a pack. Never panics; collects all findings.
pub fn preflight(pack: &MaterialPack) -> PreflightReport {
    let mut errors: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    let mut coverage: Vec<String> = Vec::new();

    if let Err(e) = pack.verify() {
        errors.push(format!("pack verification failed: {e}"));
    }
    for m in &pack.materials {
        // verify() already validated records; surface provenance gaps as errors.
        if m.provenance.citation.trim().is_empty() {
            errors.push(format!("{}: missing citation", m.id));
        }
        if m.provenance.license.trim().is_empty() {
            errors.push(format!("{}: missing license", m.id));
        }
        if m.provenance.class == ProvenanceClass::Computed {
            warnings.push(format!(
                "{}: Computed record — screening/draft quality, not a datasheet",
                m.id
            ));
        }
        // Adapter coverage tokens (which existing structs this record can feed).
        let mut feeds: Vec<&str> = Vec::new();
        if m.try_panel_material(0.018).is_ok() {
            feeds.push("PanelMaterial");
        }
        if m.try_isotropic().is_ok() {
            feeds.push("IsotropicMaterial");
        }
        if m.try_thermal().is_ok() {
            feeds.push("ThermalMaterial");
        }
        if m.try_ambient_medium().is_ok() {
            feeds.push("AmbientMedium");
        }
        if m.try_magnetic().is_ok() {
            feeds.push("MagneticMaterial");
        }
        if m.try_bh_curve().is_ok() {
            feeds.push("BhCurve");
        }
        if m.try_porous_layer(0.05, 0.0).is_ok() {
            feeds.push("PorousLayer");
        }
        if m.try_speaker_impedance().is_ok() {
            feeds.push("SpeakerMaterial");
        }
        if m.try_cabinet_kind().is_ok() {
            feeds.push("CabinetMaterial");
        }
        if feeds.is_empty() {
            warnings.push(format!("{}: feeds no existing adapter (data-only record)", m.id));
        }
        coverage.push(format!("{} -> {}", m.id, feeds.join(",")));
    }
    coverage.sort();
    PreflightReport {
        pack_name: pack.name.clone(),
        schema_version: pack.schema_version,
        material_count: pack.materials.len(),
        errors,
        warnings,
        adapter_coverage: coverage,
    }
}
