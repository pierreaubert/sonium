//! `sonium-materials` — unified, versioned material records for Sonium.
//!
//! # Why this crate exists
//!
//! Material data was previously scattered as incompatible per-physics
//! structs: FEM `IsotropicMaterial`/`ViscoelasticMaterial` (elasticity),
//! `MagneticMaterial` + `BhCurve` (magnetostatics), `PanelMaterial`
//! (cabinet radiation), `ThermalMaterial` (heat conduction), BEM
//! `PorousLayer`/`AmbientMedium` (porous acoustics),
//! `SpeakerMaterial` (speaker BEM boundaries), and
//! `CabinetMaterial` (speaker-modeling cabinet design). Each hard-coded
//! its own constants with no shared provenance, units, or tolerances.
//!
//! This crate is the single record ([`Material`]) plus fail-closed
//! [`adapters`] back to those structs' fields. See `adapters` module docs
//! for the mapping table.
//!
//! # Pure-Rust split
//!
//! Per the pure-Rust architecture split, the core compiles with Rust
//! alone: this crate depends only on `serde`, `serde_json`, `toml`,
//! `sha2`, `thiserror`, and `num-complex` — all pure Rust, no native
//! (C/CUDA/Metal) dependencies. Network fetchers for OPTIMADE /
//! Materials Project screening drafts are outer tools; the core only
//! validates the [`ScreeningDraft`] they produce.
//!
//! # Performance
//!
//! Hot path: [`ComplexModulusCurve::interpolate`] and
//! [`BhCurveData::field_strength`] (called inside assembly/evaluation
//! loops) are allocation-free binary searches over borrowed slices.
//! Everything else (pack I/O, checksum, validation, preflight) is cold.
//! No allocation occurs in interpolation; callers reuse their own
//! buffers for assembly/GMRES as with `sonium-compute` kernels.
//!
//! # Provenance fidelity
//!
//! Every record carries source, citation, grade, reference temperature,
//! method, class (`Measured`/`Curated`/`Computed`), and licence.
//! Computed screening drafts (OPTIMADE, Materials Project) are
//! inorganic-computed only, are never datasheets, and can never
//! auto-promote — see [`screening`]. No COMSOL parity is claimed.
//! Validation tiers: Tier A analytical (Wolfram), Tier B spinorama.org
//! CEA-2034 matched without fitting, Tier C self-measured (future).
//!
//! # Example
//!
//! ```rust
//! use sonium_materials::seed_pack;
//!
//! let pack = seed_pack().expect("seed pack validates");
//! let mdf = pack.get("mdf-18mm").expect("seed present");
//! let panel = mdf.try_panel_material(0.018).expect("MDF adapts");
//! assert!((panel.youngs_modulus - 3.5e9).abs() < 1.0);
//! let report = sonium_materials::preflight(&pack);
//! assert!(report.passed());
//! ```

#![warn(missing_docs)]

mod adapters;
mod curves;
mod error;
mod material;
mod pack;
mod preflight;
mod quantity;
mod screening;
mod seed;

pub use adapters::{
    AmbientMediumView, CabinetMaterialKind, IsotropicView, MagneticView, PanelMaterialView,
    PorousLayerView, SpeakerImpedanceView, ThermalView,
};
pub use curves::{BhCurveData, BhPoint, ComplexModulusCurve, ComplexModulusPoint};
pub use error::MaterialError;
pub use material::{
    Acoustic, Electric, Elastic, JcaParams, Magnetic, Material, Provenance, ProvenanceClass,
    Thermal,
};
pub use pack::{MaterialPack, SCHEMA_VERSION};
pub use preflight::{PreflightReport, preflight};
pub use quantity::Quantity;
pub use screening::{ScreeningDraft, ScreeningSource};
pub use seed::seed_pack;

#[cfg(test)]
mod tests {
    use super::*;

    fn seed() -> MaterialPack {
        seed_pack().expect("seed pack must validate")
    }

    #[test]
    fn seed_pack_contains_required_materials() {
        let pack = seed();
        for id in [
            "air",
            "copper",
            "aluminium",
            "steel-1008",
            "ferrite-y30",
            "ndfeb-n42",
            "mdf-18mm",
            "plywood-18mm",
            "cone-paper",
            "cone-composite-pp",
            "surround-rubber",
            "surround-foam",
            "spider-cloth",
            "former-kapton",
            "former-aluminium",
            "film-nomex",
            "abs-plastic",
            "absorber-melamine-foam",
            "absorber-pet-fibre",
            "absorber-mineral-wool",
            "absorber-glass-fibre",
        ] {
            assert!(pack.get(id).is_ok(), "missing seed {id}");
        }
        assert!(pack.materials.len() >= 20);
    }

    #[test]
    fn seed_pack_preflight_passes() {
        let pack = seed();
        let report = preflight(&pack);
        assert!(report.passed(), "errors: {:?}", report.errors);
        assert_eq!(report.material_count, pack.materials.len());
    }

    #[test]
    fn json_and_toml_round_trip() {
        let pack = seed();
        let json = pack.to_json().expect("json");
        let back = MaterialPack::from_json(&json).expect("json round trip");
        assert_eq!(back, pack);
        let toml_text = pack.to_toml().expect("toml");
        let back2 = MaterialPack::from_toml(&toml_text).expect("toml round trip");
        assert_eq!(back2, pack);
    }

    #[test]
    fn checksum_tamper_is_rejected() {
        let pack = seed();
        let mut json = pack.to_json().expect("json");
        // Flip a density digit: checksum must catch it.
        json = json.replacen("8960.0", "8961.0", 1);
        assert!(matches!(
            MaterialPack::from_json(&json),
            Err(MaterialError::ChecksumMismatch { .. })
        ));
    }

    #[test]
    fn complex_modulus_interpolation_and_no_extrapolation() {
        let pack = seed();
        let rubber = pack.get("surround-rubber").expect("rubber");
        let Elastic::Isotropic { complex_modulus, .. } = rubber.elastic.as_ref().expect("elastic") else {
            panic!("rubber must be isotropic");
        };
        let curve = complex_modulus.as_ref().expect("curve");
        let mid = curve.interpolate("surround-rubber", 100.0, 293.15).expect("inside");
        assert!((mid.re - 5.0e6).abs() < 1.0);
        assert!(matches!(
            curve.interpolate("surround-rubber", 1.0e9, 293.15),
            Err(MaterialError::OutOfDomain { .. })
        ));
    }

    #[test]
    fn adapters_succeed_for_mdf_and_air() {
        let pack = seed();
        let mdf = pack.get("mdf-18mm").expect("mdf");
        let panel = mdf.try_panel_material(0.018).expect("panel");
        assert!((panel.density - 750.0).abs() < 1e-9);
        assert_eq!(mdf.try_cabinet_kind().expect("kind"), CabinetMaterialKind::Mdf18mm);
        let air = pack.get("air").expect("air");
        let medium = air.try_ambient_medium().expect("medium");
        assert!((medium.speed_of_sound - 343.0).abs() < 1e-9);
        let steel = pack.get("steel-1008").expect("steel");
        let bh = steel.try_bh_curve().expect("bh");
        assert!(bh.field_strength(1.0) > 0.0);
        let foam = pack.get("absorber-melamine-foam").expect("foam");
        let porous = foam.try_porous_layer(0.05, 0.01).expect("porous");
        assert!((porous.flow_resistivity - 12000.0).abs() < 1e-9);
    }

    #[test]
    fn adapters_fail_closed_with_named_gaps() {
        let pack = seed();
        // Orthotropic cone paper cannot feed the isotropic panel model.
        let paper = pack.get("cone-paper").expect("paper");
        assert!(matches!(
            paper.try_panel_material(0.001),
            Err(MaterialError::AdapterGap { gap: "anisotropy_mismatch", .. })
        ));
        assert!(matches!(
            paper.try_isotropic(),
            Err(MaterialError::AdapterGap { gap: "anisotropy_mismatch", .. })
        ));
        // Air has no elastic block.
        let air = pack.get("air").expect("air");
        assert!(matches!(
            air.try_isotropic(),
            Err(MaterialError::AdapterGap { gap: "missing_elastic", .. })
        ));
        // Ferrite has µr/Br but no B–H table.
        let ferrite = pack.get("ferrite-y30").expect("ferrite");
        assert!(matches!(
            ferrite.try_bh_curve(),
            Err(MaterialError::AdapterGap { gap: "missing_bh_curve", .. })
        ));
        // Copper is not a cabinet sheet good.
        let copper = pack.get("copper").expect("copper");
        assert!(matches!(
            copper.try_cabinet_kind(),
            Err(MaterialError::AdapterGap { gap: "unmapped_cabinet_material", .. })
        ));
    }

    #[test]
    fn screening_draft_never_auto_promotes() {
        let mut computed = seed().get("copper").expect("copper").clone();
        computed.provenance.class = ProvenanceClass::Computed;
        computed.provenance.source = "Materials Project".into();
        let draft = ScreeningDraft {
            source: ScreeningSource::MaterialsProject,
            external_id: "mp-30".into(),
            material: computed,
            screening_note: "computed inorganic screening only".into(),
        };
        draft.validate().expect("draft validates");
        assert_eq!(draft.source.docs_url(), "https://docs.materialsproject.org/");
        assert!(matches!(
            draft.auto_promote(),
            Err(MaterialError::ScreeningPromotionRefused { .. })
        ));
        let opt = ScreeningDraft {
            source: ScreeningSource::Optimade,
            ..draft.clone()
        };
        assert_eq!(opt.source.docs_url(), "https://www.optimade.org/");
        // A Curated record smuggled in as a draft is rejected.
        let mut bad = draft.clone();
        bad.material.provenance.class = ProvenanceClass::Curated;
        assert!(bad.validate().is_err());
    }

    #[test]
    fn invalid_records_are_rejected() {
        let pack = seed();
        let mut broken = pack.get("copper").expect("copper").clone();
        broken.density.value = -1.0;
        assert!(broken.validate().is_err());
        let mut bad_bh = pack.get("steel-1008").expect("steel").clone();
        if let Some(m) = bad_bh.magnetic.as_mut()
            && let Some(c) = m.bh_curve.as_mut()
        {
            c.points[1].b_t = 0.0; // break monotonicity / origin rule
        }
        assert!(bad_bh.validate().is_err());
    }
}
