//! The unified [`Material`] record and its provenance model.
//!
//! One record carries every physics domain; each domain is `Option` so a
//! fluid (air) need not invent elastic constants. Adapters fail closed with
//! a named gap when a required domain is absent — they never substitute a
//! default.

use serde::{Deserialize, Serialize};

use crate::curves::{BhCurveData, ComplexModulusCurve};
use crate::{MaterialError, Quantity};

/// How a record's values were obtained. Controls promotion rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProvenanceClass {
    /// Direct laboratory measurement with a stated method.
    Measured,
    /// Human-reviewed value from a cited handbook/datasheet/textbook.
    Curated,
    /// Computed value (DFT, simulation, screening database). Never a datasheet.
    Computed,
}

/// Full provenance for a [`Material`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    /// Measured / Curated / Computed classification.
    pub class: ProvenanceClass,
    /// Origin name, for example `CRC Handbook of Chemistry and Physics`.
    pub source: String,
    /// Full bibliographic citation (authors, title, edition/year, page/table).
    pub citation: String,
    /// Material grade, for example `N42`, `6061-T6`, `1008`. `None` if generic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grade: Option<String>,
    /// Reference temperature (K) at which the nominal values apply.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_temperature_k: Option<f64>,
    /// Measurement/computation method, for example `DMA`, `impedance tube`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    /// Licence / redistribution note. Seed pack entries are permissive.
    pub license: String,
}

impl Provenance {
    /// Validate provenance completeness (fail-closed preflight input).
    pub fn validate(&self, material_id: &str) -> Result<(), MaterialError> {
        let bad = |field: &'static str, reason: String| MaterialError::InvalidField {
            material_id: material_id.to_owned(),
            field,
            reason,
        };
        if self.source.trim().is_empty() {
            return Err(bad("provenance.source", "source must not be empty".into()));
        }
        if self.citation.trim().is_empty() {
            return Err(bad("provenance.citation", "citation must not be empty".into()));
        }
        if self.license.trim().is_empty() {
            return Err(bad("provenance.license", "license must not be empty".into()));
        }
        if let Some(t) = self.reference_temperature_k
            && (!t.is_finite() || t <= 0.0)
        {
            return Err(bad(
                "provenance.reference_temperature_k",
                format!("reference temperature {t} K must be finite and > 0"),
            ));
        }
        if self.class == ProvenanceClass::Measured
            && self.method.as_deref().is_none_or(|m| m.trim().is_empty())
        {
            return Err(bad(
                "provenance.method",
                "Measured records must state a measurement method".into(),
            ));
        }
        Ok(())
    }
}

/// Linear-elastic description (isotropic or orthotropic) plus optional damping.
///
/// `Material` records are cold configuration data (never in assembly/GMRES
/// hot loops), so the orthotropic variant is kept inline rather than boxed.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "symmetry", rename_all = "snake_case")]
pub enum Elastic {
    /// Isotropic: `E`, `ν`, optional loss factor and complex-modulus table.
    Isotropic {
        /// Young's modulus.
        youngs_modulus: Quantity,
        /// Poisson's ratio in (−1, 0.5).
        poisson_ratio: f64,
        /// Structural loss factor η (dimensionless), if known.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        loss_factor: Option<f64>,
        /// Tabulated `E*(f, T)`, if measured/modelled.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        complex_modulus: Option<ComplexModulusCurve>,
    },
    /// Orthotropic (ply/wood/paper): nine engineering constants.
    Orthotropic {
        /// Young's modulus along 1 (Pa).
        e1: Quantity,
        /// Young's modulus along 2 (Pa).
        e2: Quantity,
        /// Young's modulus along 3 (Pa).
        e3: Quantity,
        /// Shear modulus G12 (Pa).
        g12: Quantity,
        /// Shear modulus G13 (Pa).
        g13: Quantity,
        /// Shear modulus G23 (Pa).
        g23: Quantity,
        /// Poisson ratio ν12.
        nu12: f64,
        /// Poisson ratio ν13.
        nu13: f64,
        /// Poisson ratio ν23.
        nu23: f64,
        /// Structural loss factor η, if known.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        loss_factor: Option<f64>,
        /// Tabulated `E*(f, T)` (per-axis tables use the dominant axis).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        complex_modulus: Option<ComplexModulusCurve>,
    },
}

impl Elastic {
    /// Validate all elastic quantities and curve domains.
    pub fn validate(&self, material_id: &str) -> Result<(), MaterialError> {
        let bad = |field: &'static str, reason: String| MaterialError::InvalidField {
            material_id: material_id.to_owned(),
            field,
            reason,
        };
        let check_nu = |nu: f64, field: &'static str| -> Result<(), MaterialError> {
            if !nu.is_finite() || !(-1.0..0.5).contains(&nu) {
                return Err(bad(field, format!("Poisson ratio {nu} must lie in (-1, 0.5)")));
            }
            Ok(())
        };
        let check_loss = |eta: Option<f64>| -> Result<(), MaterialError> {
            if let Some(e) = eta
                && (!e.is_finite() || e < 0.0)
            {
                return Err(bad("elastic.loss_factor", format!("loss factor {e} must be finite and >= 0")));
            }
            Ok(())
        };
        match self {
            Elastic::Isotropic {
                youngs_modulus,
                poisson_ratio,
                loss_factor,
                complex_modulus,
            } => {
                youngs_modulus.validate(material_id, "elastic.youngs_modulus", "Pa", true)?;
                check_nu(*poisson_ratio, "elastic.poisson_ratio")?;
                check_loss(*loss_factor)?;
                if let Some(c) = complex_modulus {
                    c.validate(material_id)?;
                }
            }
            Elastic::Orthotropic {
                e1,
                e2,
                e3,
                g12,
                g13,
                g23,
                nu12,
                nu13,
                nu23,
                loss_factor,
                complex_modulus,
            } => {
                for (q, f) in [
                    (e1, "elastic.e1"),
                    (e2, "elastic.e2"),
                    (e3, "elastic.e3"),
                    (g12, "elastic.g12"),
                    (g13, "elastic.g13"),
                    (g23, "elastic.g23"),
                ] {
                    q.validate(material_id, f, "Pa", true)?;
                }
                check_nu(*nu12, "elastic.nu12")?;
                check_nu(*nu13, "elastic.nu13")?;
                check_nu(*nu23, "elastic.nu23")?;
                check_loss(*loss_factor)?;
                if let Some(c) = complex_modulus {
                    c.validate(material_id)?;
                }
            }
        }
        Ok(())
    }

    /// The tabulated complex-modulus curve, if present.
    pub fn complex_modulus(&self) -> Option<&ComplexModulusCurve> {
        match self {
            Elastic::Isotropic { complex_modulus, .. } => complex_modulus.as_ref(),
            Elastic::Orthotropic { complex_modulus, .. } => complex_modulus.as_ref(),
        }
    }

    /// Structural loss factor, if present.
    pub fn loss_factor(&self) -> Option<f64> {
        match self {
            Elastic::Isotropic { loss_factor, .. } => *loss_factor,
            Elastic::Orthotropic { loss_factor, .. } => *loss_factor,
        }
    }
}

/// Magnetic description: linear `µr` and/or tabulated B–H, remanence, loss.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Magnetic {
    /// Relative permeability µr (linear regime), if applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relative_permeability: Option<f64>,
    /// Tabulated first-quadrant B–H curve (soft magnets / steel).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bh_curve: Option<BhCurveData>,
    /// Remanent flux density Br (T) for permanent magnets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remanence_t: Option<Quantity>,
    /// Coercivity Hc (A/m) for permanent magnets.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coercivity_a_per_m: Option<f64>,
    /// Saturation flux density (T), if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub saturation_t: Option<f64>,
    /// Specific core loss (W/kg) at a stated operating point, if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub core_loss_w_per_kg: Option<Quantity>,
    /// Curie temperature (K), if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub curie_temperature_k: Option<f64>,
}

impl Magnetic {
    /// Validate magnetic fields.
    pub fn validate(&self, material_id: &str) -> Result<(), MaterialError> {
        let bad = |field: &'static str, reason: String| MaterialError::InvalidField {
            material_id: material_id.to_owned(),
            field,
            reason,
        };
        if self.relative_permeability.is_none() && self.bh_curve.is_none() {
            return Err(bad(
                "magnetic",
                "magnetic block needs relative_permeability and/or bh_curve".into(),
            ));
        }
        if let Some(mu) = self.relative_permeability
            && (!mu.is_finite() || mu <= 0.0)
        {
            return Err(bad("magnetic.relative_permeability", format!("µr {mu} must be finite and > 0")));
        }
        if let Some(c) = &self.bh_curve {
            c.validate(material_id)?;
        }
        if let Some(br) = &self.remanence_t {
            br.validate(material_id, "magnetic.remanence_t", "T", true)?;
        }
        if let Some(hc) = self.coercivity_a_per_m
            && (!hc.is_finite() || hc <= 0.0)
        {
            return Err(bad("magnetic.coercivity_a_per_m", format!("Hc {hc} must be finite and > 0")));
        }
        if let Some(bs) = self.saturation_t
            && (!bs.is_finite() || bs <= 0.0)
        {
            return Err(bad("magnetic.saturation_t", format!("Bs {bs} must be finite and > 0")));
        }
        if let Some(l) = &self.core_loss_w_per_kg {
            l.validate(material_id, "magnetic.core_loss_w_per_kg", "W/kg", true)?;
        }
        if let Some(tc) = self.curie_temperature_k
            && (!tc.is_finite() || tc <= 0.0)
        {
            return Err(bad("magnetic.curie_temperature_k", format!("Tc {tc} K must be finite and > 0")));
        }
        Ok(())
    }
}

/// Electrical description.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Electric {
    /// Electrical conductivity σ (S/m).
    pub conductivity: Quantity,
    /// Temperature coefficient of resistivity (1/K), if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature_coefficient_per_k: Option<f64>,
}

impl Electric {
    /// Validate electrical fields. Resistivity is derived, never stored.
    pub fn validate(&self, material_id: &str) -> Result<(), MaterialError> {
        self.conductivity
            .validate(material_id, "electric.conductivity", "S/m", true)?;
        if let Some(a) = self.temperature_coefficient_per_k
            && !a.is_finite()
        {
            return Err(MaterialError::InvalidField {
                material_id: material_id.to_owned(),
                field: "electric.temperature_coefficient_per_k",
                reason: format!("coefficient {a} must be finite"),
            });
        }
        Ok(())
    }

    /// Resistivity ρ = 1/σ (Ω·m).
    pub fn resistivity_ohm_m(&self) -> f64 {
        1.0 / self.conductivity.value
    }
}

/// Thermal description.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Thermal {
    /// Thermal conductivity k (W/(m·K)).
    pub conductivity: Quantity,
    /// Specific heat capacity cp (J/(kg·K)).
    pub specific_heat: Quantity,
    /// Linear thermal expansion coefficient (1/K), if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expansion_per_k: Option<f64>,
    /// Surface emissivity in [0, 1], if known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub emissivity: Option<f64>,
    /// Sound speed (m/s) — fluids only (air). Kept here so `AmbientMedium`
    /// adaptation has a single home.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed_of_sound: Option<Quantity>,
}

impl Thermal {
    /// Validate thermal fields.
    pub fn validate(&self, material_id: &str) -> Result<(), MaterialError> {
        let bad = |field: &'static str, reason: String| MaterialError::InvalidField {
            material_id: material_id.to_owned(),
            field,
            reason,
        };
        self.conductivity
            .validate(material_id, "thermal.conductivity", "W/(m*K)", true)?;
        self.specific_heat
            .validate(material_id, "thermal.specific_heat", "J/(kg*K)", true)?;
        if let Some(a) = self.expansion_per_k
            && !a.is_finite()
        {
            return Err(bad("thermal.expansion_per_k", format!("expansion {a} must be finite")));
        }
        if let Some(e) = self.emissivity
            && (!e.is_finite() || !(0.0..=1.0).contains(&e))
        {
            return Err(bad("thermal.emissivity", format!("emissivity {e} must lie in [0, 1]")));
        }
        if let Some(c) = &self.speed_of_sound {
            c.validate(material_id, "thermal.speed_of_sound", "m/s", true)?;
        }
        Ok(())
    }

    /// Thermal diffusivity α = k/(ρ·cp) (m²/s). Needs density from the record.
    pub fn diffusivity(&self, density_kg_per_m3: f64) -> f64 {
        self.conductivity.value / (density_kg_per_m3 * self.specific_heat.value)
    }
}

/// Johnson–Champoux–Allard 5-parameter porous model.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct JcaParams {
    /// Static flow resistivity σ (Pa·s/m²).
    pub flow_resistivity_pa_s_per_m2: f64,
    /// Open porosity φ in (0, 1].
    pub porosity: f64,
    /// Tortuosity α∞ ≥ 1.
    pub tortuosity: f64,
    /// Viscous characteristic length Λ (m).
    pub viscous_characteristic_length_m: f64,
    /// Thermal characteristic length Λ′ (m).
    pub thermal_characteristic_length_m: f64,
}

impl JcaParams {
    /// Validate JCA domains.
    pub fn validate(&self, material_id: &str) -> Result<(), MaterialError> {
        let bad = |field: &'static str, reason: String| MaterialError::InvalidField {
            material_id: material_id.to_owned(),
            field,
            reason,
        };
        if !self.flow_resistivity_pa_s_per_m2.is_finite() || self.flow_resistivity_pa_s_per_m2 <= 0.0 {
            return Err(bad("acoustic.jca.flow_resistivity", "flow resistivity must be finite and > 0".into()));
        }
        if !self.porosity.is_finite() || !(0.0..=1.0).contains(&self.porosity) || self.porosity == 0.0 {
            return Err(bad("acoustic.jca.porosity", format!("porosity {} must lie in (0, 1]", self.porosity)));
        }
        if !self.tortuosity.is_finite() || self.tortuosity < 1.0 {
            return Err(bad("acoustic.jca.tortuosity", format!("tortuosity {} must be >= 1", self.tortuosity)));
        }
        for (v, f) in [
            (self.viscous_characteristic_length_m, "acoustic.jca.viscous_length"),
            (self.thermal_characteristic_length_m, "acoustic.jca.thermal_length"),
        ] {
            if !v.is_finite() || v <= 0.0 {
                return Err(bad(f, format!("characteristic length {v} must be finite and > 0")));
            }
        }
        Ok(())
    }
}

/// Acoustic description: porous absorber and/or surface impedance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Acoustic {
    /// Static flow resistivity (Pa·s/m²) for Delany–Bazley/Miki models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flow_resistivity: Option<Quantity>,
    /// Full JCA 5-parameter set, if characterised.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jca: Option<JcaParams>,
    /// Specific acoustic impedance Zs (Pa·s/m) for constant-impedance walls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub specific_impedance: Option<Quantity>,
}

impl Acoustic {
    /// Validate acoustic fields.
    pub fn validate(&self, material_id: &str) -> Result<(), MaterialError> {
        if self.flow_resistivity.is_none() && self.jca.is_none() && self.specific_impedance.is_none() {
            return Err(MaterialError::InvalidField {
                material_id: material_id.to_owned(),
                field: "acoustic",
                reason: "acoustic block needs flow_resistivity, jca, and/or specific_impedance".into(),
            });
        }
        if let Some(q) = &self.flow_resistivity {
            q.validate(material_id, "acoustic.flow_resistivity", "Pa*s/m^2", true)?;
        }
        if let Some(j) = &self.jca {
            j.validate(material_id)?;
            // Cross-check: JCA resistivity must agree with the scalar if both present.
            if let Some(q) = &self.flow_resistivity
                && (q.value - j.flow_resistivity_pa_s_per_m2).abs()
                    > 1e-9 * j.flow_resistivity_pa_s_per_m2.max(1.0)
            {
                return Err(MaterialError::InvalidField {
                    material_id: material_id.to_owned(),
                    field: "acoustic.flow_resistivity",
                    reason: "flow_resistivity disagrees with jca.flow_resistivity_pa_s_per_m2".into(),
                });
            }
        }
        if let Some(q) = &self.specific_impedance {
            q.validate(material_id, "acoustic.specific_impedance", "Pa*s/m", true)?;
        }
        Ok(())
    }
}

/// The unified material record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Material {
    /// Stable kebab-case identifier, unique inside a pack.
    pub id: String,
    /// Human-readable name.
    pub display_name: String,
    /// Provenance (source, citation, grade, temperature, method, class).
    pub provenance: Provenance,
    /// Mass density ρ (kg/m³).
    pub density: Quantity,
    /// Elastic domain, if solid.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elastic: Option<Elastic>,
    /// Magnetic domain, if applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub magnetic: Option<Magnetic>,
    /// Electrical domain, if applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub electric: Option<Electric>,
    /// Thermal domain, if characterised.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thermal: Option<Thermal>,
    /// Acoustic domain, if characterised.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub acoustic: Option<Acoustic>,
    /// Free-form notes (processing, anisotropy caveats, …).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

impl Material {
    /// Full record validation (used by pack load and preflight).
    pub fn validate(&self) -> Result<(), MaterialError> {
        if self.id.trim().is_empty()
            || !self
                .id
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
        {
            return Err(MaterialError::InvalidField {
                material_id: self.id.clone(),
                field: "id",
                reason: format!("id `{}` must be non-empty kebab-case [a-z0-9-_]", self.id),
            });
        }
        if self.display_name.trim().is_empty() {
            return Err(MaterialError::InvalidField {
                material_id: self.id.clone(),
                field: "display_name",
                reason: "display_name must not be empty".into(),
            });
        }
        self.provenance.validate(&self.id)?;
        self.density.validate(&self.id, "density", "kg/m^3", true)?;
        if let Some(e) = &self.elastic {
            e.validate(&self.id)?;
        }
        if let Some(m) = &self.magnetic {
            m.validate(&self.id)?;
        }
        if let Some(e) = &self.electric {
            e.validate(&self.id)?;
        }
        if let Some(t) = &self.thermal {
            t.validate(&self.id)?;
        }
        if let Some(a) = &self.acoustic {
            a.validate(&self.id)?;
        }
        Ok(())
    }
}
