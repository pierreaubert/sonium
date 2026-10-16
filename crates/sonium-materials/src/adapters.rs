//! Fail-closed adapters to the existing Sonium material structs.
//!
//! Mapping table (existing struct ← required `Material` fields):
//!
//! | Adapter view | Existing struct | Needs |
//! |---|---|---|
//! | [`PanelMaterialView`] | `sonium_fem::…::cabinet_radiation::PanelMaterial` | isotropic E, ν, density, η, thickness (arg) |
//! | [`IsotropicView`] | `sonium_fem::…::elasticity::IsotropicMaterial` | isotropic E, ν |
//! | [`MagneticView`] / [`BhCurveData`] | `sonium_fem::…::magnetostatics::{MagneticMaterial, BhCurve}` | µr and/or B–H, Br |
//! | [`PorousLayerView`] | `sonium_bem::materials::PorousLayer` | flow resistivity, thickness (arg) |
//! | [`ThermalView`] | `sonium_fem::…::heat_conduction::ThermalMaterial` | k, ρ, cp |
//! | [`AmbientMediumView`] | `sonium_bem::materials::AmbientMedium` | ρ, c (thermal.speed_of_sound) |
//! | [`SpeakerImpedanceView`] | `sonium_sim_speaker::model::SpeakerMaterial` | Zs or porous |
//! | [`CabinetMaterialKind`] | `sonium_sim_speaker_modeling::CabinetMaterial` | id `mdf-18mm` / `plywood-18mm` |
//!
//! This crate stays pure-Rust (no FEM/BEM dependency, per the pure-Rust
//! split: core compiles alone, native deps live in outer tools), so
//! adapters return field-identical *views*. Callers in `sonium-fem` /
//! `sonium-bem` copy the view fields into the real struct in one line;
//! the validation and gap logic lives here and is unit-tested here.
//! Every adapter returns [`MaterialError::AdapterGap`] with a stable gap
//! token when the record cannot satisfy it — never a default.

use crate::curves::BhCurveData;
use crate::{Elastic, Material, MaterialError};

fn gap(
    adapter: &'static str,
    material: &Material,
    gap: &'static str,
    detail: &str,
) -> MaterialError {
    MaterialError::AdapterGap {
        adapter,
        material_id: material.id.clone(),
        gap,
        detail: detail.to_owned(),
    }
}

/// Field-identical view of `PanelMaterial` (cabinet radiation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PanelMaterialView {
    /// Young's modulus (Pa).
    pub youngs_modulus: f64,
    /// Poisson's ratio.
    pub poissons_ratio: f64,
    /// Density (kg/m³).
    pub density: f64,
    /// Panel thickness (m) — supplied by the caller, not the material.
    pub thickness: f64,
    /// Structural loss factor η.
    pub loss_factor: f64,
}

/// Field-identical view of FEM `IsotropicMaterial`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IsotropicView {
    /// Young's modulus (Pa).
    pub youngs_modulus: f64,
    /// Poisson's ratio.
    pub poisson_ratio: f64,
}

/// Field-identical view of FEM `ThermalMaterial`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermalView {
    /// Thermal conductivity (W/(m·K)).
    pub conductivity: f64,
    /// Density (kg/m³).
    pub density: f64,
    /// Specific heat (J/(kg·K)).
    pub specific_heat: f64,
}

/// Field-identical view of BEM `AmbientMedium`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AmbientMediumView {
    /// Density (kg/m³).
    pub density: f64,
    /// Speed of sound (m/s).
    pub speed_of_sound: f64,
}

/// Magnetic adaptation result (linear and/or nonlinear + magnet data).
#[derive(Debug, Clone, PartialEq)]
pub struct MagneticView {
    /// Relative permeability, if linear data exists.
    pub relative_permeability: Option<f64>,
    /// B–H curve, if tabulated data exists.
    pub bh_curve: Option<BhCurveData>,
    /// Remanence Br (T), if a permanent magnet.
    pub remanence_t: Option<f64>,
}

/// Field-identical view of BEM `PorousLayer` (Delany–Bazley/Miki input).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PorousLayerView {
    /// Layer thickness (m) — supplied by the caller.
    pub thickness: f64,
    /// Static flow resistivity (Pa·s/m²).
    pub flow_resistivity: f64,
    /// Air gap behind the layer (m).
    pub air_gap: f64,
}

/// Speaker boundary adaptation: constant impedance or porous layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpeakerImpedanceView {
    /// Constant specific impedance (Pa·s/m) → `SpeakerMaterial::ConstantImpedance`.
    ConstantImpedance {
        /// Specific impedance (Pa·s/m).
        specific_impedance: f64,
    },
    /// Porous layer → `SpeakerMaterial::PorousLayer` (thickness from caller).
    Porous {
        /// Flow resistivity (Pa·s/m²).
        flow_resistivity: f64,
    },
}

/// Mirror of `sonium_sim_speaker_modeling::CabinetMaterial`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CabinetMaterialKind {
    /// `CabinetMaterial::Mdf18mm`.
    Mdf18mm,
    /// `CabinetMaterial::Plywood18mm`.
    Plywood18mm,
}

impl Material {
    /// Adapt to a cabinet `PanelMaterial` view. Thickness is a part
    /// property, not a material property, so it is an argument.
    pub fn try_panel_material(&self, thickness_m: f64) -> Result<PanelMaterialView, MaterialError> {
        const A: &str = "PanelMaterial";
        if !thickness_m.is_finite() || thickness_m <= 0.0 {
            return Err(gap(A, self, "invalid_thickness", "thickness must be finite and > 0"));
        }
        let (e, nu, eta) = match &self.elastic {
            Some(Elastic::Isotropic {
                youngs_modulus,
                poisson_ratio,
                loss_factor,
                ..
            }) => (youngs_modulus.value, *poisson_ratio, *loss_factor),
            Some(Elastic::Orthotropic { .. }) => {
                return Err(gap(
                    A,
                    self,
                    "anisotropy_mismatch",
                    "PanelMaterial is isotropic; record is orthotropic — homogenise explicitly first",
                ));
            }
            None => return Err(gap(A, self, "missing_elastic", "record has no elastic block")),
        };
        let loss_factor = eta.ok_or_else(|| {
            gap(A, self, "missing_loss_factor", "PanelMaterial requires a structural loss factor η")
        })?;
        Ok(PanelMaterialView {
            youngs_modulus: e,
            poissons_ratio: nu,
            density: self.density.value,
            thickness: thickness_m,
            loss_factor,
        })
    }

    /// Adapt to an FEM `IsotropicMaterial` view.
    pub fn try_isotropic(&self) -> Result<IsotropicView, MaterialError> {
        const A: &str = "IsotropicMaterial";
        match &self.elastic {
            Some(Elastic::Isotropic {
                youngs_modulus,
                poisson_ratio,
                ..
            }) => Ok(IsotropicView {
                youngs_modulus: youngs_modulus.value,
                poisson_ratio: *poisson_ratio,
            }),
            Some(Elastic::Orthotropic { .. }) => Err(gap(
                A,
                self,
                "anisotropy_mismatch",
                "record is orthotropic; IsotropicMaterial cannot represent it",
            )),
            None => Err(gap(A, self, "missing_elastic", "record has no elastic block")),
        }
    }

    /// Adapt to a `ThermalMaterial` view.
    pub fn try_thermal(&self) -> Result<ThermalView, MaterialError> {
        const A: &str = "ThermalMaterial";
        let t = self
            .thermal
            .as_ref()
            .ok_or_else(|| gap(A, self, "missing_thermal", "record has no thermal block"))?;
        Ok(ThermalView {
            conductivity: t.conductivity.value,
            density: self.density.value,
            specific_heat: t.specific_heat.value,
        })
    }

    /// Adapt to an `AmbientMedium` view (fluids: air).
    pub fn try_ambient_medium(&self) -> Result<AmbientMediumView, MaterialError> {
        const A: &str = "AmbientMedium";
        let c = self
            .thermal
            .as_ref()
            .and_then(|t| t.speed_of_sound.as_ref())
            .ok_or_else(|| {
                gap(A, self, "missing_speed_of_sound", "record has no thermal.speed_of_sound (not a characterised fluid)")
            })?;
        Ok(AmbientMediumView {
            density: self.density.value,
            speed_of_sound: c.value,
        })
    }

    /// Adapt to magnetic views (`MagneticMaterial` / `BhCurve` inputs).
    pub fn try_magnetic(&self) -> Result<MagneticView, MaterialError> {
        const A: &str = "MagneticMaterial";
        let m = self
            .magnetic
            .as_ref()
            .ok_or_else(|| gap(A, self, "missing_magnetic", "record has no magnetic block"))?;
        Ok(MagneticView {
            relative_permeability: m.relative_permeability,
            bh_curve: m.bh_curve.clone(),
            remanence_t: m.remanence_t.as_ref().map(|q| q.value),
        })
    }

    /// Adapt to just the B–H curve (nonlinear magnetostatics input).
    pub fn try_bh_curve(&self) -> Result<BhCurveData, MaterialError> {
        const A: &str = "BhCurve";
        let m = self
            .magnetic
            .as_ref()
            .ok_or_else(|| gap(A, self, "missing_magnetic", "record has no magnetic block"))?;
        m.bh_curve.clone().ok_or_else(|| {
            gap(A, self, "missing_bh_curve", "record has µr only; no tabulated B-H curve")
        })
    }

    /// Adapt to a BEM `PorousLayer` view. Thickness/air-gap are layer
    /// (part) properties, supplied by the caller.
    pub fn try_porous_layer(
        &self,
        thickness_m: f64,
        air_gap_m: f64,
    ) -> Result<PorousLayerView, MaterialError> {
        const A: &str = "PorousLayer";
        if !thickness_m.is_finite() || thickness_m <= 0.0 {
            return Err(gap(A, self, "invalid_thickness", "thickness must be finite and > 0"));
        }
        if !air_gap_m.is_finite() || air_gap_m < 0.0 {
            return Err(gap(A, self, "invalid_air_gap", "air gap must be finite and >= 0"));
        }
        let a = self
            .acoustic
            .as_ref()
            .ok_or_else(|| gap(A, self, "missing_acoustic", "record has no acoustic block"))?;
        let flow = a
            .flow_resistivity
            .as_ref()
            .map(|q| q.value)
            .or_else(|| a.jca.map(|j| j.flow_resistivity_pa_s_per_m2))
            .ok_or_else(|| {
                gap(A, self, "missing_flow_resistivity", "record has neither flow_resistivity nor JCA parameters")
            })?;
        Ok(PorousLayerView {
            thickness: thickness_m,
            flow_resistivity: flow,
            air_gap: air_gap_m,
        })
    }

    /// Adapt to a `SpeakerMaterial` view (constant impedance preferred,
    /// porous otherwise).
    pub fn try_speaker_impedance(&self) -> Result<SpeakerImpedanceView, MaterialError> {
        const A: &str = "SpeakerMaterial";
        let a = self
            .acoustic
            .as_ref()
            .ok_or_else(|| gap(A, self, "missing_acoustic", "record has no acoustic block"))?;
        if let Some(z) = &a.specific_impedance {
            return Ok(SpeakerImpedanceView::ConstantImpedance {
                specific_impedance: z.value,
            });
        }
        if let Some(q) = &a.flow_resistivity {
            return Ok(SpeakerImpedanceView::Porous {
                flow_resistivity: q.value,
            });
        }
        if let Some(j) = &a.jca {
            return Ok(SpeakerImpedanceView::Porous {
                flow_resistivity: j.flow_resistivity_pa_s_per_m2,
            });
        }
        Err(gap(
            A,
            self,
            "missing_specific_impedance",
            "record has neither specific_impedance nor flow resistivity",
        ))
    }

    /// Adapt to `CabinetMaterial` (only the two sheet goods exist there).
    pub fn try_cabinet_kind(&self) -> Result<CabinetMaterialKind, MaterialError> {
        const A: &str = "CabinetMaterial";
        match self.id.as_str() {
            "mdf-18mm" => Ok(CabinetMaterialKind::Mdf18mm),
            "plywood-18mm" => Ok(CabinetMaterialKind::Plywood18mm),
            _ => Err(gap(
                A,
                self,
                "unmapped_cabinet_material",
                "CabinetMaterial only has Mdf18mm and Plywood18mm; extend the enum to map this record",
            )),
        }
    }
}
