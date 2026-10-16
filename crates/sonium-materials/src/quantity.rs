//! Scalar quantities with explicit units and tolerances.
//!
//! The schema rule is *curves, not single numbers, where physics is
//! frequency/temperature dependent, and units + tolerances on every scalar*.
//! A bare `f64` never appears as a physical property in [`crate::Material`].

use serde::{Deserialize, Serialize};

use crate::MaterialError;

/// A physical scalar with unit and optional symmetric/asymmetric tolerance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Quantity {
    /// Nominal value in `unit`.
    pub value: f64,
    /// SI unit string, for example `kg/m^3`, `Pa`, `W/(m*K)`, `S/m`.
    pub unit: String,
    /// Relative tolerance (fraction, `0.05` = ±5 %). `None` = unstated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tolerance_rel: Option<f64>,
    /// Absolute tolerance in the same unit. `None` = unstated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tolerance_abs: Option<f64>,
}

impl Quantity {
    /// Construct a quantity.
    pub fn new(value: f64, unit: &str) -> Self {
        Self {
            value,
            unit: unit.to_owned(),
            tolerance_rel: None,
            tolerance_abs: None,
        }
    }

    /// Attach a relative tolerance (fraction).
    pub fn with_rel_tol(mut self, tol: f64) -> Self {
        self.tolerance_rel = Some(tol);
        self
    }

    /// Attach an absolute tolerance in the quantity's unit.
    pub fn with_abs_tol(mut self, tol: f64) -> Self {
        self.tolerance_abs = Some(tol);
        self
    }

    /// Validate finiteness, expected unit, and tolerance domains.
    pub fn validate(
        &self,
        material_id: &str,
        field: &'static str,
        expected_unit: &str,
        must_be_positive: bool,
    ) -> Result<(), MaterialError> {
        let bad = |reason: String| MaterialError::InvalidField {
            material_id: material_id.to_owned(),
            field,
            reason,
        };
        if !self.value.is_finite() {
            return Err(bad(format!("value {} is not finite", self.value)));
        }
        if must_be_positive && self.value <= 0.0 {
            return Err(bad(format!("value {} must be > 0", self.value)));
        }
        if self.unit != expected_unit {
            return Err(bad(format!(
                "unit `{}` does not match expected `{expected_unit}`",
                self.unit
            )));
        }
        if let Some(t) = self.tolerance_rel
            && (!t.is_finite() || t < 0.0)
        {
            return Err(bad(format!("tolerance_rel {t} must be finite and >= 0")));
        }
        if let Some(t) = self.tolerance_abs
            && (!t.is_finite() || t < 0.0)
        {
            return Err(bad(format!("tolerance_abs {t} must be finite and >= 0")));
        }
        Ok(())
    }

    /// Inclusive `[low, high]` interval implied by the tolerances.
    pub fn interval(&self) -> (f64, f64) {
        let mut lo = self.value;
        let mut hi = self.value;
        if let Some(r) = self.tolerance_rel {
            lo -= self.value.abs() * r;
            hi += self.value.abs() * r;
        }
        if let Some(a) = self.tolerance_abs {
            lo -= a;
            hi += a;
        }
        (lo, hi)
    }
}
