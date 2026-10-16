//! Tabulated curves: complex modulus vs (frequency, temperature) and B–H.
//!
//! Hot path: [`ComplexModulusCurve::interpolate`] and
//! [`BhCurveData::field_strength`] are allocation-free (binary search over a
//! borrowed slice, no `Vec` growth) so they can sit inside FEM assembly /
//! GMRES-adjacent evaluation loops without allocator pressure. All other
//! code in this crate is cold (I/O, validation, preflight).

use num_complex::Complex64;
use serde::{Deserialize, Serialize};

use crate::MaterialError;

/// One point of a tabulated complex modulus `E* = E' + i·E''`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ComplexModulusPoint {
    /// Frequency (Hz).
    pub frequency_hz: f64,
    /// Temperature (K).
    pub temperature_k: f64,
    /// Storage modulus E′ (Pa).
    pub storage_pa: f64,
    /// Loss modulus E″ (Pa).
    pub loss_pa: f64,
}

/// Frequency/temperature tabulated complex modulus.
///
/// Points must be sorted by `(temperature_k, frequency_hz)` with strictly
/// increasing frequency inside each temperature slice. Interpolation is
/// linear in log₁₀(frequency) within the nearest temperature slice and
/// **fails closed** outside the tabulated domain — no extrapolation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComplexModulusCurve {
    /// Tabulated points, sorted as documented.
    pub points: Vec<ComplexModulusPoint>,
}

impl ComplexModulusCurve {
    /// Validate ordering, finiteness, and positivity domains.
    pub fn validate(&self, material_id: &str) -> Result<(), MaterialError> {
        let bad = |reason: String| MaterialError::InvalidField {
            material_id: material_id.to_owned(),
            field: "elastic.complex_modulus",
            reason,
        };
        if self.points.len() < 2 {
            return Err(bad("complex-modulus curve needs at least 2 points".into()));
        }
        for (i, p) in self.points.iter().enumerate() {
            if !p.frequency_hz.is_finite() || p.frequency_hz <= 0.0 {
                return Err(bad(format!("point {i}: frequency must be finite and > 0")));
            }
            if !p.temperature_k.is_finite() || p.temperature_k <= 0.0 {
                return Err(bad(format!("point {i}: temperature must be finite and > 0 K")));
            }
            if !p.storage_pa.is_finite() || p.storage_pa <= 0.0 {
                return Err(bad(format!("point {i}: storage modulus must be finite and > 0")));
            }
            if !p.loss_pa.is_finite() || p.loss_pa < 0.0 {
                return Err(bad(format!("point {i}: loss modulus must be finite and >= 0")));
            }
        }
        // Strictly increasing (temperature, frequency) lexicographic order.
        for w in self.points.windows(2) {
            let (a, b) = (w[0], w[1]);
            let ordered = b.temperature_k > a.temperature_k
                || (b.temperature_k == a.temperature_k && b.frequency_hz > a.frequency_hz);
            if !ordered {
                return Err(bad(
                    "points must be sorted by (temperature_k, frequency_hz) with strictly increasing frequency per temperature".into(),
                ));
            }
        }
        Ok(())
    }

    /// Distinct sorted temperatures present in the table.
    pub fn temperatures(&self) -> Vec<f64> {
        let mut out: Vec<f64> = Vec::new();
        for p in &self.points {
            if out.last() != Some(&p.temperature_k) {
                out.push(p.temperature_k);
            }
        }
        out
    }

    /// Interpolate `E*` at `(frequency_hz, temperature_k)`.
    ///
    /// Picks the nearest temperature slice, then log-frequency linear
    /// interpolation. Allocation-free. Returns
    /// [`MaterialError::OutOfDomain`] outside the tabulated range.
    pub fn interpolate(
        &self,
        material_id: &str,
        frequency_hz: f64,
        temperature_k: f64,
    ) -> Result<Complex64, MaterialError> {
        let ood = |detail: String| MaterialError::OutOfDomain {
            material_id: material_id.to_owned(),
            detail,
        };
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(ood(format!("frequency {frequency_hz} must be finite and > 0")));
        }
        if !temperature_k.is_finite() || temperature_k <= 0.0 {
            return Err(ood(format!("temperature {temperature_k} must be finite and > 0 K")));
        }
        // Nearest temperature slice (linear scan over distinct temps — small).
        let temps = self.temperatures();
        let mut best = temps[0];
        let mut best_d = (temperature_k - best).abs();
        for &t in &temps[1..] {
            let d = (temperature_k - t).abs();
            if d < best_d {
                best = t;
                best_d = d;
            }
        }
        // Slice bounds without allocation.
        let start = self.points.iter().position(|p| p.temperature_k == best).unwrap_or(0);
        let end = self
            .points
            .iter()
            .rposition(|p| p.temperature_k == best)
            .map_or(start, |i| i + 1);
        let slice = &self.points[start..end];
        if slice.len() < 2 {
            return Err(ood(format!(
                "temperature slice at {best} K has fewer than 2 points"
            )));
        }
        let f_lo = slice[0].frequency_hz;
        let f_hi = slice[slice.len() - 1].frequency_hz;
        if frequency_hz < f_lo || frequency_hz > f_hi {
            return Err(ood(format!(
                "frequency {frequency_hz} Hz outside tabulated [{f_lo}, {f_hi}] Hz at {best} K (no extrapolation)"
            )));
        }
        // Binary search for bracketing pair (hot path, no alloc).
        let idx = slice.partition_point(|p| p.frequency_hz <= frequency_hz);
        if idx == 0 {
            let p = slice[0];
            return Ok(Complex64::new(p.storage_pa, p.loss_pa));
        }
        if idx >= slice.len() {
            let p = slice[slice.len() - 1];
            return Ok(Complex64::new(p.storage_pa, p.loss_pa));
        }
        let (a, b) = (slice[idx - 1], slice[idx]);
        if (frequency_hz - a.frequency_hz).abs() < f64::EPSILON {
            return Ok(Complex64::new(a.storage_pa, a.loss_pa));
        }
        let t = (frequency_hz.log10() - a.frequency_hz.log10())
            / (b.frequency_hz.log10() - a.frequency_hz.log10());
        Ok(Complex64::new(
            a.storage_pa + t * (b.storage_pa - a.storage_pa),
            a.loss_pa + t * (b.loss_pa - a.loss_pa),
        ))
    }
}

/// One `(B, H)` point of a magnetic B–H curve.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BhPoint {
    /// Flux density B (T).
    pub b_t: f64,
    /// Field strength H (A/m).
    pub h_a_per_m: f64,
}

/// First-quadrant monotone B–H data, mirroring `sonium-fem` `BhCurve`
/// semantics: starts at `(0, 0)`, both coordinates strictly increasing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BhCurveData {
    /// Curve points.
    pub points: Vec<BhPoint>,
}

impl BhCurveData {
    /// Validate exactly as `sonium_fem` `BhCurve::new` does.
    pub fn validate(&self, material_id: &str) -> Result<(), MaterialError> {
        let bad = |detail: String| MaterialError::InvalidBhCurve {
            material_id: material_id.to_owned(),
            detail,
        };
        if self.points.len() < 2 {
            return Err(bad("a B-H curve requires at least two points".into()));
        }
        if self.points[0].b_t != 0.0 || self.points[0].h_a_per_m != 0.0 {
            return Err(bad("a B-H curve must start at (0, 0)".into()));
        }
        for (i, p) in self.points.iter().enumerate() {
            if !p.b_t.is_finite() || !p.h_a_per_m.is_finite() || p.b_t < 0.0 || p.h_a_per_m < 0.0 {
                return Err(bad(format!("B-H point {i} is non-finite or outside the first quadrant")));
            }
        }
        for i in 1..self.points.len() {
            if self.points[i].b_t <= self.points[i - 1].b_t {
                return Err(bad(format!("flux density must increase strictly at B-H point {i}")));
            }
            if self.points[i].h_a_per_m <= self.points[i - 1].h_a_per_m {
                return Err(bad(format!("field strength must increase strictly at B-H point {i}")));
            }
        }
        Ok(())
    }

    fn segment(&self, b: f64) -> usize {
        let ins = self.points.partition_point(|p| p.b_t <= b);
        ins.saturating_sub(1).min(self.points.len() - 2)
    }

    fn segment_slope(&self, seg: usize) -> f64 {
        let (a, b) = (self.points[seg], self.points[seg + 1]);
        (b.h_a_per_m - a.h_a_per_m) / (b.b_t - a.b_t)
    }

    /// Field magnitude `H(B)` (piecewise-linear, `|B|` symmetric).
    pub fn field_strength(&self, flux_density_t: f64) -> f64 {
        let b = flux_density_t.abs();
        let seg = self.segment(b);
        let p0 = self.points[seg];
        p0.h_a_per_m + self.segment_slope(seg) * (b - p0.b_t)
    }

    /// Secant reluctivity `ν = H/B` (origin defined by the first slope).
    pub fn reluctivity(&self, flux_density_t: f64) -> f64 {
        let b = flux_density_t.abs();
        if b <= f64::EPSILON {
            self.segment_slope(0)
        } else {
            self.field_strength(b) / b
        }
    }

    /// Differential reluctivity `dH/dB` (Newton tangent).
    pub fn differential_reluctivity(&self, flux_density_t: f64) -> f64 {
        self.segment_slope(self.segment(flux_density_t.abs()))
    }
}
