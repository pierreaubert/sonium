//! Irreversible magnet demagnetisation: the per-element remanence
//! ratchet of the reference solver (`simulation/demag.py`).
//!
//! The state is a per-element remanence factor that only ever falls.
//! Reading the material curve at the element's present `H` is wrong
//! for a magnet that demagnetises itself (`H` is produced mostly by
//! the element's own magnetisation), so the operating point is the
//! classic graphical one: where the element's load line crosses the
//! material's intrinsic curve. Only the irreversible part is a loss:
//! the recoil line is followed from the operating point back to
//! `H = 0`, so an element that never left the reversible branch
//! keeps its full strength.
//!
//! The `H` the rule reads is inverted out of `B` through the same
//! constitutive law the field was solved under,
//! `B = mu0 * mu_rec * H + Br_vec * br`, projected on the
//! magnetisation direction. The rule must only ever see a converged
//! field: an intermediate Newton iterate has not built up the full
//! armature reaction, and a monotone rule shown a numerical transient
//! burns it in permanently.

// Rust guideline compliant 2026-02-21

use super::super::magnetostatics::MU_0;

/// Convergence tolerance on the per-element remanence factor: the
/// magnet counts as settled once no element still moves by more.
pub const DEMAG_TOL: f64 = 1.0e-4;

/// A magnet's second-quadrant demagnetisation curve and recoil data.
#[derive(Debug, Clone)]
pub struct MagnetDemagCurve {
    /// Field strengths `H` in A/m, sorted ascending.
    hs: Vec<f64>,
    /// Intrinsic polarisation `J = B - mu0 * H` in tesla, paired with `hs`.
    js: Vec<f64>,
    /// Declared recoil relative permeability the field solve used.
    mu_rec: f64,
    /// Recoil relative permeability from the curve's own top segment.
    mu_rec_curve: f64,
    /// Full-strength remanence `|B_r|` in tesla.
    br0_t: f64,
}

impl MagnetDemagCurve {
    /// Builds the curve from `(H, B)` table points (any order).
    ///
    /// # Errors
    ///
    /// Returns [`DemagError`] for fewer than two points, non-finite
    /// data, or a non-positive recoil permeability / remanence.
    pub fn new(points_h_b: &[[f64; 2]], mu_rec: f64, br0_t: f64) -> Result<Self, DemagError> {
        if points_h_b.len() < 2 {
            return Err(DemagError::TooFewPoints);
        }
        if !mu_rec.is_finite() || mu_rec <= 0.0 || !br0_t.is_finite() || br0_t <= 0.0 {
            return Err(DemagError::InvalidMaterial);
        }
        let mut points: Vec<[f64; 2]> = points_h_b.to_vec();
        if points.iter().flatten().any(|value| !value.is_finite()) {
            return Err(DemagError::InvalidMaterial);
        }
        points.sort_by(|a, b| a[0].total_cmp(&b[0]));
        let hs: Vec<f64> = points.iter().map(|point| point[0]).collect();
        let bs: Vec<f64> = points.iter().map(|point| point[1]).collect();
        let js: Vec<f64> = hs.iter().zip(&bs).map(|(h, b)| b - MU_0 * h).collect();
        // Recoil slope from the curve's own top segment: using the
        // declared mu_rec here instead would book the mismatch as
        // spurious permanent loss on healthy elements.
        let last = hs.len() - 1;
        let dh = (hs[last] - hs[last - 1]).abs().max(1.0e-9);
        let mu_rec_curve = (bs[last] - bs[last - 1]).abs() / dh / MU_0;
        Ok(Self {
            hs,
            js,
            mu_rec,
            mu_rec_curve,
            br0_t,
        })
    }

    /// Declared recoil relative permeability.
    pub fn mu_rec(&self) -> f64 {
        self.mu_rec
    }

    /// Full-strength remanence in tesla.
    pub fn br0_t(&self) -> f64 {
        self.br0_t
    }

    /// Where the load line `J = k * H` first crosses the intrinsic
    /// curve, walking segments from `H = 0` downwards. An element
    /// whose line never crosses stays on the reversible branch,
    /// `(0, Br0)`.
    fn cross(&self, k: f64) -> (f64, f64) {
        for i in (0..self.hs.len() - 1).rev() {
            let (h0, h1) = (self.hs[i], self.hs[i + 1]);
            if h1 <= h0 {
                continue;
            }
            let slope = (self.js[i + 1] - self.js[i]) / (h1 - h0);
            let denom = slope - k;
            if denom.abs() <= 1.0e-30 {
                continue;
            }
            let hc = (slope * h0 - self.js[i]) / denom;
            if hc.is_finite() && hc >= h0 - 1.0e-6 && hc <= h1 + 1.0e-6 {
                return (hc, k * hc);
            }
        }
        (0.0, self.br0_t)
    }

    /// New remanence factor for one element at element-mean `B`,
    /// given magnetisation unit vector `m_hat` and present factor.
    /// The result never exceeds the present factor.
    pub fn updated_factor(&self, b_t: [f64; 2], m_hat: [f64; 2], current: f64) -> f64 {
        let b_along = b_t[0] * m_hat[0] + b_t[1] * m_hat[1];
        // Invert the solver's own magnet law along M.
        let h = (b_along - self.br0_t * current) / (MU_0 * self.mu_rec);
        // Intrinsic polarisation now; the recoil term matters once
        // mu_rec > 1 (about 2.4 % of Br at a knee-deep -500 kA/m).
        let j_now = self.br0_t * current + MU_0 * (self.mu_rec - 1.0) * h;
        // H >= 0: not being demagnetised; the clip makes the load
        // line vertical, crossing at H = 0 (full strength).
        let k = j_now / h.min(-1.0e-9);
        let (h_op, j_op) = self.cross(k);
        // Recoil intercept at H = 0: the irreversible part alone.
        let br_new = j_op - (self.mu_rec_curve - 1.0) * MU_0 * h_op;
        current.min((br_new / self.br0_t.max(1.0e-12)).clamp(0.0, 1.0))
    }
}

/// Applies one converged field to the per-element factors.
///
/// `factors` covers every mesh element; `magnet_elements` lists
/// `(element_index, m_hat)` for the elements of ONE magnet material
/// (call once per material when several share the mesh). Returns
/// true when any element weakened by more than [`DEMAG_TOL`].
pub fn apply_demag_update(
    curve: &MagnetDemagCurve,
    element_mean_b_t: &[[f64; 2]],
    magnet_elements: &[(usize, [f64; 2])],
    factors: &mut [f64],
) -> bool {
    let mut moved = false;
    for &(element, m_hat) in magnet_elements {
        let current = factors[element];
        let new = curve.updated_factor(element_mean_b_t[element], m_hat, current);
        if new < current - DEMAG_TOL {
            moved = true;
        }
        factors[element] = new;
    }
    moved
}

/// Invalid demagnetisation-curve data.
#[derive(Debug, thiserror::Error)]
pub enum DemagError {
    /// Fewer than two curve points.
    #[error("a demagnetisation curve needs at least two points")]
    TooFewPoints,
    /// Non-finite point, or non-positive recoil data.
    #[error("demagnetisation material data must be finite and positive")]
    InvalidMaterial,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-checkable two-segment intrinsic curve: reversible top
    /// segment of slope mu0*(mu_rec_c - 1) in J, then a knee dropping
    /// steeply. Points are (H, B); J = B - mu0 H.
    fn curve() -> MagnetDemagCurve {
        // Br0 = 1.2 T at H=0 (B=J). Top segment H in [-200k, 0]:
        // B = 1.2 + mu0*1.05*H (declared mu_rec 1.05). Knee segment
        // from (-200k, B_top) down to (-800k, -0.9) (steep in J).
        let b_top = 1.2 + MU_0 * 1.05 * -200_000.0;
        MagnetDemagCurve::new(
            &[[-800_000.0, -0.9], [-200_000.0, b_top], [0.0, 1.2]],
            1.05,
            1.2,
        )
        .expect("valid curve")
    }

    #[test]
    fn reversible_excursion_keeps_full_strength() {
        let curve = curve();
        // Operating point on the reversible segment: choose B so the
        // inverted H is -100 kA/m at full strength:
        // B = mu0*mu_rec*H + Br0 = 1.2 - 0.132 = 1.068.
        let b = 1.2 + MU_0 * 1.05 * -100_000.0;
        let factor = curve.updated_factor([b, 0.0], [1.0, 0.0], 1.0);
        assert!(
            (factor - 1.0).abs() < 1.0e-9,
            "reversible element derated to {factor}"
        );
    }

    #[test]
    fn positive_field_never_derates() {
        let curve = curve();
        let factor = curve.updated_factor([1.5, 0.0], [1.0, 0.0], 1.0);
        assert_eq!(factor, 1.0);
    }

    #[test]
    fn knee_crossing_derates_irreversibly_and_monotonically() {
        let curve = curve();
        // Deep demagnetising field: inverted H = -600 kA/m.
        let b = 1.2 + MU_0 * 1.05 * -600_000.0;
        let factor = curve.updated_factor([b, 0.0], [1.0, 0.0], 1.0);
        assert!(factor < 0.9, "expected strong de-rating, got {factor}");
        assert!(factor > 0.0, "expected surviving remanence, got {factor}");
        // A later benign field cannot restore it.
        let after = curve.updated_factor([1.2, 0.0], [1.0, 0.0], factor);
        assert_eq!(after, factor);
    }

    #[test]
    fn apply_update_reports_motion() {
        let curve = curve();
        let b_deep = 1.2 + MU_0 * 1.05 * -600_000.0;
        let mean_b = vec![[b_deep, 0.0], [1.2, 0.0]];
        let mut factors = vec![1.0, 1.0];
        let moved = apply_demag_update(
            &curve,
            &mean_b,
            &[(0, [1.0, 0.0]), (1, [1.0, 0.0])],
            &mut factors,
        );
        assert!(moved);
        assert!(factors[0] < 1.0 && factors[1] == 1.0);
        // Second application of the same field: the weakened element
        // sees a milder inverted H (its own remanence fell), so the
        // ratchet does not run away.
        let settled = factors[0];
        apply_demag_update(&curve, &mean_b, &[(0, [1.0, 0.0])], &mut factors);
        assert!(factors[0] <= settled);
    }

    #[test]
    fn invalid_curves_are_rejected() {
        assert!(MagnetDemagCurve::new(&[[0.0, 1.2]], 1.05, 1.2).is_err());
        assert!(MagnetDemagCurve::new(&[[0.0, 1.2], [-1.0, 1.0]], 0.0, 1.2).is_err());
        assert!(MagnetDemagCurve::new(&[[0.0, 1.2], [-1.0, f64::NAN]], 1.05, 1.2).is_err());
    }
}
