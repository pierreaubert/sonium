//! Copper and iron losses from solved `A_z` field histories.
//!
//! Iron loss has two models, chosen per material record as in the
//! reference solver: the three-coefficient Bertotti separation, and
//! direct interpolation of a measured `P(B, f)` surface (log-log PCHIP,
//! blended into Bertotti outside the measured envelope) summed over the
//! harmonics of each element's flux locus. Lamination is handled in the
//! homogenised-field convention: the solve reports `B_homog = k_f *
//! B_steel`, so loss is evaluated at `B / k_f` and billed on the steel
//! volume `k_f * V_geom`.

// Rust guideline compliant 2026-02-21

/// Copper resistivity at 20 °C in Ω·m (reference-solver value).
pub const RHO_CU_20: f64 = 1.724e-8;
/// Copper resistivity temperature coefficient in 1/K.
pub const ALPHA_CU: f64 = 0.00393;

/// Bertotti coefficients in the reference solver's W/m³ conventions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BertottiCoefficients {
    /// Hysteresis coefficient, W/(m³·Hz·T²).
    pub kh: f64,
    /// Classical eddy coefficient, W/(m³·Hz²·T²).
    pub kc: f64,
    /// Excess coefficient, W/(m³·Hz^1.5·T^1.5).
    pub ke: f64,
}

impl BertottiCoefficients {
    /// Sinusoidal loss density `kh f B² + kc f² B² + ke f^1.5 B^1.5`.
    pub fn sinusoidal_w_per_m3(&self, b_peak_t: f64, frequency_hz: f64) -> f64 {
        let b = b_peak_t.abs();
        self.kh * frequency_hz * b * b
            + self.kc * frequency_hz * frequency_hz * b * b
            + self.ke * frequency_hz.powf(1.5) * b.powf(1.5)
    }
}

/// Monotone cubic Hermite (PCHIP) through `(x, y)` nodes.
///
/// Slopes follow SciPy's `PchipInterpolator`: Fritsch–Carlson weighted
/// harmonic means inside, one-sided three-point rules with
/// shape-preserving clamps at the ends. Outside the node range the
/// curve continues linearly with the end slope (a power law in the
/// log-log loss application), not with the end cubic.
#[derive(Debug, Clone)]
struct Pchip {
    x: Vec<f64>,
    y: Vec<f64>,
    d: Vec<f64>,
}

impl Pchip {
    fn new(x: Vec<f64>, y: Vec<f64>) -> Result<Self, LossError> {
        if x.len() != y.len() || x.len() < 2 {
            return Err(LossError::InvalidCurve);
        }
        if x.windows(2).any(|pair| pair[1] <= pair[0]) {
            return Err(LossError::InvalidCurve);
        }
        let n = x.len();
        let h: Vec<f64> = (0..n - 1).map(|i| x[i + 1] - x[i]).collect();
        let secant: Vec<f64> = (0..n - 1).map(|i| (y[i + 1] - y[i]) / h[i]).collect();
        let mut d = vec![0.0; n];
        d[0] = edge_slope(
            h[0],
            *h.get(1).unwrap_or(&h[0]),
            secant[0],
            *secant.get(1).unwrap_or(&secant[0]),
        );
        d[n - 1] = edge_slope(
            h[n - 2],
            *h.get(n - 3).unwrap_or(&h[n - 2]),
            secant[n - 2],
            *secant.get(n - 3).unwrap_or(&secant[n - 2]),
        );
        for i in 1..n - 1 {
            d[i] = interior_slope(h[i - 1], h[i], secant[i - 1], secant[i]);
        }
        Ok(Self { x, y, d })
    }

    fn eval(&self, xq: f64) -> f64 {
        let n = self.x.len();
        if xq <= self.x[0] {
            return self.y[0] + self.d[0] * (xq - self.x[0]);
        }
        if xq >= self.x[n - 1] {
            return self.y[n - 1] + self.d[n - 1] * (xq - self.x[n - 1]);
        }
        let i = self
            .x
            .partition_point(|x| *x <= xq)
            .saturating_sub(1)
            .min(n - 2);
        let h = self.x[i + 1] - self.x[i];
        let t = (xq - self.x[i]) / h;
        hermite(t, h, self.y[i], self.y[i + 1], self.d[i], self.d[i + 1])
    }
}

fn edge_slope(h0: f64, h1: f64, d0: f64, d1: f64) -> f64 {
    let mut slope = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
    if slope.signum() != d0.signum() {
        slope = 0.0;
    } else if d0.signum() != d1.signum() && slope.abs() > 3.0 * d0.abs() {
        slope = 3.0 * d0;
    }
    slope
}

fn interior_slope(h_prev: f64, h_next: f64, d_prev: f64, d_next: f64) -> f64 {
    if d_prev * d_next <= 0.0 {
        return 0.0;
    }
    let w1 = 2.0 * h_next + h_prev;
    let w2 = h_next + 2.0 * h_prev;
    (w1 + w2) / (w1 / d_prev + w2 / d_next)
}

fn hermite(t: f64, h: f64, y0: f64, y1: f64, d0: f64, d1: f64) -> f64 {
    let t2 = t * t;
    let t3 = t2 * t;
    (2.0 * t3 - 3.0 * t2 + 1.0) * y0
        + (t3 - 2.0 * t2 + t) * h * d0
        + (-2.0 * t3 + 3.0 * t2) * y1
        + (t3 - t2) * h * d1
}

fn smoothstep(u: f64) -> f64 {
    let u = u.clamp(0.0, 1.0);
    u * u * (3.0 - 2.0 * u)
}

/// Blend factor in induction: pure Bertotti at `1.35 * B_env(f)`.
const B_BLEND_FACTOR: f64 = 1.35;
/// Blend band in frequency: one octave beyond the table edge.
const F_BLEND_OCTAVES: f64 = 1.0;

/// Measured `P(B, f)` surface of one steel record, in W/m³ of steel.
///
/// Interpolation is PCHIP in `(ln B, ln P)` per measured frequency, then
/// PCHIP across `ln f` (local slopes, so four curves suffice at any
/// `f`). At every measured anchor the surface returns the table value
/// exactly. Outside the measured envelope the log-loss blends smoothly
/// into the record's Bertotti extrapolation.
#[derive(Debug, Clone)]
pub struct MeasuredLossSurface {
    frequencies_hz: Vec<f64>,
    ln_f: Vec<f64>,
    curves: Vec<Pchip>,
    envelope: Pchip,
    bertotti: BertottiCoefficients,
    /// Total measured points across all curves.
    pub n_points: usize,
}

impl MeasuredLossSurface {
    /// Builds a surface from measured curves.
    ///
    /// `curves[i]` holds `(B_t, P)` points at `frequencies_hz[i]`, with
    /// `P` in W/kg when `w_per_kg` is set (converted once with
    /// `density_kg_per_m3`), else W/m³.
    ///
    /// # Errors
    ///
    /// Returns [`LossError::InvalidCurve`] for fewer than three usable
    /// frequency curves or a curve with fewer than three positive points.
    pub fn new(
        frequencies_hz: &[f64],
        curves: &[Vec<(f64, f64)>],
        density_kg_per_m3: f64,
        w_per_kg: bool,
        bertotti: BertottiCoefficients,
    ) -> Result<Self, LossError> {
        if frequencies_hz.len() != curves.len() {
            return Err(LossError::InvalidCurve);
        }
        let scale = if w_per_kg { density_kg_per_m3 } else { 1.0 };
        let mut order: Vec<usize> = (0..frequencies_hz.len()).collect();
        order.sort_by(|&a, &b| frequencies_hz[a].total_cmp(&frequencies_hz[b]));
        let mut frequencies = Vec::new();
        let mut built = Vec::new();
        let mut b_max = Vec::new();
        let mut n_points = 0;
        for &index in &order {
            let frequency = frequencies_hz[index];
            if frequency <= 0.0 {
                continue;
            }
            let mut points: Vec<(f64, f64)> = curves[index]
                .iter()
                .filter(|point| point.0 > 0.0 && point.1 > 0.0)
                .map(|point| (point.0, point.1 * scale))
                .collect();
            points.sort_by(|a, b| a.0.total_cmp(&b.0));
            points.dedup_by(|next, prev| next.0 == prev.0);
            if points.len() < 3 {
                continue;
            }
            n_points += curves[index].len();
            let ln_b: Vec<f64> = points.iter().map(|point| point.0.ln()).collect();
            let ln_p: Vec<f64> = points.iter().map(|point| point.1.ln()).collect();
            b_max.push(points.last().expect("non-empty curve").0);
            built.push(Pchip::new(ln_b, ln_p)?);
            frequencies.push(frequency);
        }
        if frequencies.len() < 3 {
            return Err(LossError::InvalidCurve);
        }
        let ln_f: Vec<f64> = frequencies.iter().map(|f| f.ln()).collect();
        let envelope = Pchip::new(ln_f.clone(), b_max.iter().map(|b| b.ln()).collect())?;
        Ok(Self {
            frequencies_hz: frequencies,
            ln_f,
            curves: built,
            envelope,
            bertotti,
            n_points,
        })
    }

    /// Highest measured induction at `frequency_hz`, in tesla.
    pub fn envelope_b_t(&self, frequency_hz: f64) -> f64 {
        let ln_f = frequency_hz
            .max(1.0e-9)
            .ln()
            .clamp(self.ln_f[0], *self.ln_f.last().expect("non-empty surface"));
        self.envelope.eval(ln_f).exp()
    }

    /// 0 inside the measured envelope, 1 at the blend band's far edge.
    pub fn outsideness(&self, b_peak_t: f64, frequency_hz: f64) -> f64 {
        // 1e-12 of slack so a top-of-curve anchor reads as inside.
        let u_b = ((b_peak_t.max(1.0e-300).ln() - self.envelope_b_t(frequency_hz).ln() - 1.0e-12)
            .max(0.0))
            / B_BLEND_FACTOR.ln();
        let ln_f = frequency_hz.max(1.0e-9).ln();
        let last = *self.ln_f.last().expect("non-empty surface");
        let u_f =
            ((ln_f - last).max(self.ln_f[0] - ln_f).max(0.0)) / (F_BLEND_OCTAVES * 2.0f64.ln());
        u_b.hypot(u_f).min(1.0)
    }

    fn ln_surface(&self, ln_b: f64, frequency_hz: f64) -> f64 {
        let ln_f = frequency_hz.max(1.0e-9).ln();
        let n = self.frequencies_hz.len();
        if n == 1 {
            return self.curves[0].eval(ln_b);
        }
        let i = self
            .ln_f
            .partition_point(|value| *value <= ln_f)
            .saturating_sub(1)
            .min(n - 2);
        let lo = i.saturating_sub(1);
        let hi = (i + 2).min(n - 1);
        let value = |j: usize| self.curves[j].eval(ln_b);
        let secant = |j: usize| (value(j + 1) - value(j)) / (self.ln_f[j + 1] - self.ln_f[j]);
        let slope = |j: usize| -> f64 {
            if j == 0 {
                if n > 2 {
                    edge_slope(
                        self.ln_f[1] - self.ln_f[0],
                        self.ln_f[2] - self.ln_f[1],
                        secant(0),
                        secant(1),
                    )
                } else {
                    secant(0)
                }
            } else if j == n - 1 {
                if n > 2 {
                    edge_slope(
                        self.ln_f[n - 1] - self.ln_f[n - 2],
                        self.ln_f[n - 2] - self.ln_f[n - 3],
                        secant(n - 2),
                        secant(n - 3),
                    )
                } else {
                    secant(0)
                }
            } else {
                interior_slope(
                    self.ln_f[j] - self.ln_f[j - 1],
                    self.ln_f[j + 1] - self.ln_f[j],
                    secant(j - 1),
                    secant(j),
                )
            }
        };
        let _ = (lo, hi);
        let (d_i, d_j) = (slope(i), slope(i + 1));
        if ln_f < self.ln_f[0] {
            return value(0) + d_i * (ln_f - self.ln_f[0]);
        }
        if ln_f > self.ln_f[n - 1] {
            return value(n - 1) + d_j * (ln_f - self.ln_f[n - 1]);
        }
        let h = self.ln_f[i + 1] - self.ln_f[i];
        hermite(
            (ln_f - self.ln_f[i]) / h,
            h,
            value(i),
            value(i + 1),
            d_i,
            d_j,
        )
    }

    /// Sinusoidal loss density and its Bertotti blend weight `s`.
    ///
    /// Zero induction returns exactly zero loss. The blend weight lets
    /// callers account how many watts came from extrapolation.
    pub fn w_per_m3(&self, b_peak_t: f64, frequency_hz: f64) -> (f64, f64) {
        let b = b_peak_t.abs();
        if b <= 0.0 {
            return (0.0, 0.0);
        }
        let ln_surface = self.ln_surface(b.ln(), frequency_hz);
        let s = smoothstep(self.outsideness(b, frequency_hz));
        if s > 0.0 {
            let p_bertotti = self.bertotti.sinusoidal_w_per_m3(b, frequency_hz);
            if p_bertotti > 0.0 {
                return (((1.0 - s) * ln_surface + s * p_bertotti.ln()).exp(), s);
            }
        }
        (ln_surface.exp(), s)
    }
}

/// Steel loss record: Bertotti coefficients, lamination and surface.
#[derive(Debug, Clone)]
pub struct SteelLoss {
    /// Three-coefficient Bertotti fit (the fallback and blend target).
    pub bertotti: BertottiCoefficients,
    /// Lamination stacking factor `k_f` in `(0, 1]`.
    pub stacking_factor: f64,
    /// Measured surface, when the record opts in.
    pub surface: Option<MeasuredLossSurface>,
}

/// Peak harmonic amplitudes of one component history, per element.
///
/// `history` is frames × elements row-major over `n_periods` electrical
/// periods. Returns `(amplitudes, frequencies)` with amplitudes laid out
/// harmonics × elements (DC dropped): bin `m` (1-based) sits at
/// `m * f_elec / n_periods` and carries peak amplitude `2|C_m|` (`|C_m|`
/// at Nyquist). The window is used exactly as captured — no detrending.
pub fn harmonic_amplitudes(
    history: &[f64],
    frames: usize,
    elements: usize,
    f_elec_hz: f64,
    n_periods: f64,
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(history.len(), frames * elements);
    assert!(frames >= 2);
    let n_bins = frames / 2;
    let mut amplitudes = vec![0.0; n_bins * elements];
    let frequencies: Vec<f64> = (1..=n_bins)
        .map(|m| m as f64 * f_elec_hz / n_periods.max(1.0e-9))
        .collect();
    for element in 0..elements {
        for m in 1..=n_bins {
            let mut re = 0.0;
            let mut im = 0.0;
            for frame in 0..frames {
                let angle = 2.0 * std::f64::consts::PI * (m * frame) as f64 / frames as f64;
                let value = history[frame * elements + element];
                re += value * angle.cos();
                im -= value * angle.sin();
            }
            let magnitude = (re * re + im * im).sqrt() / frames as f64;
            let is_nyquist = frames.is_multiple_of(2) && m == n_bins;
            amplitudes[(m - 1) * elements + element] = if is_nyquist {
                magnitude
            } else {
                2.0 * magnitude
            };
        }
    }
    (amplitudes, frequencies)
}

/// Iron-loss result for one material group over a captured window.
#[derive(Debug, Clone)]
pub struct IronLossReport {
    /// Classical (eddy) loss per frame in watts — the rippling series.
    pub classical_series_w: Vec<f64>,
    /// Per-cycle remainder in watts (hysteresis + excess bucket).
    pub per_cycle_w: f64,
    /// Total iron loss in watts (series mean plus per-cycle bucket).
    pub total_w: f64,
    /// Hysteresis share of the total, in watts.
    pub hysteresis_w: f64,
    /// Mean classical eddy loss, in watts.
    pub eddy_w: f64,
    /// Excess share of the total, in watts.
    pub excess_w: f64,
    /// Per-element mean loss density in W/m³ of steel.
    pub density_w_per_m3_steel: Vec<f64>,
    /// Fraction of surface-path watts blended from extrapolation.
    pub envelope_out_frac: f64,
    /// Fundamental-only surface total (surface path only), in watts.
    pub fundamental_only_w: Option<f64>,
    /// What plain Bertotti would have said, in watts.
    pub bertotti_w: f64,
}

/// Iron loss from per-element `B(t)` histories of one steel group.
///
/// Histories are frames × elements row-major in the homogenised field;
/// `areas_m2` are the geometric element areas. The classical term uses
/// the periodic central difference of the full waveform; hysteresis and
/// excess use each component's peak-to-peak amplitude. When the steel
/// carries a measured surface, the total is replaced by the harmonic
/// surface sum and the classical series is kept as its shape (scaled
/// down if it alone would exceed the surface total).
///
/// # Errors
///
/// Returns [`LossError`] on shape mismatches or invalid parameters.
#[allow(clippy::too_many_arguments)]
pub fn iron_loss(
    history_bx_t: &[f64],
    history_by_t: &[f64],
    areas_m2: &[f64],
    steel: &SteelLoss,
    stack_length_m: f64,
    f_elec_hz: f64,
    n_periods: f64,
    dt_s: f64,
) -> Result<IronLossReport, LossError> {
    let elements = areas_m2.len();
    if elements == 0 || history_bx_t.len() != history_by_t.len() {
        return Err(LossError::ShapeMismatch);
    }
    if !history_bx_t.len().is_multiple_of(elements) {
        return Err(LossError::ShapeMismatch);
    }
    let frames = history_bx_t.len() / elements;
    if frames < 2
        || !stack_length_m.is_finite()
        || stack_length_m <= 0.0
        || !f_elec_hz.is_finite()
        || f_elec_hz < 0.0
        || !dt_s.is_finite()
        || dt_s <= 0.0
    {
        return Err(LossError::InvalidParameter);
    }
    let kf = if steel.stacking_factor > 0.0 {
        steel.stacking_factor
    } else {
        1.0
    };
    let inv_kf = 1.0 / kf;
    let vol_steel: Vec<f64> = areas_m2
        .iter()
        .map(|area| area * stack_length_m * kf)
        .collect();
    let (kh, kc, ke) = (steel.bertotti.kh, steel.bertotti.kc, steel.bertotti.ke);
    const TWO_PI_SQ: f64 = 2.0 * std::f64::consts::PI * std::f64::consts::PI;

    // Classical term: kc/(2 pi^2) * sum_e vol_e * |dB_steel/dt|^2 per frame.
    let mut classical = vec![0.0; frames];
    let mut classical_per_element = vec![0.0; elements];
    for element in 0..elements {
        let mut mean_density = 0.0;
        for (frame, classical_frame) in classical.iter_mut().enumerate() {
            let next = (frame + 1) % frames;
            let prev = (frame + frames - 1) % frames;
            let dbx = (history_bx_t[next * elements + element]
                - history_bx_t[prev * elements + element])
                / (2.0 * dt_s)
                * inv_kf;
            let dby = (history_by_t[next * elements + element]
                - history_by_t[prev * elements + element])
                / (2.0 * dt_s)
                * inv_kf;
            let density = kc / TWO_PI_SQ * (dbx * dbx + dby * dby);
            *classical_frame += density * vol_steel[element];
            mean_density += density;
        }
        classical_per_element[element] = mean_density / frames as f64;
    }
    let p_eddy = classical.iter().sum::<f64>() / frames as f64;

    // Per-cycle terms from peak-to-peak component amplitudes (in steel).
    let mut hyst = 0.0;
    let mut excess = 0.0;
    let mut per_cycle_per_element = vec![0.0; elements];
    for element in 0..elements {
        let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
        for frame in 0..frames {
            let bx = history_bx_t[frame * elements + element];
            let by = history_by_t[frame * elements + element];
            min_x = min_x.min(bx);
            max_x = max_x.max(bx);
            min_y = min_y.min(by);
            max_y = max_y.max(by);
        }
        let bac2 =
            (((max_x - min_x) * 0.5).powi(2) + ((max_y - min_y) * 0.5).powi(2)) * inv_kf * inv_kf;
        let density = kh * f_elec_hz * bac2 + ke * f_elec_hz.powf(1.5) * bac2.max(0.0).powf(0.75);
        per_cycle_per_element[element] = density;
        hyst += kh * f_elec_hz * bac2 * vol_steel[element];
        excess += ke * f_elec_hz.powf(1.5) * bac2.max(0.0).powf(0.75) * vol_steel[element];
    }
    let bertotti_w = p_eddy + hyst + excess;

    let (
        total_w,
        per_cycle_w,
        hysteresis_w,
        excess_w,
        eddy_w,
        mut density,
        envelope_out_frac,
        fundamental_only_w,
    ) = if let Some(surface) = &steel.surface {
        let mut surface_density = vec![0.0; elements];
        let mut fundamental_density = vec![0.0; elements];
        let (mut watts_all, mut watts_out) = (0.0, 0.0);
        for history in [history_bx_t, history_by_t] {
            let (amplitudes, frequencies) =
                harmonic_amplitudes(history, frames, elements, f_elec_hz, n_periods);
            for (bin, frequency) in frequencies.iter().enumerate() {
                for element in 0..elements {
                    let amplitude = amplitudes[bin * elements + element] * inv_kf;
                    if amplitude <= 0.0 {
                        continue;
                    }
                    let (value, blend) = surface.w_per_m3(amplitude, *frequency);
                    surface_density[element] += value;
                    if bin == 0 {
                        fundamental_density[element] += value;
                    }
                    let watts = value * vol_steel[element];
                    watts_all += watts;
                    watts_out += watts * blend;
                }
            }
        }
        let p_surf: f64 = surface_density
            .iter()
            .zip(&vol_steel)
            .map(|(density, volume)| density * volume)
            .sum();
        let p_fund: f64 = fundamental_density
            .iter()
            .zip(&vol_steel)
            .map(|(density, volume)| density * volume)
            .sum();
        let (rest, eddy) = if p_surf <= 0.0 {
            classical.iter_mut().for_each(|value| *value = 0.0);
            classical_per_element
                .iter_mut()
                .for_each(|value| *value = 0.0);
            (0.0, 0.0)
        } else if p_eddy > p_surf {
            let ratio = p_surf / p_eddy;
            classical.iter_mut().for_each(|value| *value *= ratio);
            classical_per_element
                .iter_mut()
                .for_each(|value| *value *= ratio);
            (0.0, p_surf)
        } else {
            (p_surf - p_eddy, p_eddy)
        };
        let share = if hyst + excess > 0.0 {
            hyst / (hyst + excess)
        } else {
            1.0
        };
        // Per-element total density: surface density already is the
        // element total; use it directly for thermal coupling.
        (
            eddy + rest,
            rest,
            rest * share,
            rest * (1.0 - share),
            eddy,
            surface_density,
            watts_out / watts_all.max(1.0e-30),
            Some(p_fund),
        )
    } else {
        let density: Vec<f64> = classical_per_element
            .iter()
            .zip(&per_cycle_per_element)
            .map(|(classical, per_cycle)| classical + per_cycle)
            .collect();
        (
            bertotti_w,
            hyst + excess,
            hyst,
            excess,
            p_eddy,
            density,
            0.0,
            None,
        )
    };
    let _ = &mut density;
    Ok(IronLossReport {
        classical_series_w: classical,
        per_cycle_w,
        total_w,
        hysteresis_w,
        eddy_w,
        excess_w,
        density_w_per_m3_steel: density,
        envelope_out_frac,
        fundamental_only_w,
        bertotti_w,
    })
}

/// End-winding length factor for a tooth-coil concentrated winding.
///
/// `k_end = (pi * (tooth_width + bundle_width) / 2 + L_stack) / L_stack`:
/// each end-turn is a half-loop whose centreline diameter is the tooth
/// plus one wire bundle. Returns 1 for a non-positive stack length.
pub fn end_winding_factor(stack_length_m: f64, tooth_width_m: f64, bundle_width_m: f64) -> f64 {
    if stack_length_m <= 0.0 {
        return 1.0;
    }
    let span = tooth_width_m + bundle_width_m.max(0.0);
    (std::f64::consts::PI * span / 2.0 + stack_length_m) / stack_length_m
}

/// DC copper loss, end factor and effective phase resistance.
///
/// `P = rho(T) * J^2 * V_cu * k_end` with `J` the branch-current density
/// in one strand. The strand section is the measured total copper area
/// shared over all conductors when `copper_area_m2` is supplied, else
/// the nominal wire rectangle. At zero current the loss is zero but
/// `k_end` and `R_phase` are still returned (computed on a 1 A probe):
/// a voltage-drive circuit launched at `I = 0` must not lose its
/// winding resistance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CopperLoss {
    /// Total three-phase DC copper loss in watts.
    pub power_w: f64,
    /// End-winding factor actually used.
    pub end_factor: f64,
    /// Effective phase resistance in ohms.
    pub r_phase_ohm: f64,
}

/// Inputs for [`copper_loss_dc`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CopperLossInput {
    /// Number of stator slots.
    pub num_slots: usize,
    /// Conductors per slot (wire rows × strips per row).
    pub conductors_per_slot: f64,
    /// Nominal strand cross-section in m².
    pub nominal_wire_area_m2: f64,
    /// Measured total copper cross-section in m², when known.
    pub measured_copper_area_m2: Option<f64>,
    /// Stack length in metres.
    pub stack_length_m: f64,
    /// Parallel paths per phase.
    pub n_parallel: f64,
    /// Phase current in A rms.
    pub i_phase_rms_a: f64,
    /// Coil temperature in °C.
    pub coil_temp_c: f64,
    /// End-winding factor; non-positive values mean "compute from
    /// `tooth_width_m`/`bundle_width_m`".
    pub end_factor: f64,
    /// Tooth width in metres (end-factor geometry).
    pub tooth_width_m: f64,
    /// Wire-bundle width in metres (end-factor geometry).
    pub bundle_width_m: f64,
}

/// Computes the DC copper loss triple `(P, k_end, R_phase)`.
pub fn copper_loss_dc(input: &CopperLossInput) -> CopperLoss {
    let n_conductors = input.num_slots as f64 * input.conductors_per_slot;
    let mut wire_area = input.nominal_wire_area_m2;
    if let Some(measured) = input.measured_copper_area_m2
        && measured > 0.0
        && n_conductors > 0.0
    {
        wire_area = measured / n_conductors;
    }
    if wire_area <= 0.0 {
        return CopperLoss {
            power_w: 0.0,
            end_factor: 1.0,
            r_phase_ohm: 0.0,
        };
    }
    let probe = if input.i_phase_rms_a > 0.0 {
        input.i_phase_rms_a
    } else {
        1.0
    };
    let volume =
        input.num_slots as f64 * wire_area * input.conductors_per_slot * input.stack_length_m;
    let k_end = if input.end_factor > 0.0 {
        input.end_factor
    } else {
        end_winding_factor(
            input.stack_length_m,
            input.tooth_width_m,
            input.bundle_width_m,
        )
    };
    let rho = RHO_CU_20 * (1.0 + ALPHA_CU * (input.coil_temp_c - 20.0));
    let branch_current = probe / input.n_parallel.max(1.0);
    let density = branch_current / wire_area;
    let power = rho * density * density * volume * k_end;
    CopperLoss {
        power_w: if input.i_phase_rms_a <= 0.0 {
            0.0
        } else {
            power
        },
        end_factor: k_end,
        r_phase_ohm: power / (3.0 * probe * probe),
    }
}

/// Invalid loss-model data or history shape.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LossError {
    /// A loss curve had too few or non-increasing points.
    #[error("loss curve needs at least 2-3 strictly increasing points")]
    InvalidCurve,
    /// History and area shapes disagreed.
    #[error("flux history shape disagrees with element areas")]
    ShapeMismatch,
    /// A scalar parameter was invalid.
    #[error("loss parameter must be finite and in range")]
    InvalidParameter,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn pchip_interpolates_nodes_and_is_monotone() {
        let curve = Pchip::new(vec![0.0, 1.0, 2.0, 3.0], vec![0.0, 1.0, 1.5, 4.0]).unwrap();
        for (x, y) in [(0.0, 0.0), (1.0, 1.0), (2.0, 1.5), (3.0, 4.0)] {
            assert!((curve.eval(x) - y).abs() < 1.0e-12);
        }
        let mut previous = f64::NEG_INFINITY;
        for step in 0..=100 {
            let value = curve.eval(3.0 * step as f64 / 100.0);
            assert!(value >= previous - 1.0e-12, "not monotone at {step}");
            previous = value;
        }
    }

    fn synthetic_surface() -> MeasuredLossSurface {
        // P = 100 * f * B^2 W/kg exactly on the anchors.
        let freqs = [50.0, 400.0, 1000.0];
        let curves: Vec<Vec<(f64, f64)>> = freqs
            .iter()
            .map(|f| {
                [0.1, 0.5, 1.0, 1.5]
                    .iter()
                    .map(|b| (*b, 100.0 * f * b * b / 7600.0))
                    .collect()
            })
            .collect();
        MeasuredLossSurface::new(
            &freqs,
            &curves,
            7600.0,
            true,
            BertottiCoefficients {
                kh: 100.0,
                kc: 0.0,
                ke: 0.0,
            },
        )
        .unwrap()
    }

    #[test]
    fn surface_returns_anchors_exactly() {
        let surface = synthetic_surface();
        let (value, blend) = surface.w_per_m3(1.0, 400.0);
        assert!((value - 100.0 * 400.0).abs() / 40000.0 < 1.0e-9, "{value}");
        assert_eq!(blend, 0.0);
        assert_eq!(surface.w_per_m3(0.0, 400.0), (0.0, 0.0));
    }

    #[test]
    fn harmonic_amplitudes_recover_two_tones() {
        let frames = 64;
        let history: Vec<f64> = (0..frames)
            .map(|k| {
                let phase = 2.0 * PI * k as f64 / frames as f64;
                1.2 * phase.cos() + 0.3 * (3.0 * phase + 0.5).sin() + 0.7
            })
            .collect();
        let (amplitudes, frequencies) = harmonic_amplitudes(&history, frames, 1, 100.0, 1.0);
        assert!((amplitudes[0] - 1.2).abs() < 1.0e-12);
        assert!((amplitudes[2] - 0.3).abs() < 1.0e-12);
        assert!((frequencies[2] - 300.0).abs() < 1.0e-12);
    }

    #[test]
    fn bertotti_sinusoid_matches_hand_value() {
        // Pure sinusoid B_x = B0 sin(2 pi f t), one element: classical
        // mean = kc/(2 pi^2) * <(dB/dt)^2> = kc f^2 B0^2 (the discrete
        // central difference reads slightly low; bound it).
        let frames = 720;
        let (b0, f) = (1.3, 250.0);
        let dt = 1.0 / (f * frames as f64);
        let bx: Vec<f64> = (0..frames)
            .map(|k| b0 * (2.0 * PI * k as f64 / frames as f64).sin())
            .collect();
        let by = vec![0.0; frames];
        let steel = SteelLoss {
            bertotti: BertottiCoefficients {
                kh: 120.0,
                kc: 0.05,
                ke: 1.8,
            },
            stacking_factor: 1.0,
            surface: None,
        };
        let report = iron_loss(&bx, &by, &[1.0e-4], &steel, 0.01, f, 1.0, dt).unwrap();
        let volume = 1.0e-6;
        let expected = volume
            * (120.0 * f * b0 * b0 + 0.05 * f * f * b0 * b0 + 1.8 * f.powf(1.5) * b0.powf(1.5));
        assert!(
            (report.total_w - expected).abs() / expected < 2.0e-3,
            "{} vs {expected}",
            report.total_w
        );
    }

    #[test]
    fn copper_loss_at_zero_current_keeps_resistance() {
        let input = CopperLossInput {
            num_slots: 12,
            conductors_per_slot: 6.0,
            nominal_wire_area_m2: 1.0e-6,
            measured_copper_area_m2: Some(72.0e-6),
            stack_length_m: 0.01,
            n_parallel: 1.0,
            i_phase_rms_a: 0.0,
            coil_temp_c: 120.0,
            end_factor: 0.0,
            tooth_width_m: 2.6e-3,
            bundle_width_m: 2.3e-3,
        };
        let at_zero = copper_loss_dc(&input);
        assert_eq!(at_zero.power_w, 0.0);
        assert!(at_zero.r_phase_ohm > 0.0);
        let at_60 = copper_loss_dc(&CopperLossInput {
            i_phase_rms_a: 60.0,
            ..input
        });
        assert!((at_60.r_phase_ohm - at_zero.r_phase_ohm).abs() < 1.0e-15);
        assert!((at_60.power_w - 3.0 * 3600.0 * at_60.r_phase_ohm).abs() < 1.0e-9);
    }
}
