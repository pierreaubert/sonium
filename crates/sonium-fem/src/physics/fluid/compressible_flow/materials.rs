//! Ideal-gas material laws for compressible flow.

/// Calorically perfect gas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IdealGas {
    pub gamma: f64,
    pub gas_constant: f64,
}

impl IdealGas {
    pub const AIR: Self = Self {
        gamma: 1.4,
        gas_constant: 287.05,
    };

    pub fn pressure(&self, density: f64, internal_energy_per_mass: f64) -> f64 {
        (self.gamma - 1.0) * density * internal_energy_per_mass
    }

    pub fn speed_of_sound(&self, pressure: f64, density: f64) -> f64 {
        (self.gamma * pressure / density).sqrt()
    }

    pub fn temperature(&self, pressure: f64, density: f64) -> f64 {
        pressure / (density * self.gas_constant)
    }

    pub fn specific_heat_cp(&self) -> f64 {
        self.gamma * self.gas_constant / (self.gamma - 1.0)
    }
}

/// Dynamic-viscosity law used by the compressible Navier--Stokes DG solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ViscosityModel {
    Constant(f64),
    Sutherland {
        reference_viscosity: f64,
        reference_temperature: f64,
        sutherland_temperature: f64,
    },
}

impl ViscosityModel {
    pub const AIR: Self = Self::Sutherland {
        reference_viscosity: 1.716e-5,
        reference_temperature: 273.15,
        sutherland_temperature: 110.4,
    };

    pub fn dynamic_viscosity(self, temperature: f64) -> f64 {
        match self {
            Self::Constant(viscosity) => viscosity,
            Self::Sutherland {
                reference_viscosity,
                reference_temperature,
                sutherland_temperature,
            } => {
                let temperature = temperature.max(f64::MIN_POSITIVE);
                reference_viscosity
                    * (temperature / reference_temperature).powf(1.5)
                    * (reference_temperature + sutherland_temperature)
                    / (temperature + sutherland_temperature)
            }
        }
    }

    pub fn thermal_conductivity(self, temperature: f64, gas: IdealGas, prandtl: f64) -> f64 {
        self.dynamic_viscosity(temperature) * gas.specific_heat_cp() / prandtl
    }

    pub fn is_valid(self) -> bool {
        match self {
            Self::Constant(viscosity) => viscosity.is_finite() && viscosity > 0.0,
            Self::Sutherland {
                reference_viscosity,
                reference_temperature,
                sutherland_temperature,
            } => {
                reference_viscosity.is_finite()
                    && reference_viscosity > 0.0
                    && reference_temperature.is_finite()
                    && reference_temperature > 0.0
                    && sutherland_temperature.is_finite()
                    && sutherland_temperature >= 0.0
            }
        }
    }
}

/// Conservative Euler state `[rho, rho*u, rho*v, rho*w, rho*E]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConservativeState {
    pub density: f64,
    pub momentum: [f64; 3],
    pub total_energy: f64,
}

impl ConservativeState {
    pub fn from_primitive(density: f64, velocity: [f64; 3], pressure: f64, gas: IdealGas) -> Self {
        let kinetic = 0.5 * density * velocity.iter().map(|u| u * u).sum::<f64>();
        let internal = pressure / (gas.gamma - 1.0);
        Self {
            density,
            momentum: [
                density * velocity[0],
                density * velocity[1],
                density * velocity[2],
            ],
            total_energy: internal + kinetic,
        }
    }

    pub fn velocity(&self) -> [f64; 3] {
        [
            self.momentum[0] / self.density,
            self.momentum[1] / self.density,
            self.momentum[2] / self.density,
        ]
    }

    pub fn kinetic_energy_per_volume(&self) -> f64 {
        let m2 = self.momentum.iter().map(|m| m * m).sum::<f64>();
        0.5 * m2 / self.density
    }

    pub fn pressure(&self, gas: IdealGas) -> f64 {
        (gas.gamma - 1.0) * (self.total_energy - self.kinetic_energy_per_volume())
    }

    pub fn euler_flux(&self, gas: IdealGas, normal: [f64; 3]) -> [f64; 5] {
        let velocity = self.velocity();
        let un = velocity
            .iter()
            .zip(normal.iter())
            .map(|(u, n)| u * n)
            .sum::<f64>();
        let p = self.pressure(gas);
        [
            self.density * un,
            self.momentum[0] * un + p * normal[0],
            self.momentum[1] * un + p * normal[1],
            self.momentum[2] * un + p * normal[2],
            (self.total_energy + p) * un,
        ]
    }

    pub fn max_wave_speed(&self, gas: IdealGas, normal: [f64; 3]) -> f64 {
        let velocity = self.velocity();
        let un = velocity
            .iter()
            .zip(normal.iter())
            .map(|(u, n)| u * n)
            .sum::<f64>();
        un.abs() + gas.speed_of_sound(self.pressure(gas), self.density)
    }

    pub fn as_array(self) -> [f64; 5] {
        [
            self.density,
            self.momentum[0],
            self.momentum[1],
            self.momentum[2],
            self.total_energy,
        ]
    }

    pub fn from_array(values: [f64; 5]) -> Self {
        Self {
            density: values[0],
            momentum: [values[1], values[2], values[3]],
            total_energy: values[4],
        }
    }

    pub fn is_physical(self, gas: IdealGas, density_floor: f64, pressure_floor: f64) -> bool {
        self.density.is_finite()
            && self.total_energy.is_finite()
            && self.momentum.iter().all(|value| value.is_finite())
            && self.density >= density_floor
            && self.pressure(gas).is_finite()
            && self.pressure(gas) >= pressure_floor
    }
}

pub fn rusanov_flux(
    left: ConservativeState,
    right: ConservativeState,
    gas: IdealGas,
    normal: [f64; 3],
) -> [f64; 5] {
    let fl = left.euler_flux(gas, normal);
    let fr = right.euler_flux(gas, normal);
    let ul = [
        left.density,
        left.momentum[0],
        left.momentum[1],
        left.momentum[2],
        left.total_energy,
    ];
    let ur = [
        right.density,
        right.momentum[0],
        right.momentum[1],
        right.momentum[2],
        right.total_energy,
    ];
    let smax = left
        .max_wave_speed(gas, normal)
        .max(right.max_wave_speed(gas, normal));
    let mut flux = [0.0; 5];
    for i in 0..5 {
        flux[i] = 0.5 * (fl[i] + fr[i]) - 0.5 * smax * (ur[i] - ul[i]);
    }
    flux
}

/// HLLC numerical flux (Toro, *Riemann Solvers*, Ch. 10) with Einfeldt
/// wave-speed estimates. Restores the contact wave that the Rusanov flux
/// dissipates at the sound speed. Returns `None` — signalling a Rusanov
/// fallback — whenever a wave-speed denominator degenerates or a
/// star-region density/pressure would be nonphysical.
pub fn hllc_flux(
    left: ConservativeState,
    right: ConservativeState,
    gas: IdealGas,
    normal: [f64; 3],
) -> Option<[f64; 5]> {
    // (density, velocity, normal velocity, pressure, sound speed, enthalpy)
    let unpack = |state: ConservativeState| {
        let velocity = state.velocity();
        let pressure = state.pressure(gas);
        let un: f64 = velocity.iter().zip(normal.iter()).map(|(u, n)| u * n).sum();
        let sound = gas.speed_of_sound(pressure, state.density);
        let enthalpy = (state.total_energy + pressure) / state.density;
        (state.density, velocity, un, pressure, sound, enthalpy)
    };
    let (rho_l, u_l, un_l, p_l, c_l, h_l) = unpack(left);
    let (rho_r, u_r, un_r, p_r, c_r, h_r) = unpack(right);
    if !rho_l.is_finite() || !rho_r.is_finite() || rho_l <= 0.0 || rho_r <= 0.0 {
        return None;
    }
    // Einfeldt (Roe-averaged) wave-speed estimates.
    let (sqrt_l, sqrt_r) = (rho_l.sqrt(), rho_r.sqrt());
    let denom = sqrt_l + sqrt_r;
    let u_tilde: [f64; 3] = std::array::from_fn(|i| (sqrt_l * u_l[i] + sqrt_r * u_r[i]) / denom);
    let h_tilde = (sqrt_l * h_l + sqrt_r * h_r) / denom;
    let kinetic_tilde: f64 = u_tilde.iter().map(|u| u * u).sum::<f64>() * 0.5;
    let c_tilde_sq = (gas.gamma - 1.0) * (h_tilde - kinetic_tilde);
    if !c_tilde_sq.is_finite() || c_tilde_sq <= 0.0 {
        return None;
    }
    let c_tilde = c_tilde_sq.sqrt();
    let un_tilde: f64 = u_tilde.iter().zip(normal.iter()).map(|(u, n)| u * n).sum();
    let s_l = (un_l - c_l).min(un_tilde - c_tilde);
    let s_r = (un_r + c_r).max(un_tilde + c_tilde);
    // Contact (middle) wave speed.
    let numerator = p_r - p_l + rho_l * un_l * (s_l - un_l) - rho_r * un_r * (s_r - un_r);
    let denominator = rho_l * (s_l - un_l) - rho_r * (s_r - un_r);
    if denominator.abs() < 1.0e-300 {
        return None;
    }
    let s_m = numerator / denominator;
    let star_state = |state: ConservativeState,
                      rho: f64,
                      velocity: [f64; 3],
                      un: f64,
                      pressure: f64,
                      s_k: f64|
     -> Option<ConservativeState> {
        let factor = rho * (s_k - un) / (s_k - s_m);
        if !factor.is_finite() || factor <= 0.0 {
            return None;
        }
        let p_star = pressure + rho * (s_k - un) * (s_m - un);
        if !p_star.is_finite() || p_star <= 0.0 {
            return None;
        }
        let mut velocity_star = velocity;
        for (component, n) in velocity_star.iter_mut().zip(normal.iter()) {
            *component += (s_m - un) * n;
        }
        let energy_star = factor
            * (state.total_energy / rho + (s_m - un) * (s_m + pressure / (rho * (s_k - un))));
        let candidate = ConservativeState {
            density: factor,
            momentum: [
                factor * velocity_star[0],
                factor * velocity_star[1],
                factor * velocity_star[2],
            ],
            total_energy: energy_star,
        };
        if candidate.is_physical(gas, 1.0e-14, 1.0e-14) {
            Some(candidate)
        } else {
            None
        }
    };
    let flux_l = left.euler_flux(gas, normal);
    let flux_r = right.euler_flux(gas, normal);
    let add_scaled_flux =
        |flux: [f64; 5], star: ConservativeState, base: ConservativeState, s: f64| {
            let star = star.as_array();
            let base = base.as_array();
            std::array::from_fn(|i| flux[i] + s * (star[i] - base[i]))
        };
    if s_l >= 0.0 {
        Some(flux_l)
    } else if s_m >= 0.0 {
        let star = star_state(left, rho_l, u_l, un_l, p_l, s_l)?;
        Some(add_scaled_flux(flux_l, star, left, s_l))
    } else if s_r >= 0.0 {
        let star = star_state(right, rho_r, u_r, un_r, p_r, s_r)?;
        Some(add_scaled_flux(flux_r, star, right, s_r))
    } else {
        Some(flux_r)
    }
}

pub fn local_lax_friedrichs_speed(
    left: ConservativeState,
    right: ConservativeState,
    gas: IdealGas,
    normal: [f64; 3],
) -> f64 {
    left.max_wave_speed(gas, normal)
        .max(right.max_wave_speed(gas, normal))
}

pub fn isentropic_area_ratio(mach: f64, gamma: f64) -> f64 {
    let m = mach.max(1.0e-12);
    let factor = (2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * m * m);
    (1.0 / m) * factor.powf((gamma + 1.0) / (2.0 * (gamma - 1.0)))
}

pub fn sod_shock_tube_states(gas: IdealGas) -> (ConservativeState, ConservativeState) {
    (
        ConservativeState::from_primitive(1.0, [0.0, 0.0, 0.0], 1.0, gas),
        ConservativeState::from_primitive(0.125, [0.0, 0.0, 0.0], 0.1, gas),
    )
}

#[cfg(test)]
mod hllc_tests {
    use super::*;

    #[test]
    fn hllc_matches_physical_flux_for_equal_states() {
        let gas = IdealGas::AIR;
        let state = ConservativeState::from_primitive(1.2, [30.0, -5.0, 2.0], 101_325.0, gas);
        let numerical = hllc_flux(state, state, gas, [1.0, 0.0, 0.0]).expect("HLLC flux");
        let physical = state.euler_flux(gas, [1.0, 0.0, 0.0]);
        for i in 0..5 {
            assert!(
                (numerical[i] - physical[i]).abs() < 1.0e-6,
                "component {i}: {} vs {}",
                numerical[i],
                physical[i]
            );
        }
    }

    #[test]
    fn hllc_preserves_stationary_contact_where_rusanov_does_not() {
        let gas = IdealGas {
            gamma: 1.4,
            gas_constant: 1.0,
        };
        let left = ConservativeState::from_primitive(1.0, [0.0; 3], 1.0, gas);
        let right = ConservativeState::from_primitive(0.4, [0.0; 3], 1.0, gas);
        let hllc = hllc_flux(left, right, gas, [1.0, 0.0, 0.0]).expect("HLLC contact flux");
        let rusanov = rusanov_flux(left, right, gas, [1.0, 0.0, 0.0]);
        assert!(
            hllc[0].abs() < 1.0e-12,
            "stationary contact must have zero HLLC mass flux, got {}",
            hllc[0]
        );
        assert!(
            rusanov[0].abs() > 0.1,
            "Rusanov should visibly diffuse the same contact, got {}",
            rusanov[0]
        );
    }
}

#[cfg(test)]
mod transport_tests {
    use super::*;

    #[test]
    fn sutherland_air_matches_reference_and_increases_with_temperature() {
        let law = ViscosityModel::AIR;
        let reference = law.dynamic_viscosity(273.15);
        assert!((reference - 1.716e-5).abs() < 1.0e-12);
        assert!(law.dynamic_viscosity(500.0) > reference);
        assert!(law.thermal_conductivity(300.0, IdealGas::AIR, 0.72) > 0.0);
    }
}
