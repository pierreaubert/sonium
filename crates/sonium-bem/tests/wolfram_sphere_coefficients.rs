//! Rigid-sphere Mie reference replayed from the Wolfram special-function
//! golden (`sonium-wave/fixtures/special-golden-wolfram-v1.json`).
//!
//! This is the BEM-side consumption of the independent oracle required by
//! audit-sonium-wave-20260921 ("independent goldens consumed by both wave
//! and BEM"): the scattering coefficients b_n = -j_n'(ka)/h_n'(ka) and the
//! kr-dependent Bessel values both come from Wolfram data, combined here
//! through the textbook partial-wave sum. The same truncation is applied to
//! `sphere_scattering_3d`, so any difference exposes an implementation or
//! reference divergence rather than series-tail effects.
//!
//! ka = pi exercises the first interior-resonance neighborhood of the unit
//! sphere; ka = pi/2 is the off-resonance control. Receiver radii are chosen
//! so every kr argument is tabulated in the fixture (kr = 2*pi and pi).
//!
//! The reference coefficients use a_n = j_n'(ka)/h_n'(ka) from the Neumann
//! condition, derived independently in the test body.

use num_complex::Complex64;
use serde::Deserialize;
use sonium_wave::analytical::sphere_scattering_3d;
use sonium_wave::special::legendre_polynomials;
use std::f64::consts::PI;

const WOLFRAM_FIXTURE: &str =
    include_str!("../../sonium-wave/fixtures/special-golden-wolfram-v1.json");
const NUM_TERMS: usize = 14;
const TOLERANCE: f64 = 1e-10;

#[derive(Debug, Deserialize)]
struct Fixture {
    schema: String,
    precision_digits: u32,
    spherical: Vec<SphericalCase>,
}

#[derive(Debug, Deserialize)]
struct SphericalCase {
    name: String,
    x: String,
    j: Vec<String>,
    y: Vec<String>,
    j_derivative: Vec<String>,
    y_derivative: Vec<String>,
}

fn parse_f64(value: &str) -> f64 {
    value.parse().expect("fixture value is a decimal")
}

fn case<'a>(fixture: &'a Fixture, name: &str) -> &'a SphericalCase {
    fixture
        .spherical
        .iter()
        .find(|case| case.name == name)
        .unwrap_or_else(|| panic!("wolfram fixture lacks case {name}"))
}

/// Rigid-sphere scattering coefficients a_n = j_n'(ka)/h_n'(ka) from the
/// Wolfram j/y/j'/y' values (independent formula and independent data).
/// Derived from the Neumann condition d/dr [j_n(kr) - a_n h_n(kr)] = 0 at
/// r = a for the incident expansion e^{ikz} = sum_n (2n+1) i^n j_n(kr) P_n.
fn wolfram_coefficients(ka: &SphericalCase) -> Vec<Complex64> {
    ka.j.iter()
        .zip(ka.y.iter())
        .zip(ka.j_derivative.iter())
        .zip(ka.y_derivative.iter())
        .take(NUM_TERMS)
        .map(|(((_, _), jd), yd)| {
            let (jd, yd) = (parse_f64(jd), parse_f64(yd));
            let hn_prime = Complex64::new(jd, yd);
            assert!(
                hn_prime.norm() > 0.0,
                "non-degenerate Hankel derivative at {}",
                ka.name
            );
            Complex64::new(jd, 0.0) / hn_prime
        })
        .collect()
}

/// Total field p = sum_n (2n+1) i^n [j_n(kr) - a_n h_n(kr)] P_n(cos theta)
/// with every Bessel value taken from the Wolfram fixture.
fn wolfram_field(
    coefficients: &[Complex64],
    kr_case: &SphericalCase,
    cos_theta: f64,
    num_terms: usize,
) -> Complex64 {
    let legendre = legendre_polynomials(num_terms, cos_theta);
    let mut total = Complex64::new(0.0, 0.0);
    for n in 0..num_terms {
        let n_f64 = n as f64;
        let prefactor = 2.0 * n_f64 + 1.0;
        let i_power_n = Complex64::new((n_f64 * PI / 2.0).cos(), (n_f64 * PI / 2.0).sin());
        let jn = parse_f64(&kr_case.j[n]);
        let hn = Complex64::new(jn, parse_f64(&kr_case.y[n]));
        total +=
            prefactor * i_power_n * (Complex64::new(jn, 0.0) - coefficients[n] * hn) * legendre[n];
    }
    total
}

fn check_ka(ka_name: &str, kr_name: &str, wave_number: f64, radius: f64) {
    let fixture: Fixture = serde_json::from_str(WOLFRAM_FIXTURE).expect("valid Wolfram fixture");
    assert_eq!(fixture.schema, "sonium-wave.special-golden-wolfram.v1");
    assert_eq!(fixture.precision_digits, 50);
    let ka = case(&fixture, ka_name);
    let kr_case = case(&fixture, kr_name);
    assert!(
        (parse_f64(&ka.x) - wave_number * radius).abs() <= 1.0e-12,
        "ka case {ka_name} carries x={} but the run requests ka={}",
        ka.x,
        wave_number * radius
    );
    let receiver_r = 2.0 * radius;
    assert!(
        (parse_f64(&kr_case.x) - wave_number * receiver_r).abs() <= 1.0e-12,
        "kr case {kr_name} carries x={} but the run requests kr={}",
        kr_case.x,
        wave_number * receiver_r
    );
    let coefficients = wolfram_coefficients(ka);
    let thetas = [0.0, PI / 3.0, PI / 2.0, PI];
    let reference = sphere_scattering_3d(
        wave_number,
        radius,
        NUM_TERMS,
        vec![receiver_r],
        thetas.to_vec(),
    );
    assert_eq!(reference.pressure.len(), thetas.len());

    for (index, theta) in thetas.iter().enumerate() {
        let expected = wolfram_field(&coefficients, kr_case, theta.cos(), NUM_TERMS);
        let actual = reference.pressure[index];
        let scale = expected.norm().max(1.0);
        assert!(
            (actual - expected).norm() <= TOLERANCE * scale,
            "ka={ka_name} theta={theta:.4}: BEM-reference {actual:?} != wolfram {expected:?}"
        );
    }
}

#[test]
fn rigid_sphere_field_matches_wolfram_reference_off_resonance() {
    // ka = pi/2, receiver kr = pi: both tabulated in the fixture.
    check_ka("pi_over_2", "pi", PI / 2.0, 1.0);
}

#[test]
fn rigid_sphere_field_matches_wolfram_reference_at_first_resonance() {
    // ka = pi is the first interior Dirichlet resonance of the unit sphere;
    // receiver kr = 2*pi is tabulated in the fixture.
    check_ka("pi", "two_pi", PI, 1.0);
}
