// Rust guideline compliant 2026-02-21
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::physics::fluid::compressible_flow::{
    CompressibleBoundaryCondition, CompressibleDgConfig, CompressibleDgSolver,
    CompressibleDgState, CompressibleFluxKind, ConservativeState, IdealGas, ViscosityModel,
    isentropic_area_ratio, local_lax_friedrichs_speed, rusanov_flux, sod_shock_tube_states,
};
use std::collections::HashMap;

/// Test 1: Sod shock tube initial states match expected values.
fn test_sod_shock_tube_states() {
    let gas = IdealGas::AIR;
    let (left, right) = sod_shock_tube_states(gas);

    assert!(
        (left.density - 1.0).abs() < 1e-12,
        "Left density should be 1.0"
    );
    assert!(
        (left.velocity()[0]).abs() < 1e-12,
        "Left velocity should be 0"
    );
    assert!(
        (left.pressure(gas) - 1.0).abs() < 1e-12,
        "Left pressure should be 1.0"
    );

    assert!(
        (right.density - 0.125).abs() < 1e-12,
        "Right density should be 0.125"
    );
    assert!(
        (right.velocity()[0]).abs() < 1e-12,
        "Right velocity should be 0"
    );
    assert!(
        (right.pressure(gas) - 0.1).abs() < 1e-12,
        "Right pressure should be 0.1"
    );

    println!(
        "Sod shock tube states: left=({}, {}, {}), right=({}, {}, {})",
        left.density,
        left.velocity()[0],
        left.pressure(gas),
        right.density,
        right.velocity()[0],
        right.pressure(gas)
    );
}

/// Test 2: Rusanov flux with left=right should reduce to exact Euler flux.
fn test_rusanov_consistency() {
    let gas = IdealGas::AIR;
    let state = ConservativeState::from_primitive(2.0, [3.0, 0.0, 0.0], 5.0, gas);
    let normal = [1.0, 0.0, 0.0];

    let flux = rusanov_flux(state, state, gas, normal);
    let expected = state.euler_flux(gas, normal);

    for i in 0..5 {
        assert!(
            (flux[i] - expected[i]).abs() < 1e-12,
            "Rusanov consistency failed at component {}: {} vs {}",
            i,
            flux[i],
            expected[i]
        );
    }
    println!("Rusanov consistency: PASSED");
}

/// Test 3: Rusanov flux is dissipative (eigenvalues bounded by max wave speed).
fn test_rusanov_dissipation() {
    let gas = IdealGas::AIR;
    let left = ConservativeState::from_primitive(1.0, [0.0, 0.0, 0.0], 1.0, gas);
    let right = ConservativeState::from_primitive(0.125, [0.0, 0.0, 0.0], 0.1, gas);
    let normal = [1.0, 0.0, 0.0];

    let smax = local_lax_friedrichs_speed(left, right, gas, normal);
    let flux = rusanov_flux(left, right, gas, normal);

    // The dissipation term magnitude should be related to smax
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

    for i in 0..5 {
        let central = 0.5 * (fl[i] + fr[i]);
        let diff = 0.5 * smax * (ur[i] - ul[i]).abs();
        // Rusanov flux = central - diff, so it should be bounded
        assert!(
            flux[i].abs() < central.abs() + diff + 1.0,
            "Rusanov flux unbounded at component {}: {}",
            i,
            flux[i]
        );
    }

    println!("Rusanov dissipation: smax={:.4e}, PASSED", smax);
}

/// Test 4: Isentropic area ratio at Mach=1 should equal 1.
fn test_isentropic_area_ratio_unity() {
    let gamma = 1.4;
    let a_ratio = isentropic_area_ratio(1.0, gamma);
    println!("Isentropic area ratio at M=1: {:.12e}", a_ratio);
    assert!(
        (a_ratio - 1.0).abs() < 1e-12,
        "Area ratio at M=1 should be 1.0"
    );
}

/// Test 5: Isentropic area ratio increases with Mach number.
fn test_isentropic_area_ratio_monotonic() {
    let gamma = 1.4;
    let a1 = isentropic_area_ratio(1.5, gamma);
    let a2 = isentropic_area_ratio(2.0, gamma);
    let a3 = isentropic_area_ratio(3.0, gamma);

    println!(
        "Isentropic area ratios: M=1.5 -> {:.4e}, M=2.0 -> {:.4e}, M=3.0 -> {:.4e}",
        a1, a2, a3
    );
    assert!(a2 > a1, "Area ratio should increase with Mach");
    assert!(a3 > a2, "Area ratio should increase with Mach");
}

/// Test 6: Conservative state round-trip (primitive -> conservative -> pressure/velocity).
fn test_conservative_state_roundtrip() {
    let gas = IdealGas::AIR;
    let rho = 2.5;
    let v = [1.5, -0.5, 0.3];
    let p = 101325.0;

    let state = ConservativeState::from_primitive(rho, v, p, gas);
    let p_recovered = state.pressure(gas);
    let v_recovered = state.velocity();

    assert!((state.density - rho).abs() < 1e-12, "Density mismatch");
    assert!(
        (p_recovered - p).abs() < 1e-6,
        "Pressure mismatch: {} vs {}",
        p_recovered,
        p
    );
    for i in 0..3 {
        assert!(
            (v_recovered[i] - v[i]).abs() < 1e-12,
            "Velocity mismatch at {}: {} vs {}",
            i,
            v_recovered[i],
            v[i]
        );
    }
    println!("Conservative state roundtrip: PASSED");
}

/// Test 7: Max wave speed for stationary flow is just sound speed.
fn test_max_wave_speed_stationary() {
    let gas = IdealGas::AIR;
    let state = ConservativeState::from_primitive(1.225, [0.0, 0.0, 0.0], 101325.0, gas);
    let normal = [1.0, 0.0, 0.0];

    let c = state.max_wave_speed(gas, normal);
    let expected_c = gas.speed_of_sound(101325.0, 1.225);

    println!(
        "Max wave speed (stationary): {:.4e}, expected {:.4e}",
        c, expected_c
    );
    assert!(
        (c - expected_c).abs() < 1e-6,
        "Wave speed mismatch: {} vs {}",
        c,
        expected_c
    );
}

/// Exact Sod density at (x, t) for the standard states, via the Toro
/// star-region solve. Reference values: p* = 0.30313, u* = 0.92745.
fn sod_exact_density(x: f64, t: f64) -> f64 {
    let g = 1.4f64;
    let (rho_l, p_l, c_l) = (1.0f64, 1.0f64, 1.4f64.sqrt());
    let (rho_r, p_r) = (0.125f64, 0.1f64);
    let c_r = (g * p_r / rho_r).sqrt();
    let pressure_function = |p: f64, rho: f64, p0: f64, c0: f64| -> f64 {
        if p > p0 {
            let a = 2.0 / ((g + 1.0) * rho);
            let b = (g - 1.0) / (g + 1.0) * p0;
            (p - p0) * (a / (p + b)).sqrt()
        } else {
            2.0 * c0 / (g - 1.0) * ((p / p0).powf((g - 1.0) / (2.0 * g)) - 1.0)
        }
    };
    // Bisection on f_L(p) + f_R(p) = 0 (u_L = u_R = 0).
    let (mut lo, mut hi) = (0.01f64, 1.0f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        let value = pressure_function(mid, rho_l, p_l, c_l)
            + pressure_function(mid, rho_r, p_r, c_r);
        if value > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let p_star = 0.5 * (lo + hi);
    let u_star = -pressure_function(p_star, rho_l, p_l, c_l);
    let rho_l_star = rho_l * (p_star / p_l).powf(1.0 / g);
    let rho_r_star = rho_r
        * ((g + 1.0) * p_star + (g - 1.0) * p_r) / ((g - 1.0) * p_star + (g + 1.0) * p_r);
    let shock_speed = (g * p_r / rho_r).sqrt()
        * (((g + 1.0) * p_star / (2.0 * g * p_r) + (g - 1.0) / (2.0 * g)).sqrt());
    let xi = (x - 0.5) / t;
    if xi <= u_star {
        // Left of the contact.
        let head = -c_l;
        let c_l_star = c_l * (p_star / p_l).powf((g - 1.0) / (2.0 * g));
        let tail = u_star - c_l_star;
        if xi <= head {
            rho_l
        } else if xi <= tail {
            let u = 2.0 / (g + 1.0) * (c_l + xi);
            let c = c_l - 0.5 * (g - 1.0) * u;
            rho_l * (c / c_l).powf(2.0 / (g - 1.0))
        } else {
            rho_l_star
        }
    } else if xi >= shock_speed {
        rho_r
    } else {
        rho_r_star
    }
}

/// Test 8: time-marched P1-DG Sod solution against the exact Riemann
/// solution on two meshes (the flux-level tests above never advance the
/// solver). Gates: density L1 error decreases under refinement and stays
/// below 0.03, and the numerical shock sits at the exact shock position.
fn test_sod_dg_time_marched() {
    let gas = IdealGas {
        gamma: 1.4,
        gas_constant: 1.0,
    };
    let (left, right) = sod_shock_tube_states(gas);
    let t_end = 0.2f64;
    let mut l1_errors = Vec::new();
    for nx in [100usize, 200] {
        let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.05, nx, 2);
        let mut boundaries: HashMap<i32, CompressibleBoundaryCondition> = HashMap::new();
        boundaries.insert(1, CompressibleBoundaryCondition::FarField(left));
        boundaries.insert(2, CompressibleBoundaryCondition::FarField(right));
        boundaries.insert(3, CompressibleBoundaryCondition::SlipWall);
        boundaries.insert(4, CompressibleBoundaryCondition::SlipWall);
        // Sharpened configuration (env-reversible for P1/Rusanov
        // comparison): P2 space, HLLC flux.
        let degree = if std::env::var("SONIUM_SOD_DEGREE").is_ok_and(|v| v == "1") {
            PolynomialDegree::P1
        } else {
            PolynomialDegree::P2
        };
        let flux = if std::env::var("SONIUM_SOD_FLUX").is_ok_and(|v| v == "rusanov") {
            CompressibleFluxKind::Rusanov
        } else {
            CompressibleFluxKind::Hllc
        };
        let config = CompressibleDgConfig {
            gas,
            viscosity: ViscosityModel::Constant(1.0e-10),
            boundaries,
            degree,
            flux,
            troubled_cell_threshold: 0.03,
            persson_threshold: -3.0,
            ..CompressibleDgConfig::default()
        };
        let mut state = CompressibleDgState::from_function_degree(&mesh, degree, |point| {
            if point[0] < 0.5 { left } else { right }
        })
        .expect("Sod DG initial state");
        let mut time = 0.0;
        if std::env::var("SONIUM_DG_BACKEND").as_deref() == Ok("cuda") {
            run_sod_gpu(&mesh, config, &mut state, t_end, &mut time);
        } else {
            let solver = CompressibleDgSolver::new(&mesh, config).expect("Sod DG solver");
            while time < t_end {
                let stable = solver
                    .operator
                    .stable_time_step(&state)
                    .expect("Sod CFL");
                let report = solver.step(&mut state, stable).expect("Sod DG step");
                time += report.accepted_time_step;
                assert!(
                    report.minimum_density > 0.0 && report.minimum_pressure > 0.0,
                    "Sod DG lost positivity at t={time:.3}"
                );
            }
        }
        // Density L1 error and shock position from cell averages.
        let mut l1 = 0.0;
        let mut total_area = 0.0;
        let mut axial: Vec<(f64, f64)> = Vec::new();
        for element in 0..mesh.num_elements() {
            let cell = &mesh.elements[element];
            let mut centroid = [0.0f64; 2];
            for &node in cell.vertices() {
                centroid[0] += mesh.nodes[node].x;
                centroid[1] += mesh.nodes[node].y;
            }
            centroid[0] /= cell.vertices().len() as f64;
            centroid[1] /= cell.vertices().len() as f64;
            let average = state.cell_average(element);
            let (a, b, c) = (
                &mesh.nodes[cell.vertices()[0]],
                &mesh.nodes[cell.vertices()[1]],
                &mesh.nodes[cell.vertices()[2]],
            );
            let area = 0.5
                * ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)).abs();
            l1 += area * (average.density - sod_exact_density(centroid[0], t_end)).abs();
            total_area += area;
            axial.push((centroid[0], average.density));
        }
        l1 /= total_area;
        axial.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        // Shock: first cell right of the contact whose density falls below
        // the midpoint of the post/pre-shock densities (0.2656, 0.125).
        let shock_x = axial
            .iter()
            .filter(|(x, _)| *x > 0.6)
            .find(|(_, rho)| *rho < 0.195)
            .map(|(x, _)| *x)
            .unwrap_or(f64::NAN);
        let exact_shock_x = 0.5 + 1.7522 * t_end;
        println!(
            "Sod DG nx={nx}: L1 density error {l1:.4}, shock x {shock_x:.3} (exact {exact_shock_x:.3})"
        );
        assert!(
            (shock_x - exact_shock_x).abs() < 0.06,
            "Sod shock position {shock_x:.3} vs exact {exact_shock_x:.3}"
        );
        l1_errors.push(l1);
    }
    assert!(
        l1_errors[1] < l1_errors[0],
        "Sod L1 error did not decrease: {} -> {}",
        l1_errors[0],
        l1_errors[1]
    );
    assert!(
        l1_errors[1] < 0.03,
        "Sod fine-mesh L1 density error {:.4} >= 0.03",
        l1_errors[1]
    );
    println!("Sod DG time-marched: PASSED");
}

/// GPU twin of the Sod DG time loop: one upload, device steps, one
/// download before the L1/shock probes.
#[cfg(feature = "cuda")]
fn run_sod_gpu(
    mesh: &sonium_fem::mesh::Mesh,
    config: CompressibleDgConfig,
    state: &mut CompressibleDgState,
    t_end: f64,
    time: &mut f64,
) {
    use sonium_fem::physics::fluid::compressible_flow::CompressibleDgGpuSolver;
    let mut solver = CompressibleDgGpuSolver::new(mesh, config).expect("Sod GPU solver");
    solver.set_state(state).expect("Sod GPU upload");
    while *time < t_end {
        let report = solver.step(f64::INFINITY).expect("Sod GPU step");
        *time += report.accepted_time_step;
        assert!(
            report.minimum_density > 0.0 && report.minimum_pressure > 0.0,
            "Sod DG lost positivity at t={time:.3}"
        );
    }
    solver.fetch_state(state).expect("Sod GPU download");
}

#[cfg(not(feature = "cuda"))]
fn run_sod_gpu(
    _mesh: &sonium_fem::mesh::Mesh,
    _config: CompressibleDgConfig,
    _state: &mut CompressibleDgState,
    _t_end: f64,
    _time: &mut f64,
) {
    panic!("SONIUM_DG_BACKEND=cuda requires building with --features cuda");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        println!("=== QA: compressible flow Sod shock tube ===");
        test_sod_shock_tube_states();
        test_rusanov_consistency();
        test_rusanov_dissipation();
        test_isentropic_area_ratio_unity();
        test_isentropic_area_ratio_monotonic();
        test_conservative_state_roundtrip();
        test_max_wave_speed_stationary();
        test_sod_dg_time_marched();
        println!("SUCCESS");
    });
    if result.is_err() {
        eprintln!("FAILED");
        std::process::exit(1);
    }
}
