//! Solved 3D porous-electrode macro model with thermal feedback and CN history.
//!
//! C=F*c_e is a scaled electrolyte concentration. The declared linearized
//! interfacial law is a*j = a*L*(phi_s-phi_e-u*C), not full nonlinear DFN.
//! Three native macro operators and the reaction Jacobian are composed by
//! this caller; heat uses actual FE currents, overpotential and temperature.
use sonium_fem::assembly::{TripletMatrix, assemble_bilinear, assemble_linear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::forms::{LinearForm, LocalVector};
use sonium_fem::mesh::unit_cube_tetrahedra;
use sonium_fem::physics::coupled::battery::{
    BatteryCapacityForm, BatteryHeatBreakdown, BatteryHeatSourceForm, BatteryLocalHeatState,
    BatteryMaterial, BatteryRegionKind, BatteryScalarDiffusionForm,
    InterfacialReactionJacobianForm, assemble_battery_macro_operators,
};
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{BasisTabulation, FunctionSpace, H1Space};
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::{CsrMatrix, SparseCholeskyFactorization, SparseLuFactorization};

const CASE: &str = "fem.battery-3d-electrothermal-history.v1";
const F: f64 = 96_485.332_12;
const U: f64 = 0.01;
const ENTROPY: f64 = 0.01;
const ORDER: usize = 6;
const MATERIAL: BatteryMaterial = BatteryMaterial {
    region: BatteryRegionKind::NegativeElectrode,
    porosity: 0.5,
    bruggeman_exponent: 1.0,
    electrolyte_diffusivity: 1.0,
    electrolyte_conductivity: 1.4,
    solid_conductivity: 2.0,
    transference_number: 0.4,
    particle_radius: 1.0,
    solid_diffusivity: 0.2,
    maximum_solid_concentration: 10.0,
    specific_surface_area: 0.5 / F,
    exchange_current_density: F,
    charge_transfer_coefficient: 0.5,
    density: 1.0,
    specific_heat: 1.3,
    thermal_conductivity: 0.7,
};
#[derive(Debug, Clone, Copy)]
enum Law {
    Linear,
    Exponential,
}
impl Law {
    fn value(self, t: f64) -> f64 {
        match self {
            Self::Linear => 1.0 + t,
            Self::Exponential => (-t).exp(),
        }
    }
    fn derivative(self, t: f64) -> f64 {
        match self {
            Self::Linear => 1.0,
            Self::Exponential => -(-t).exp(),
        }
    }
}
#[derive(Debug, Clone, Copy)]
enum Control {
    Full,
    NoCapacity,
    NoReaction,
    NoConcentrationCoupling,
    NoHeat,
    NoPorosity,
}
// Independent polynomials and gradients for C, phi_s, phi_e, temperature excess.
fn reference(p: [f64; 3]) -> ([f64; 4], [[f64; 3]; 4]) {
    let [x, y, z] = p;
    let a = 1.0 + x * x + 2.0 * y * y + 3.0 * z * z + x * y + x * z + y * z;
    let b = 1.0 + 2.0 * x * x + y * y + z * z + x * y + 2.0 * x * z + y * z;
    let da = [2.0 * x + y + z, 4.0 * y + x + z, 6.0 * z + x + y];
    let db = [
        4.0 * x + y + 2.0 * z,
        2.0 * y + x + z,
        2.0 * z + 2.0 * x + y,
    ];
    (
        [1.0 + 0.1 * a, 0.04 * a, 0.01 * b, 0.1 * b],
        [
            da.map(|v| 0.1 * v),
            da.map(|v| 0.04 * v),
            db.map(|v| 0.01 * v),
            db.map(|v| 0.1 * v),
        ],
    )
}
fn heat(value: [f64; 4], gradient: [[f64; 3]; 4], control: Control) -> f64 {
    if matches!(control, Control::NoHeat) {
        return 0.0;
    }
    let eta = value[1] - value[2] - U * value[0];
    let sigma = MATERIAL.solid_conductivity;
    let kappa = if matches!(control, Control::NoPorosity) {
        MATERIAL.electrolyte_conductivity
    } else {
        MATERIAL.effective_electrolyte_conductivity()
    };
    BatteryHeatBreakdown::from_local_state(BatteryLocalHeatState {
        solid_current: gradient[1].map(|v| -sigma * v),
        electrolyte_current: gradient[2].map(|v| -kappa * v),
        solid_conductivity: sigma,
        electrolyte_conductivity: kappa,
        interfacial_current_density: eta, // a*L=1: volumetric reaction current
        overpotential: eta,
        temperature: 300.0 + value[3],
        entropy_coefficient: ENTROPY,
    })
    .total()
}
fn exact_heat(value: [f64; 4], gradient: [[f64; 3]; 4]) -> f64 {
    let eta = value[1] - value[2] - 0.01 * value[0];
    2.0 * gradient[1].iter().map(|v| v * v).sum::<f64>()
        + 0.7 * gradient[2].iter().map(|v| v * v).sum::<f64>()
        + eta * eta
        + eta * (300.0 + value[3]) * 0.01
}
fn source(p: [f64; 3], t: f64, law: Law) -> [f64; 4] {
    let (v, d) = reference(p);
    let g = law.value(t);
    let gp = law.derivative(t);
    let value = v.map(|q| g * q);
    let gradient = d.map(|row| row.map(|q| g * q));
    let j = value[1] - value[2] - 0.01 * value[0];
    // Independently evaluated strong forms. Laplacians=(1.2,.48,.08,.8).
    [
        0.5 * gp * v[0] - 0.5 * g * 1.2 - 0.6 * j,
        -2.0 * g * 0.48 + j,
        -0.7 * g * 0.08 - j,
        1.3 * gp * v[3] - 0.7 * g * 0.8 - exact_heat(value, gradient),
    ]
}
fn add_block(
    out: &mut TripletMatrix<f64>,
    a: &TripletMatrix<f64>,
    r: usize,
    c: usize,
    n: usize,
    s: f64,
) {
    for k in 0..a.nnz() {
        out.push(r * n + a.rows[k], c * n + a.cols[k], s * a.values[k]);
    }
}
fn lift(
    mut a: TripletMatrix<f64>,
    mut rhs: Vec<f64>,
    fixed: &[bool],
    prescribed: &[f64],
) -> (CsrMatrix<f64>, Vec<f64>) {
    for k in 0..a.nnz() {
        let (r, c) = (a.rows[k], a.cols[k]);
        if fixed[c] && !fixed[r] {
            rhs[r] -= a.values[k] * prescribed[c];
        }
        if fixed[r] || fixed[c] {
            a.values[k] = 0.0;
        }
    }
    for (d, &b) in fixed.iter().enumerate() {
        if b {
            a.push(d, d, 1.0);
            rhs[d] = prescribed[d];
        }
    }
    (a.to_csr(), rhs)
}
#[derive(Clone)]
struct Point {
    tab: BasisTabulation,
    dofs: Vec<usize>,
    weight: f64,
}
fn evaluate(point: &Point, state: &[f64], n: usize) -> ([f64; 4], [[f64; 3]; 4]) {
    let value = std::array::from_fn(|f| {
        point
            .dofs
            .iter()
            .enumerate()
            .map(|(i, &d)| state[f * n + d] * point.tab.value(i))
            .sum()
    });
    let gradient = std::array::from_fn(|f| {
        std::array::from_fn(|a| {
            point
                .dofs
                .iter()
                .enumerate()
                .map(|(i, &d)| state[f * n + d] * point.tab.grad(i, a))
                .sum()
        })
    });
    (value, gradient)
}
fn heat_load(points: &[Point], state: &[f64], n: usize, control: Control) -> Vec<f64> {
    // Per-point locals are independent; collect in index order (rayon
    // above the machine-adaptive gate, serial below) and scatter in
    // point order, so results are bitwise identical to the serial loop.
    let locals: Vec<LocalVector<f64>> = sonium_fem::parallelism::collect_indexed(
        points.len(),
        sonium_fem::parallelism::ELEMENTS_PER_THREAD,
        |i| {
            let point = &points[i];
            let (v, d) = evaluate(point, state, n);
            let density = heat(v, d, control);
            let mut local = LocalVector::zeros(point.dofs.len());
            // Invoke the production heat source form with solved FE data.
            BatteryHeatSourceForm {
                heat_density: |_, _, _| density,
            }
            .integrate_point(&point.tab, point.weight, &mut local);
            local
        },
    );
    let mut out = vec![0.0; n];
    for (point, local) in points.iter().zip(locals.iter()) {
        for (i, &d) in point.dofs.iter().enumerate() {
            out[d] += local.values[i];
        }
    }
    out
}
#[derive(Debug)]
struct Metrics {
    field: [f64; 4],
    flux: [f64; 4],
    history: f64,
    residual: f64,
    balance: f64,
    boundary: f64,
    change: f64,
    heat_error: f64,
    steps: usize,
    free: usize,
    iterations: usize,
}
fn solve(
    nmesh: usize,
    degree: PolynomialDegree,
    steps: usize,
    law: Law,
    control: Control,
) -> Result<Metrics, String> {
    let mut mesh = unit_cube_tetrahedra(nmesh);
    for face in &mut mesh.boundaries {
        face.marker = 1;
    }
    let space = H1Space::new(degree);
    let n = space.num_global_dofs(&mesh);
    let coords = space.dof_coordinates(&mesh).ok_or("missing coordinates")?;
    let mut fixed = vec![false; 4 * n];
    for d in space.boundary_dofs(&mesh, 1) {
        for f in 0..4 {
            fixed[f * n + d] = true;
        }
    }
    let free = fixed[..n].iter().filter(|&&b| !b).count();
    if free == 0 {
        return Err("no free battery fields".into());
    }
    let prescribed = |t: f64| {
        (0..4 * n)
            .map(|d| law.value(t) * reference(coords[d % n]).0[d / n])
            .collect::<Vec<_>>()
    };
    let mut material = MATERIAL
        .validate()
        .map_err(|e| format!("material: {e:?}"))?;
    if matches!(control, Control::NoPorosity) {
        material.bruggeman_exponent = 0.0;
    }
    let operators = assemble_battery_macro_operators(&mesh, degree, material);
    let capacity = if matches!(control, Control::NoCapacity) {
        0.0
    } else {
        1.0
    };
    let reaction = if matches!(control, Control::NoReaction) {
        0.0
    } else {
        2.0 * F
    };
    let u = if matches!(control, Control::NoConcentrationCoupling) {
        0.0
    } else {
        U
    };
    let r = assemble_bilinear(
        &InterfacialReactionJacobianForm {
            specific_surface_area: material.specific_surface_area,
            current_slope: reaction,
        },
        &space,
        &space,
        &mesh,
        ORDER,
    );
    let dt = 1.0 / steps as f64;
    let mut lhs = TripletMatrix::new(3 * n, 3 * n);
    add_block(
        &mut lhs,
        &operators.electrolyte_capacity,
        0,
        0,
        n,
        capacity / dt,
    );
    add_block(&mut lhs, &operators.electrolyte_diffusion, 0, 0, n, 0.5);
    add_block(&mut lhs, &r, 0, 0, n, 0.3 * u);
    add_block(&mut lhs, &r, 0, 1, n, -0.3);
    add_block(&mut lhs, &r, 0, 2, n, 0.3);
    add_block(&mut lhs, &operators.solid_conduction, 1, 1, n, 1.0);
    add_block(&mut lhs, &operators.electrolyte_conduction, 2, 2, n, 1.0);
    for (row, sgn) in [(1, 1.0), (2, -1.0)] {
        for (col, scale) in [(0, -u), (1, 1.0), (2, -1.0)] {
            add_block(&mut lhs, &r, row, col, n, sgn * scale);
        }
    }
    let raw = lhs.to_csr();
    let mass = operators.electrolyte_capacity.to_csr();
    let diff = operators.electrolyte_diffusion.to_csr();
    let react = r.to_csr();
    let macro_fixed = &fixed[..3 * n];
    let factor = SparseLuFactorization::factorize(
        &lift(
            lhs.clone(),
            vec![0.0; 3 * n],
            macro_fixed,
            &prescribed(0.0)[..3 * n],
        )
        .0,
    )
    .map_err(|e| e.to_string())?;
    let tm = assemble_bilinear(
        &BatteryCapacityForm {
            coefficient: capacity * material.density * material.specific_heat,
        },
        &space,
        &space,
        &mesh,
        ORDER,
    );
    let tk = assemble_bilinear(
        &BatteryScalarDiffusionForm {
            coefficient: material.thermal_conductivity,
        },
        &space,
        &space,
        &mesh,
        ORDER,
    );
    let mut tlhs = TripletMatrix::new(n, n);
    add_block(&mut tlhs, &tm, 0, 0, n, 1.0 / dt);
    add_block(&mut tlhs, &tk, 0, 0, n, 0.5);
    let traw = tlhs.to_csr();
    let tmat = tm.to_csr();
    let thermal_fixed = &fixed[3 * n..];
    let tfactor = SparseCholeskyFactorization::factorize(
        &lift(
            tlhs.clone(),
            vec![0.0; n],
            thermal_fixed,
            &prescribed(0.0)[3 * n..],
        )
        .0,
    )
    .map_err(|e| e.to_string())?;
    let mut points = Vec::new();
    for element in 0..mesh.num_elements() {
        let dofs = space.local_to_global(&mesh, element);
        for qp in QuadratureRule::new(mesh.elements[element].element_type, ORDER).iter() {
            points.push(Point {
                tab: space.tabulate(&mesh, element, qp.xi(), qp.eta(), qp.zeta()),
                dofs: dofs.clone(),
                weight: qp.weight,
            });
        }
    }
    let sources = |time: f64| {
        std::array::from_fn::<_, 4, _>(|f| {
            assemble_linear(
                &BatteryHeatSourceForm {
                    heat_density: |x, y, z| source([x, y, z], time, law)[f],
                },
                &space,
                &mesh,
                ORDER,
            )
        })
    };
    let mut state = prescribed(0.0);
    let initial = state.clone();
    let mut oldsource = sources(0.0);
    let mut oldheat = heat_load(&points, &state, n, control);
    let mut residual = 0.0_f64;
    let mut balance = 0.0_f64;
    let mut boundary = 0.0_f64;
    let mut history = 0.0_f64;
    let mut iterations = 0;
    for step in 1..=steps {
        let time = step as f64 * dt;
        let newsource = sources(time);
        let exact = prescribed(time);
        let old = state.clone();
        let oldmass = mass.matvec(&old[..n]);
        let olddiff = diff.matvec(&old[..n]);
        let eta: Vec<_> = (0..n)
            .map(|d| old[n + d] - old[2 * n + d] - u * old[d])
            .collect();
        let oldreaction = react.matvec(&eta);
        let mut rhs = vec![0.0; 3 * n];
        for d in 0..n {
            rhs[d] = capacity * oldmass[d] / dt - 0.5 * olddiff[d]
                + 0.3 * oldreaction[d]
                + 0.5 * (oldsource[0][d] + newsource[0][d]);
            rhs[n + d] = newsource[1][d];
            rhs[2 * n + d] = newsource[2][d];
        }
        let solution = factor
            .solve(&lift(lhs.clone(), rhs.clone(), macro_fixed, &exact[..3 * n]).1)
            .map_err(|e| e.to_string())?;
        state[..3 * n].copy_from_slice(&solution);
        let action = raw.matvec(&solution);
        for f in 0..3 {
            let numerator: f64 = (0..n)
                .filter(|&d| !fixed[d])
                .map(|d| (action[f * n + d] - rhs[f * n + d]).powi(2))
                .sum();
            let scale = rhs[f * n..(f + 1) * n]
                .iter()
                .map(|v| v.abs())
                .sum::<f64>()
                .max(1e-12);
            residual = residual.max(numerator.sqrt() / scale);
            let outward = -(0..n)
                .filter(|&d| fixed[d])
                .map(|d| action[f * n + d] - rhs[f * n + d])
                .sum::<f64>();
            let current_eta: Vec<_> = (0..n)
                .map(|d| state[n + d] - state[2 * n + d] - u * state[d])
                .collect();
            let current_reaction: f64 = react.matvec(&current_eta).iter().sum();
            let budget = match f {
                0 => {
                    let change: Vec<_> = (0..n).map(|d| (state[d] - old[d]) / dt).collect();
                    let stored: f64 = capacity * mass.matvec(&change).iter().sum::<f64>();
                    let injected: f64 = (0..n)
                        .map(|d| 0.5 * (oldsource[0][d] + newsource[0][d]))
                        .sum();
                    stored + outward
                        - 0.3 * (oldreaction.iter().sum::<f64>() + current_reaction)
                        - injected
                }
                1 => outward + current_reaction - newsource[1].iter().sum::<f64>(),
                _ => outward - current_reaction - newsource[2].iter().sum::<f64>(),
            };
            balance = balance.max(budget.abs() / scale);
        }
        let oldtm = tmat.matvec(&old[3 * n..]);
        let oldtlhs = traw.matvec(&old[3 * n..]);
        let base: Vec<_> = (0..n)
            .map(|d| {
                2.0 * oldtm[d] / dt - oldtlhs[d]
                    + 0.5 * (oldsource[3][d] + newsource[3][d] + oldheat[d])
            })
            .collect();
        let mut converged = false;
        for _ in 0..80 {
            iterations += 1;
            let currentheat = heat_load(&points, &state, n, control);
            let trhs: Vec<_> = (0..n).map(|d| base[d] + 0.5 * currentheat[d]).collect();
            let solution = tfactor
                .solve(&lift(tlhs.clone(), trhs, thermal_fixed, &exact[3 * n..]).1)
                .map_err(|e| e.to_string())?;
            let change = solution
                .iter()
                .zip(&state[3 * n..])
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
                / solution.iter().map(|a| a * a).sum::<f64>().sqrt();
            state[3 * n..].copy_from_slice(&solution);
            if change < 1e-13 {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err("battery thermal feedback did not converge".into());
        }
        let currentheat = heat_load(&points, &state, n, control);
        let action = traw.matvec(&state[3 * n..]);
        let trhs: Vec<_> = (0..n).map(|d| base[d] + 0.5 * currentheat[d]).collect();
        let scale = trhs.iter().map(|v| v.abs()).sum::<f64>().max(1e-12);
        residual = residual.max(
            (0..n)
                .filter(|&d| !fixed[d])
                .map(|d| (action[d] - trhs[d]).powi(2))
                .sum::<f64>()
                .sqrt()
                / scale,
        );
        let change: Vec<_> = (0..n)
            .map(|d| (state[3 * n + d] - old[3 * n + d]) / dt)
            .collect();
        let stored: f64 = tmat.matvec(&change).iter().sum();
        let input: f64 = (0..n)
            .map(|d| 0.5 * (oldsource[3][d] + newsource[3][d] + oldheat[d] + currentheat[d]))
            .sum();
        let outward = -(0..n)
            .filter(|&d| fixed[d])
            .map(|d| action[d] - trhs[d])
            .sum::<f64>();
        balance = balance.max(
            (stored + outward - input).abs()
                / stored.abs().max(input.abs()).max(outward.abs()).max(1e-12),
        );
        for f in 0..4 {
            let mut e = 0.0;
            let mut norm = 0.0;
            for d in 0..n {
                if fixed[d] {
                    boundary = boundary.max((state[f * n + d] - exact[f * n + d]).abs());
                } else {
                    e += (state[f * n + d] - exact[f * n + d]).powi(2);
                    norm += exact[f * n + d].powi(2);
                }
            }
            history = history.max((e / norm).sqrt());
        }
        if state.iter().any(|v| !v.is_finite())
            || (matches!(control, Control::Full) && state[..n].iter().any(|&v| v <= 0.0))
        {
            return Err("invalid battery state".into());
        }
        oldsource = newsource;
        oldheat = currentheat;
    }
    let mut error = [0.0; 4];
    let mut norm = [0.0; 4];
    let mut fe = [0.0; 4];
    let mut fnorm = [0.0; 4];
    let mut he = 0.0;
    let mut hn = 0.0;
    for point in &points {
        let (v, d) = evaluate(point, &state, n);
        let (rv, rd) = reference(point.tab.physical_coords);
        let g = law.value(1.0);
        let rv = rv.map(|a| g * a);
        let rd = rd.map(|r| r.map(|a| g * a));
        let w = point.weight * point.tab.det_j;
        for f in 0..4 {
            error[f] += w * (v[f] - rv[f]).powi(2);
            norm[f] += w * rv[f].powi(2);
            for a in 0..3 {
                fe[f] += w * (d[f][a] - rd[f][a]).powi(2);
                fnorm[f] += w * rd[f][a].powi(2);
            }
        }
        he += w * (heat(v, d, control) - exact_heat(rv, rd)).powi(2);
        hn += w * exact_heat(rv, rd).powi(2);
    }
    let change = state
        .iter()
        .zip(&initial)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
        / initial.iter().map(|a| a * a).sum::<f64>().sqrt();
    Ok(Metrics {
        field: std::array::from_fn(|f| (error[f] / norm[f]).sqrt()),
        flux: std::array::from_fn(|f| (fe[f] / fnorm[f]).sqrt()),
        history,
        residual,
        balance,
        boundary,
        change,
        heat_error: (he / hn).sqrt(),
        steps,
        free,
        iterations,
    })
}
fn run() -> Result<QaResult, String> {
    let mut spatial = Vec::new();
    for n in [3, 6, 12] {
        let m = solve(n, PolynomialDegree::P1, 8, Law::Linear, Control::Full)?;
        println!("battery P1 n={n}: {m:?}");
        spatial.push(m);
    }
    let mut temporal = Vec::new();
    for steps in [8, 16, 32] {
        let m = solve(
            2,
            PolynomialDegree::P2,
            steps,
            Law::Exponential,
            Control::Full,
        )?;
        println!("battery P2 steps={steps}: {m:?}");
        temporal.push(m);
    }
    let patch = solve(2, PolynomialDegree::P2, 8, Law::Linear, Control::Full)?;
    println!("battery P2 exact: {patch:?}");
    let mut negative = f64::INFINITY;
    for c in [
        Control::NoCapacity,
        Control::NoReaction,
        Control::NoConcentrationCoupling,
        Control::NoHeat,
        Control::NoPorosity,
    ] {
        let m = solve(2, PolynomialDegree::P2, 8, Law::Linear, c)?;
        println!("battery {c:?}: {m:?}");
        negative = negative.min(m.history);
    }
    let all: Vec<_> = spatial.iter().chain(&temporal).chain([&patch]).collect();
    let mut result = QaResult::passed(CASE)
        .requested_backend("cpu")
        .actual_backend("cpu");
    for (name, value, bound) in [
        (
            "fine_field_error",
            spatial[2].field.into_iter().fold(0.0_f64, f64::max),
            0.005,
        ),
        (
            "fine_flux_error",
            spatial[2].flux.into_iter().fold(0.0_f64, f64::max),
            0.1,
        ),
        // Measured fine-grid heat error = 2.01e-2; gate at ~2x.
        ("fine_heat_error", spatial[2].heat_error, 0.04),
        (
            "fine_temporal_error",
            temporal[2]
                .field
                .into_iter()
                .chain(temporal[2].flux)
                .chain([temporal[2].heat_error])
                .fold(0.0_f64, f64::max),
            0.001,
        ),
        (
            "exact_p2_error",
            patch
                .field
                .into_iter()
                .chain(patch.flux)
                .chain([patch.heat_error])
                .fold(0.0_f64, f64::max),
            1e-9,
        ),
        (
            "physical_residual",
            all.iter().map(|m| m.residual).fold(0.0_f64, f64::max),
            1e-9,
        ),
        (
            "weak_inventory_heat_balance",
            all.iter().map(|m| m.balance).fold(0.0_f64, f64::max),
            1e-9,
        ),
        (
            "boundary_error",
            all.iter().map(|m| m.boundary).fold(0.0_f64, f64::max),
            1e-10,
        ),
    ] {
        if !value.is_finite() || value > bound {
            return Err(format!("{name}={value:e} > {bound}"));
        }
        result = result.metric(name, value).maximum(name, bound);
    }
    for f in 0..4 {
        for (prefix, levels, flux, limit) in [
            ("spatial_field", &spatial, false, 1.8),
            ("spatial_flux", &spatial, true, 0.9),
            ("temporal_field", &temporal, false, 1.8),
            ("temporal_flux", &temporal, true, 1.8),
        ] {
            let rate = levels
                .windows(2)
                .map(|m| {
                    let select = |a: &Metrics| if flux { a.flux[f] } else { a.field[f] };
                    (select(&m[0]) / select(&m[1])).log2()
                })
                .fold(f64::INFINITY, f64::min);
            let name = format!("{prefix}_{f}_rate");
            if !rate.is_finite() || rate < limit {
                return Err(format!("{name}={rate} below {limit}"));
            }
            result = result.metric(&name, rate).minimum(&name, limit);
        }
    }
    for (name, levels, bound) in [
        ("spatial_heat_rate", &spatial, 0.9),
        ("temporal_heat_rate", &temporal, 1.8),
    ] {
        let rate = levels
            .windows(2)
            .map(|m| (m[0].heat_error / m[1].heat_error).log2())
            .fold(f64::INFINITY, f64::min);
        if !rate.is_finite() || rate < bound {
            return Err(format!("{name}={rate} below {bound}"));
        }
        result = result.metric(name, rate).minimum(name, bound);
    }
    for (name, value, bound) in [
        ("minimum_omitted_term_error", negative, 1e-3),
        (
            "minimum_history_change",
            all.iter().map(|m| m.change).fold(f64::INFINITY, f64::min),
            0.1,
        ),
    ] {
        if !value.is_finite() || value < bound {
            return Err(format!("{name}={value:e} below {bound}"));
        }
        result = result.metric(name, value).minimum(name, bound);
    }
    Ok(result.metric("executed_time_steps",all.iter().map(|m|m.steps).sum::<usize>() as f64).minimum("executed_time_steps",88.0)
        .metric("minimum_free_field_unknowns",all.iter().map(|m|m.free).min().unwrap_or(0) as f64).minimum("minimum_free_field_unknowns",1.0)
        .metric("thermal_feedback_iterations",all.iter().map(|m|m.iterations).sum::<usize>() as f64).minimum("thermal_feedback_iterations",88.0)
        .metric("negative_controls",5.0).minimum("negative_controls",5.0)
        .reference("independent quadratic scaled electrolyte concentration, two potentials and temperature; porous-electrode strong sources and ohmic/reaction/entropic heat")
        .reference_checksum("analytic:3d-linearized-porous-electrode-cn-electrothermal:v1")
        .input_checksum("P1n3-6-12:linear8:P2n2:exp8-16-32:linear8:porosity.5:D1:kappa1.4:sigma2:aL1:u.01:entropy.01:five-controls")
        .dimension("3D tetrahedral cube; four xyz-curved fields, each with free interior unknowns")
        .formulation("caller-composed native battery macro operators/reaction Jacobian; CN concentration, coupled quasi-static potentials and CN temperature with solved-current/temperature heat; linearized reaction and prescribed U(C); excludes nonlinear DFN/particle coupling")
        .tolerance("all value space/time rates>=1.8; flux space/time rates>=.9/1.8; P2/residual/weak inventory<=1e-9; omitted-term controls>=1e-3"))
}
fn main() {
    sonium_qa::fem_physics_3d::run_bilinear_case(
        "fem.battery-3d.v1",
        &BatteryScalarDiffusionForm { coefficient: 1.0 },
        true,
    );
    let result = match run() {
        Ok(r) => r,
        Err(e) => QaResult::failed(CASE, e),
    };
    match result.to_json_line() {
        Ok(j) => println!("QA_RESULT: {j}"),
        Err(e) => {
            eprintln!("FAILED: {e}");
            std::process::exit(1);
        }
    }
    if result.status == QaStatus::Passed {
        println!("SUCCESS");
    } else {
        std::process::exit(1);
    }
}
