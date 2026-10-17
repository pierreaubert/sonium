//! Transient incompressible Stokes/inductionless MHD manufactured solution.
//! Native MHD blocks couple three velocity components to electric potential;
//! Taylor-Hood pressure enforces incompressibility. Low-Re advection and
//! magnetic induction are excluded explicitly, not approximated by zero solves.
use sonium_fem::assembly::mixed::assemble_bilinear_two_spaces_checked;
use sonium_fem::assembly::{TripletMatrix, assemble_bilinear, assemble_linear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::forms::{SourceForm, VectorMassForm};
use sonium_fem::mesh::unit_cube_tetrahedra;
use sonium_fem::physics::coupled::mhd::{
    MhdMaterial, MhdPotentialForm, assemble_mhd_electromagnetic_blocks,
};
use sonium_fem::physics::fluid::stokes::forms::{
    StokesBodyForceForm, StokesDivergenceForm, StokesViscousForm,
};
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::{CsrMatrix, SparseLuFactorization};
const CASE: &str = "fem.mhd-3d-stokes-history.v1";
const MATERIAL: MhdMaterial = MhdMaterial {
    density: 1.3,
    dynamic_viscosity: 0.3,
    electrical_conductivity: 2.0,
    imposed_magnetic_flux_density: [0.4, 0.7, 1.1],
};
const ORDER: usize = 6;
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
    NoInertia,
    NoDamping,
    NoMotionalEmf,
    NoLorentzPotential,
    NoZField,
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn reference(p: [f64; 3]) -> ([f64; 3], [[f64; 3]; 3], f64, [f64; 3], f64) {
    let [x, y, z] = p;
    (
        [
            0.1 * (1.0 + y * y * y + y * z * z),
            0.1 * (2.0 + z * z * z + z * x * x),
            0.1 * (3.0 + x * x * x + x * y * y),
        ],
        [
            [0.0, 0.1 * (3.0 * y * y + z * z), 0.2 * y * z],
            [0.2 * x * z, 0.0, 0.1 * (3.0 * z * z + x * x)],
            [0.1 * (3.0 * x * x + y * y), 0.2 * x * y, 0.0],
        ],
        0.2 * (1.0 + x * x * x + 2.0 * y * y * y + 3.0 * z * z * z + x * y * z),
        [
            0.2 * (3.0 * x * x + y * z),
            0.2 * (6.0 * y * y + x * z),
            0.2 * (9.0 * z * z + x * y),
        ],
        0.05 * (x + 2.0 * y + 3.0 * z),
    )
}
fn reference_current(u: [f64; 3], dphi: [f64; 3]) -> [f64; 3] {
    let emf = cross(u, [0.4, 0.7, 1.1]);
    std::array::from_fn(|a| 2.0 * (emf[a] - dphi[a]))
}
fn sources(p: [f64; 3], t: f64, law: Law) -> ([f64; 3], f64) {
    let [x, y, z] = p;
    let (u, du, _, dp, _) = reference(p);
    let g = law.value(t);
    let gp = law.derivative(t);
    let lorentz = cross(reference_current(u, dp), [0.4, 0.7, 1.1]);
    let lap = [0.8 * y, 0.8 * z, 0.8 * x];
    let gradp = [0.05, 0.1, 0.15];
    let momentum =
        std::array::from_fn(|a| 1.3 * gp * u[a] + g * (-0.3 * lap[a] + gradp[a] - lorentz[a]));
    let curl = [
        du[2][1] - du[1][2],
        du[0][2] - du[2][0],
        du[1][0] - du[0][1],
    ];
    let charge = g
        * 2.0
        * (-0.2 * (6.0 * x + 12.0 * y + 18.0 * z) + 0.4 * curl[0] + 0.7 * curl[1] + 1.1 * curl[2]);
    (momentum, charge)
}
fn block(out: &mut TripletMatrix<f64>, a: &TripletMatrix<f64>, ro: usize, co: usize, scale: f64) {
    for k in 0..a.nnz() {
        out.push(ro + a.rows[k], co + a.cols[k], scale * a.values[k]);
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
#[derive(Debug)]
struct Metrics {
    velocity: f64,
    velocity_gradient: f64,
    potential: f64,
    potential_gradient: f64,
    pressure: f64,
    current: f64,
    force: f64,
    joule: f64,
    divergence: f64,
    residual: f64,
    boundary: f64,
    charge_balance: f64,
    power: f64,
    history: f64,
    change: f64,
    steps: usize,
    free: usize,
}
fn solve(
    nmesh: usize,
    high: PolynomialDegree,
    low: PolynomialDegree,
    steps: usize,
    law: Law,
    control: Control,
) -> Result<Metrics, String> {
    let mut mesh = unit_cube_tetrahedra(nmesh);
    for face in &mut mesh.boundaries {
        face.marker = 1;
    }
    let vspace = VectorH1Space::new(high, 3);
    let fspace = H1Space::new(high);
    let pspace = H1Space::new(low);
    let nu = vspace.num_global_dofs(&mesh);
    let nf = fspace.num_global_dofs(&mesh);
    let np = pspace.num_global_dofs(&mesh);
    let total = nu + nf + np;
    let vc = vspace
        .dof_coordinates(&mesh)
        .ok_or("missing velocity coordinates")?;
    let fc = fspace
        .dof_coordinates(&mesh)
        .ok_or("missing potential coordinates")?;
    let pc = pspace
        .dof_coordinates(&mesh)
        .ok_or("missing pressure coordinates")?;
    let mut fixed = vec![false; total];
    for d in vspace.boundary_dofs(&mesh, 1) {
        fixed[d] = true;
    }
    for d in fspace.boundary_dofs(&mesh, 1) {
        fixed[nu + d] = true;
    }
    fixed[nu + nf] = true;
    let free = [
        fixed[..nu].iter().filter(|&&b| !b).count() / 3,
        fixed[nu..nu + nf].iter().filter(|&&b| !b).count(),
        np - 1,
    ]
    .into_iter()
    .min()
    .unwrap_or(0);
    if free == 0 {
        return Err("no free velocity/potential/pressure unknowns".into());
    }
    let prescribed = |t: f64| {
        let g = law.value(t);
        let mut v = Vec::with_capacity(total);
        for (d, p) in vc.iter().enumerate() {
            v.push(g * reference(*p).0[d % 3]);
        }
        for p in &fc {
            v.push(g * reference(*p).2);
        }
        for p in &pc {
            v.push(g * reference(*p).4);
        }
        v
    };
    let mut material = MATERIAL
        .validate()
        .map_err(|e| format!("material: {e:?}"))?;
    if matches!(control, Control::NoZField) {
        material.imposed_magnetic_flux_density[2] = 0.0;
    }
    let em = assemble_mhd_electromagnetic_blocks(&mesh, high, 3, material);
    let capacity = if matches!(control, Control::NoInertia) {
        0.0
    } else {
        1.0
    };
    let damping = if matches!(control, Control::NoDamping) {
        0.0
    } else {
        1.0
    };
    let emf = if matches!(control, Control::NoMotionalEmf) {
        0.0
    } else {
        1.0
    };
    let lorentz = if matches!(control, Control::NoLorentzPotential) {
        0.0
    } else {
        1.0
    };
    let mass = assemble_bilinear(
        &VectorMassForm {
            coefficient: capacity * material.density,
        },
        &vspace,
        &vspace,
        &mesh,
        ORDER,
    );
    let viscous = assemble_bilinear(
        &StokesViscousForm::new(material.dynamic_viscosity),
        &vspace,
        &vspace,
        &mesh,
        ORDER,
    );
    let divergence =
        assemble_bilinear_two_spaces_checked(&StokesDivergenceForm, &vspace, &pspace, &mesh, ORDER)
            .map_err(|e| e.to_string())?;
    let dt = 1.0 / steps as f64;
    let mut lhs = TripletMatrix::new(total, total);
    block(&mut lhs, &mass, 0, 0, 1.0 / dt);
    block(&mut lhs, &viscous, 0, 0, 0.5);
    block(&mut lhs, &em.velocity_damping, 0, 0, 0.5 * damping);
    block(&mut lhs, &em.velocity_from_potential, 0, nu, 0.5 * lorentz);
    block(&mut lhs, &em.potential_from_velocity, nu, 0, emf);
    block(&mut lhs, &em.potential_conduction, nu, nu, 1.0);
    block(&mut lhs, &divergence, 0, nu + nf, 0.5);
    for k in 0..divergence.nnz() {
        lhs.push(
            nu + nf + divergence.cols[k],
            divergence.rows[k],
            divergence.values[k],
        );
    }
    let raw = lhs.to_csr();
    let factor = SparseLuFactorization::factorize(
        &lift(lhs.clone(), vec![0.0; total], &fixed, &prescribed(0.0)).0,
    )
    .map_err(|e| e.to_string())?;
    let m = mass.to_csr();
    let k = viscous.to_csr();
    let d = em.velocity_damping.to_csr();
    let lp = em.velocity_from_potential.to_csr();
    let bp = divergence.to_csr();
    let source = |t: f64| {
        let body = assemble_linear(
            &StokesBodyForceForm::new(move |p| sources(p, t, law).0.to_vec()),
            &vspace,
            &mesh,
            ORDER,
        );
        let charge = assemble_linear(
            &SourceForm::new(move |x, y, z| sources([x, y, z], t, law).1),
            &fspace,
            &mesh,
            ORDER,
        );
        (body, charge)
    };
    let mut state = prescribed(0.0);
    let initial = state.clone();
    let mut old_source = source(0.0);
    let mut residual = 0.0_f64;
    let mut boundary = 0.0_f64;
    let mut charge_balance = 0.0_f64;
    let mut power = 0.0_f64;
    let mut history = 0.0_f64;
    for step in 1..=steps {
        let time = step as f64 * dt;
        let exact = prescribed(time);
        let old = state.clone();
        let new_source = source(time);
        let oldm = m.matvec(&old[..nu]);
        let oldk = k.matvec(&old[..nu]);
        let oldd = d.matvec(&old[..nu]);
        let oldlp = lp.matvec(&old[nu..nu + nf]);
        let oldbp = bp.matvec(&old[nu + nf..]);
        let mut rhs = vec![0.0; total];
        for a in 0..nu {
            rhs[a] = oldm[a] / dt
                - 0.5 * (oldk[a] + damping * oldd[a] + lorentz * oldlp[a] + oldbp[a])
                + 0.5 * (old_source.0[a] + new_source.0[a]);
        }
        rhs[nu..nu + nf].copy_from_slice(&new_source.1);
        state = factor
            .solve(&lift(lhs.clone(), rhs.clone(), &fixed, &exact).1)
            .map_err(|e| e.to_string())?;
        let action = raw.matvec(&state);
        let scale = rhs.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
        // The pressure gauge row's continuity equation must hold too: gauge fixing
        // cannot hide a discrete mass leak. Only prescribed velocity/potential rows
        // are excluded from the physical residual.
        residual = residual.max(
            (0..total)
                .filter(|&a| !fixed[a] || a >= nu + nf)
                .map(|a| (action[a] - rhs[a]).powi(2))
                .sum::<f64>()
                .sqrt()
                / scale,
        );
        let injection: f64 = new_source.1.iter().sum();
        let outward: f64 = (nu..nu + nf)
            .filter(|&a| fixed[a])
            .map(|a| rhs[a] - action[a])
            .sum();
        charge_balance = charge_balance
            .max((outward - injection).abs() / injection.abs().max(outward.abs()).max(1e-12));
        let mean: Vec<_> = (0..nu).map(|a| 0.5 * (state[a] + old[a])).collect();
        let change: Vec<_> = (0..nu).map(|a| (state[a] - old[a]) / dt).collect();
        let mean_phi: Vec<_> = (0..nf)
            .map(|a| 0.5 * (state[nu + a] + old[nu + a]))
            .collect();
        let mean_p: Vec<_> = (0..np)
            .map(|a| 0.5 * (state[nu + nf + a] + old[nu + nf + a]))
            .collect();
        let ma = m.matvec(&change);
        let ka = k.matvec(&mean);
        let da = d.matvec(&mean);
        let la = lp.matvec(&mean_phi);
        let pa = bp.matvec(&mean_p);
        let kinetic: f64 = mean.iter().zip(ma).map(|(u, a)| u * a).sum();
        let dissipated: f64 = (0..nu)
            .map(|a| mean[a] * (ka[a] + damping * da[a] + lorentz * la[a] + pa[a]))
            .sum();
        let input: f64 = (0..nu)
            .map(|a| mean[a] * 0.5 * (old_source.0[a] + new_source.0[a]))
            .sum();
        let wall: f64 = (0..nu)
            .filter(|&a| fixed[a])
            .map(|a| mean[a] * (action[a] - rhs[a]))
            .sum();
        power = power.max(
            (kinetic + dissipated - input - wall).abs()
                / kinetic
                    .abs()
                    .max(dissipated.abs())
                    .max(input.abs())
                    .max(wall.abs())
                    .max(1e-12),
        );
        for (range, _) in [(0..nu, 0), (nu..nu + nf, 1), (nu + nf..total, 2)] {
            let mut e = 0.0;
            let mut norm = 0.0;
            for a in range {
                if fixed[a] {
                    boundary = boundary.max((state[a] - exact[a]).abs());
                } else {
                    e += (state[a] - exact[a]).powi(2);
                    norm += exact[a].powi(2);
                }
            }
            history = history.max((e / norm).sqrt());
        }
        if state.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite MHD history".into());
        }
        old_source = new_source;
    }
    let mut error = [0.0; 7];
    let mut norm = [0.0; 7];
    let mut div = 0.0;
    let mut gu = 0.0;
    let mut joule = 0.0;
    let mut refjoule = 0.0;
    let g = law.value(1.0);
    for e in 0..mesh.num_elements() {
        let vd = vspace.local_to_global(&mesh, e);
        let fd = fspace.local_to_global(&mesh, e);
        let pd = pspace.local_to_global(&mesh, e);
        for qp in QuadratureRule::new(mesh.elements[e].element_type, ORDER).iter() {
            let vt = vspace.tabulate(&mesh, e, qp.xi(), qp.eta(), qp.zeta());
            let ft = fspace.tabulate(&mesh, e, qp.xi(), qp.eta(), qp.zeta());
            let pt = pspace.tabulate(&mesh, e, qp.xi(), qp.eta(), qp.zeta());
            let w = qp.weight * ft.det_j;
            let u: [f64; 3] = std::array::from_fn(|a| {
                vd.iter()
                    .enumerate()
                    .map(|(i, &d)| state[d] * vt.vector_value(i, a))
                    .sum()
            });
            let du: [[f64; 3]; 3] = std::array::from_fn(|a| {
                std::array::from_fn(|b| {
                    vd.iter()
                        .enumerate()
                        .filter(|(i, _)| i % 3 == a)
                        .map(|(i, &d)| state[d] * vt.grad(i, b))
                        .sum()
                })
            });
            let phi: f64 = fd
                .iter()
                .enumerate()
                .map(|(i, &d)| state[nu + d] * ft.value(i))
                .sum();
            let dp: [f64; 3] = std::array::from_fn(|a| {
                fd.iter()
                    .enumerate()
                    .map(|(i, &d)| state[nu + d] * ft.grad(i, a))
                    .sum()
            });
            let p: f64 = pd
                .iter()
                .enumerate()
                .map(|(i, &d)| state[nu + nf + d] * pt.value(i))
                .sum();
            let (ru, rdu, rphi, rdp, rp) = reference(ft.physical_coords);
            let ru = ru.map(|v| g * v);
            let rdu = rdu.map(|row| row.map(|v| g * v));
            let rdp = rdp.map(|v| g * v);
            let ux_b = cross(u, material.imposed_magnetic_flux_density);
            let current: [f64; 3] =
                std::array::from_fn(|a| material.electrical_conductivity * (ux_b[a] - dp[a]));
            let rcurrent = reference_current(ru, rdp);
            let force = cross(current, material.imposed_magnetic_flux_density);
            let rforce = cross(rcurrent, [0.4, 0.7, 1.1]);
            for a in 0..3 {
                error[0] += w * (u[a] - ru[a]).powi(2);
                norm[0] += w * ru[a].powi(2);
                error[3] += w * (dp[a] - rdp[a]).powi(2);
                norm[3] += w * rdp[a].powi(2);
                error[5] += w * (current[a] - rcurrent[a]).powi(2);
                norm[5] += w * rcurrent[a].powi(2);
                error[6] += w * (force[a] - rforce[a]).powi(2);
                norm[6] += w * rforce[a].powi(2);
                for b in 0..3 {
                    error[1] += w * (du[a][b] - rdu[a][b]).powi(2);
                    norm[1] += w * rdu[a][b].powi(2);
                    gu += w * du[a][b].powi(2);
                }
            }
            error[2] += w * (phi - g * rphi).powi(2);
            norm[2] += w * (g * rphi).powi(2);
            error[4] += w * (p - g * rp).powi(2);
            norm[4] += w * (g * rp).powi(2);
            div += w * (du[0][0] + du[1][1] + du[2][2]).powi(2);
            joule += w * current
                .iter()
                .map(|j| j * j / material.electrical_conductivity)
                .sum::<f64>();
            refjoule += w * rcurrent.iter().map(|j| j * j / 2.0).sum::<f64>();
        }
    }
    let errors: [f64; 7] = std::array::from_fn(|a| {
        if norm[a] > 0.0 {
            (error[a] / norm[a]).sqrt()
        } else {
            0.0
        }
    });
    let change = state
        .iter()
        .zip(&initial)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
        / initial.iter().map(|a| a * a).sum::<f64>().sqrt();
    Ok(Metrics {
        velocity: errors[0],
        velocity_gradient: errors[1],
        potential: errors[2],
        potential_gradient: errors[3],
        pressure: errors[4],
        current: errors[5],
        force: errors[6],
        joule: (joule / refjoule - 1.0).abs(),
        divergence: (div / gu).sqrt(),
        residual,
        boundary,
        charge_balance,
        power,
        history,
        change,
        steps,
        free,
    })
}
fn run() -> Result<QaResult, String> {
    let patch = solve(
        2,
        PolynomialDegree::P3,
        PolynomialDegree::P2,
        8,
        Law::Linear,
        Control::Full,
    )?;
    println!("MHD exact P3/P2: {patch:?}");
    let exact = patch
        .velocity
        .max(patch.velocity_gradient)
        .max(patch.potential)
        .max(patch.potential_gradient)
        .max(patch.pressure)
        .max(patch.current)
        .max(patch.force)
        .max(patch.joule)
        .max(patch.divergence);
    if exact > 1e-9 {
        return Err(format!("exact cubic MHD patch error={exact:e}"));
    }
    // Independent solves (own mesh, locals only): run each group
    // concurrently and rejoin in order; prints stay in schedule order.
    use rayon::prelude::*;
    let spatial: Vec<_> = [2, 4, 6]
        .into_par_iter()
        .map(|n| {
            solve(
                n,
                PolynomialDegree::P2,
                PolynomialDegree::P1,
                8,
                Law::Linear,
                Control::Full,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (n, m) in [2, 4, 6].iter().zip(spatial.iter()) {
        println!("MHD P2/P1 n={n}: {m:?}");
    }
    let temporal: Vec<_> = [8, 16, 32]
        .into_par_iter()
        .map(|steps| {
            solve(
                2,
                PolynomialDegree::P3,
                PolynomialDegree::P2,
                steps,
                Law::Exponential,
                Control::Full,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (steps, m) in [8, 16, 32].iter().zip(temporal.iter()) {
        println!("MHD P3/P2 steps={steps}: {m:?}");
    }
    let controls = [
        Control::NoInertia,
        Control::NoDamping,
        Control::NoMotionalEmf,
        Control::NoLorentzPotential,
        Control::NoZField,
    ];
    let control_runs: Vec<_> = controls
        .into_par_iter()
        .map(|c| {
            solve(
                2,
                PolynomialDegree::P3,
                PolynomialDegree::P2,
                8,
                Law::Linear,
                c,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut negative = f64::INFINITY;
    for (c, m) in [
        Control::NoInertia,
        Control::NoDamping,
        Control::NoMotionalEmf,
        Control::NoLorentzPotential,
        Control::NoZField,
    ]
    .iter()
    .zip(control_runs.iter())
    {
        println!("MHD {c:?}: {m:?}");
        negative = negative.min(m.history);
    }
    let all: Vec<_> = spatial.iter().chain(&temporal).chain([&patch]).collect();
    let fine = &spatial[2];
    let fine_time = &temporal[2];
    let mut result = QaResult::passed(CASE)
        .requested_backend("cpu")
        .actual_backend("cpu");
    for (name, v, bound) in [
        ("fine_velocity_error", fine.velocity, 0.005),
        ("fine_potential_error", fine.potential, 0.005),
        // Measured fine-grid errors: pressure 1.2e-4, current 4.6e-3,
        // force 7.5e-3, divergence 2.2e-4; gates at ~2-8x measured.
        ("fine_pressure_error", fine.pressure, 0.001),
        ("fine_current_error", fine.current, 0.01),
        ("fine_lorentz_force_error", fine.force, 0.02),
        ("fine_joule_error", fine.joule, 0.01),
        ("fine_divergence_error", fine.divergence, 0.001),
        ("exact_p3_error", exact, 1e-9),
        (
            "fine_temporal_error",
            fine_time
                .velocity
                .max(fine_time.potential)
                .max(fine_time.pressure)
                .max(fine_time.current)
                .max(fine_time.force),
            0.005,
        ),
        (
            "physical_residual",
            all.iter().map(|m| m.residual).fold(0.0_f64, f64::max),
            1e-9,
        ),
        (
            "weak_charge_balance",
            all.iter().map(|m| m.charge_balance).fold(0.0_f64, f64::max),
            1e-9,
        ),
        (
            "kinetic_lorentz_viscous_power_balance",
            all.iter().map(|m| m.power).fold(0.0_f64, f64::max),
            1e-9,
        ),
        (
            "boundary_error",
            all.iter().map(|m| m.boundary).fold(0.0_f64, f64::max),
            1e-10,
        ),
    ] {
        if !v.is_finite() || v > bound {
            return Err(format!("{name}={v:e} > {bound}"));
        }
        result = result.metric(name, v).maximum(name, bound);
    }
    for (name, select, limit) in [
        (
            "velocity",
            (|m: &Metrics| m.velocity) as fn(&Metrics) -> f64,
            2.7,
        ),
        ("potential", |m: &Metrics| m.potential, 2.7),
        ("velocity_gradient", |m: &Metrics| m.velocity_gradient, 1.8),
        (
            "potential_gradient",
            |m: &Metrics| m.potential_gradient,
            1.8,
        ),
        ("pressure", |m: &Metrics| m.pressure, 1.7),
        ("current", |m: &Metrics| m.current, 1.8),
        ("lorentz_force", |m: &Metrics| m.force, 1.8),
    ] {
        for (prefix, levels, ns, bound) in [
            ("spatial", &spatial, [2.0_f64, 4.0, 6.0], limit),
            ("temporal", &temporal, [8.0_f64, 16.0, 32.0], 1.8),
        ] {
            let rate = levels
                .windows(2)
                .enumerate()
                .map(|(i, m)| (select(&m[0]) / select(&m[1])).ln() / (ns[i + 1] / ns[i]).ln())
                .fold(f64::INFINITY, f64::min);
            let key = format!("{prefix}_{name}_rate");
            if !rate.is_finite() || rate < bound {
                return Err(format!("{key}={rate} below {bound}"));
            }
            result = result.metric(&key, rate).minimum(&key, bound);
        }
    }
    for (name, levels, ns, select, bound) in [
        (
            "spatial_joule_rate",
            &spatial,
            [2.0_f64, 4.0, 6.0],
            (|m: &Metrics| m.joule) as fn(&Metrics) -> f64,
            3.0,
        ),
        (
            "temporal_joule_rate",
            &temporal,
            [8.0_f64, 16.0, 32.0],
            |m: &Metrics| m.joule,
            1.8,
        ),
        (
            "spatial_divergence_rate",
            &spatial,
            [2.0_f64, 4.0, 6.0],
            |m: &Metrics| m.divergence,
            1.8,
        ),
    ] {
        let rate = levels
            .windows(2)
            .enumerate()
            .map(|(i, m)| (select(&m[0]) / select(&m[1])).ln() / (ns[i + 1] / ns[i]).ln())
            .fold(f64::INFINITY, f64::min);
        if !rate.is_finite() || rate < bound {
            return Err(format!("{name}={rate} below {bound}"));
        }
        result = result.metric(name, rate).minimum(name, bound);
    }
    if !negative.is_finite() || negative < 1e-3 {
        return Err(format!("omitted-term error {negative:e} below 1e-3"));
    }
    Ok(result.metric("minimum_omitted_term_error",negative).minimum("minimum_omitted_term_error",1e-3)
  .metric("minimum_history_change",all.iter().map(|m|m.change).fold(f64::INFINITY,f64::min)).minimum("minimum_history_change",0.1)
  .metric("minimum_free_field_unknowns",all.iter().map(|m|m.free).min().unwrap_or(0) as f64).minimum("minimum_free_field_unknowns",1.0)
  .metric("executed_time_steps",all.iter().map(|m|m.steps).sum::<usize>() as f64).minimum("executed_time_steps",88.0)
  .metric("negative_controls",5.0).minimum("negative_controls",5.0)
  .reference("independent divergence-free cubic velocity, cubic potential and linear pressure; strong momentum/charge sources, current, Lorentz force and Joule dissipation")
  .reference_checksum("analytic:3d-transient-incompressible-stokes-inductionless-mhd:v1")
  .input_checksum("P2-P1:n2-4-6:linear8:P3-P2:n2:exp8-16-32:linear8:rho1.3:mu.3:sigma2:B.4,.7,1.1:five-controls")
  .dimension("3D tetrahedral cube, three free velocity components, free potential and pressure with one gauge")
  .formulation("native electromagnetic two-way blocks and native mixed Stokes forms; Taylor-Hood incompressibility, CN velocity inertia/viscosity/Lorentz force and quasi-static potential; prescribed charge source; low-Re/low magnetic-Re model")
  .tolerance("P2 value/gradient rates>=2.7/1.8; pressure>=1.7; temporal rates>=1.8; exact P3/raw continuity/power/charge<=1e-9; five all-history controls>=1e-3"))
}
fn main() {
    sonium_qa::fem_physics_3d::run_bilinear_case(
        "fem.mhd-3d.v1",
        &MhdPotentialForm {
            electrical_conductivity: 1.0,
        },
        true,
    );
    let r = match run() {
        Ok(r) => r,
        Err(e) => QaResult::failed(CASE, e),
    };
    match r.to_json_line() {
        Ok(j) => println!("QA_RESULT: {j}"),
        Err(e) => {
            eprintln!("FAILED: {e}");
            std::process::exit(1);
        }
    }
    if r.status == QaStatus::Passed {
        println!("SUCCESS");
    } else {
        std::process::exit(1);
    }
}
