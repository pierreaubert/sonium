//! Nonlinear Cahn-Hilliard / quasi-static incompressible Stokes 3D MMS.
//! The caller composes the native mixed linearization, quartic bulk/tangent,
//! solved-velocity advection and solved-phase/chemical-gradient capillary load.
//! Manufactured phase and chemical Dirichlet data provide a forced open system;
//! mass balance includes volume source, advection and boundary diffusive flux.
use sonium_fem::assembly::mixed::assemble_bilinear_two_spaces_checked;
use sonium_fem::assembly::{TripletMatrix, assemble_bilinear, assemble_linear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::forms::{BilinearForm, LinearForm, LocalMatrix, LocalVector, SourceForm};
use sonium_fem::mesh::unit_cube_tetrahedra;
use sonium_fem::physics::fluid::phase_field_flow::{
    CapillaryForceForm, ChemicalPotentialDiffusionForm, ChemicalPotentialTangentForm,
    PhaseAdvectionForm, PhaseFieldFluid, PhaseMassForm, assemble_linearized_cahn_hilliard,
};
use sonium_fem::physics::fluid::stokes::forms::{
    StokesBodyForceForm, StokesDivergenceForm, StokesViscousForm,
};
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{BasisTabulation, FunctionSpace, H1Space, VectorH1Space};
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::{
    BlockPreconditionerConfig, CsrMatrix, FgmresConfig, IlutConfig, SaddlePointConfig,
    SaddlePointPreconditioner, SchurStrategy, SparseLuFactorization, fgmres_preconditioned,
};
const CASE: &str = "fem.phase-field-flow-3d-quartic-history.v1";
const ORDER: usize = 8;
const MATERIAL: PhaseFieldFluid = PhaseFieldFluid {
    density_minus: 1.3,
    density_plus: 1.3,
    viscosity_minus: 0.3,
    viscosity_plus: 0.3,
    surface_tension: 0.188_561_808_316_412_7,
    interface_thickness: 0.5,
    mobility: 0.2,
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
    NoAdvection,
    NoMobility,
    NoBulk,
    NoGradientEnergy,
    NoCapillary,
}
#[derive(Clone, Copy)]
struct Reference {
    phase: f64,
    phase_gradient: [f64; 3],
    chemical: f64,
    chemical_gradient: [f64; 3],
    velocity: [f64; 3],
    velocity_gradient: [[f64; 3]; 3],
    pressure: f64,
}
fn reference(p: [f64; 3], t: f64, law: Law) -> Reference {
    let [x, y, z] = p;
    let g = law.value(t);
    Reference {
        phase: g * 0.02 * (1.0 + x * x + 2.0 * y * y + 3.0 * z * z + x * y + x * z + y * z),
        phase_gradient: [
            g * 0.02 * (2.0 * x + y + z),
            g * 0.02 * (4.0 * y + x + z),
            g * 0.02 * (6.0 * z + x + y),
        ],
        chemical: g * 0.1 * (1.0 + 2.0 * x * x + y * y + z * z + x * y + 2.0 * x * z + y * z),
        chemical_gradient: [
            g * 0.1 * (4.0 * x + y + 2.0 * z),
            g * 0.1 * (2.0 * y + x + z),
            g * 0.1 * (2.0 * z + 2.0 * x + y),
        ],
        velocity: [
            g * 0.02 * (1.0 + y * y + y * z),
            g * 0.02 * (2.0 + z * z + z * x),
            g * 0.02 * (3.0 + x * x + x * y),
        ],
        velocity_gradient: [
            [0.0, g * 0.02 * (2.0 * y + z), g * 0.02 * y],
            [g * 0.02 * z, 0.0, g * 0.02 * (2.0 * z + x)],
            [g * 0.02 * (2.0 * x + y), g * 0.02 * x, 0.0],
        ],
        pressure: g * 0.05 * (x + 2.0 * y + 3.0 * z),
    }
}
fn source(p: [f64; 3], t: f64, law: Law) -> (f64, f64, [f64; 3]) {
    let r = reference(p, t, law);
    let g = law.value(t);
    let phase0 = reference(p, 0.0, law).phase;
    let transport = law.derivative(t) * phase0
        + (0..3)
            .map(|a| r.velocity[a] * r.phase_gradient[a])
            .sum::<f64>()
        - 0.2 * g * 0.8;
    let definition = 0.4 * (r.phase.powi(3) - r.phase) - 0.1 * g * 0.24 - r.chemical;
    let gradp = [0.05, 0.1, 0.15];
    let body =
        std::array::from_fn(|a| -0.3 * g * 0.04 + g * gradp[a] + r.phase * r.chemical_gradient[a]);
    (transport, definition, body)
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
    values: &[f64],
) -> (CsrMatrix<f64>, Vec<f64>) {
    for k in 0..a.nnz() {
        let (r, c) = (a.rows[k], a.cols[k]);
        if fixed[c] && !fixed[r] {
            rhs[r] -= a.values[k] * values[c];
        }
        if fixed[r] || fixed[c] {
            a.values[k] = 0.0;
        }
    }
    for (d, &b) in fixed.iter().enumerate() {
        if b {
            a.push(d, d, 1.0);
            rhs[d] = values[d];
        }
    }
    (a.to_csr(), rhs)
}

fn newton_solve(
    cached: &mut Option<SparseLuFactorization<f64>>,
    matrix: &CsrMatrix<f64>,
    rhs: &[f64],
    factorizations: &mut usize,
    refinements: &mut usize,
) -> Result<Vec<f64>, String> {
    if let Some(factor) = cached.as_ref()
        && factor.accepts_value_update(matrix)
        && let Ok(solved) = factor.solve_refined(matrix, rhs, 1e-12, 16)
    {
        *refinements += solved.iterations;
        if solved.converged {
            return Ok(solved.x);
        }
    }
    // Changed sparsity or unsuccessful residual correction requires a fresh
    // factorization. Every reused solve checks the current Newton matrix.
    let factor = SparseLuFactorization::factorize(matrix).map_err(|e| e.to_string())?;
    let solution = factor.solve(rhs).map_err(|e| e.to_string())?;
    *cached = Some(factor);
    *factorizations += 1;
    Ok(solution)
}
struct Point {
    phase: BasisTabulation,
    velocity: BasisTabulation,
    phase_dofs: Vec<usize>,
    velocity_dofs: Vec<usize>,
    weight: f64,
}
fn phase_values(p: &Point, state: &[f64], n: usize) -> (f64, [f64; 3]) {
    let c = p
        .phase_dofs
        .iter()
        .enumerate()
        .map(|(i, &d)| state[d] * p.phase.value(i))
        .sum();
    let dm = std::array::from_fn(|a| {
        p.phase_dofs
            .iter()
            .enumerate()
            .map(|(i, &d)| state[n + d] * p.phase.grad(i, a))
            .sum()
    });
    (c, dm)
}
fn velocity_values(p: &Point, state: &[f64]) -> [f64; 3] {
    std::array::from_fn(|a| {
        p.velocity_dofs
            .iter()
            .enumerate()
            .map(|(i, &d)| state[d] * p.velocity.vector_value(i, a))
            .sum()
    })
}
fn advection(points: &[Point], flow: &[f64], n: usize, control: Control) -> TripletMatrix<f64> {
    let mut matrix = TripletMatrix::new(n, n);
    if matches!(control, Control::NoAdvection) {
        return matrix;
    }
    for p in points {
        let u = velocity_values(p, flow);
        let mut local = LocalMatrix::zeros(p.phase_dofs.len());
        PhaseAdvectionForm {
            velocity: |_, _, _| u,
        }
        .integrate_point(&p.phase, &p.phase, p.weight, &mut local);
        for (i, &r) in p.phase_dofs.iter().enumerate() {
            for (j, &c) in p.phase_dofs.iter().enumerate() {
                matrix.push(r, c, local.get(i, j));
            }
        }
    }
    matrix
}
// Return native nonlinear bulk correction and its exact native point tangent;
// the cached mixed block already includes -beta*M + kappa*K.
fn nonlinear(
    points: &[Point],
    state: &[f64],
    n: usize,
    control: Control,
) -> (Vec<f64>, TripletMatrix<f64>) {
    let mut load = vec![0.0; n];
    let mut matrix = TripletMatrix::new(n, n);
    if matches!(control, Control::NoBulk) {
        return (load, matrix);
    }
    let beta = MATERIAL.double_well_coefficient();
    for p in points {
        let (c, _) = phase_values(p, state, n);
        let correction = MATERIAL.chemical_potential_bulk(c) + beta * c;
        let mut local = LocalMatrix::zeros(p.phase_dofs.len());
        ChemicalPotentialTangentForm {
            bulk_second_derivative: 3.0 * beta * c * c,
            gradient_energy_coefficient: 0.0,
        }
        .integrate_point(&p.phase, &p.phase, p.weight, &mut local);
        for (i, &r) in p.phase_dofs.iter().enumerate() {
            load[r] += correction * p.phase.value(i) * p.weight * p.phase.det_j;
            for (j, &c) in p.phase_dofs.iter().enumerate() {
                matrix.push(r, c, local.get(i, j));
            }
        }
    }
    (load, matrix)
}
fn capillary(points: &[Point], state: &[f64], n: usize, nu: usize, control: Control) -> Vec<f64> {
    let mut load = vec![0.0; nu];
    if matches!(control, Control::NoCapillary) {
        return load;
    }
    for p in points {
        let (c, dm) = phase_values(p, state, n);
        let mut local = LocalVector::zeros(p.velocity_dofs.len());
        CapillaryForceForm {
            phase: |_, _, _| c,
            chemical_potential_gradient: |_, _, _| dm,
        }
        .integrate_point(&p.velocity, p.weight, &mut local);
        for (i, &d) in p.velocity_dofs.iter().enumerate() {
            load[d] += local.values[i];
        }
    }
    load
}
#[derive(Debug)]
struct Metrics {
    phase: f64,
    chemical: f64,
    phase_flux: f64,
    chemical_flux: f64,
    velocity: f64,
    velocity_gradient: f64,
    pressure: f64,
    capillary: f64,
    energy: f64,
    mass: f64,
    divergence: f64,
    residual: f64,
    boundary: f64,
    mass_balance: f64,
    chemical_balance: f64,
    power: f64,
    history: f64,
    change: f64,
    steps: usize,
    free: usize,
    newton: usize,
    picard: usize,
    factorizations: usize,
    refinements: usize,
    flow_iterations: usize,
}
fn solve(
    nmesh: usize,
    degree: PolynomialDegree,
    steps: usize,
    law: Law,
    control: Control,
) -> Result<Metrics, String> {
    let mut mesh = unit_cube_tetrahedra(nmesh);
    for b in &mut mesh.boundaries {
        b.marker = 1;
    }
    let scalar = H1Space::new(degree);
    let vector = VectorH1Space::new(PolynomialDegree::P2, 3);
    let pressure = H1Space::new(PolynomialDegree::P1);
    let n = scalar.num_global_dofs(&mesh);
    let nu = vector.num_global_dofs(&mesh);
    let np = pressure.num_global_dofs(&mesh);
    let coords = scalar
        .dof_coordinates(&mesh)
        .ok_or("phase coordinates missing")?;
    let vc = vector
        .dof_coordinates(&mesh)
        .ok_or("velocity coordinates missing")?;
    let pc = pressure
        .dof_coordinates(&mesh)
        .ok_or("pressure coordinates missing")?;
    let mut fixed = vec![false; 2 * n];
    for d in scalar.boundary_dofs(&mesh, 1) {
        fixed[d] = true;
        fixed[n + d] = true;
    }
    let mut ffixed = vec![false; nu + np];
    for d in vector.boundary_dofs(&mesh, 1) {
        ffixed[d] = true;
    }
    ffixed[nu] = true;
    let free = [
        fixed[..n].iter().filter(|&&b| !b).count(),
        ffixed[..nu].iter().filter(|&&b| !b).count() / 3,
        np - 1,
    ]
    .into_iter()
    .min()
    .unwrap_or(0);
    if free == 0 {
        return Err("no free phase/chemical/flow/pressure unknowns".into());
    }
    let prescribed = |t: f64| {
        (0..2 * n)
            .map(|d| {
                let r = reference(coords[d % n], t, law);
                if d < n { r.phase } else { r.chemical }
            })
            .collect::<Vec<_>>()
    };
    let fprescribed = |t: f64| {
        let mut v = Vec::with_capacity(nu + np);
        for (d, p) in vc.iter().enumerate() {
            v.push(reference(*p, t, law).velocity[d % 3]);
        }
        for p in &pc {
            v.push(reference(*p, t, law).pressure);
        }
        v
    };
    let material = MATERIAL.validate().map_err(|e| format!("material {e:?}"))?;
    let dt = 1.0 / steps as f64;
    let base =
        assemble_linearized_cahn_hilliard(&mesh, degree, material, dt / 2.0, 0.0, |_, _, _| {
            [0.0; 3]
        });
    let mass = assemble_bilinear(
        &PhaseMassForm { coefficient: 1.0 },
        &scalar,
        &scalar,
        &mesh,
        ORDER,
    );
    let m = mass.to_csr();
    let capacity = if matches!(control, Control::NoCapacity) {
        0.0
    } else {
        1.0
    };
    let mut definition = base.block_10.clone();
    if matches!(control, Control::NoBulk | Control::NoGradientEnergy) {
        definition = assemble_bilinear(
            &ChemicalPotentialTangentForm {
                bulk_second_derivative: if matches!(control, Control::NoBulk) {
                    0.0
                } else {
                    -material.double_well_coefficient()
                },
                gradient_energy_coefficient: if matches!(control, Control::NoGradientEnergy) {
                    0.0
                } else {
                    material.gradient_energy_coefficient()
                },
            },
            &scalar,
            &scalar,
            &mesh,
            ORDER,
        );
    }
    let b = definition.to_csr();
    let diffusion = base.block_01.clone();
    let diffusion_scale = if matches!(control, Control::NoMobility) {
        0.0
    } else {
        1.0
    };
    let d = diffusion.to_csr();
    let viscosity = material.viscosity(0.0);
    let viscous = assemble_bilinear(
        &StokesViscousForm::new(viscosity),
        &vector,
        &vector,
        &mesh,
        4,
    );
    let v = viscous.to_csr();
    let divergence =
        assemble_bilinear_two_spaces_checked(&StokesDivergenceForm, &vector, &pressure, &mesh, 4)
            .map_err(|e| e.to_string())?;
    let g = divergence.to_csr();
    let mut flow_matrix = TripletMatrix::new(nu + np, nu + np);
    block(&mut flow_matrix, &viscous, 0, 0, 1.0);
    block(&mut flow_matrix, &divergence, 0, nu, 1.0);
    for k in 0..divergence.nnz() {
        flow_matrix.push(
            nu + divergence.cols[k],
            divergence.rows[k],
            divergence.values[k],
        );
    }
    let flow_raw = flow_matrix.to_csr();
    let constrained_flow = lift(
        flow_matrix.clone(),
        vec![0.0; nu + np],
        &ffixed,
        &fprescribed(0.0),
    )
    .0;
    let flow_factor = if nmesh < 6 {
        Some(SparseLuFactorization::factorize(&constrained_flow).map_err(|e| e.to_string())?)
    } else {
        None
    };
    let flow_preconditioner = if nmesh >= 6 {
        let schur = assemble_bilinear(
            &PhaseMassForm {
                coefficient: 1.0 / viscosity,
            },
            &pressure,
            &pressure,
            &mesh,
            2,
        );
        let mut pinned = vec![false; np];
        pinned[0] = true;
        let pressure_schur = lift(schur, vec![0.0; np], &pinned, &vec![0.0; np]).0;
        Some(SaddlePointPreconditioner::from_matrix_with_schur(
            &constrained_flow,
            &pressure_schur,
            SaddlePointConfig {
                block1_size: nu,
                block1_config: BlockPreconditionerConfig::Ilut(IlutConfig::default()),
                block2_config: BlockPreconditionerConfig::Ilut(IlutConfig::default()),
                schur_strategy: SchurStrategy::External,
                upper_triangular: true,
            },
        ))
    } else {
        None
    };
    let mut points = Vec::new();
    let coupling_order = if degree == PolynomialDegree::P1 {
        4
    } else {
        ORDER
    };
    for e in 0..mesh.num_elements() {
        let pd = scalar.local_to_global(&mesh, e);
        let vd = vector.local_to_global(&mesh, e);
        for qp in QuadratureRule::new(mesh.elements[e].element_type, coupling_order).iter() {
            points.push(Point {
                phase: scalar.tabulate(&mesh, e, qp.xi(), qp.eta(), qp.zeta()),
                velocity: vector.tabulate(&mesh, e, qp.xi(), qp.eta(), qp.zeta()),
                phase_dofs: pd.clone(),
                velocity_dofs: vd.clone(),
                weight: qp.weight,
            });
        }
    }
    let sources = |t: f64| {
        let fc = assemble_linear(
            &SourceForm::new(move |x, y, z| source([x, y, z], t, law).0),
            &scalar,
            &mesh,
            ORDER,
        );
        let fm = assemble_linear(
            &SourceForm::new(move |x, y, z| source([x, y, z], t, law).1),
            &scalar,
            &mesh,
            ORDER,
        );
        let fu = assemble_linear(
            &StokesBodyForceForm::new(move |p| source(p, t, law).2.to_vec()),
            &vector,
            &mesh,
            ORDER,
        );
        (fc, fm, fu)
    };
    let mut state = prescribed(0.0);
    let initial = state.clone();
    let mut flow = fprescribed(0.0);
    let mut old_source = sources(0.0);
    let mut residual = 0.0_f64;
    let mut boundary = 0.0_f64;
    let mut mass_balance = 0.0_f64;
    let mut chemical_balance = 0.0_f64;
    let mut power = 0.0_f64;
    let mut history = 0.0_f64;
    let mut newton = 0;
    let mut picard = 0;
    let mut cached_phase_factor = None;
    let mut factorizations = 0;
    let mut refinements = 0;
    let mut flow_iterations = 0;
    for step in 1..=steps {
        let time = step as f64 * dt;
        let exact = prescribed(time);
        let fexact = fprescribed(time);
        let old = state.clone();
        let new_source = sources(time);
        let old_adv = advection(&points, &flow, n, control).to_csr();
        let oldm = m.matvec(&old[..n]);
        let olda = old_adv.matvec(&old[..n]);
        let oldd = d.matvec(&old[n..]);
        let rhs: Vec<_> = (0..n)
            .map(|a| {
                capacity * oldm[a] / dt - 0.5 * (olda[a] + diffusion_scale * oldd[a])
                    + 0.5 * (old_source.0[a] + new_source.0[a])
            })
            .collect();
        let mut converged = false;
        for _ in 0..24 {
            picard += 1;
            let oldflow = flow.clone();
            let adv = advection(&points, &flow, n, control);
            let mut operator = TripletMatrix::new(2 * n, 2 * n);
            block(&mut operator, &mass, 0, 0, capacity / dt);
            block(&mut operator, &adv, 0, 0, 0.5);
            block(&mut operator, &diffusion, 0, n, 0.5 * diffusion_scale);
            block(&mut operator, &definition, n, 0, 1.0);
            block(&mut operator, &base.block_11, n, n, 1.0);
            let raw = operator.to_csr();
            let mut solved = false;
            for _ in 0..16 {
                let (bulk, tangent) = nonlinear(&points, &state, n, control);
                let action = raw.matvec(&state);
                let mut r: Vec<_> = (0..2 * n)
                    .map(|a| {
                        if a < n {
                            action[a] - rhs[a]
                        } else {
                            action[a] + bulk[a - n] - new_source.1[a - n]
                        }
                    })
                    .collect();
                let norm = r
                    .iter()
                    .enumerate()
                    .filter(|(a, _)| !fixed[*a])
                    .map(|(_, v)| v * v)
                    .sum::<f64>()
                    .sqrt();
                let scale = rhs
                    .iter()
                    .chain(&new_source.1)
                    .map(|v| v * v)
                    .sum::<f64>()
                    .sqrt()
                    .max(1e-12);
                let boundary_error = state
                    .iter()
                    .zip(&exact)
                    .enumerate()
                    .filter(|(a, _)| fixed[*a])
                    .map(|(_, (a, b))| (a - b).abs())
                    .fold(0.0_f64, f64::max);
                if norm / scale < 1e-12 && boundary_error < 1e-12 {
                    solved = true;
                    break;
                }
                for x in &mut r {
                    *x = -*x;
                }
                let mut jac = operator.clone();
                block(&mut jac, &tangent, n, 0, 1.0);
                let delta_bc: Vec<_> = exact.iter().zip(&state).map(|(a, b)| a - b).collect();
                let (j, load) = lift(jac, r, &fixed, &delta_bc);
                let delta = newton_solve(
                    &mut cached_phase_factor,
                    &j,
                    &load,
                    &mut factorizations,
                    &mut refinements,
                )?;
                for (a, dx) in state.iter_mut().zip(delta) {
                    *a += dx;
                }
                newton += 1;
            }
            if !solved {
                return Err("quartic CH Newton failed".into());
            }
            let cap = capillary(&points, &state, n, nu, control);
            let mut frhs = vec![0.0; nu + np];
            for a in 0..nu {
                frhs[a] = new_source.2[a] + cap[a];
            }
            let flow_rhs = lift(flow_matrix.clone(), frhs.clone(), &ffixed, &fexact).1;
            flow = if let Some(factor) = &flow_factor {
                factor.solve(&flow_rhs).map_err(|e| e.to_string())?
            } else {
                let preconditioner = flow_preconditioner
                    .as_ref()
                    .ok_or("missing Stokes preconditioner")?;
                let config = FgmresConfig {
                    max_iterations: 1000,
                    restart: 100,
                    tolerance: 1e-13,
                    print_interval: 0,
                    convergence_check_interval: 1,
                    exact_residual_check_interval: 1,
                };
                let solved = fgmres_preconditioned(
                    &constrained_flow,
                    preconditioner,
                    &flow_rhs,
                    Some(&flow),
                    &config,
                );
                if !solved.converged {
                    return Err(format!("Stokes FGMRES failed at n={nmesh}"));
                }
                flow_iterations += solved.iterations;
                solved.x
            };
            let change = flow
                .iter()
                .zip(&oldflow)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
                / flow.iter().map(|v| v * v).sum::<f64>().sqrt();
            if change < 1e-12 {
                // Reassemble transport using the final solved velocity, not the previous
                // Picard input. Fresh physical residual covers the coupled system.
                let actual_adv = advection(&points, &flow, n, control).to_csr();
                let mc = m.matvec(&state[..n]);
                let ac = actual_adv.matvec(&state[..n]);
                let dm = d.matvec(&state[n..]);
                let bc = b.matvec(&state[..n]);
                let mm = m.matvec(&state[n..]);
                let (bulk, _) = nonlinear(&points, &state, n, control);
                let rc: Vec<_> = (0..n)
                    .map(|a| {
                        capacity * mc[a] / dt + 0.5 * (ac[a] + diffusion_scale * dm[a]) - rhs[a]
                    })
                    .collect();
                let rm: Vec<_> = (0..n)
                    .map(|a| bc[a] - mm[a] + bulk[a] - new_source.1[a])
                    .collect();
                let scale = rhs
                    .iter()
                    .chain(&new_source.1)
                    .map(|v| v * v)
                    .sum::<f64>()
                    .sqrt()
                    .max(1e-12);
                residual = residual.max(
                    (0..n)
                        .filter(|&a| !fixed[a])
                        .map(|a| rc[a] * rc[a] + rm[a] * rm[a])
                        .sum::<f64>()
                        .sqrt()
                        / scale,
                );
                let stored = capacity * (mc.iter().sum::<f64>() - oldm.iter().sum::<f64>()) / dt;
                let transport = 0.5 * (ac.iter().sum::<f64>() + olda.iter().sum::<f64>());
                let injected =
                    0.5 * (old_source.0.iter().sum::<f64>() + new_source.0.iter().sum::<f64>());
                let outward = -(0..n).filter(|&a| fixed[a]).map(|a| rc[a]).sum::<f64>();
                mass_balance = mass_balance.max(
                    (stored + transport + outward - injected).abs()
                        / stored
                            .abs()
                            .max(transport.abs())
                            .max(outward.abs())
                            .max(injected.abs())
                            .max(1e-12),
                );
                let beta = if matches!(control, Control::NoBulk) {
                    0.0
                } else {
                    material.double_well_coefficient()
                };
                let net_bulk = bulk.iter().sum::<f64>() - beta * mc.iter().sum::<f64>();
                let chemical_out = -(0..n).filter(|&a| fixed[a]).map(|a| rm[a]).sum::<f64>();
                let chemical_in = new_source.1.iter().sum::<f64>();
                chemical_balance = chemical_balance.max(
                    (net_bulk + chemical_out - mm.iter().sum::<f64>() - chemical_in).abs()
                        / chemical_in
                            .abs()
                            .max(net_bulk.abs())
                            .max(mm.iter().sum::<f64>().abs())
                            .max(1e-12),
                );
                let faction = flow_raw.matvec(&flow);
                let fscale = frhs.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
                residual = residual.max(
                    (0..nu + np)
                        .filter(|&a| !ffixed[a] || a >= nu)
                        .map(|a| (faction[a] - frhs[a]).powi(2))
                        .sum::<f64>()
                        .sqrt()
                        / fscale,
                );
                let viscous_action = v.matvec(&flow[..nu]);
                let pressure_action = g.matvec(&flow[nu..]);
                let dissipation = (0..nu).map(|a| flow[a] * viscous_action[a]).sum::<f64>();
                let pressure_work = (0..nu).map(|a| flow[a] * pressure_action[a]).sum::<f64>();
                let body_work = (0..nu).map(|a| flow[a] * new_source.2[a]).sum::<f64>();
                let cap_work = (0..nu).map(|a| flow[a] * cap[a]).sum::<f64>();
                let wall = (0..nu)
                    .filter(|&a| ffixed[a])
                    .map(|a| flow[a] * (faction[a] - frhs[a]))
                    .sum::<f64>();
                power = power.max(
                    (dissipation + pressure_work - body_work - cap_work - wall).abs()
                        / dissipation
                            .abs()
                            .max(body_work.abs())
                            .max(cap_work.abs())
                            .max(wall.abs())
                            .max(1e-12),
                );
                converged = true;
                break;
            }
        }
        if !converged {
            return Err("CH/capillary/advection Picard failed".into());
        }
        for ranges in [0..n, n..2 * n] {
            let mut e = 0.0;
            let mut norm = 0.0;
            for a in ranges {
                if fixed[a] {
                    boundary = boundary.max((state[a] - exact[a]).abs());
                } else {
                    e += (state[a] - exact[a]).powi(2);
                    norm += exact[a].powi(2);
                }
            }
            history = history.max((e / norm).sqrt());
        }
        for ranges in [0..nu, nu..nu + np] {
            let mut e = 0.0;
            let mut norm = 0.0;
            for a in ranges {
                if ffixed[a] {
                    boundary = boundary.max((flow[a] - fexact[a]).abs());
                } else {
                    e += (flow[a] - fexact[a]).powi(2);
                    norm += fexact[a].powi(2);
                }
            }
            history = history.max((e / norm).sqrt());
        }
        if state.iter().chain(&flow).any(|v| !v.is_finite()) {
            return Err("nonfinite CH/Stokes history".into());
        }
        if matches!(control, Control::Full) && state[..n].iter().any(|c| c.abs() > 1.0) {
            return Err("reference phase outside admissible interval".into());
        }
        old_source = new_source;
    }
    let mut error = [0.0; 8];
    let mut norm = [0.0; 8];
    let mut energy = 0.0;
    let mut re = 0.0;
    let mut mass_value = 0.0;
    let mut refmass = 0.0;
    let mut div = 0.0;
    let mut gu = 0.0;
    for e in 0..mesh.num_elements() {
        let pd = scalar.local_to_global(&mesh, e);
        let vd = vector.local_to_global(&mesh, e);
        let prd = pressure.local_to_global(&mesh, e);
        for qp in QuadratureRule::new(mesh.elements[e].element_type, ORDER).iter() {
            let tab = scalar.tabulate(&mesh, e, qp.xi(), qp.eta(), qp.zeta());
            let vt = vector.tabulate(&mesh, e, qp.xi(), qp.eta(), qp.zeta());
            let pt = pressure.tabulate(&mesh, e, qp.xi(), qp.eta(), qp.zeta());
            let w = qp.weight * tab.det_j;
            let cv = pd
                .iter()
                .enumerate()
                .map(|(i, &d)| state[d] * tab.value(i))
                .sum::<f64>();
            let mv = pd
                .iter()
                .enumerate()
                .map(|(i, &d)| state[n + d] * tab.value(i))
                .sum::<f64>();
            let dc: [f64; 3] = std::array::from_fn(|a| {
                pd.iter()
                    .enumerate()
                    .map(|(i, &d)| state[d] * tab.grad(i, a))
                    .sum()
            });
            let dm: [f64; 3] = std::array::from_fn(|a| {
                pd.iter()
                    .enumerate()
                    .map(|(i, &d)| state[n + d] * tab.grad(i, a))
                    .sum()
            });
            let uv: [f64; 3] = std::array::from_fn(|a| {
                vd.iter()
                    .enumerate()
                    .map(|(i, &d)| flow[d] * vt.vector_value(i, a))
                    .sum()
            });
            let du: [[f64; 3]; 3] = std::array::from_fn(|a| {
                std::array::from_fn(|b| {
                    vd.iter()
                        .enumerate()
                        .filter(|(i, _)| i % 3 == a)
                        .map(|(i, &d)| flow[d] * vt.grad(i, b))
                        .sum()
                })
            });
            let pv = prd
                .iter()
                .enumerate()
                .map(|(i, &d)| flow[nu + d] * pt.value(i))
                .sum::<f64>();
            let r = reference(tab.physical_coords, 1.0, law);
            for (a, (actual, expected)) in [(cv, r.phase), (mv, r.chemical), (pv, r.pressure)]
                .into_iter()
                .enumerate()
            {
                let idx = [0, 1, 6][a];
                error[idx] += w * (actual - expected).powi(2);
                norm[idx] += w * expected.powi(2);
            }
            for a in 0..3 {
                for (idx, actual, expected) in [
                    (2, dc[a], r.phase_gradient[a]),
                    (3, dm[a], r.chemical_gradient[a]),
                    (4, uv[a], r.velocity[a]),
                    (7, -cv * dm[a], -r.phase * r.chemical_gradient[a]),
                ] {
                    error[idx] += w * (actual - expected).powi(2);
                    norm[idx] += w * expected.powi(2);
                }
                for b in 0..3 {
                    error[5] += w * (du[a][b] - r.velocity_gradient[a][b]).powi(2);
                    norm[5] += w * r.velocity_gradient[a][b].powi(2);
                    gu += w * du[a][b].powi(2);
                }
            }
            energy += w
                * (material.bulk_free_energy(cv)
                    + 0.5
                        * material.gradient_energy_coefficient()
                        * dc.iter().map(|v| v * v).sum::<f64>());
            re += w
                * (0.1 * (r.phase * r.phase - 1.0).powi(2)
                    + 0.05 * r.phase_gradient.iter().map(|v| v * v).sum::<f64>());
            mass_value += w * cv;
            refmass += w * r.phase;
            div += w * (du[0][0] + du[1][1] + du[2][2]).powi(2);
        }
    }
    let error = std::array::from_fn::<_, 8, _>(|a| (error[a] / norm[a]).sqrt());
    let change = state[..n]
        .iter()
        .zip(&initial[..n])
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
        / initial[..n].iter().map(|v| v * v).sum::<f64>().sqrt();
    Ok(Metrics {
        phase: error[0],
        chemical: error[1],
        phase_flux: error[2],
        chemical_flux: error[3],
        velocity: error[4],
        velocity_gradient: error[5],
        pressure: error[6],
        capillary: error[7],
        energy: (energy / re - 1.0).abs(),
        mass: (mass_value / refmass - 1.0).abs(),
        divergence: (div / gu).sqrt(),
        residual,
        boundary,
        mass_balance,
        chemical_balance,
        power,
        history,
        change,
        steps,
        free,
        newton,
        picard,
        factorizations,
        refinements,
        flow_iterations,
    })
}
fn run() -> Result<QaResult, String> {
    let patch = solve(2, PolynomialDegree::P2, 8, Law::Linear, Control::Full)?;
    println!("phase-field exact P2: {patch:?}");
    let exact = patch
        .phase
        .max(patch.chemical)
        .max(patch.phase_flux)
        .max(patch.chemical_flux)
        .max(patch.velocity)
        .max(patch.velocity_gradient)
        .max(patch.pressure)
        .max(patch.capillary)
        .max(patch.energy)
        .max(patch.mass)
        .max(patch.divergence);
    if exact > 1e-9 {
        return Err(format!("exact quartic/P2 patch error={exact:e}"));
    }
    // The spatial, temporal, and control legs are independent solves
    // (own mesh, locals only); run them concurrently and rejoin in
    // order. Each solve is deterministic, so results match the serial
    // schedule up to floating-point summation order inside shared
    // parallel kernels — well within the budgeted tolerances below.
    use rayon::prelude::*;
    let spatial: Vec<Metrics> = [6, 9, 12]
        .into_par_iter()
        .map(|n| solve(n, PolynomialDegree::P1, 8, Law::Linear, Control::Full))
        .collect::<Result<Vec<_>, _>>()?;
    for (n, m) in [6, 9, 12].iter().zip(spatial.iter()) {
        println!("phase-field P1/P2 n={n}: {m:?}");
    }
    let temporal: Vec<Metrics> = [16, 32, 64]
        .into_par_iter()
        .map(|steps| {
            solve(
                2,
                PolynomialDegree::P2,
                steps,
                Law::Exponential,
                Control::Full,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (steps, m) in [16, 32, 64].iter().zip(temporal.iter()) {
        println!("phase-field P2 time={steps}: {m:?}");
    }
    let controls = [
        Control::NoCapacity,
        Control::NoAdvection,
        Control::NoMobility,
        Control::NoBulk,
        Control::NoGradientEnergy,
        Control::NoCapillary,
    ];
    let control_runs: Vec<Metrics> = controls
        .into_par_iter()
        .map(|c| solve(2, PolynomialDegree::P2, 8, Law::Linear, c))
        .collect::<Result<Vec<_>, _>>()?;
    let mut negative = f64::INFINITY;
    for (c, m) in [
        Control::NoCapacity,
        Control::NoAdvection,
        Control::NoMobility,
        Control::NoBulk,
        Control::NoGradientEnergy,
        Control::NoCapillary,
    ]
    .iter()
    .zip(control_runs.iter())
    {
        println!("phase-field {c:?}: {m:?}");
        negative = negative.min(m.history);
    }
    let all: Vec<_> = spatial.iter().chain(&temporal).chain([&patch]).collect();
    let f = &spatial[2];
    let t = &temporal[2];
    let mut result = QaResult::passed(CASE)
        .requested_backend("cpu")
        .actual_backend("cpu");
    for (name, value, bound) in [
        ("fine_phase_error", f.phase, 0.005),
        ("fine_chemical_error", f.chemical, 0.005),
        ("fine_phase_flux_error", f.phase_flux, 0.1),
        ("fine_chemical_flux_error", f.chemical_flux, 0.1),
        ("fine_velocity_error", f.velocity, 0.01),
        ("fine_pressure_error", f.pressure, 0.1),
        ("fine_capillary_error", f.capillary, 0.1),
        ("fine_free_energy_error", f.energy, 0.005),
        ("fine_mass_error", f.mass, 0.01),
        // Measured fine-grid divergence error = 9.0e-5; gate at ~5x.
        ("fine_divergence_error", f.divergence, 5.0e-4),
        ("exact_p2_error", exact, 1e-9),
        (
            "fine_temporal_error",
            t.phase
                .max(t.chemical)
                .max(t.velocity)
                .max(t.pressure)
                .max(t.capillary),
            0.005,
        ),
        (
            "physical_coupled_residual",
            all.iter().map(|m| m.residual).fold(0.0_f64, f64::max),
            1e-9,
        ),
        (
            "source_aware_phase_mass_balance",
            all.iter().map(|m| m.mass_balance).fold(0.0_f64, f64::max),
            1e-9,
        ),
        (
            "weak_chemical_definition_balance",
            all.iter()
                .map(|m| m.chemical_balance)
                .fold(0.0_f64, f64::max),
            1e-9,
        ),
        (
            "stokes_capillary_power_balance",
            all.iter().map(|m| m.power).fold(0.0_f64, f64::max),
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
    for (name, select, limit) in [
        ("phase", (|m: &Metrics| m.phase) as fn(&Metrics) -> f64, 1.8),
        ("chemical", |m: &Metrics| m.chemical, 1.8),
        ("phase_flux", |m: &Metrics| m.phase_flux, 0.9),
        ("chemical_flux", |m: &Metrics| m.chemical_flux, 0.9),
        ("velocity", |m: &Metrics| m.velocity, 1.7),
        ("capillary", |m: &Metrics| m.capillary, 0.9),
    ] {
        for (prefix, levels, ns, bound) in [
            ("spatial", &spatial, [6.0_f64, 9.0, 12.0], limit),
            ("temporal", &temporal, [16.0_f64, 32.0, 64.0], 1.8),
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
    if !negative.is_finite() || negative < 1e-3 {
        return Err(format!(
            "minimum omitted-term error={negative:e} below 1e-3"
        ));
    }
    Ok(result.metric("minimum_omitted_term_error",negative).minimum("minimum_omitted_term_error",1e-3).metric("minimum_history_change",all.iter().map(|m|m.change).fold(f64::INFINITY,f64::min)).minimum("minimum_history_change",0.1).metric("minimum_free_field_unknowns",all.iter().map(|m|m.free).min().unwrap_or(0)as f64).minimum("minimum_free_field_unknowns",1.0).metric("executed_time_steps",all.iter().map(|m|m.steps).sum::<usize>()as f64).minimum("executed_time_steps",144.0).metric("executed_newton_solves",all.iter().map(|m|m.newton).sum::<usize>()as f64).minimum("executed_newton_solves",144.0).metric("executed_flow_picard_solves",all.iter().map(|m|m.picard).sum::<usize>()as f64).minimum("executed_flow_picard_solves",144.0).metric("phase_factorizations",all.iter().map(|m|m.factorizations).sum::<usize>()as f64).minimum("phase_factorizations",7.0).metric("jacobian_refinement_corrections",all.iter().map(|m|m.refinements).sum::<usize>()as f64).minimum("jacobian_refinement_corrections",1.0).metric("stokes_fgmres_iterations",all.iter().map(|m|m.flow_iterations).sum::<usize>()as f64).minimum("stokes_fgmres_iterations",1.0).metric("negative_controls",6.0).minimum("negative_controls",6.0)
 .reference("independent quadratic phase/chemical/three divergence-free velocity fields and linear pressure, quartic-bulk strong sources, capillary load, source-aware inventory and quartic free energy")
 .reference_checksum("analytic:3d-quartic-cahn-hilliard-stokes-cn:v1")
 .input_checksum("phaseP1:n6-9-12:linear8:phaseP2:n2:exp16-32-64:linear8:StokesP2-P1:beta.4:kappa.1:mobility.2:viscosity.3:six-controls")
 .dimension("3D tetrahedral cube; free phase, chemical potential, three velocity components and pressure with one gauge")
 .formulation("native mixed CH baseline, quartic material bulk/native Newton tangent, actual solved-velocity advection and solved-phase/chemical-gradient capillary load; CN phase with coupled quasi-static Taylor-Hood Stokes Picard; equal phase densities/viscosities; prescribed volume sources and Dirichlet data")
 .tolerance("phase/chemical values>=1.8 spatial/time rates, flux>=.9/1.8, velocity>=1.7/1.8; exact P2/raw residual/mass/chemical/power<=1e-9; six history controls>=1e-3"))
}
fn main() {
    sonium_qa::fem_physics_3d::run_bilinear_case(
        "fem.phase-field-flow-3d.v1",
        &ChemicalPotentialDiffusionForm { mobility: 1.0 },
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
