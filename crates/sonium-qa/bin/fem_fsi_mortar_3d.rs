//! Dynamic linear compressible viscous fluid / elastic solid mortar benchmark.
//! Both 3D volumes evolve on a fixed reference geometry. Natural exterior
//! tractions and volume sources are derived independently of assembled matrices.
//! Affine spatial fields are exactly representable on nonmatching P1 meshes;
//! exponential time histories isolate CN error, and linear time gives an exact patch.
//!
//! Scope is deliberately limited to that affine patch contract on a fixed
//! reference geometry: it exercises mortar coupling assembly and Crank-
//! Nicolson time error for exactly representable fields. It does not solve
//! a moving-mesh (ALE) problem, nonlinear flow, or non-affine discretization
//! accuracy; do not cite this binary as evidence for those.
use sonium_fem::assembly::mixed::assemble_bilinear_two_spaces_checked;
use sonium_fem::assembly::{TripletMatrix, assemble_bilinear, assemble_linear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, try_apply_traction_vector};
use sonium_fem::forms::{MassForm, SourceForm, VectorMassForm};
use sonium_fem::mesh::boundary_mesh::create_boundary_submesh;
use sonium_fem::mesh::{Mesh, unit_cube_tetrahedra};
use sonium_fem::physics::coupled::fsi::{FsiMortarConfig, FsiMortarOperators, FsiMortarSurface};
use sonium_fem::physics::fluid::stokes::forms::StokesDivergenceForm;
use sonium_fem::physics::solid::elasticity::{BodyForceForm, ElasticityForm, IsotropicMaterial};
use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::{CsrMatrix, SparseLuFactorization};
const CASE: &str = "fem.fsi-mortar-3d-dynamic.v1";
#[derive(Clone, Copy, Debug)]
enum Law {
    Linear,
    Exponential,
}
#[derive(Clone, Copy, Debug)]
enum Control {
    Full,
    NoCoupling,
    NoFluidInertia,
    NoSolidInertia,
    NoElasticity,
}
struct Reference {
    fluid: [f64; 3],
    pressure: f64,
    solid: [f64; 3],
    displacement: [f64; 3],
    fluid_stress: [[f64; 3]; 3],
    solid_stress: [[f64; 3]; 3],
    multiplier: [f64; 3],
}
fn reference(p: [f64; 3], t: f64, law: Law) -> Reference {
    let g = match law {
        Law::Linear => 1.0 + t,
        Law::Exponential => t.exp(),
    };
    let velocity_scale = match law {
        Law::Linear => 1.0,
        Law::Exponential => g,
    };
    let mut solid_gradient = [[0.02, 0.0, 0.0], [0.0, 0.03, 0.0], [0.0, 0.0, 0.04]];
    let mut fluid_gradient = [[0.02, 0.0, 0.0], [0.0, 0.03, 0.0], [0.0, 0.0, 0.02]];
    if matches!(law, Law::Exponential) {
        solid_gradient[0][2] = 0.01;
        solid_gradient[1][2] = 0.015;
        solid_gradient[2][0] = 0.005;
        solid_gradient[2][1] = -0.006;
        fluid_gradient[0][2] = 0.035;
        fluid_gradient[1][2] = 0.030;
        fluid_gradient[2][0] = 0.005;
        fluid_gradient[2][1] = -0.006;
    }
    let constants = [0.1, 0.2, 0.3];
    let w: [f64; 3] = std::array::from_fn(|a| {
        constants[a] + (0..3).map(|b| solid_gradient[a][b] * p[b]).sum::<f64>()
    });
    let u: [f64; 3] = std::array::from_fn(|a| {
        constants[a] + (0..3).map(|b| fluid_gradient[a][b] * p[b]).sum::<f64>()
    });
    let pressure = match law {
        Law::Linear => 0.012 - 0.136 * g,
        Law::Exponential => -0.124 * g,
    };
    let fluid_stress = std::array::from_fn(|a| {
        std::array::from_fn(|b| {
            0.3 * velocity_scale * (fluid_gradient[a][b] + fluid_gradient[b][a])
                - if a == b { pressure } else { 0.0 }
        })
    });
    let solid_stress = std::array::from_fn(|a| {
        std::array::from_fn(|b| {
            g * (0.8 * (solid_gradient[a][b] + solid_gradient[b][a])
                + if a == b { 0.8 * 0.09 } else { 0.0 })
        })
    });
    Reference {
        fluid: u.map(|v| velocity_scale * v),
        pressure,
        solid: w.map(|v| velocity_scale * v),
        displacement: w.map(|v| g * v),
        fluid_stress,
        solid_stress,
        multiplier: std::array::from_fn(|a| -solid_stress[a][2]),
    }
}
fn volume_sources(p: [f64; 3], t: f64, law: Law) -> (Vec<f64>, f64, Vec<f64>) {
    let r = reference(p, t, law);
    match law {
        Law::Linear => (vec![0.0; 3], 0.002, vec![0.0; 3]),
        Law::Exponential => (
            r.fluid.map(|v| 1.3 * v).to_vec(),
            0.008 * t.exp(),
            r.solid.map(|v| 1.7 * v).to_vec(),
        ),
    }
}
fn mesh(n: usize, fluid: bool) -> Mesh {
    let mut mesh = unit_cube_tetrahedra(n);
    if fluid {
        for p in &mut mesh.nodes {
            p.z -= 1.0;
        }
    }
    for b in &mut mesh.boundaries {
        let center: [f64; 3] = std::array::from_fn(|a| {
            b.nodes
                .iter()
                .map(|&j| {
                    let p = &mesh.nodes[j];
                    [p.x, p.y, p.z][a]
                })
                .sum::<f64>()
                / b.nodes.len() as f64
        });
        b.marker = if center[2].abs() < 1e-10 {
            7
        } else if center[0].abs() < 1e-10 {
            1
        } else if (center[0] - 1.0).abs() < 1e-10 {
            2
        } else if center[1].abs() < 1e-10 {
            3
        } else if (center[1] - 1.0).abs() < 1e-10 {
            4
        } else if center[2] < 0.0 {
            5
        } else {
            6
        };
    }
    mesh
}
fn load(
    mesh: &Mesh,
    space: &VectorH1Space,
    t: f64,
    law: Law,
    fluid: bool,
) -> Result<Vec<f64>, String> {
    let body = assemble_linear(
        &BodyForceForm {
            force_fn: move |x, y, z| {
                let (f, _, s) = volume_sources([x, y, z], t, law);
                if fluid { f } else { s }
            },
        },
        space,
        mesh,
        3,
    );
    let n = space.num_global_dofs(mesh);
    let mut system = FemSystem::new(TripletMatrix::new(n, n), body);
    for (marker, normal) in [
        (1, [-1.0, 0.0, 0.0]),
        (2, [1.0, 0.0, 0.0]),
        (3, [0.0, -1.0, 0.0]),
        (4, [0.0, 1.0, 0.0]),
        (5, [0.0, 0.0, -1.0]),
        (6, [0.0, 0.0, 1.0]),
    ] {
        if !mesh.boundaries.iter().any(|b| b.marker == marker) {
            continue;
        }
        try_apply_traction_vector(&mut system, mesh, space, marker, &|x, y, z| {
            let r = reference([x, y, z], t, law);
            let stress = if fluid {
                r.fluid_stress
            } else {
                r.solid_stress
            };
            (0..3)
                .map(|a| (0..3).map(|b| stress[a][b] * normal[b]).sum())
                .collect()
        })
        .map_err(|e| e.to_string())?;
    }
    Ok(system.rhs)
}
fn block(out: &mut TripletMatrix<f64>, a: &TripletMatrix<f64>, row: usize, col: usize, scale: f64) {
    for k in 0..a.nnz() {
        out.push(row + a.rows[k], col + a.cols[k], scale * a.values[k]);
    }
}
fn transpose_block(
    out: &mut TripletMatrix<f64>,
    a: &TripletMatrix<f64>,
    row: usize,
    col: usize,
    scale: f64,
) {
    for k in 0..a.nnz() {
        out.push(row + a.cols[k], col + a.rows[k], scale * a.values[k]);
    }
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn quadratic(a: &CsrMatrix<f64>, x: &[f64]) -> f64 {
    dot(x, &a.matvec(x))
}
fn relative(a: &[f64], b: &[f64]) -> f64 {
    (a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>()
        / b.iter().map(|v| v * v).sum::<f64>().max(1e-30))
    .sqrt()
}
fn analytic_energy(t: f64, law: Law) -> f64 {
    // Exact box integral of affine component squared: mean^2 + |gradient|^2/12.
    let square_integral = |fluid: bool| {
        let center = [0.5, 0.5, if fluid { -0.5 } else { 0.5 }];
        let r = reference(center, t, law);
        let value = if fluid { r.fluid } else { r.solid };
        let gradient: [[f64; 3]; 3] = std::array::from_fn(|a| {
            std::array::from_fn(|b| {
                let mut p = center;
                p[b] += 1.0;
                let q = reference(p, t, law);
                (if fluid { q.fluid[a] } else { q.solid[a] }) - value[a]
            })
        });
        dot(&value, &value) + gradient.iter().flatten().map(|v| v * v / 12.0).sum::<f64>()
    };
    let r = reference([0.0; 3], t, law);
    let gradient: [[f64; 3]; 3] = std::array::from_fn(|a| {
        std::array::from_fn(|b| {
            let mut p = [0.0; 3];
            p[b] = 1.0;
            reference(p, t, law).displacement[a] - r.displacement[a]
        })
    });
    let elastic = (0..3)
        .map(|a| {
            (0..3)
                .map(|b| 0.5 * r.solid_stress[a][b] * gradient[a][b])
                .sum::<f64>()
        })
        .sum::<f64>();
    0.65 * square_integral(true)
        + 0.25 * r.pressure * r.pressure
        + 0.85 * square_integral(false)
        + elastic
}
#[derive(Debug, Default)]
struct Metrics {
    fluid: f64,
    pressure: f64,
    solid: f64,
    displacement: f64,
    multiplier: f64,
    energy: f64,
    residual: f64,
    interface: f64,
    power: f64,
    action_reaction: f64,
    history: f64,
    change: f64,
    steps: usize,
    free: usize,
}
fn solve(
    nf: usize,
    ns: usize,
    steps: usize,
    law: Law,
    control: Control,
) -> Result<Metrics, String> {
    let fm = mesh(nf, true);
    let sm = mesh(ns, false);
    let vector = VectorH1Space::new(PolynomialDegree::P1, 3);
    let scalar = H1Space::new(PolynomialDegree::P1);
    let ftrace = FsiMortarSurface::from_boundary_submesh(&create_boundary_submesh(&fm, 7))
        .map_err(|e| format!("{e:?}"))?;
    let strace = FsiMortarSurface::from_boundary_submesh(&create_boundary_submesh(&sm, 7))
        .map_err(|e| format!("{e:?}"))?;
    if ftrace.coordinates.len() == strace.coordinates.len() {
        return Err("matching interfaces do not exercise mortar".into());
    }
    let mortar = FsiMortarOperators::build(&ftrace, &strace, FsiMortarConfig::default())
        .map_err(|e| format!("{e:?}"))?;
    let nu = vector.num_global_dofs(&fm);
    let np = scalar.num_global_dofs(&fm);
    let nv = vector.num_global_dofs(&sm);
    let po = nu;
    let vo = po + np;
    let wo = vo + nv;
    let lo = wo + nv;
    let total = lo + mortar.multiplier_dofs;
    let fc = vector.dof_coordinates(&fm).ok_or("fluid coordinates")?;
    let pc = scalar.dof_coordinates(&fm).ok_or("pressure coordinates")?;
    let sc = vector.dof_coordinates(&sm).ok_or("solid coordinates")?;
    let prescribed = |t: f64| {
        let mut x = vec![0.0; total];
        for (i, &p) in fc.iter().enumerate() {
            x[i] = reference(p, t, law).fluid[i % 3];
        }
        for (i, &p) in pc.iter().enumerate() {
            x[po + i] = reference(p, t, law).pressure;
        }
        for (i, &p) in sc.iter().enumerate() {
            let r = reference(p, t, law);
            x[vo + i] = r.solid[i % 3];
            x[wo + i] = r.displacement[i % 3];
        }
        for (i, v) in x[lo..].iter_mut().enumerate() {
            *v = reference([0.0; 3], t, law).multiplier[i % 3];
        }
        x
    };
    let fmass = assemble_bilinear(
        &VectorMassForm { coefficient: 1.3 },
        &vector,
        &vector,
        &fm,
        2,
    );
    let pmass = assemble_bilinear(&MassForm, &scalar, &scalar, &fm, 2);
    let smass = assemble_bilinear(
        &VectorMassForm { coefficient: 1.7 },
        &vector,
        &vector,
        &sm,
        2,
    );
    let kinematic = assemble_bilinear(
        &VectorMassForm { coefficient: 1.0 },
        &vector,
        &vector,
        &sm,
        2,
    );
    // The isotropic native form with lambda=0, mu=.3 is the Newtonian
    // symmetric-gradient viscous form, with no elastic state in the fluid.
    let viscous = assemble_bilinear(
        &ElasticityForm::new_3d(IsotropicMaterial::new(0.6, 0.0)),
        &vector,
        &vector,
        &fm,
        2,
    );
    let elastic = assemble_bilinear(
        &ElasticityForm::new_3d(IsotropicMaterial::new(2.0, 0.25)),
        &vector,
        &vector,
        &sm,
        2,
    );
    let divergence =
        assemble_bilinear_two_spaces_checked(&StokesDivergenceForm, &vector, &scalar, &fm, 2)
            .map_err(|e| e.to_string())?;
    let mut m = TripletMatrix::new(total, total);
    let mut k = TripletMatrix::new(total, total);
    block(
        &mut m,
        &fmass,
        0,
        0,
        if matches!(control, Control::NoFluidInertia) {
            0.0
        } else {
            1.0
        },
    );
    block(&mut m, &pmass, po, po, 0.5);
    block(
        &mut m,
        &smass,
        vo,
        vo,
        if matches!(control, Control::NoSolidInertia) {
            0.0
        } else {
            1.0
        },
    );
    block(&mut m, &kinematic, wo, wo, 1.0);
    block(&mut k, &viscous, 0, 0, 1.0);
    block(&mut k, &divergence, 0, po, 1.0);
    transpose_block(&mut k, &divergence, po, 0, -1.0);
    block(
        &mut k,
        &elastic,
        vo,
        wo,
        if matches!(control, Control::NoElasticity) {
            0.0
        } else {
            1.0
        },
    );
    block(&mut k, &kinematic, wo, vo, -1.0);
    let dt = 1.0 / steps as f64;
    let mut a = TripletMatrix::new(total, total);
    block(&mut a, &m, 0, 0, 1.0 / dt);
    block(&mut a, &k, 0, 0, 0.5);
    if matches!(control, Control::NoCoupling) {
        for i in lo..total {
            a.push(i, i, 1.0);
        }
    } else {
        let interface = mortar.assemble_saddle_operator();
        let map = |j: usize| {
            if j < mortar.fluid_trace_dofs {
                3 * ftrace.parent_nodes[j / 3] + j % 3
            } else if j < mortar.fluid_trace_dofs + mortar.solid_trace_dofs {
                let d = j - mortar.fluid_trace_dofs;
                vo + 3 * strace.parent_nodes[d / 3] + d % 3
            } else {
                lo + j - mortar.fluid_trace_dofs - mortar.solid_trace_dofs
            }
        };
        for j in 0..interface.matrix.nnz() {
            a.push(
                map(interface.matrix.rows[j]),
                map(interface.matrix.cols[j]),
                interface.matrix.values[j],
            );
        }
    }
    let a = a.to_csr();
    let factor = SparseLuFactorization::factorize(&a).map_err(|e| e.to_string())?;
    let m = m.to_csr();
    let k = k.to_csr();
    let mf = fmass.to_csr();
    let mp = pmass.to_csr();
    let ms = smass.to_csr();
    let ks = elastic.to_csr();
    let vf = viscous.to_csr();
    let energy = |x: &[f64]| {
        0.5 * (quadratic(&mf, &x[..nu])
            + 0.5 * quadratic(&mp, &x[po..vo])
            + quadratic(&ms, &x[vo..wo])
            + quadratic(&ks, &x[wo..lo]))
    };
    let forcing = |t: f64| -> Result<Vec<f64>, String> {
        let mut f = vec![0.0; total];
        f[..nu].copy_from_slice(&load(&fm, &vector, t, law, true)?);
        f[vo..wo].copy_from_slice(&load(&sm, &vector, t, law, false)?);
        let pressure = assemble_linear(
            &SourceForm::new(|x, y, z| volume_sources([x, y, z], t, law).1),
            &scalar,
            &fm,
            2,
        );
        f[po..vo].copy_from_slice(&pressure);
        Ok(f)
    };
    let initial = prescribed(0.0);
    let mut x = initial.clone();
    let mut old_force = forcing(0.0)?;
    let mut metrics = Metrics {
        free: nu.min(np).min(nv).min(mortar.multiplier_dofs),
        ..Metrics::default()
    };
    for step in 1..=steps {
        let t = step as f64 * dt;
        let f = forcing(t)?;
        let old = x.clone();
        let mx = m.matvec(&old);
        let kx = k.matvec(&old);
        let mean_force: Vec<_> = f
            .iter()
            .zip(&old_force)
            .map(|(a, b)| 0.5 * (a + b))
            .collect();
        let rhs: Vec<_> = mx
            .iter()
            .zip(&kx)
            .zip(&mean_force)
            .map(|((m, k), f)| m / dt - 0.5 * k + f)
            .collect();
        x = factor.solve(&rhs).map_err(|e| e.to_string())?;
        if x.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite FSI state".into());
        }
        let applied = a.matvec(&x);
        metrics.residual = metrics.residual.max(relative(&applied, &rhs));
        let exact = prescribed(t);
        let exact_old = prescribed(t - dt);
        let lambda_exact: Vec<_> = exact[lo..]
            .iter()
            .zip(&exact_old[lo..])
            .map(|(a, b)| 0.5 * (a + b))
            .collect();
        let errors = [
            relative(&x[..nu], &exact[..nu]),
            relative(&x[po..vo], &exact[po..vo]),
            relative(&x[vo..wo], &exact[vo..wo]),
            relative(&x[wo..lo], &exact[wo..lo]),
            relative(&x[lo..], &lambda_exact),
        ];
        metrics.fluid = metrics.fluid.max(errors[0]);
        metrics.pressure = metrics.pressure.max(errors[1]);
        metrics.solid = metrics.solid.max(errors[2]);
        metrics.displacement = metrics.displacement.max(errors[3]);
        metrics.multiplier = metrics.multiplier.max(errors[4]);
        metrics.history = metrics
            .history
            .max(errors.iter().copied().fold(0.0, f64::max));
        let ft: Vec<_> = ftrace
            .parent_nodes
            .iter()
            .flat_map(|&j| x[3 * j..3 * j + 3].iter().copied())
            .collect();
        let st: Vec<_> = strace
            .parent_nodes
            .iter()
            .flat_map(|&j| x[vo + 3 * j..vo + 3 * j + 3].iter().copied())
            .collect();
        let residual = mortar.constraint_residual(&ft, &st);
        metrics.interface = metrics
            .interface
            .max(residual.iter().map(|v| v.abs()).fold(0.0, f64::max));
        let transfer = mortar.multiplier_forces(&x[lo..]);
        let force_scale = transfer
            .fluid_forces
            .iter()
            .chain(&transfer.solid_forces)
            .map(|v| v.abs())
            .sum::<f64>()
            .max(1e-12);
        for c in 0..3 {
            let net = transfer
                .fluid_forces
                .iter()
                .enumerate()
                .filter(|(i, _)| i % 3 == c)
                .map(|(_, v)| v)
                .sum::<f64>()
                + transfer
                    .solid_forces
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i % 3 == c)
                    .map(|(_, v)| v)
                    .sum::<f64>();
            metrics.action_reaction = metrics.action_reaction.max(net.abs() / force_scale);
        }
        let mid: Vec<_> = x.iter().zip(&old).map(|(a, b)| 0.5 * (a + b)).collect();
        let work = dot(&mean_force[..nu], &mid[..nu])
            + dot(&mean_force[po..vo], &mid[po..vo])
            + dot(&mean_force[vo..wo], &mid[vo..wo]);
        let dissipation = quadratic(&vf, &mid[..nu]);
        let balance = (energy(&x) - energy(&old)) / dt + dissipation - work;
        metrics.power = metrics
            .power
            .max(balance.abs() / work.abs().max(dissipation).max(1e-12));
        let reference_energy = analytic_energy(t, law);
        metrics.energy = metrics
            .energy
            .max((energy(&x) - reference_energy).abs() / reference_energy);
        metrics.change = metrics.change.max(relative(&x[..lo], &initial[..lo]));
        metrics.steps += 1;
        old_force = f;
    }
    Ok(metrics)
}
fn run() -> Result<QaResult, String> {
    let patch = solve(1, 2, 8, Law::Linear, Control::Full)?;
    println!("FSI linear patch: {patch:?}");
    let mut temporal = Vec::new();
    for n in [8, 16, 32] {
        let m = solve(2, 3, n, Law::Exponential, Control::Full)?;
        println!("FSI time={n}: {m:?}");
        temporal.push(m);
    }
    let other = solve(3, 4, 32, Law::Exponential, Control::Full)?;
    println!("FSI other nonmatching mesh: {other:?}");
    let mut negative = f64::INFINITY;
    for c in [
        Control::NoCoupling,
        Control::NoFluidInertia,
        Control::NoSolidInertia,
        Control::NoElasticity,
    ] {
        let m = solve(1, 2, 8, Law::Exponential, c)?;
        println!("FSI {c:?}: {m:?}");
        // Removing the multiplier cannot prove coupling by itself. Require
        // a change in solved volume-field histories for every control.
        negative = negative.min(m.fluid.max(m.pressure).max(m.solid).max(m.displacement));
    }
    let all: Vec<_> = temporal.iter().chain([&patch, &other]).collect();
    let mut r = QaResult::passed(CASE)
        .requested_backend("cpu")
        .actual_backend("cpu");
    for (name, value, bound) in [
        (
            "exact_linear_patch_error",
            patch.history.max(patch.energy),
            1e-9,
        ),
        (
            "fine_field_error",
            temporal[2].history.max(other.history),
            0.005,
        ),
        (
            "fine_energy_error",
            temporal[2].energy.max(other.energy),
            0.005,
        ),
        (
            "physical_coupled_residual",
            all.iter().map(|m| m.residual).fold(0.0, f64::max),
            1e-9,
        ),
        (
            "mortar_velocity_constraint",
            all.iter().map(|m| m.interface).fold(0.0, f64::max),
            1e-10,
        ),
        (
            "dynamic_power_balance",
            all.iter().map(|m| m.power).fold(0.0, f64::max),
            1e-9,
        ),
        (
            "interface_action_reaction",
            all.iter().map(|m| m.action_reaction).fold(0.0, f64::max),
            1e-10,
        ),
    ] {
        if !value.is_finite() || value > bound {
            return Err(format!("{name}={value:e} > {bound}"));
        }
        r = r.metric(name, value).maximum(name, bound);
    }
    for (name, select) in [
        ("fluid", (|m: &Metrics| m.fluid) as fn(&Metrics) -> f64),
        ("pressure", |m: &Metrics| m.pressure),
        ("solid_velocity", |m: &Metrics| m.solid),
        ("displacement", |m: &Metrics| m.displacement),
        ("multiplier", |m: &Metrics| m.multiplier),
        ("energy", |m: &Metrics| m.energy),
    ] {
        let rate = temporal
            .windows(2)
            .map(|v| (select(&v[0]) / select(&v[1])).log2())
            .fold(f64::INFINITY, f64::min);
        if !rate.is_finite() || rate < 1.8 {
            return Err(format!("temporal {name} rate={rate}"));
        }
        let key = format!("temporal_{name}_rate");
        r = r.metric(&key, rate).minimum(&key, 1.8);
    }
    if !negative.is_finite() || negative < 1e-3 {
        return Err(format!("omitted coupling/history error={negative}"));
    }
    Ok(r.metric("minimum_omitted_term_error",negative).minimum("minimum_omitted_term_error",1e-3).metric("negative_controls",4.0).minimum("negative_controls",4.0)
        .metric("executed_time_steps",all.iter().map(|m|m.steps).sum::<usize>() as f64).minimum("executed_time_steps",96.0)
        .metric("minimum_free_field_unknowns",all.iter().map(|m|m.free).min().unwrap_or(0) as f64).minimum("minimum_free_field_unknowns",1.0)
        .metric("minimum_history_change",all.iter().map(|m|m.change).fold(f64::INFINITY,f64::min)).minimum("minimum_history_change",0.1)
        .reference("independent affine 3D velocity/displacement, constant pressure and interface traction; exact exterior stresses, volume sources and integrated kinetic/compressive/elastic energy")
        .reference_checksum("analytic:3d-linear-compressible-viscous-fluid-elastic-solid-mortar-cn:v1")
        .input_checksum("P1:fluid1-solid2:linear8:fluid2-solid3:exp8-16-32:fluid3-solid4:exp32:rhoF1.3:rhoS1.7:chi.5:muF.3:lambdaS.8:muS.8")
        .dimension("two 3D tetrahedral volumes and nonmatching planar full-vector P1 mortar interface; all fluid velocity/pressure and solid velocity/displacement/multiplier unknowns free")
        .formulation("linear compressible symmetric-gradient viscous fluid and isotropic elastic solid on fixed reference geometry; native mass, divergence, stiffness, traction and mortar saddle blocks; CN volume histories and midpoint traction")
        .tolerance("exact affine-space linear-time patch/residual/power<=1e-9; six temporal rates>=1.8; fine errors<=.005; constraint/action-reaction<=1e-10; four controls>=1e-3"))
}
fn main() {
    sonium_qa::fem_physics_3d::run_fsi_case();
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
