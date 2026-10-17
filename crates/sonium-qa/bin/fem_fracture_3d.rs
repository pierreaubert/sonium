//! Coupled 3D AT2 damage and degraded elasticity manufactured histories.
//! Positive strain makes the full elastic energy the tensile history driver.
//! Prescribed volume sources and boundary damage define a forced smooth problem;
//! this is a history/coupling accuracy gate, not a crack-propagation experiment.
use sonium_fem::assembly::{TripletMatrix, assemble_linear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::forms::{BilinearForm, LinearForm, LocalMatrix, LocalVector, SourceForm};
use sonium_fem::mesh::unit_cube_tetrahedra;
use sonium_fem::physics::solid::elasticity::{BodyForceForm, IsotropicMaterial};
use sonium_fem::physics::solid::fracture::{
    DamageHistoryRhsForm, DamageHistoryTangentForm, DegradedElasticityForm, PhaseFieldDamageForm,
    PhaseFieldFractureMaterial, assemble_damage_operator_with_history,
};
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{BasisTabulation, FunctionSpace, H1Space, VectorH1Space};
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::{CsrMatrix, SparseCholeskyFactorization};

const CASE: &str = "fem.fracture-3d-at2-history.v1";
const LOADS: [f64; 8] = [0.0, 0.4, 0.8, 0.3, 1.0, 0.2, 1.2, 0.5];
const MATERIAL: PhaseFieldFractureMaterial = PhaseFieldFractureMaterial {
    critical_energy_release_rate: 0.2,
    length_scale: 0.25,
    residual_stiffness: 1e-6,
};
#[derive(Clone, Copy, Debug)]
enum Control {
    Full,
    NoHistoryMaximum,
    NoHistoryTangent,
    NoDegradation,
    NoDamageSource,
}
struct Reference {
    damage: f64,
    damage_gradient: [f64; 3],
    displacement: [f64; 3],
    strain: [f64; 3],
    history: f64,
    stress: [f64; 3],
}
fn reference(p: [f64; 3], load: f64, peak: f64) -> Reference {
    let [x, y, z] = p;
    let q = 0.1 + 0.02 * (x * x + 2.0 * y * y + 3.0 * z * z + x * y + x * z + y * z);
    let strain = p.map(|v| 0.05 * load * (1.0 + v));
    let trace = strain.iter().sum::<f64>();
    let peak_strain = p.map(|v| 0.05 * peak * (1.0 + v));
    let h = 4.0 * peak_strain.iter().sum::<f64>().powi(2)
        + 8.0 * peak_strain.iter().map(|v| v * v).sum::<f64>();
    let damage = peak * peak * q;
    let degradation = (1.0 - 1e-6) * (1.0 - damage).powi(2) + 1e-6;
    Reference {
        damage,
        damage_gradient: [2.0 * x + y + z, 4.0 * y + x + z, 6.0 * z + x + y]
            .map(|v| peak * peak * 0.02 * v),
        displacement: p.map(|v| 0.05 * load * (v + 0.5 * v * v)),
        strain,
        history: h,
        stress: strain.map(|e| degradation * (8.0 * trace + 16.0 * e)),
    }
}
fn sources(p: [f64; 3], load: f64, peak: f64) -> (f64, Vec<f64>) {
    let r = reference(p, load, peak);
    let trace = r.strain.iter().sum::<f64>();
    let g = (1.0 - 1e-6) * (1.0 - r.damage).powi(2) + 1e-6;
    let body = (0..3)
        .map(|a| {
            let dg = -2.0 * (1.0 - 1e-6) * (1.0 - r.damage) * r.damage_gradient[a];
            -g * 24.0 * 0.05 * load - dg * (8.0 * trace + 16.0 * r.strain[a])
        })
        .collect();
    let damage = (0.8 + 2.0 * r.history) * r.damage - 0.05 * peak * peak * 0.24 - 2.0 * r.history;
    (damage, body)
}
struct Point {
    scalar: BasisTabulation,
    vector: BasisTabulation,
    weight: f64,
}
struct ElementPoints {
    scalar_dofs: Vec<usize>,
    vector_dofs: Vec<usize>,
    points: Vec<Point>,
}
fn damage_at(e: &ElementPoints, p: &Point, d: &[f64]) -> (f64, [f64; 3]) {
    let v = e
        .scalar_dofs
        .iter()
        .enumerate()
        .map(|(i, &j)| d[j] * p.scalar.value(i))
        .sum();
    let g = std::array::from_fn(|a| {
        e.scalar_dofs
            .iter()
            .enumerate()
            .map(|(i, &j)| d[j] * p.scalar.grad(i, a))
            .sum()
    });
    (v, g)
}
fn strain_at(e: &ElementPoints, p: &Point, u: &[f64]) -> [[f64; 3]; 3] {
    let grad: [[f64; 3]; 3] = std::array::from_fn(|a| {
        std::array::from_fn(|b| {
            e.vector_dofs
                .iter()
                .enumerate()
                .filter(|(_, j)| **j % 3 == a)
                .map(|(i, &j)| u[j] * p.vector.grad(i, b))
                .sum()
        })
    });
    std::array::from_fn(|a| std::array::from_fn(|b| 0.5 * (grad[a][b] + grad[b][a])))
}
fn energy_and_stress(strain: [[f64; 3]; 3], elastic: IsotropicMaterial) -> (f64, [f64; 6]) {
    let eps = [
        strain[0][0],
        strain[1][1],
        strain[2][2],
        2.0 * strain[1][2],
        2.0 * strain[0][2],
        2.0 * strain[0][1],
    ];
    let c = elastic.voigt_3d();
    let stress = std::array::from_fn(|i| (0..6).map(|j| c[i * 6 + j] * eps[j]).sum());
    (
        0.5 * eps.iter().zip(stress).map(|(a, b)| a * b).sum::<f64>(),
        stress,
    )
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
    for (i, &b) in fixed.iter().enumerate() {
        if b {
            a.push(i, i, 1.0);
            rhs[i] = values[i];
        }
    }
    (a.to_csr(), rhs)
}
fn solve_spd(
    a: TripletMatrix<f64>,
    rhs: Vec<f64>,
    fixed: &[bool],
    values: &[f64],
) -> Result<Vec<f64>, String> {
    let (matrix, b) = lift(a, rhs, fixed, values);
    SparseCholeskyFactorization::factorize(&matrix)
        .map_err(|e| e.to_string())?
        .solve(&b)
        .map_err(|e| e.to_string())
}
fn mechanics(
    points: &[ElementPoints],
    d: &[f64],
    elastic: IsotropicMaterial,
    nu: usize,
    control: Control,
) -> TripletMatrix<f64> {
    let mut a = TripletMatrix::new(nu, nu);
    for e in points {
        let mut local = LocalMatrix::zeros(e.vector_dofs.len());
        for p in &e.points {
            let damage = if matches!(control, Control::NoDegradation) {
                0.0
            } else {
                damage_at(e, p, d).0
            };
            DegradedElasticityForm {
                elastic_material: elastic,
                fracture_material: MATERIAL,
                damage,
                plane: None,
            }
            .integrate_point(&p.vector, &p.vector, p.weight, &mut local);
        }
        for (i, &r) in e.vector_dofs.iter().enumerate() {
            for (j, &c) in e.vector_dofs.iter().enumerate() {
                a.push(r, c, local.get(i, j));
            }
        }
    }
    a
}
fn update_history(
    points: &[ElementPoints],
    u: &[f64],
    old: &[Vec<f64>],
    elastic: IsotropicMaterial,
    control: Control,
) -> Vec<Vec<f64>> {
    points
        .iter()
        .zip(old)
        .map(|(e, previous)| {
            e.points
                .iter()
                .zip(previous)
                .map(|(p, &h)| {
                    let energy = energy_and_stress(strain_at(e, p, u), elastic).0.max(0.0);
                    if matches!(control, Control::NoHistoryMaximum) {
                        energy
                    } else {
                        energy.max(h)
                    }
                })
                .collect()
        })
        .collect()
}
fn damage_operator(
    points: &[ElementPoints],
    base: &TripletMatrix<f64>,
    history: &[Vec<f64>],
    source: &[f64],
    control: Control,
) -> (TripletMatrix<f64>, Vec<f64>) {
    let mut a = base.clone();
    let mut rhs = source.to_vec();
    for (e, hist) in points.iter().zip(history) {
        let mut local = LocalMatrix::zeros(e.scalar_dofs.len());
        let mut load = LocalVector::zeros(e.scalar_dofs.len());
        for (p, &h) in e.points.iter().zip(hist) {
            if !matches!(control, Control::NoHistoryTangent) {
                DamageHistoryTangentForm {
                    coefficient: 2.0 * h,
                }
                .integrate_point(&p.scalar, &p.scalar, p.weight, &mut local);
            }
            DamageHistoryRhsForm {
                history_energy_density: h,
            }
            .integrate_point(&p.scalar, p.weight, &mut load);
        }
        for (i, &r) in e.scalar_dofs.iter().enumerate() {
            rhs[r] += load.values[i];
            for (j, &c) in e.scalar_dofs.iter().enumerate() {
                a.push(r, c, local.get(i, j));
            }
        }
    }
    (a, rhs)
}
fn difference(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}
fn raw_residual(a: &CsrMatrix<f64>, x: &[f64], rhs: &[f64], fixed: &[bool]) -> (f64, f64) {
    let applied = a.matvec(x);
    let mut norm = 0.0;
    let mut scale = 0.0;
    let mut budget = 0.0;
    let mut budget_scale = 0.0;
    for i in 0..x.len() {
        if !fixed[i] {
            norm += (applied[i] - rhs[i]).powi(2);
            scale += rhs[i].powi(2);
            budget += x[i] * (applied[i] - rhs[i]);
            budget_scale += (x[i] * rhs[i]).abs();
        }
    }
    (
        (norm / scale.max(1e-30)).sqrt(),
        budget.abs() / budget_scale.max(1e-12),
    )
}
#[derive(Debug, Default)]
struct Metrics {
    damage: f64,
    damage_gradient: f64,
    displacement: f64,
    strain: f64,
    history: f64,
    stress: f64,
    elastic_energy: f64,
    crack_energy: f64,
    residual: f64,
    power: f64,
    damage_balance: f64,
    boundary: f64,
    irreversibility: f64,
    unload_change: f64,
    peak_change: f64,
    history_error: f64,
    min_strain: f64,
    free: usize,
    steps: usize,
    iterations: usize,
}
fn solve(nmesh: usize, degree: PolynomialDegree, control: Control) -> Result<Metrics, String> {
    let mut mesh = unit_cube_tetrahedra(nmesh);
    for b in &mut mesh.boundaries {
        b.marker = 1;
    }
    let scalar = H1Space::new(degree);
    let vector = VectorH1Space::new(degree, 3);
    let nd = scalar.num_global_dofs(&mesh);
    let nu = vector.num_global_dofs(&mesh);
    let dc = scalar
        .dof_coordinates(&mesh)
        .ok_or("damage coordinates missing")?;
    let uc = vector
        .dof_coordinates(&mesh)
        .ok_or("displacement coordinates missing")?;
    let mut dfixed = vec![false; nd];
    let mut ufixed = vec![false; nu];
    for j in scalar.boundary_dofs(&mesh, 1) {
        dfixed[j] = true;
    }
    for j in vector.boundary_dofs(&mesh, 1) {
        ufixed[j] = true;
    }
    let free = dfixed
        .iter()
        .filter(|&&b| !b)
        .count()
        .min(ufixed.iter().filter(|&&b| !b).count() / 3);
    if free == 0 {
        return Err("no free coupled unknowns".into());
    }
    let elastic = IsotropicMaterial::new(20.0, 0.25);
    let order = if degree == PolynomialDegree::P1 { 2 } else { 6 };
    let mut points = Vec::new();
    let mut errors = Vec::new();
    for e in 0..mesh.num_elements() {
        let sd = scalar.local_to_global(&mesh, e);
        let ud = vector.local_to_global(&mesh, e);
        let tabulate = |order| ElementPoints {
            scalar_dofs: sd.clone(),
            vector_dofs: ud.clone(),
            points: QuadratureRule::new(mesh.elements[e].element_type, order)
                .iter()
                .map(|q| Point {
                    scalar: scalar.tabulate(&mesh, e, q.xi(), q.eta(), q.zeta()),
                    vector: vector.tabulate(&mesh, e, q.xi(), q.eta(), q.zeta()),
                    weight: q.weight,
                })
                .collect(),
        };
        points.push(tabulate(order));
        errors.push(tabulate(8));
    }
    let mut history: Vec<Vec<f64>> = points.iter().map(|e| vec![0.0; e.points.len()]).collect();
    let mut error_history: Vec<Vec<f64>> =
        errors.iter().map(|e| vec![0.0; e.points.len()]).collect();
    let base = assemble_damage_operator_with_history(
        &mesh,
        degree,
        MATERIAL,
        &vec![0.0; mesh.num_elements()],
    );
    let mut d = vec![0.0; nd];
    let mut u = vec![0.0; nu];
    let mut peak = 0.0_f64;
    let mut metrics = Metrics {
        free,
        min_strain: f64::INFINITY,
        ..Metrics::default()
    };
    for step in 1..LOADS.len() {
        let load = LOADS[step];
        let old_peak = peak;
        peak = peak.max(load);
        let previous_d = d.clone();
        let old_history = history.clone();
        let old_error_history = error_history.clone();
        let dv: Vec<_> = dc
            .iter()
            .map(|&p| reference(p, load, peak).damage)
            .collect();
        let uv: Vec<_> = uc
            .iter()
            .enumerate()
            .map(|(i, &p)| reference(p, load, peak).displacement[i % 3])
            .collect();
        let mut source = assemble_linear(
            &SourceForm::new(|x, y, z| sources([x, y, z], load, peak).0),
            &scalar,
            &mesh,
            8,
        );
        if matches!(control, Control::NoDamageSource) {
            source.fill(0.0);
        }
        let body = assemble_linear(
            &BodyForceForm {
                force_fn: |x, y, z| sources([x, y, z], load, peak).1,
            },
            &vector,
            &mesh,
            8,
        );
        let mut converged = false;
        for _ in 0..240 {
            let unew = solve_spd(
                mechanics(&points, &d, elastic, nu, control),
                body.clone(),
                &ufixed,
                &uv,
            )?;
            let h = update_history(&points, &unew, &old_history, elastic, control);
            let (a, b) = damage_operator(&points, &base, &h, &source, control);
            let dnew = solve_spd(a, b, &dfixed, &dv)?;
            if unew
                .iter()
                .chain(&dnew)
                .chain(h.iter().flatten())
                .any(|v| !v.is_finite())
            {
                return Err("nonfinite displacement, damage or history".into());
            }
            let change = difference(&d, &dnew).max(difference(&u, &unew));
            d = dnew;
            u = unew;
            history = h;
            metrics.iterations += 1;
            if change < 2e-13 {
                converged = true;
                break;
            }
        }
        if !converged {
            return Err(format!(
                "staggered coupling failed n={nmesh} {degree:?} {control:?} load={load}"
            ));
        }
        error_history = update_history(&errors, &u, &old_error_history, elastic, control);
        // Admit the tensile-energy assumption at the actual history quadrature,
        // in addition to the independent error quadrature below. Gershgorin's
        // lower bound verifies positive semidefinite symmetric strain.
        for e in &points {
            for p in &e.points {
                let strain = strain_at(e, p, &u);
                for a in 0..3 {
                    let lower = strain[a][a]
                        - (0..3)
                            .filter(|&b| b != a)
                            .map(|b| strain[a][b].abs())
                            .sum::<f64>();
                    metrics.min_strain = metrics.min_strain.min(lower);
                }
            }
        }
        let mech = mechanics(&points, &d, elastic, nu, control).to_csr();
        let fresh_history = update_history(&points, &u, &old_history, elastic, control);
        let (damage_matrix, damage_rhs) =
            damage_operator(&points, &base, &fresh_history, &source, control);
        let (ur, up) = raw_residual(&mech, &u, &body, &ufixed);
        let (dr, dp) = raw_residual(&damage_matrix.to_csr(), &d, &damage_rhs, &dfixed);
        metrics.residual = metrics.residual.max(ur).max(dr);
        metrics.power = metrics.power.max(up);
        metrics.damage_balance = metrics.damage_balance.max(dp);
        metrics.boundary = metrics.boundary.max(
            d.iter()
                .zip(&dv)
                .zip(&dfixed)
                .filter(|(_, b)| **b)
                .map(|((x, y), _)| (x - y).abs())
                .fold(0.0, f64::max),
        );
        metrics.boundary = metrics.boundary.max(
            u.iter()
                .zip(&uv)
                .zip(&ufixed)
                .filter(|(_, b)| **b)
                .map(|((x, y), _)| (x - y).abs())
                .fold(0.0, f64::max),
        );
        metrics.irreversibility = metrics.irreversibility.max(
            previous_d
                .iter()
                .zip(&d)
                .map(|(a, b)| (a - b).max(0.0))
                .fold(0.0, f64::max),
        );
        if load < LOADS[step - 1] {
            metrics.unload_change = metrics.unload_change.max(difference(&d, &previous_d));
        }
        if peak > old_peak {
            let free_change = d
                .iter()
                .zip(&previous_d)
                .zip(&dfixed)
                .filter(|(_, fixed)| !**fixed)
                .map(|((a, b), _)| (a - b).abs())
                .fold(0.0, f64::max);
            metrics.peak_change = metrics.peak_change.max(free_change);
        }
        if matches!(control, Control::Full)
            && d.iter()
                .any(|v| !v.is_finite() || *v < -1e-10 || *v > 1.0 + 1e-10)
        {
            return Err("damage outside [0,1]".into());
        }
        let mut numerator = [0.0; 6];
        let mut denominator = [0.0; 6];
        let mut ee = 0.0;
        let mut er = 0.0;
        let mut ce = 0.0;
        let mut cr = 0.0;
        for (e, h) in errors.iter().zip(&error_history) {
            for (p, &hist) in e.points.iter().zip(h) {
                let w = p.weight * p.scalar.det_j;
                let r = reference(p.scalar.physical_coords, load, peak);
                let (actual_d, dg) = damage_at(e, p, &d);
                let strain = strain_at(e, p, &u);
                let (energy, stress) = energy_and_stress(strain, elastic);
                let g = MATERIAL.degradation(actual_d);
                let actual_u: [f64; 3] = std::array::from_fn(|a| {
                    e.vector_dofs
                        .iter()
                        .enumerate()
                        .map(|(i, &j)| u[j] * p.vector.vector_value(i, a))
                        .sum()
                });
                numerator[0] += (actual_d - r.damage).powi(2) * w;
                denominator[0] += r.damage.powi(2) * w;
                numerator[4] += (hist - r.history).powi(2) * w;
                denominator[4] += r.history.powi(2) * w;
                for a in 0..3 {
                    numerator[1] += (dg[a] - r.damage_gradient[a]).powi(2) * w;
                    denominator[1] += r.damage_gradient[a].powi(2) * w;
                    numerator[2] += (actual_u[a] - r.displacement[a]).powi(2) * w;
                    denominator[2] += r.displacement[a].powi(2) * w;
                    numerator[5] += (g * stress[a] - r.stress[a]).powi(2) * w;
                    denominator[5] += r.stress[a].powi(2) * w;
                    let lower = strain[a][a]
                        - (0..3)
                            .filter(|&b| b != a)
                            .map(|b| strain[a][b].abs())
                            .sum::<f64>();
                    metrics.min_strain = metrics.min_strain.min(lower);
                    for b in 0..3 {
                        let expected = if a == b { r.strain[a] } else { 0.0 };
                        numerator[3] += (strain[a][b] - expected).powi(2) * w;
                        denominator[3] += expected.powi(2) * w;
                    }
                }
                for a in 3..6 {
                    numerator[5] += (g * stress[a]).powi(2) * w;
                }
                ee += g * energy * w;
                er += (r
                    .stress
                    .iter()
                    .zip(r.strain)
                    .map(|(s, e)| 0.5 * s * e)
                    .sum::<f64>())
                    * w;
                ce += MATERIAL.crack_surface_density(actual_d, dg.iter().map(|v| v * v).sum()) * w;
                // Independently derived Gc/(2*l)=.4 and Gc*l/2=.025.
                cr += (0.4 * r.damage * r.damage
                    + 0.025 * r.damage_gradient.iter().map(|v| v * v).sum::<f64>())
                    * w;
            }
        }
        let error: Vec<_> = numerator
            .iter()
            .zip(denominator)
            .map(|(a, b)| (a / b.max(1e-30)).sqrt())
            .collect();
        metrics.damage = metrics.damage.max(error[0]);
        metrics.damage_gradient = metrics.damage_gradient.max(error[1]);
        metrics.displacement = metrics.displacement.max(error[2]);
        metrics.strain = metrics.strain.max(error[3]);
        metrics.history = metrics.history.max(error[4]);
        metrics.stress = metrics.stress.max(error[5]);
        metrics.history_error = metrics
            .history_error
            .max(error.iter().copied().fold(0.0, f64::max));
        metrics.elastic_energy = metrics.elastic_energy.max((ee - er).abs() / er.max(1e-30));
        metrics.crack_energy = metrics.crack_energy.max((ce - cr).abs() / cr.max(1e-30));
        metrics.steps += 1;
    }
    if matches!(control, Control::Full) && metrics.min_strain < -1e-9 {
        return Err(format!("strain not tensile: {}", metrics.min_strain));
    }
    Ok(metrics)
}
fn run() -> Result<QaResult, String> {
    let patch = solve(2, PolynomialDegree::P2, Control::Full)?;
    println!("fracture P2 patch: {patch:?}");
    let exact = patch
        .history_error
        .max(patch.elastic_energy)
        .max(patch.crack_energy);
    // Independent solves (own mesh, locals only): run concurrently and
    // rejoin in order. Prints stay in schedule order after the join.
    use rayon::prelude::*;
    let spatial: Vec<Metrics> = [3, 6, 9]
        .into_par_iter()
        .map(|n| solve(n, PolynomialDegree::P1, Control::Full))
        .collect::<Result<Vec<_>, _>>()?;
    for (n, m) in [3, 6, 9].iter().zip(spatial.iter()) {
        println!("fracture P1 n={n}: {m:?}");
    }
    let controls = [
        Control::NoHistoryMaximum,
        Control::NoHistoryTangent,
        Control::NoDegradation,
        Control::NoDamageSource,
    ];
    let control_runs: Vec<Metrics> = controls
        .into_par_iter()
        .map(|control| solve(2, PolynomialDegree::P2, control))
        .collect::<Result<Vec<_>, _>>()?;
    let mut negative = f64::INFINITY;
    for (control, m) in [
        Control::NoHistoryMaximum,
        Control::NoHistoryTangent,
        Control::NoDegradation,
        Control::NoDamageSource,
    ]
    .iter()
    .zip(control_runs.iter())
    {
        println!("fracture {control:?}: {m:?}");
        negative = negative.min(m.history_error);
    }
    let all: Vec<_> = spatial.iter().chain([&patch]).collect();
    let fine = &spatial[2];
    let mut r = QaResult::passed(CASE)
        .requested_backend("cpu")
        .actual_backend("cpu");
    for (name, value, bound) in [
        ("fine_damage_error", fine.damage, 0.005),
        ("fine_damage_gradient_error", fine.damage_gradient, 0.1),
        ("fine_displacement_error", fine.displacement, 0.005),
        ("fine_strain_error", fine.strain, 0.1),
        // Measured fine-grid history error = 2.38e-2; gate at ~1.7x.
        ("fine_history_error", fine.history, 0.04),
        ("fine_stress_error", fine.stress, 0.1),
        ("fine_elastic_energy_error", fine.elastic_energy, 0.01),
        ("fine_crack_energy_error", fine.crack_energy, 0.01),
        ("exact_p2_error", exact, 1e-9),
        (
            "physical_coupled_residual",
            all.iter().map(|m| m.residual).fold(0.0, f64::max),
            1e-9,
        ),
        (
            "elastic_power_balance",
            all.iter().map(|m| m.power).fold(0.0, f64::max),
            1e-9,
        ),
        (
            "weak_damage_balance",
            all.iter().map(|m| m.damage_balance).fold(0.0, f64::max),
            1e-9,
        ),
        (
            "boundary_error",
            all.iter().map(|m| m.boundary).fold(0.0, f64::max),
            1e-10,
        ),
        (
            "damage_irreversibility_error",
            all.iter().map(|m| m.irreversibility).fold(0.0, f64::max),
            1e-9,
        ),
        (
            "unload_damage_change",
            all.iter().map(|m| m.unload_change).fold(0.0, f64::max),
            1e-9,
        ),
    ] {
        if !value.is_finite() || value > bound {
            return Err(format!("{name}={value:e} > {bound}"));
        }
        r = r.metric(name, value).maximum(name, bound);
    }
    for (name, select, bound) in [
        (
            "damage",
            (|m: &Metrics| m.damage) as fn(&Metrics) -> f64,
            1.8,
        ),
        ("damage_gradient", |m: &Metrics| m.damage_gradient, 0.9),
        ("displacement", |m: &Metrics| m.displacement, 1.8),
        ("strain", |m: &Metrics| m.strain, 0.9),
        ("history", |m: &Metrics| m.history, 0.9),
        ("stress", |m: &Metrics| m.stress, 0.9),
    ] {
        let ns = [3.0_f64, 6.0, 9.0];
        let rate = spatial
            .windows(2)
            .enumerate()
            .map(|(i, m)| (select(&m[0]) / select(&m[1])).ln() / (ns[i + 1] / ns[i]).ln())
            .fold(f64::INFINITY, f64::min);
        if !rate.is_finite() || rate < bound {
            return Err(format!("spatial_{name}_rate={rate} < {bound}"));
        }
        let key = format!("spatial_{name}_rate");
        r = r.metric(&key, rate).minimum(&key, bound);
    }
    if !negative.is_finite() || negative < 1e-3 {
        return Err(format!("omitted term error {negative}"));
    }
    Ok(r.metric("minimum_omitted_term_error",negative).minimum("minimum_omitted_term_error",1e-3)
        .metric("negative_controls",4.0).minimum("negative_controls",4.0)
        .metric("minimum_free_field_unknowns",all.iter().map(|m|m.free).min().unwrap_or(0) as f64).minimum("minimum_free_field_unknowns",1.0)
        .metric("executed_load_steps",all.iter().map(|m|m.steps).sum::<usize>() as f64).minimum("executed_load_steps",28.0)
        .metric("executed_coupled_iterations",all.iter().map(|m|m.iterations).sum::<usize>() as f64).minimum("executed_coupled_iterations",28.0)
        .metric("minimum_peak_damage_change",all.iter().map(|m|m.peak_change).fold(f64::INFINITY,f64::min)).minimum("minimum_peak_damage_change",0.01)
        .reference("independent 3D quadratic displacement/damage, positive diagonal strain, maximum-load elastic history and degraded stress; strong body/damage sources and elastic/crack energies")
        .reference_checksum("analytic:3d-forced-at2-degraded-elasticity-load-unload:v1")
        .input_checksum("P1:n3-6-9:P2:n2:loads0,.4,.8,.3,1,.2,1.2,.5:E20:nu.25:Gc.2:ell.25:four-controls")
        .dimension("3D tetrahedral cube; free damage and three displacement components")
        .formulation("native AT2 regularization/history RHS/tangent and solved-damage degraded elasticity; staggered SPD solves, quadrature history from native elastic stress and actual solved strain; verified tensile strains; prescribed body/damage sources and Dirichlet data")
        .tolerance("P1 values>=1.8 and gradients/history/stress>=.9 convergence; P2/residual/budgets/irreversibility<=1e-9; four controls>=1e-3"))
}
fn main() {
    sonium_qa::fem_physics_3d::run_bilinear_case(
        "fem.fracture-3d.v1",
        &PhaseFieldDamageForm {
            material: PhaseFieldFractureMaterial {
                critical_energy_release_rate: 2.0,
                length_scale: 0.1,
                residual_stiffness: 1e-8,
            },
        },
        false,
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
