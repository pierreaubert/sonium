// Temporary probe (deleted before finishing).
use sonium_fem::physics::solid::plasticity::funct::Funct1D;
use sonium_fem::physics::solid::plasticity::law_foams::{
    HoneycombLaw, HoneycombState, JwlEos, Multimat51, MultimatState,
};
use sonium_fem::physics::solid::plasticity::law_metals::{Law44, Law44State, Law62, Law62State};
use sonium_fem::physics::solid::plasticity::update::{StressUpdateModel, von_mises_c1};

#[test]
fn probe_law44() {
    let law = Law44::card_metal();
    println!(
        "flow(0,0)={:.6e} flow(0.01,0)={:.6e}",
        law.flow_stress(0.0, 0.0),
        law.flow_stress(0.01, 0.0)
    );
    println!(
        "rate_factor(1e-3)={:.6e} rate_factor(1e2)={:.6e}",
        law.rate_factor(1e-3),
        law.rate_factor(1e2)
    );
    // uniaxial with lateral solve, emax=0.02
    let de = 0.02 / 100.0;
    let mut state = Law44State::new();
    let mut ny = 0;
    for _ in 0..100 {
        let (mut lo, mut hi) = (-de, 0.0);
        let probe = |dl: f64| law.update(&state, &[de, dl, dl, 0.0, 0.0, 0.0]).stress[1];
        let mut blo = probe(lo);
        let mut bhi = probe(hi);
        for _ in 0..20 {
            if blo < 0.0 && bhi > 0.0 {
                break;
            }
            let w = (hi - lo).abs();
            lo -= w;
            hi += w;
            blo = probe(lo);
            bhi = probe(hi);
        }
        assert!(blo < 0.0 && bhi > 0.0);
        for _ in 0..50 {
            let mid = 0.5 * (lo + hi);
            if probe(mid) > 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let dl = 0.5 * (lo + hi);
        let u = law.update(&state, &[de, dl, dl, 0.0, 0.0, 0.0]);
        if u.yielding {
            ny += 1;
        }
        law.commit(&mut state, &u);
    }
    println!(
        "law44 uni: yielding_steps={ny} ep={:.6e} sig={:.6e} {:.6e} {:.6e} scale={} rupt={}",
        state.equiv_plastic_strain,
        state.stress[0],
        state.stress[1],
        state.stress[2],
        state.tensile_scale,
        state.ruptured
    );
    println!(
        "  back={:.3e} {:.3e} {:.3e}",
        state.back_stress[0], state.back_stress[1], state.back_stress[2]
    );
    let rel = std::array::from_fn(|i| state.stress[i] - state.back_stress[i]);
    println!(
        "  q_rel={:.6e} sig0={:.6e}",
        von_mises_c1(&rel),
        law.flow_stress(0.0, 0.0)
    );
    // volumetric: elastic?
    let mut state = Law44State::new();
    let dv = 0.02 / 100.0;
    let mut any_y = false;
    for _ in 0..100 {
        let u = law.update(&state, &[dv, dv, dv, 0.0, 0.0, 0.0]);
        any_y |= u.yielding;
        law.commit(&mut state, &u);
    }
    println!(
        "law44 vol: any_yield={any_y} ep={:.3e} sig0={:.6e}",
        state.equiv_plastic_strain, state.stress[0]
    );
    // shear
    let mut state = Law44State::new();
    let dg = 0.02 / 100.0;
    let mut ny = 0;
    for _ in 0..100 {
        let u = law.update(&state, &[0.0, 0.0, 0.0, dg, 0.0, 0.0]);
        if u.yielding {
            ny += 1;
        }
        law.commit(&mut state, &u);
    }
    println!(
        "law44 shear: yielding_steps={ny} ep={:.6e} sig3={:.6e}",
        state.equiv_plastic_strain, state.stress[3]
    );
    // rate leg factor check: total-norm rate
    let dt = de / 1e2;
    let d = [de, -0.5 * de, -0.5 * de, 0.0, 0.0, 0.0];
    let shear = 0.5 * (d[3] * d[3] + d[4] * d[4] + d[5] * d[5]);
    let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + shear).sqrt() / dt;
    println!(
        "law44 fast total-rate={r:.6e} factor={:.6e}",
        law.rate_factor(r)
    );
}

#[test]
fn probe_honeycomb() {
    let lo = Funct1D::new(vec![0.0, 1.0], vec![1.0, 3.0]).unwrap();
    let hi = Funct1D::new(vec![0.0, 1.0], vec![2.0, 4.0]).unwrap();
    let rated = core::array::from_fn(|_| vec![(lo.clone(), 0.0), (hi.clone(), 10.0)]);
    let law = HoneycombLaw::law50([100.0; 6], rated).unwrap();
    // axis-1 compression, emax=0.3, rate 5
    let de = 0.3 / 100.0;
    let dt = de / 5.0;
    let mut state = HoneycombState::new();
    for i in 0..100 {
        let d = [-de, 0.0, 0.0, 0.0, 0.0, 0.0];
        let u = law.update_with_rate(&state, &d, dt);
        if i < 2 || i == 99 {
            println!(
                "hc step {}: stress={:.6e} {:.6e} y={} dep={:.3e}",
                i + 1,
                u.stress[0],
                u.stress[1],
                u.yielding,
                u.delta_equiv_plastic_strain
            );
        }
        law.commit(&mut state, &u);
        law.advance_totals(&mut state, &d, dt);
    }
    println!(
        "hc ax1: sig={:.6e} ep={:.6e} strains={:.4}",
        state.stresses[0], state.equiv_plastic, state.strains[0]
    );
}

#[test]
fn probe_multimat() {
    let eos = JwlEos::new(2.0e9, 5.0e8, 4.5, 1.2, 0.25, 1500.0, 1.0e5).unwrap();
    let reacted = JwlEos::new(1.0e9, 5.0e8, 4.5, 1.2, 0.25, 1500.0, 1.0e5).unwrap();
    println!(
        "virgin P unreacted={:.6e} reacted={:.6e}",
        eos.pressure(0.0, 1.0e5),
        reacted.pressure(0.0, 1.0e5)
    );
    println!(
        "P(mu=0.3,e=2e5)={:.6e} P(mu=-0.3,e=5e4)={:.6e}",
        eos.pressure(0.3, 2.0e5),
        eos.pressure(-0.3, 5.0e4)
    );
    let law = Multimat51::new(eos, reacted, 0.0, 0.0, 100.0 * 1.0e-6).unwrap();
    // volumetric compression mu -> 0.3
    let dv = 0.001;
    let mut state = MultimatState::new(eos.e0, reacted.e0);
    for _ in 0..100 {
        let d = [-dv, -dv, -dv, 0.0, 0.0, 0.0];
        let u = law.update(&state, &d);
        law.commit(&mut state, &u);
        law.advance_totals(&mut state, &d, 1.0e-6);
    }
    println!(
        "mm vol_c: sig0={:.6e} mu={:.4} eu={:.6e} er={:.6e} burn={} P={:.6e}",
        -law.mixture_pressure(state.mu, state.energy_u, state.energy_r, state.burn),
        state.mu,
        state.energy_u,
        state.energy_r,
        state.burn,
        state.pressure
    );
    // uniaxial tension mu -> -0.02
    let mut state = MultimatState::new(eos.e0, reacted.e0);
    let de = 0.0002;
    for _ in 0..100 {
        let d = [de, 0.0, 0.0, 0.0, 0.0, 0.0];
        let u = law.update(&state, &d);
        law.commit(&mut state, &u);
        law.advance_totals(&mut state, &d, 1.0e-6);
    }
    println!(
        "mm uni_t: mu={:.5} eu={:.6e} P={:.6e} burn={}",
        state.mu, state.energy_u, state.pressure, state.burn
    );
    // shear: zero deviatoric?
    let mut state = MultimatState::new(eos.e0, reacted.e0);
    for _ in 0..100 {
        let d = [0.0, 0.0, 0.0, 0.002, 0.0, 0.0];
        let u = law.update(&state, &d);
        assert!(u.stress[3] == 0.0);
        law.commit(&mut state, &u);
        law.advance_totals(&mut state, &d, 1.0e-6);
    }
    println!(
        "mm shear: sig={:.6e} {:.6e} {:.6e} {:.6e} mu={} P={:.6e} burn={}",
        -state.pressure,
        -state.pressure,
        -state.pressure,
        0.0,
        state.mu,
        state.pressure,
        state.burn
    );
}

#[test]
fn probe_law62() {
    let law = Law62::rubber_prony();
    println!(
        "G0={:.3e} gamma_inf={} bulk={:.6e}",
        law.ground_shear(),
        law.gamma_infinity(),
        law.bulk_modulus()
    );
    // shear path dt=1e-3
    let dt = 1.0e-3;
    let dg = 0.002;
    let mut state = Law62State::new(law.prony.len());
    for i in 0..100 {
        let d = [0.0, 0.0, 0.0, dg, 0.0, 0.0];
        let u = law.update_with_rate(&state, &d, dt);
        if i == 0 || i == 99 {
            println!(
                "l62 shear step {}: sig3={:.6e} y={} dep={}",
                i + 1,
                u.stress[3],
                u.yielding,
                u.delta_equiv_plastic_strain
            );
        }
        law.commit(&mut state, &u);
    }
    println!(
        "l62 shear end: sig3={:.6e} br3={:.6e} hencky3={:.6e}",
        state.stress[3], state.branches[0][3], state.hencky[3]
    );
    // volumetric small
    let mut state = Law62State::new(law.prony.len());
    for _ in 0..100 {
        let d = [0.0001, 0.0001, 0.0001, 0.0, 0.0, 0.0];
        let u = law.update_with_rate(&state, &d, dt);
        law.commit(&mut state, &u);
    }
    println!(
        "l62 vol: sig0={:.6e} hencky0={:.6e}",
        state.stress[0], state.hencky[0]
    );
    // uniaxial lateral solve probe
    let state = Law62State::new(law.prony.len());
    let de = 0.002;
    let probe = |dl: f64| {
        law.update_with_rate(&state, &[de, dl, dl, 0.0, 0.0, 0.0], dt)
            .stress[1]
    };
    let (mut lo, mut hi) = (-de, 0.0);
    let mut blo = probe(lo);
    let mut bhi = probe(hi);
    for _ in 0..20 {
        if blo < 0.0 && bhi > 0.0 {
            break;
        }
        let w = (hi - lo).abs();
        lo -= w;
        hi += w;
        blo = probe(lo);
        bhi = probe(hi);
    }
    println!("l62 lat bracket: blo={blo:.3e} bhi={bhi:.3e}");
    for _ in 0..50 {
        let mid = 0.5 * (lo + hi);
        if probe(mid) > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    println!(
        "l62 lat dl={:.6e} sig1={:.6e}",
        0.5 * (lo + hi),
        probe(0.5 * (lo + hi))
    );
}
