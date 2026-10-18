//! Tier-2 flare probe: one mouth-flared profile versus the straight tube.
//!
//! Screening evidence only (coarse fidelity, 2D slot surrogate): runs both
//! profiles at equal voltage and at equal airflow (one linear rescale plus
//! a verification run) and prints THD, the `U^8/L^3` trend, bulk Mach, and
//! compression. Ignored by default; run explicitly in release mode:
//!
//! ```sh
//! cargo test -p sonium-sim-speaker-modeling --release \
//!   --test port_optim_flare_probe -- --ignored --nocapture
//! ```

use sonium_sim_speaker_modeling::port_flow::{
    PortEvalDrive, PortEvalFidelity, PortOptimReferenceCase, ProfileOptions, SplinePortProfile,
    Tier2EvalContext,
};

/// Interior-only profile at slot scale: both mouths pinned to the bore,
/// the three interior knots set from `interior` (signed flare variables).
fn interior_profile(
    context: &Tier2EvalContext,
    interior: [f64; 3],
) -> Result<SplinePortProfile, sonium_sim_speaker_modeling::port_flow::PortFlowError> {
    let bore = context.bore_radius_m();
    SplinePortProfile::from_normalized_flared(
        5,
        context.duct_length_m(),
        bore,
        0.6 * bore,
        1.4 * bore,
        &[0.0, interior[0], interior[1], interior[2], 0.0],
        ProfileOptions::default(),
    )
}

fn coarse_fidelity() -> PortEvalFidelity {
    PortEvalFidelity {
        samples_per_cycle: 48,
        total_cycles: 10,
        tail_cycles: 4,
        snapshot_interval_steps: 0,
        compression_reference: None,
    }
}

#[test]
#[ignore = "Tier-2 flare probe; run explicitly in release mode"]
fn flare_probe_equal_airflow() {
    let case = PortOptimReferenceCase::default();
    let context = Tier2EvalContext::from_reference_case(case.clone()).expect("Tier 2 context");
    let bore = context.bore_radius_m();
    let length = context.duct_length_m();
    let options = ProfileOptions::default();
    let straight = context
        .candidate_profile(5, &[0.0, 0.0], options)
        .expect("straight profile");
    // Mouth flare at slot scale: both mouths widen toward 1.4 * bore,
    // throat section stays at the bore.
    let flared = SplinePortProfile::from_normalized_flared(
        5,
        length,
        bore,
        0.6 * bore,
        1.4 * bore,
        &[-0.6, -0.3, 0.0, -0.3, -0.6],
        options,
    )
    .expect("flared profile must construct at slot scale");
    let frequency_hz = case.tier2_drive_frequency_hz().expect("drive frequency");
    let fidelity = coarse_fidelity();

    for voltage in [1.0_f64, 5.0, 10.0] {
        let drive = PortEvalDrive {
            voltage_peak_v: voltage,
            frequency_hz,
            second_tone: None,
        };
        let straight_metrics = context
            .evaluate(&straight, &drive, &fidelity)
            .expect("straight Tier 2 run")
            .metrics;
        let flared_metrics = context
            .evaluate(&flared, &drive, &fidelity)
            .expect("flared Tier 2 run")
            .metrics;
        // Equal airflow: rescale the flared drive by the flow ratio and
        // verify with a second run.
        let v_flared =
            voltage * straight_metrics.port_flow_rms_m3s / flared_metrics.port_flow_rms_m3s;
        let matched = context
            .evaluate(
                &flared,
                &PortEvalDrive {
                    voltage_peak_v: v_flared,
                    frequency_hz,
                    second_tone: None,
                },
                &fidelity,
            )
            .expect("matched flared Tier 2 run")
            .metrics;
        let match_pct =
            100.0 * (matched.port_flow_rms_m3s / straight_metrics.port_flow_rms_m3s - 1.0);
        eprintln!(
            "V={voltage}: straight THD={:.3e} trend={:.3e} Mach={:.4} | \
             flared(equal-V) THD={:.3e} trend={:.3e} Mach={:.4} | \
             flared(equal-Q, V={v_flared:.3}) THD={:.3e} ({:+.1}%) trend={:.3e} ({:+.2} dB) match={match_pct:+.2}%",
            straight_metrics.thd_fraction,
            straight_metrics.turbulence_power_trend,
            straight_metrics.peak_mach_bulk,
            flared_metrics.thd_fraction,
            flared_metrics.turbulence_power_trend,
            flared_metrics.peak_mach_bulk,
            matched.thd_fraction,
            100.0 * (matched.thd_fraction / straight_metrics.thd_fraction - 1.0),
            matched.turbulence_power_trend,
            10.0 * (matched.turbulence_power_trend / straight_metrics.turbulence_power_trend)
                .log10(),
        );
        assert!(
            straight_metrics.health.passed,
            "straight run must be healthy"
        );
        assert!(matched.health.passed, "matched flared run must be healthy");
        assert!(
            match_pct.abs() < 5.0,
            "equal-airflow match must land within 5%, got {match_pct}%"
        );
    }
}

/// Separation-aware leg: snapshot fidelity at the highest probe drive, so
/// the separation fraction and field Mach are measured, not trended.
#[test]
#[ignore = "Tier-2 flare separation probe; run explicitly in release mode"]
fn flare_probe_separation_snapshots() {
    let case = PortOptimReferenceCase::default();
    let context = Tier2EvalContext::from_reference_case(case.clone()).expect("Tier 2 context");
    let bore = context.bore_radius_m();
    let length = context.duct_length_m();
    let options = ProfileOptions::default();
    let straight = context
        .candidate_profile(5, &[0.0, 0.0], options)
        .expect("straight profile");
    let flared = SplinePortProfile::from_normalized_flared(
        5,
        length,
        bore,
        0.6 * bore,
        1.4 * bore,
        &[-0.6, -0.3, 0.0, -0.3, -0.6],
        options,
    )
    .expect("flared profile must construct at slot scale");
    let fidelity = PortEvalFidelity {
        samples_per_cycle: 48,
        total_cycles: 10,
        tail_cycles: 4,
        snapshot_interval_steps: 4,
        compression_reference: None,
    };
    let drive = PortEvalDrive {
        voltage_peak_v: 10.0,
        frequency_hz: case.tier2_drive_frequency_hz().expect("drive frequency"),
        second_tone: None,
    };
    let straight_metrics = context
        .evaluate(&straight, &drive, &fidelity)
        .expect("straight snapshot run")
        .metrics;
    let flared_metrics = context
        .evaluate(&flared, &drive, &fidelity)
        .expect("flared snapshot run")
        .metrics;
    eprintln!(
        "separation @10V equal-V: straight sep={:?} Mach_field={:?} THD={:.3e} | \
         flared sep={:?} Mach_field={:?} THD={:.3e} (snapshots {} vs {})",
        straight_metrics.separation_fraction,
        straight_metrics.peak_mach_field,
        straight_metrics.thd_fraction,
        flared_metrics.separation_fraction,
        flared_metrics.peak_mach_field,
        flared_metrics.thd_fraction,
        straight_metrics.n_tail_snapshots,
        flared_metrics.n_tail_snapshots,
    );
    assert!(straight_metrics.health.passed && flared_metrics.health.passed);
    assert!(
        straight_metrics.separation_fraction.is_some()
            && flared_metrics.separation_fraction.is_some(),
        "snapshot fidelity must produce separation fractions"
    );
}

/// Interior-only shapes at Tier 2: gentle interior bulge and pinch versus
/// the straight tube, both mouths pinned to the bore and the port length
/// fixed (the cabinet constraints). Equal airflow at 5 V and 10 V, with
/// snapshots so separation and field Mach are measured.
#[test]
#[ignore = "Tier-2 interior-shape probe; run explicitly in release mode"]
fn tier2_interior_shapes_equal_airflow() {
    let case = PortOptimReferenceCase::default();
    let context = Tier2EvalContext::from_reference_case(case.clone()).expect("Tier 2 context");
    let frequency_hz = case.tier2_drive_frequency_hz().expect("drive frequency");
    let fidelity = PortEvalFidelity {
        samples_per_cycle: 48,
        total_cycles: 10,
        tail_cycles: 4,
        snapshot_interval_steps: 4,
        compression_reference: None,
    };
    let shapes: [(&str, SplinePortProfile); 3] = [
        (
            "bulge",
            interior_profile(&context, [-0.5, -0.5, -0.5]).expect("bulge profile"),
        ),
        (
            "pinch",
            interior_profile(&context, [0.5, 0.5, 0.5]).expect("pinch profile"),
        ),
        (
            "straight",
            interior_profile(&context, [0.0, 0.0, 0.0]).expect("straight profile"),
        ),
    ];
    let straight = &shapes[2].1;
    for voltage in [5.0_f64, 10.0] {
        let drive = PortEvalDrive {
            voltage_peak_v: voltage,
            frequency_hz,
            second_tone: None,
        };
        let straight_metrics = context
            .evaluate(straight, &drive, &fidelity)
            .expect("straight Tier 2 run")
            .metrics;
        for (name, profile) in &shapes {
            let equal_v = context
                .evaluate(profile, &drive, &fidelity)
                .expect("shape Tier 2 run")
                .metrics;
            // Equal airflow: rescale this shape's drive by the flow ratio
            // against the straight tube at the same voltage, then verify.
            let v_matched =
                voltage * straight_metrics.port_flow_rms_m3s / equal_v.port_flow_rms_m3s;
            let matched = context
                .evaluate(
                    profile,
                    &PortEvalDrive {
                        voltage_peak_v: v_matched,
                        frequency_hz,
                        second_tone: None,
                    },
                    &fidelity,
                )
                .expect("matched Tier 2 run")
                .metrics;
            let match_pct =
                100.0 * (matched.port_flow_rms_m3s / straight_metrics.port_flow_rms_m3s - 1.0);
            eprintln!(
                "V={voltage} {name}: equal-Q THD={:.3e} ({:+.1}% vs straight) trend dB={:+.2} \
                 sep={:?} Mach_field={:?} Mach_bulk={:.4} match={match_pct:+.2}%",
                matched.thd_fraction,
                100.0 * (matched.thd_fraction / straight_metrics.thd_fraction - 1.0),
                10.0 * (matched.turbulence_power_trend / straight_metrics.turbulence_power_trend)
                    .log10(),
                matched.separation_fraction,
                matched.peak_mach_field,
                matched.peak_mach_bulk,
            );
            assert!(matched.health.passed, "{name} matched run must be healthy");
            assert!(
                matched.separation_fraction.is_some(),
                "{name} must produce a separation fraction with snapshots"
            );
            assert!(
                match_pct.abs() < 5.0,
                "{name} equal-airflow match must land within 5%, got {match_pct}%"
            );
        }
    }
}

/// Interior-shape screen at Tier 2: bulge family versus straight at 10 V,
/// equal airflow, coarse fidelity without snapshots (fast ranking).
/// Prints THD and trend for every shape; the winner feeds the
/// production-fidelity verification test.
#[test]
#[ignore = "Tier-2 interior screen; run explicitly in release mode"]
fn tier2_interior_search_screen() {
    let case = PortOptimReferenceCase::default();
    let context = Tier2EvalContext::from_reference_case(case.clone()).expect("Tier 2 context");
    let frequency_hz = case.tier2_drive_frequency_hz().expect("drive frequency");
    let fidelity = coarse_fidelity();
    let voltage = 10.0_f64;
    let drive = PortEvalDrive {
        voltage_peak_v: voltage,
        frequency_hz,
        second_tone: None,
    };
    let shapes: [(&str, [f64; 3]); 12] = [
        ("straight", [0.0, 0.0, 0.0]),
        ("bulge-0.25", [-0.25, -0.25, -0.25]),
        ("bulge-0.50", [-0.5, -0.5, -0.5]),
        ("bulge-0.75", [-0.75, -0.75, -0.75]),
        ("bulge-0.40", [-0.4, -0.4, -0.4]),
        ("bulge-0.60", [-0.6, -0.6, -0.6]),
        ("bulge-0.95", [-0.95, -0.95, -0.95]),
        ("bell", [-0.25, -0.6, -0.25]),
        ("inlet-heavy", [-0.75, -0.4, -0.15]),
        ("outlet-heavy", [-0.15, -0.4, -0.75]),
        ("centre-only", [0.0, -0.75, 0.0]),
        ("pinch-0.50", [0.5, 0.5, 0.5]),
    ];
    let straight_profile = interior_profile(&context, [0.0, 0.0, 0.0]).expect("straight");
    let straight_q = context
        .evaluate(&straight_profile, &drive, &fidelity)
        .expect("straight run")
        .metrics
        .port_flow_rms_m3s;
    let mut ranked: Vec<(&str, f64, f64, f64)> = Vec::new();
    for (name, interior) in shapes {
        let profile = interior_profile(&context, interior).expect("shape profile");
        let equal_v = context
            .evaluate(&profile, &drive, &fidelity)
            .expect("shape run")
            .metrics;
        let v_matched = voltage * straight_q / equal_v.port_flow_rms_m3s;
        let matched = context
            .evaluate(
                &profile,
                &PortEvalDrive {
                    voltage_peak_v: v_matched,
                    frequency_hz,
                    second_tone: None,
                },
                &fidelity,
            )
            .expect("matched run")
            .metrics;
        let match_pct = 100.0 * (matched.port_flow_rms_m3s / straight_q - 1.0);
        eprintln!(
            "screen {name}: THD={:.3e} trend={:.3e} Mach_bulk={:.4} match={match_pct:+.2}%",
            matched.thd_fraction, matched.turbulence_power_trend, matched.peak_mach_bulk
        );
        assert!(matched.health.passed, "{name} must be healthy");
        assert!(match_pct.abs() < 5.0, "{name} airflow match {match_pct}%");
        ranked.push((
            name,
            matched.thd_fraction,
            matched.turbulence_power_trend,
            v_matched,
        ));
    }
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
    let straight_thd = ranked
        .iter()
        .find(|(name, ..)| *name == "straight")
        .map(|row| row.1)
        .expect("straight in screen");
    eprintln!("screen ranking by THD (straight THD={straight_thd:.3e}):");
    for (name, thd, trend, v) in &ranked {
        eprintln!(
            "  {name}: THD={thd:.3e} ({:+.1}%) trend={trend:.3e} V={v:.2}",
            100.0 * (thd / straight_thd - 1.0)
        );
    }
}

/// Production-fidelity verification of the screen winner versus straight:
/// 252 samples/cycle, snapshots on, equal airflow at 5 V and 10 V.
/// Significant improvement means the THD gain survives production
/// fidelity at both drives without an airflow or health regression.
#[test]
#[ignore = "Tier-2 production verification; run explicitly in release mode"]
fn tier2_interior_winner_production() {
    // Screen winner (tier2_interior_search_screen): uniform interior bulge.
    const WINNER: [f64; 3] = [-0.5, -0.5, -0.5];
    let case = PortOptimReferenceCase::default();
    let context = Tier2EvalContext::from_reference_case(case.clone()).expect("Tier 2 context");
    let frequency_hz = case.tier2_drive_frequency_hz().expect("drive frequency");
    let mut fidelity = PortEvalFidelity::from_reference_case(&case);
    fidelity.snapshot_interval_steps = 42;
    let straight = interior_profile(&context, [0.0, 0.0, 0.0]).expect("straight");
    let winner = interior_profile(&context, WINNER).expect("winner");
    for voltage in [5.0_f64, 10.0] {
        let drive = PortEvalDrive {
            voltage_peak_v: voltage,
            frequency_hz,
            second_tone: None,
        };
        let straight_metrics = context
            .evaluate(&straight, &drive, &fidelity)
            .expect("straight production run")
            .metrics;
        let equal_v = context
            .evaluate(&winner, &drive, &fidelity)
            .expect("winner production run")
            .metrics;
        let v_matched = voltage * straight_metrics.port_flow_rms_m3s / equal_v.port_flow_rms_m3s;
        let matched = context
            .evaluate(
                &winner,
                &PortEvalDrive {
                    voltage_peak_v: v_matched,
                    frequency_hz,
                    second_tone: None,
                },
                &fidelity,
            )
            .expect("winner matched production run")
            .metrics;
        let match_pct =
            100.0 * (matched.port_flow_rms_m3s / straight_metrics.port_flow_rms_m3s - 1.0);
        eprintln!(
            "production V={voltage}: straight THD={:.3e} trend={:.3e} sep={:?} | \
             winner THD={:.3e} ({:+.1}%) trend dB={:+.2} sep={:?} Mach_bulk={:.4} match={match_pct:+.2}%",
            straight_metrics.thd_fraction,
            straight_metrics.turbulence_power_trend,
            straight_metrics.separation_fraction,
            matched.thd_fraction,
            100.0 * (matched.thd_fraction / straight_metrics.thd_fraction - 1.0),
            10.0 * (matched.turbulence_power_trend / straight_metrics.turbulence_power_trend)
                .log10(),
            matched.separation_fraction,
            matched.peak_mach_bulk,
        );
        assert!(straight_metrics.health.passed && matched.health.passed);
        assert!(match_pct.abs() < 5.0, "airflow match {match_pct}%");
        assert!(
            matched.thd_fraction < 0.85 * straight_metrics.thd_fraction,
            "winner must keep a significant (>15%) THD gain at production fidelity"
        );
    }
}

/// Fine-mesh check of the screen winner: same comparison on a 2x finer
/// Tier-2 mesh (0.005 m), coarse time fidelity, 10 V equal airflow.
#[test]
#[ignore = "Tier-2 fine-mesh winner check; run explicitly in release mode"]
fn tier2_interior_winner_fine_mesh() {
    const WINNER: [f64; 3] = [-0.5, -0.5, -0.5];
    let case = PortOptimReferenceCase::default();
    let context = Tier2EvalContext::from_reference_case(case.clone())
        .expect("Tier 2 context")
        .with_mesh_size_m(0.005)
        .expect("fine mesh context");
    let frequency_hz = case.tier2_drive_frequency_hz().expect("drive frequency");
    let fidelity = PortEvalFidelity {
        samples_per_cycle: 48,
        total_cycles: 10,
        tail_cycles: 4,
        snapshot_interval_steps: 4,
        compression_reference: None,
    };
    let drive = PortEvalDrive {
        voltage_peak_v: 10.0,
        frequency_hz,
        second_tone: None,
    };
    let straight = interior_profile(&context, [0.0, 0.0, 0.0]).expect("straight");
    let winner = interior_profile(&context, WINNER).expect("winner");
    let straight_metrics = context
        .evaluate(&straight, &drive, &fidelity)
        .expect("straight fine run")
        .metrics;
    let equal_v = context
        .evaluate(&winner, &drive, &fidelity)
        .expect("winner fine run")
        .metrics;
    let v_matched = 10.0 * straight_metrics.port_flow_rms_m3s / equal_v.port_flow_rms_m3s;
    let matched = context
        .evaluate(
            &winner,
            &PortEvalDrive {
                voltage_peak_v: v_matched,
                frequency_hz,
                second_tone: None,
            },
            &fidelity,
        )
        .expect("winner matched fine run")
        .metrics;
    eprintln!(
        "fine-mesh V=10: straight THD={:.3e} sep={:?} | winner THD={:.3e} ({:+.1}%) sep={:?}",
        straight_metrics.thd_fraction,
        straight_metrics.separation_fraction,
        matched.thd_fraction,
        100.0 * (matched.thd_fraction / straight_metrics.thd_fraction - 1.0),
        matched.separation_fraction,
    );
    assert!(straight_metrics.health.passed && matched.health.passed);
    assert!(
        matched.thd_fraction < 0.85 * straight_metrics.thd_fraction,
        "winner must keep a significant THD gain on the fine mesh"
    );
}

/// Fidelity ladder for the coarse-screen winner: straight versus
/// bulge-0.50 at 10 V equal airflow across time resolutions. The coarse
/// (48 samples/cycle) ranking reversed at production fidelity, so this
/// establishes where the THD ranking stabilises before any new screen.
#[test]
#[ignore = "Tier-2 fidelity ladder; run explicitly in release mode"]
fn tier2_fidelity_ladder() {
    let case = PortOptimReferenceCase::default();
    let context = Tier2EvalContext::from_reference_case(case.clone()).expect("Tier 2 context");
    let frequency_hz = case.tier2_drive_frequency_hz().expect("drive frequency");
    let straight = interior_profile(&context, [0.0, 0.0, 0.0]).expect("straight");
    let winner = interior_profile(&context, [-0.5, -0.5, -0.5]).expect("winner");
    let drive = PortEvalDrive {
        voltage_peak_v: 10.0,
        frequency_hz,
        second_tone: None,
    };
    for samples_per_cycle in [96_usize, 144, 252] {
        let fidelity = PortEvalFidelity {
            samples_per_cycle,
            total_cycles: 10,
            tail_cycles: 4,
            snapshot_interval_steps: 0,
            compression_reference: None,
        };
        let straight_metrics = context
            .evaluate(&straight, &drive, &fidelity)
            .expect("straight ladder run")
            .metrics;
        let equal_v = context
            .evaluate(&winner, &drive, &fidelity)
            .expect("winner ladder run")
            .metrics;
        let v_matched = 10.0 * straight_metrics.port_flow_rms_m3s / equal_v.port_flow_rms_m3s;
        let matched = context
            .evaluate(
                &winner,
                &PortEvalDrive {
                    voltage_peak_v: v_matched,
                    frequency_hz,
                    second_tone: None,
                },
                &fidelity,
            )
            .expect("winner matched ladder run")
            .metrics;
        eprintln!(
            "ladder spc={samples_per_cycle}: straight THD={:.3e} | winner THD={:.3e} ({:+.1}%)",
            straight_metrics.thd_fraction,
            matched.thd_fraction,
            100.0 * (matched.thd_fraction / straight_metrics.thd_fraction - 1.0),
        );
        assert!(straight_metrics.health.passed && matched.health.passed);
    }
}

/// Production-window screen at 5 V: candidate interior shapes at the
/// production fidelity (252 samples/cycle, 40 cycles, 8-cycle tail),
/// equal airflow. The coarse-fidelity ranking reversed under this
/// window, so screening happens here despite the cost. (10 V diverges
/// late in the production window on the straight tube, step ~9147.)
#[test]
#[ignore = "Tier-2 production screen; run explicitly in release mode"]
fn tier2_production_screen() {
    let case = PortOptimReferenceCase::default();
    let context = Tier2EvalContext::from_reference_case(case.clone()).expect("Tier 2 context");
    let frequency_hz = case.tier2_drive_frequency_hz().expect("drive frequency");
    let fidelity = PortEvalFidelity::from_reference_case(&case);
    let voltage = 5.0_f64;
    let drive = PortEvalDrive {
        voltage_peak_v: voltage,
        frequency_hz,
        second_tone: None,
    };
    let shapes: [(&str, [f64; 3]); 7] = [
        ("straight", [0.0, 0.0, 0.0]),
        ("bulge-0.50", [-0.5, -0.5, -0.5]),
        ("bulge-0.25", [-0.25, -0.25, -0.25]),
        ("inlet-heavy", [-0.75, -0.4, -0.15]),
        ("outlet-pinch", [0.0, 0.0, 0.5]),
        ("pinch-0.25", [0.25, 0.25, 0.25]),
        ("inlet-bulge", [-0.5, 0.0, 0.0]),
    ];
    let straight_profile = interior_profile(&context, [0.0, 0.0, 0.0]).expect("straight");
    let straight_metrics = context
        .evaluate(&straight_profile, &drive, &fidelity)
        .expect("straight production run")
        .metrics;
    eprintln!(
        "production-screen straight: THD={:.3e} trend={:.3e}",
        straight_metrics.thd_fraction, straight_metrics.turbulence_power_trend
    );
    let mut ranked: Vec<(&str, f64, f64)> = vec![(
        "straight",
        straight_metrics.thd_fraction,
        straight_metrics.turbulence_power_trend,
    )];
    for (name, interior) in shapes.into_iter().skip(1) {
        let profile = interior_profile(&context, interior).expect("shape profile");
        let equal_v = context
            .evaluate(&profile, &drive, &fidelity)
            .expect("shape production run")
            .metrics;
        let v_matched = voltage * straight_metrics.port_flow_rms_m3s / equal_v.port_flow_rms_m3s;
        let matched = context
            .evaluate(
                &profile,
                &PortEvalDrive {
                    voltage_peak_v: v_matched,
                    frequency_hz,
                    second_tone: None,
                },
                &fidelity,
            )
            .expect("matched production run")
            .metrics;
        let match_pct =
            100.0 * (matched.port_flow_rms_m3s / straight_metrics.port_flow_rms_m3s - 1.0);
        eprintln!(
            "production-screen {name}: THD={:.3e} ({:+.1}%) trend dB={:+.2} match={match_pct:+.2}%",
            matched.thd_fraction,
            100.0 * (matched.thd_fraction / straight_metrics.thd_fraction - 1.0),
            10.0 * (matched.turbulence_power_trend / straight_metrics.turbulence_power_trend)
                .log10(),
        );
        assert!(matched.health.passed, "{name} must be healthy");
        assert!(match_pct.abs() < 5.0, "{name} airflow match {match_pct}%");
        ranked.push((name, matched.thd_fraction, matched.turbulence_power_trend));
    }
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
    eprintln!("production-screen ranking by THD:");
    for (name, thd, trend) in &ranked {
        eprintln!(
            "  {name}: THD={thd:.3e} ({:+.1}%) trend={trend:.3e}",
            100.0 * (thd / straight_metrics.thd_fraction - 1.0)
        );
    }
}
