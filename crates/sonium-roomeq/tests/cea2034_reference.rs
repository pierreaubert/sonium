use serde::Deserialize;
use sonium_roomeq::cea2034::{
    ApproxPeqScoreInputRef, PeqScoreScratch, score_peq_approx, score_peq_approx_with_scratch,
    try_cea2034, try_directivity_index, try_score,
};

#[derive(Debug, Deserialize)]
struct CeaFixture {
    source: String,
    spl_db: Vec<Vec<f64>>,
    groups: Vec<Vec<usize>>,
    weights: Vec<f64>,
    expected_spinorama_db: Vec<Vec<f64>>,
}

#[test]
fn independent_cea2034_fixture_matches_reference_values() {
    let fixture: CeaFixture =
        serde_json::from_str(include_str!("../fixtures/cea2034-reference-v1.json"))
            .expect("CEA fixture must be valid JSON");
    assert_eq!(fixture.source, "independent-cea2034-equations-v1");

    let actual = try_cea2034(&fixture.spl_db, &fixture.groups, &fixture.weights)
        .expect("fixture should satisfy the CEA-2034 input contract");
    assert_eq!(actual.len(), fixture.expected_spinorama_db.len());
    for (actual_row, expected_row) in actual.iter().zip(&fixture.expected_spinorama_db) {
        assert_eq!(actual_row.len(), expected_row.len());
        for (&actual, &expected) in actual_row.iter().zip(expected_row) {
            assert!(
                (actual - expected).abs() < 2.0e-12,
                "{actual} != {expected}"
            );
        }
    }
}

#[test]
fn score_boundary_rejects_ragged_curves_and_empty_intervals() {
    let frequency = [100.0, 1_000.0];
    let curve = [80.0, 81.0];
    assert!(try_score(&frequency, &[(0, 2)], &curve, &curve[..1], &curve, &curve,).is_err());
    assert!(try_score(&frequency, &[], &curve, &curve, &curve, &curve,).is_err());
    assert!(try_score(&frequency, &[(1, 1)], &curve, &curve, &curve, &curve,).is_err());
    assert!(try_score(&frequency, &[(0, 3)], &curve, &curve, &curve, &curve,).is_err());
}

#[test]
fn cea2034_rejects_negative_sound_power_weights() {
    let spl = vec![vec![80.0, 81.0]; 8];
    let groups = (0..8).map(|row| vec![row]).collect::<Vec<_>>();
    let mut weights = vec![1.0; 8];
    weights[7] = -1.0;
    assert!(try_cea2034(&spl, &groups, &weights).is_err());
}

#[test]
fn reusable_peq_score_scratch_matches_one_shot_without_growing() {
    let frequency = [100.0, 1_000.0, 10_000.0];
    let intervals = [(0, 3)];
    let on = [80.0, 85.0, 82.0];
    let lw = [81.0, 84.0, 83.0];
    let sp = [79.0, 83.0, 81.0];
    let pir = [80.5, 84.0, 82.0];
    let peq = [-1.0, 0.5, 1.0];
    let expected = score_peq_approx(&frequency, &intervals, &lw, &sp, &pir, &on, &peq).unwrap();
    let mut scratch = PeqScoreScratch::with_capacity(frequency.len());
    let initial_capacity = scratch.minimum_capacity();
    let input = ApproxPeqScoreInputRef {
        freq: &frequency,
        intervals: &intervals,
        lw: &lw,
        sp: &sp,
        pir: &pir,
        on: &on,
        peq: &peq,
    };
    let actual = score_peq_approx_with_scratch(input, &mut scratch).unwrap();
    let repeated = score_peq_approx_with_scratch(input, &mut scratch).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(repeated, expected);
    assert_eq!(scratch.minimum_capacity(), initial_capacity);
}

#[test]
fn realistic_size_uniform_spinorama_is_flat_with_zero_directivity_index() {
    // Realistic-size golden: 24 measurement rows x 25 frequency bins of
    // uniform 80 dB. Every RMS aggregate of a constant field is the
    // constant itself (up to float summation order), so all nine spinorama
    // rows must read 80 dB and DI = LW - SP must read 0 dB. Expectations
    // are analytic, not copied from the implementation under test.
    const ROWS: usize = 24;
    const FREQS: usize = 25;
    let spl = vec![vec![80.0; FREQS]; ROWS];
    let groups = vec![
        (0..8).collect::<Vec<_>>(),
        (8..11).collect::<Vec<_>>(),
        (11..14).collect::<Vec<_>>(),
        (14..17).collect::<Vec<_>>(),
        (17..20).collect::<Vec<_>>(),
        (20..24).collect::<Vec<_>>(),
        vec![0],
        (0..24).collect::<Vec<_>>(),
    ];
    assert_eq!(groups.len(), 8);
    let weights = vec![1.0; ROWS];
    let spinorama = try_cea2034(&spl, &groups, &weights).expect("uniform input is valid");
    assert_eq!(spinorama.len(), 9);
    for (row_index, row) in spinorama.iter().enumerate() {
        assert_eq!(row.len(), FREQS);
        for (bin, &value) in row.iter().enumerate() {
            assert!(
                (value - 80.0).abs() < 1e-9,
                "row {row_index} bin {bin} drifted from uniform field: {value}"
            );
        }
    }
    let di = try_directivity_index(&spinorama[0], &spinorama[7]).expect("LW and SP rows align");
    assert_eq!(di.len(), FREQS);
    for (bin, &value) in di.iter().enumerate() {
        assert!(
            value.abs() < 1e-9,
            "uniform field must have zero DI at bin {bin}: {value}"
        );
    }
}

#[test]
fn directivity_index_rejects_mismatched_and_non_finite_curves() {
    assert_eq!(
        try_directivity_index(&[90.0, 87.0], &[84.0, 84.0]).unwrap(),
        vec![6.0, 3.0]
    );
    assert!(try_directivity_index(&[], &[]).is_err());
    assert!(try_directivity_index(&[90.0], &[84.0, 84.0]).is_err());
    assert!(try_directivity_index(&[f64::INFINITY], &[84.0]).is_err());
    assert!(try_directivity_index(&[90.0], &[f64::NAN]).is_err());
    assert!(try_directivity_index(&[f64::MAX], &[-f64::MAX]).is_err());
}

#[test]
fn energy_aggregates_are_level_shift_invariant_far_outside_pressure_range() {
    use sonium_roomeq::cea2034::{
        ContiguousCea2034InputRef, compute_pir_from_lw_er_sp, try_cea2034_contiguous,
    };
    let spl: Vec<_> = (0..8)
        .map(|i| vec![70.0 + i as f64, 85.0 - i as f64])
        .collect();
    let groups = vec![
        vec![0, 1],
        vec![1],
        vec![2],
        vec![3],
        vec![4],
        vec![5],
        vec![6],
        (0..8).collect(),
    ];
    let weights = vec![1.0; 8];
    let baseline = try_cea2034(&spl, &groups, &weights).unwrap();
    for offset in [-10000.0, 10000.0] {
        let shifted: Vec<Vec<_>> = spl
            .iter()
            .map(|row| row.iter().map(|v| v + offset).collect())
            .collect();
        let actual = try_cea2034(&shifted, &groups, &weights).unwrap();
        let flat: Vec<_> = shifted.iter().flatten().copied().collect();
        let contiguous = try_cea2034_contiguous(
            ContiguousCea2034InputRef::new(&flat, 8, 2).unwrap(),
            &groups,
            &weights,
        )
        .unwrap();
        assert_eq!(actual, contiguous);
        for (&got, &reference) in actual.iter().flatten().zip(baseline.iter().flatten()) {
            assert!((got - offset - reference).abs() < 2e-11);
        }
        assert_eq!(
            compute_pir_from_lw_er_sp(&actual[0], &actual[1], &actual[7]),
            actual[8]
        );
    }
}

#[test]
fn finite_extremes_and_subnormal_weights_remain_finite() {
    let groups = vec![(0..8).collect::<Vec<_>>(); 8];
    for level in [-f64::MAX, f64::MAX] {
        let output = try_cea2034(&vec![vec![level]; 8], &groups, &[1.0; 8]).unwrap();
        assert!(output.iter().flatten().all(|&v| v == level));
    }
    let mut spl = vec![vec![-10000.0]; 8];
    spl[0][0] = 10000.0;
    let mut weights = vec![1.0; 8];
    weights[0] = f64::from_bits(1);
    let output = try_cea2034(&spl, &groups, &weights).unwrap();
    let expected = 10000.0 + 10.0 * (weights[0].log10() - 7.0_f64.log10());
    assert!((output[7][0] - expected).abs() < 1e-10);
    weights[0] = 0.0;
    let output = try_cea2034(&spl, &groups, &weights).unwrap();
    assert!((output[7][0] + 10000.0).abs() < 1e-10);
}
