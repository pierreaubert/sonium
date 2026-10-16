#![expect(
    deprecated,
    reason = "this suite pins the deprecated recycled-GMRES checkpoint contract for API compatibility"
)]
use num_complex::{Complex32, Complex64};
use num_traits::{FromPrimitive, ToPrimitive};
use sonium_solvers::SolverResultContract;
use sonium_solvers::{
    ComplexField, GmresConfig, RECYCLED_CHECKPOINT_FORMAT_VERSION, RecycledSubspace, SolverError,
    try_gmres_recycled,
};
use sonium_solvers_sparse::CsrMatrix;

fn operator() -> CsrMatrix<f64> {
    CsrMatrix::from_triplets(
        4,
        4,
        vec![
            (0, 0, 4.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 1, 4.5),
            (1, 2, -1.0),
            (2, 1, -1.0),
            (2, 2, 5.0),
            (2, 3, -1.0),
            (3, 2, -1.0),
            (3, 3, 5.5),
        ],
    )
}

fn config() -> GmresConfig<f64> {
    GmresConfig {
        max_iterations: 80,
        restart: 6,
        tolerance: 1.0e-11,
        ..GmresConfig::default()
    }
}

fn scalar<T: ComplexField>(value: f64) -> T
where
    T::Real: FromPrimitive,
{
    T::from_real(T::Real::from_f64(value).expect("finite scalar fixture"))
}

fn scalar_operator<T: ComplexField>() -> CsrMatrix<T>
where
    T::Real: FromPrimitive,
{
    CsrMatrix::from_raw_parts(
        4,
        4,
        vec![0, 2, 5, 8, 10],
        vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
        vec![
            scalar(4.0),
            scalar(-1.0),
            scalar(-1.0),
            scalar(4.5),
            scalar(-1.0),
            scalar(-1.0),
            scalar(5.0),
            scalar(-1.0),
            scalar(-1.0),
            scalar(5.5),
        ],
    )
}

fn scalar_config<T: ComplexField>() -> GmresConfig<T::Real>
where
    T::Real: FromPrimitive,
    GmresConfig<T::Real>: Default,
{
    GmresConfig {
        max_iterations: 80,
        restart: 6,
        tolerance: T::Real::from_f64(1.0e-5).expect("finite tolerance"),
        ..GmresConfig::default()
    }
}

fn scalar_checkpoint_case<T: ComplexField>() -> (f64, f64, f64)
where
    T::Real: FromPrimitive + ToPrimitive,
    GmresConfig<T::Real>: Default,
{
    let operator = scalar_operator::<T>();
    let first_rhs = vec![scalar(1.0), scalar(-2.0), scalar(0.5), scalar(3.0)];
    let second_rhs = vec![scalar(-0.25), scalar(1.5), scalar(2.0), scalar(-1.0)];
    let mut source = RecycledSubspace::new(3);
    let first = try_gmres_recycled(
        &operator,
        &first_rhs,
        None,
        &mut source,
        &scalar_config::<T>(),
    )
    .expect("fresh recycled solve");
    assert!(first.converged && source.is_populated());
    let bytes = source.to_checkpoint_bytes().expect("checkpoint encode");
    let mut restored =
        RecycledSubspace::<T>::from_checkpoint_bytes(&bytes).expect("checkpoint decode");
    let vector_delta = restored
        .vectors()
        .iter()
        .zip(source.vectors())
        .flat_map(|(restored, original)| restored.iter().zip(original))
        .map(|(left, right)| {
            (*left - *right)
                .norm()
                .to_f64()
                .expect("finite vector delta")
        })
        .fold(0.0, f64::max);

    let mut fresh_source = RecycledSubspace::new(3);
    let fresh = try_gmres_recycled(
        &operator,
        &second_rhs,
        None,
        &mut fresh_source,
        &scalar_config::<T>(),
    )
    .expect("fresh continuation solve");
    let resumed = restored
        .resume(&operator, &second_rhs, None, &scalar_config::<T>())
        .expect("restored continuation solve");
    assert!(fresh.converged && resumed.converged);
    SolverResultContract::validate_contract(&fresh).expect("fresh result contract");
    SolverResultContract::validate_contract(&resumed).expect("restored result contract");
    let solution_delta = fresh
        .x
        .iter()
        .zip(&resumed.x)
        .map(|(left, right)| {
            (*left - *right)
                .norm()
                .to_f64()
                .expect("finite solution delta")
        })
        .fold(0.0, f64::max);
    let residual = resumed.residual.to_f64().expect("finite residual");
    (
        solution_delta.max(vector_delta),
        residual,
        bytes.len() as f64,
    )
}

#[test]
fn recycled_checkpoint_round_trips_supported_scalar_families() {
    for (label, (solution_delta, residual, bytes)) in [
        ("f32", scalar_checkpoint_case::<f32>()),
        ("Complex32", scalar_checkpoint_case::<Complex32>()),
        ("Complex64", scalar_checkpoint_case::<Complex64>()),
    ] {
        assert!(
            solution_delta <= 1.0e-5,
            "{label} solution delta {solution_delta}"
        );
        assert!(residual <= 1.0e-5, "{label} residual {residual}");
        assert!(bytes > 0.0, "{label} checkpoint was empty");
    }
}

#[test]
fn recycled_checkpoint_round_trip_resumes_with_same_subspace() {
    let a = operator();
    let b = vec![1.0, -2.0, 0.5, 3.0];
    let mut source = RecycledSubspace::new(3);
    let first = try_gmres_recycled(&a, &b, None, &mut source, &config()).unwrap();
    assert!(first.converged);
    assert!(source.is_populated());

    let bytes = source.to_checkpoint_bytes().unwrap();
    let restored = RecycledSubspace::<f64>::from_checkpoint_bytes(&bytes).unwrap();
    assert_eq!(restored.num_vectors(), source.num_vectors());
    assert_eq!(restored.dimension(), source.dimension());
    assert_eq!(restored.vectors(), source.vectors());

    let c = vec![-0.25, 1.5, 2.0, -1.0];
    let mut resumed = restored;
    let resumed_result = try_gmres_recycled(&a, &c, None, &mut resumed, &config()).unwrap();
    let mut fresh = RecycledSubspace::new(3);
    let fresh_result = try_gmres_recycled(&a, &c, None, &mut fresh, &config()).unwrap();
    assert!(resumed_result.converged && fresh_result.converged);
    resumed_result.validate_contract().unwrap();
    assert!(
        resumed_result
            .x
            .iter()
            .zip(&fresh_result.x)
            .all(|(x, y)| { (*x - *y).abs() < 1.0e-9 })
    );
}

#[test]
fn recycled_checkpoint_preserves_complex_scalar_family_and_empty_k() {
    let mut empty = RecycledSubspace::<Complex64>::new(4);
    let empty_bytes = empty.to_checkpoint_bytes().unwrap();
    let empty_restored =
        RecycledSubspace::<Complex64>::from_checkpoint_bytes(&empty_bytes).unwrap();
    assert_eq!(empty_restored.num_vectors(), 0);
    assert_eq!(empty_restored.dimension(), 0);
    assert!(!empty_restored.is_populated());

    let a = CsrMatrix::from_triplets(
        3,
        3,
        vec![
            (0, 0, Complex64::new(3.0, 0.2)),
            (0, 1, Complex64::new(-0.5, 0.1)),
            (1, 0, Complex64::new(-0.5, -0.1)),
            (1, 1, Complex64::new(4.0, 0.0)),
            (1, 2, Complex64::new(-0.75, 0.2)),
            (2, 1, Complex64::new(-0.75, -0.2)),
            (2, 2, Complex64::new(4.5, 0.1)),
        ],
    );
    let b = vec![
        Complex64::new(1.0, -0.5),
        Complex64::new(-2.0, 0.25),
        Complex64::new(0.75, 1.0),
    ];
    let mut source = RecycledSubspace::new(2);
    let result = try_gmres_recycled(
        &a,
        &b,
        None,
        &mut source,
        &GmresConfig {
            max_iterations: 60,
            restart: 5,
            tolerance: 1.0e-10,
            ..GmresConfig::default()
        },
    )
    .unwrap();
    assert!(result.converged);
    let restored = RecycledSubspace::<Complex64>::from_checkpoint_bytes(
        &source.to_checkpoint_bytes().unwrap(),
    )
    .unwrap();
    assert_eq!(restored.vectors(), source.vectors());
    empty = restored;
    assert!(empty.is_populated());
}

#[test]
fn recycled_checkpoint_rejects_corruption_and_cross_family_payloads() {
    let mut source = RecycledSubspace::<f64>::new(2);
    let a = operator();
    let _ = try_gmres_recycled(&a, &[1.0, 0.0, 0.5, -1.0], None, &mut source, &config()).unwrap();
    let bytes = source.to_checkpoint_bytes().unwrap();

    let mut bad_magic = bytes.clone();
    bad_magic[0] ^= 1;
    assert!(matches!(
        RecycledSubspace::<f64>::from_checkpoint_bytes(&bad_magic),
        Err(SolverError::OperationFailed { .. })
    ));
    let mut bad_version = bytes.clone();
    bad_version[8..10].copy_from_slice(&(RECYCLED_CHECKPOINT_FORMAT_VERSION + 1).to_le_bytes());
    assert!(RecycledSubspace::<f64>::from_checkpoint_bytes(&bad_version).is_err());
    let mut bad_reserved = bytes.clone();
    bad_reserved[12] = 1;
    assert!(RecycledSubspace::<f64>::from_checkpoint_bytes(&bad_reserved).is_err());
    assert!(RecycledSubspace::<f64>::from_checkpoint_bytes(&bytes[..bytes.len() - 1]).is_err());
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(RecycledSubspace::<f64>::from_checkpoint_bytes(&trailing).is_err());
    assert!(RecycledSubspace::<Complex64>::from_checkpoint_bytes(&bytes).is_err());

    let empty = RecycledSubspace::<f64>::new(2)
        .to_checkpoint_bytes()
        .unwrap();
    let mut zero_count_nonzero_dimension = empty.clone();
    zero_count_nonzero_dimension[22..30].copy_from_slice(&1_u64.to_le_bytes());
    assert!(RecycledSubspace::<f64>::from_checkpoint_bytes(&zero_count_nonzero_dimension).is_err());

    let mut nonzero_count_zero_dimension = empty;
    nonzero_count_zero_dimension[30..38].copy_from_slice(&1_u64.to_le_bytes());
    assert!(RecycledSubspace::<f64>::from_checkpoint_bytes(&nonzero_count_zero_dimension).is_err());
}

#[test]
fn recycled_checkpoint_resume_accepts_exact_initial_guess() {
    let a = operator();
    let expected = [1.0, -0.5, 0.75, 2.0];
    let rhs = a.matvec(&expected);
    let mut source = RecycledSubspace::new(2);
    try_gmres_recycled(&a, &rhs, None, &mut source, &config()).unwrap();
    let bytes = source.to_checkpoint_bytes().unwrap();
    let mut restored = RecycledSubspace::<f64>::from_checkpoint_bytes(&bytes).unwrap();
    let resumed = restored
        .resume(&a, &rhs, Some(&expected), &config())
        .expect("checkpoint resume with exact initial guess");
    assert!(resumed.converged);
    assert_eq!(resumed.iterations, 0);
    assert!(resumed.residual <= config().tolerance);
    assert!(
        resumed
            .x
            .iter()
            .zip(expected)
            .all(|(actual, expected)| { (actual - expected).abs() <= 1.0e-12 })
    );
}
