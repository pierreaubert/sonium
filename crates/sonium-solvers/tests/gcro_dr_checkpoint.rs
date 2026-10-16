use num_complex::{Complex32, Complex64};
use num_traits::{FromPrimitive, ToPrimitive};
use sonium_solvers::SolverResultContract;
use sonium_solvers::sparse::CsrMatrix;
use sonium_solvers::{
    ComplexField, GcroDrCheckpoint, GcroDrConfig, GcroDrSubspace, SolverError, try_gcro_dr,
};

fn matrix() -> CsrMatrix<f64> {
    let n = 12;
    let mut triplets = Vec::new();
    for row in 0..n {
        triplets.push((row, row, 4.0));
        if row > 0 {
            triplets.push((row, row - 1, -1.0));
        }
        if row + 1 < n {
            triplets.push((row, row + 1, -1.0));
        }
    }
    CsrMatrix::from_triplets(n, n, triplets)
}

fn family_value<T: ComplexField>(real: f64, imag: f64) -> T {
    T::from_re_im(
        T::Real::from_f64(real).expect("real fixture value should convert"),
        T::Real::from_f64(imag).expect("imaginary fixture value should convert"),
    )
}

fn family_matrix<T: ComplexField>() -> CsrMatrix<T> {
    let n = 12;
    let mut triplets = Vec::new();
    for row in 0..n {
        triplets.push((row, row, family_value::<T>(4.0, 0.0)));
        if row > 0 {
            triplets.push((row, row - 1, family_value::<T>(-1.0, 0.0)));
        }
        if row + 1 < n {
            triplets.push((row, row + 1, family_value::<T>(-1.0, 0.0)));
        }
    }
    CsrMatrix::from_triplets(n, n, triplets)
}

fn family_config<T: ComplexField>() -> GcroDrConfig<T::Real> {
    let tolerance = if std::mem::size_of::<T::Real>() == 4 {
        1.0e-5
    } else {
        1.0e-11
    };
    GcroDrConfig {
        max_iterations: 60,
        restart: 4,
        tolerance: T::Real::from_f64(tolerance).expect("tolerance should convert"),
        print_interval: 0,
    }
}

fn family_residual<T: ComplexField>(matrix: &CsrMatrix<T>, x: &[T], rhs: &[T]) -> f64 {
    let action = matrix.matvec(x);
    let numerator = action
        .iter()
        .zip(rhs)
        .map(|(actual, expected)| (*actual - *expected).norm_sqr().to_f64().unwrap())
        .sum::<f64>()
        .sqrt();
    let denominator = rhs
        .iter()
        .map(|value| value.norm_sqr().to_f64().unwrap())
        .sum::<f64>()
        .sqrt();
    numerator / denominator.max(1.0)
}

fn assert_scalar_family<T: ComplexField>(tolerance: f64) {
    let operator = family_matrix::<T>();
    let expected_first: Vec<T> = (0..12)
        .map(|index| {
            family_value(
                0.2 + 0.03 * index as f64 - 0.001 * (index * index) as f64,
                0.1 - 0.004 * index as f64,
            )
        })
        .collect();
    let expected_second: Vec<T> = (0..12)
        .map(|index| {
            family_value(
                -0.4 + 0.02 * index as f64 + 0.0007 * (index * index) as f64,
                -0.08 + 0.006 * index as f64,
            )
        })
        .collect();
    let first_rhs = operator.matvec(&expected_first);
    let second_rhs = operator.matvec(&expected_second);
    let config = family_config::<T>();

    let mut first_subspace = GcroDrSubspace::new(2);
    let first = try_gcro_dr(&operator, &first_rhs, None, &mut first_subspace, &config)
        .expect("scalar-family GCRO-DR source solve should succeed");
    assert!(
        first.converged,
        "scalar-family GCRO-DR source solve must converge"
    );
    let checkpoint = GcroDrCheckpoint::from_subspace(&first_subspace)
        .expect("scalar-family GCRO-DR checkpoint capture should succeed");
    assert!(checkpoint.num_vectors() > 0);
    let bytes = checkpoint
        .to_checkpoint_bytes()
        .expect("scalar-family GCRO-DR checkpoint should serialize");
    let restored = GcroDrCheckpoint::<T>::from_checkpoint_bytes(&bytes)
        .expect("scalar-family GCRO-DR checkpoint should restore");
    assert_eq!(restored.vectors(), checkpoint.vectors());

    let (updated_subspace, resumed) = restored
        .resume(&operator, &second_rhs, None, &config)
        .expect("scalar-family GCRO-DR checkpoint resume should succeed");
    assert!(
        resumed.converged,
        "scalar-family GCRO-DR resume must converge"
    );
    resumed
        .validate_contract()
        .expect("scalar-family GCRO-DR resume must satisfy result contract");
    assert!(updated_subspace.num_vectors() <= updated_subspace.target_size());

    let mut fresh_subspace = GcroDrSubspace::new(2);
    let fresh = try_gcro_dr(&operator, &second_rhs, None, &mut fresh_subspace, &config)
        .expect("scalar-family GCRO-DR fresh solve should succeed");
    assert!(fresh.converged);
    let solution_delta = resumed
        .x
        .iter()
        .zip(&fresh.x)
        .map(|(actual, expected)| (*actual - *expected).norm().to_f64().unwrap())
        .fold(0.0, f64::max);
    let independent_residual = family_residual(&operator, &resumed.x, &second_rhs);
    assert!(
        solution_delta <= tolerance,
        "solution delta {solution_delta:e}"
    );
    assert!(
        independent_residual <= tolerance,
        "independent residual {independent_residual:e}"
    );
}

#[test]
fn gcro_dr_checkpoint_replays_scalar_families() {
    assert_scalar_family::<f32>(1.0e-4);
    assert_scalar_family::<Complex32>(1.0e-4);
    assert_scalar_family::<Complex64>(1.0e-9);
}

fn config() -> GcroDrConfig<f64> {
    GcroDrConfig {
        max_iterations: 30,
        restart: 3,
        tolerance: 1.0e-11,
        print_interval: 0,
    }
}

fn residual(matrix: &CsrMatrix<f64>, x: &[f64], rhs: &[f64]) -> f64 {
    let action = matrix.matvec(x);
    let numerator = action
        .iter()
        .zip(rhs)
        .map(|(actual, expected)| (actual - expected).powi(2))
        .sum::<f64>()
        .sqrt();
    let denominator = rhs.iter().map(|value| value.powi(2)).sum::<f64>().sqrt();
    numerator / denominator.max(1.0)
}

#[test]
fn gcro_dr_checkpoint_round_trip_resumes_and_updates_subspace() {
    let operator = matrix();
    let expected_first: Vec<f64> = (0..12)
        .map(|index| 0.2 + 0.03 * index as f64 - 0.001 * (index * index) as f64)
        .collect();
    let expected_second: Vec<f64> = (0..12)
        .map(|index| -0.4 + 0.02 * index as f64 + 0.0007 * (index * index) as f64)
        .collect();
    let first_rhs = operator.matvec(&expected_first);
    let second_rhs = operator.matvec(&expected_second);
    let mut first_subspace = GcroDrSubspace::new(2);
    let first = try_gcro_dr(&operator, &first_rhs, None, &mut first_subspace, &config())
        .expect("first GCRO-DR solve");
    assert!(first.converged);

    let checkpoint = GcroDrCheckpoint::from_subspace(&first_subspace).unwrap();
    assert_eq!(checkpoint.dimension(), 12);
    assert!(checkpoint.num_vectors() > 0);
    assert!(checkpoint.num_vectors() <= checkpoint.target_size());
    assert!(checkpoint.storage_bytes() > 0);
    let bytes = checkpoint.to_checkpoint_bytes().unwrap();
    let restored = GcroDrCheckpoint::<f64>::from_checkpoint_bytes(&bytes).unwrap();
    assert_eq!(restored.dimension(), checkpoint.dimension());
    assert_eq!(restored.num_vectors(), checkpoint.num_vectors());
    assert_eq!(restored.vectors(), checkpoint.vectors());

    let (resumed_subspace, resumed) = restored
        .resume(&operator, &second_rhs, None, &config())
        .expect("GCRO-DR checkpoint resume");
    assert!(resumed.converged);
    resumed.validate_contract().unwrap();
    assert!(residual(&operator, &resumed.x, &second_rhs) < 1.0e-9);
    assert!(resumed_subspace.num_vectors() <= resumed_subspace.target_size());

    let mut fresh_subspace = GcroDrSubspace::new(2);
    let fresh = try_gcro_dr(&operator, &second_rhs, None, &mut fresh_subspace, &config())
        .expect("fresh GCRO-DR solve");
    assert!(fresh.converged);
    let max_error = resumed
        .x
        .iter()
        .zip(&expected_second)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    assert!(max_error < 1.0e-8);
    assert!(resumed.iterations <= fresh.iterations);

    let mut destination = GcroDrSubspace::new(1);
    restored.restore_into(&mut destination).unwrap();
    assert_eq!(destination.target_size(), restored.target_size());
    assert_eq!(destination.num_vectors(), restored.num_vectors());
}

#[test]
fn gcro_dr_checkpoint_rejects_invalid_state_and_wire_data() {
    assert!(GcroDrCheckpoint::<f64>::from_vectors(0, &[vec![1.0]]).is_err());
    assert!(GcroDrCheckpoint::<f64>::from_vectors(1, &[vec![f64::NAN]]).is_err());
    let valid = GcroDrCheckpoint::<f64>::from_vectors(1, &[vec![1.0, 0.0]]).unwrap();
    let bytes = valid.to_checkpoint_bytes().unwrap();

    let mut bad_magic = bytes.clone();
    bad_magic[0] ^= 0xff;
    assert!(matches!(
        GcroDrCheckpoint::<f64>::from_checkpoint_bytes(&bad_magic),
        Err(SolverError::OperationFailed {
            operation: "GCRO-DR checkpoint",
            ..
        })
    ));
    assert!(GcroDrCheckpoint::<f64>::from_checkpoint_bytes(&bytes[..bytes.len() - 1]).is_err());
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(GcroDrCheckpoint::<f64>::from_checkpoint_bytes(&trailing).is_err());
    let mut bad_version = bytes.clone();
    bad_version[8] = 2;
    assert!(GcroDrCheckpoint::<f64>::from_checkpoint_bytes(&bad_version).is_err());

    let empty = GcroDrCheckpoint::<f64>::from_vectors(1, &[]).unwrap();
    let mut bad_empty_dimension = empty.to_checkpoint_bytes().unwrap();
    bad_empty_dimension[14..22].copy_from_slice(&1_u64.to_le_bytes());
    assert!(GcroDrCheckpoint::<f64>::from_checkpoint_bytes(&bad_empty_dimension).is_err());
}

#[test]
fn gcro_dr_checkpoint_preserves_complex_vectors() {
    let vectors = vec![
        vec![
            Complex64::new(1.0, 0.25),
            Complex64::new(-0.5, 0.1),
            Complex64::new(0.25, -0.75),
        ],
        vec![
            Complex64::new(0.1, -0.4),
            Complex64::new(0.75, 0.2),
            Complex64::new(-0.2, 0.6),
        ],
    ];
    let checkpoint = GcroDrCheckpoint::from_vectors(3, &vectors).unwrap();
    let restored = GcroDrCheckpoint::<Complex64>::from_checkpoint_bytes(
        &checkpoint.to_checkpoint_bytes().unwrap(),
    )
    .unwrap();
    for (actual, expected) in restored.vectors().iter().zip(&vectors) {
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((*actual - *expected).norm() < 1.0e-12);
        }
    }
}
