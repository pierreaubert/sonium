//! Truncation preserves aggregate support and configured row bounds.

// Rust guideline compliant 2026-02-21
use super::*;

fn check<T: ComplexField>() {
    let n = 6;
    let one = T::one();
    let matrix = CsrMatrix::from_triplets(
        n,
        n,
        (0..n)
            .map(|i| (i, i, one))
            .chain((1..n).map(|j| (0, j, one)))
            .collect(),
    );
    let tentative = build_tentative_prolongation::<T>(n, &(0..n).collect::<Vec<_>>(), n);
    for cap in [1, 2, 4, 6] {
        let p = smooth_prolongation(
            &matrix,
            &vec![one; n],
            &tentative,
            T::Real::from_f64(0.9).unwrap(),
            cap,
        );
        assert_eq!(p.row_ptrs[1] - p.row_ptrs[0], cap);
        for row in 0..n {
            let entries: Vec<_> = p.row_entries(row).collect();
            assert!(entries.len() <= cap);
            assert!(
                entries
                    .iter()
                    .any(|&(col, value)| col == row
                        && value.norm() > T::Real::from_f64(0.0).unwrap())
            );
            let sum = entries
                .iter()
                .fold(T::zero(), |sum, &(_, value)| sum + value);
            assert!((sum - one).norm() < T::Real::from_f64(1e-5).unwrap());
        }
    }
}

#[test]
fn truncation_preserves_small_aggregate_all_scalars() {
    check::<f32>();
    check::<f64>();
    check::<num_complex::Complex32>();
    check::<num_complex::Complex64>();
}
