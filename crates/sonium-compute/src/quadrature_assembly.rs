//! Fixed quadrature-to-output maps for repeated numeric FEM assembly.
//!
//! Geometry and basis products are prepared once. Each numeric update reduces
//! real-weighted complex coefficients into CSR values or a global load vector.
//! Maps retain every structural entry even when a coefficient becomes zero.

// Rust guideline compliant 2026-02-21
use crate::{BackendError, BackendResult};
use num_complex::Complex64;

pub(crate) fn invalid(reason: impl Into<String>) -> BackendError {
    BackendError::InvalidArgument {
        operation: "quadrature_assembly",
        reason: reason.into(),
    }
}

/// Deterministic weighted reductions from quadrature samples to assembled entries.
#[derive(Debug, Clone)]
pub struct QuadratureAssemblyMap {
    pub(crate) samples: usize,
    pub(crate) offsets: Vec<u32>,
    pub(crate) indices: Vec<u32>,
    pub(crate) weights: Vec<f64>,
}

impl QuadratureAssemblyMap {
    /// Prepare a fixed map from `(output, sample, weight)` contributions.
    ///
    /// Contributions to each output retain their input order. Empty outputs
    /// evaluate to zero; duplicate contributions are summed, not discarded.
    ///
    /// # Errors
    /// Rejects zero sample count, index overflow, out-of-range entries and
    /// non-finite weights. Negative weights are valid for higher-order bases.
    pub fn try_new(
        samples: usize,
        outputs: usize,
        terms: &[(usize, usize, f64)],
    ) -> BackendResult<Self> {
        if samples == 0
            || samples > i32::MAX as usize
            || outputs > i32::MAX as usize
            || terms.len() > u32::MAX as usize
        {
            return Err(invalid("map dimensions exceed supported limits"));
        }
        let mut sorted = terms.to_vec();
        if sorted.iter().any(|&(out, sample, weight)| {
            out >= outputs || sample >= samples || !weight.is_finite()
        }) {
            return Err(invalid("invalid quadrature contribution"));
        }
        sorted.sort_by_key(|term| term.0);
        let mut offsets = vec![0u32; outputs + 1];
        for &(output, _, _) in &sorted {
            offsets[output + 1] += 1;
        }
        for i in 1..offsets.len() {
            offsets[i] += offsets[i - 1];
        }
        Ok(Self {
            samples,
            offsets,
            indices: sorted.iter().map(|term| term.1 as u32).collect(),
            weights: sorted.iter().map(|term| term.2).collect(),
        })
    }

    /// Number of input quadrature samples.
    pub fn sample_count(&self) -> usize {
        self.samples
    }

    /// Number of assembled outputs, including entries with no contributions.
    pub fn output_count(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Number of retained weighted contributions.
    pub fn contribution_count(&self) -> usize {
        self.indices.len()
    }

    /// Evaluate the same fixed reductions on the host for reference checks.
    ///
    /// Non-finite coefficients propagate through arithmetic; no clipping occurs.
    ///
    /// # Errors
    /// Rejects a coefficient vector with a different sample count.
    pub fn try_assemble(&self, coefficients: &[Complex64]) -> BackendResult<Vec<Complex64>> {
        if coefficients.len() != self.samples {
            return Err(invalid("coefficient count mismatch"));
        }
        Ok((0..self.output_count())
            .map(|out| {
                (self.offsets[out] as usize..self.offsets[out + 1] as usize)
                    .map(|i| coefficients[self.indices[i] as usize] * self.weights[i])
                    .sum()
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_preserves_duplicates_empty_rows_and_negative_weights() {
        let map = QuadratureAssemblyMap::try_new(2, 4, &[(2, 0, 1.0), (0, 1, -2.0), (2, 0, 3.0)])
            .unwrap();
        let a = Complex64::new(2.0, 3.0);
        let b = Complex64::new(-1.0, 4.0);
        assert_eq!(
            map.try_assemble(&[a, b]).unwrap(),
            vec![
                -2.0 * b,
                Complex64::default(),
                4.0 * a,
                Complex64::default()
            ]
        );
        assert!(map.try_assemble(&[a]).is_err());
        assert!(QuadratureAssemblyMap::try_new(1, 1, &[(1, 0, 1.0)]).is_err());
        assert!(QuadratureAssemblyMap::try_new(1, 1, &[(0, 1, 1.0)]).is_err());
        assert!(QuadratureAssemblyMap::try_new(1, 1, &[(0, 0, f64::NAN)]).is_err());
        assert!(QuadratureAssemblyMap::try_new(0, 1, &[]).is_err());
        assert_eq!(
            QuadratureAssemblyMap::try_new(1, 0, &[])
                .unwrap()
                .output_count(),
            0
        );
    }
}
