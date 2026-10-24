//! Fixed CSR topology and deterministic element-contribution lists for tetrahedra.

// Rust guideline compliant 2026-02-21
use super::*;
use std::collections::BTreeMap;

/// Immutable sorted CSR structure and element-to-entry mapping.
#[derive(Debug, Clone)]
pub struct TetCsrPattern {
    pub(crate) row_ptrs: Vec<i32>,
    pub(crate) columns: Vec<i32>,
    pub(crate) rows: Vec<u32>,
    pub(crate) offsets: Vec<u32>,
    pub(crate) contributions: Vec<u32>,
}
impl TetCsrPattern {
    /// Borrow zero-based CSR row pointers.
    pub fn row_ptrs(&self) -> &[i32] {
        &self.row_ptrs
    }
    /// Borrow sorted zero-based CSR column indices.
    pub fn columns(&self) -> &[i32] {
        &self.columns
    }
    /// Number of stored entries, including structural zeros at constrained nodes.
    pub fn nnz(&self) -> usize {
        self.columns.len()
    }
}

impl<const N: usize> TetAcousticC64<N> {
    /// Build reusable CSR topology and deterministic contribution lists on the host.
    ///
    /// Each row includes its diagonal, including isolated nodes. Dirichlet
    /// elimination keeps structural zeros, so changing coefficients never changes
    /// the pattern. Each contribution identifies one element-local matrix entry.
    ///
    /// # Errors
    /// Rejects dimensions, nonzero counts or contribution counts outside the
    /// 32-bit kernel and signed CSR index ranges.
    pub fn csr_pattern(&self) -> BackendResult<TetCsrPattern> {
        if self.n > i32::MAX as usize
            || self
                .element_count()
                .checked_mul(N * N)
                .is_none_or(|count| count > u32::MAX as usize)
        {
            return Err(invalid("tetrahedral CSR mapping exceeds 32-bit indices"));
        }
        let mut entries: BTreeMap<(usize, usize), Vec<u32>> = BTreeMap::new();
        for i in 0..self.n {
            entries.entry((i, i)).or_default();
        }
        for (e, ids) in self.nodes.as_chunks::<N>().0.iter().enumerate() {
            for (i, &row) in ids.iter().enumerate() {
                for (j, &col) in ids.iter().enumerate() {
                    entries
                        .entry((row as usize, col as usize))
                        .or_default()
                        .push((e * N * N + i * N + j) as u32);
                }
            }
        }
        if entries.len() > i32::MAX as usize {
            return Err(invalid("CSR nnz exceeds i32"));
        }
        let mut plan = TetCsrPattern {
            row_ptrs: vec![0; self.n + 1],
            columns: Vec::with_capacity(entries.len()),
            rows: Vec::with_capacity(entries.len()),
            offsets: vec![0],
            contributions: Vec::new(),
        };
        for ((row, col), contributions) in entries {
            plan.row_ptrs[row + 1] += 1;
            plan.rows.push(row as u32);
            plan.columns.push(col as i32);
            plan.contributions.extend(contributions);
            plan.offsets.push(plan.contributions.len() as u32);
        }
        for row in 0..self.n {
            plan.row_ptrs[row + 1] += plan.row_ptrs[row];
        }
        Ok(plan)
    }

    /// Assemble a host reference CSR matrix with scaled diffusion and mass terms.
    ///
    /// Scales permit shifted-Laplacian preconditioning with the same geometry.
    /// Constrained rows/columns retain unit diagonal regardless of the scales.
    ///
    /// # Errors
    /// Rejects non-finite scales or CSR index overflow.
    pub fn assemble_csr(
        &self,
        diffusion_scale: Complex64,
        mass_scale: Complex64,
    ) -> BackendResult<(TetCsrPattern, Vec<Complex64>)> {
        if [diffusion_scale, mass_scale]
            .iter()
            .any(|z| !z.re.is_finite() || !z.im.is_finite())
        {
            return Err(invalid("CSR coefficient scales must be finite"));
        }
        let plan = self.csr_pattern()?;
        let mut values = vec![Complex64::default(); plan.nnz()];
        for (entry, value) in values.iter_mut().enumerate() {
            let row = plan.rows[entry] as usize;
            let col = plan.columns[entry] as usize;
            if self.dirichlet[row] != 0 || self.dirichlet[col] != 0 {
                if row == col {
                    *value = Complex64::new(1., 0.);
                }
                continue;
            }
            for &slot in
                &plan.contributions[plan.offsets[entry] as usize..plan.offsets[entry + 1] as usize]
            {
                let slot = slot as usize;
                let e = slot / (N * N);
                let i = slot / N % N;
                let j = slot % N;
                let g = &self.geometry[e * 13..e * 13 + 13];
                let (stiffness, mass) = if N == 4 {
                    (
                        (0..3)
                            .map(|d| g[1 + 3 * i + d] * g[1 + 3 * j + d])
                            .sum::<f64>(),
                        if i == j { 0.1 } else { 0.05 },
                    )
                } else {
                    let gi = p2_gradient_coefficients(i, g);
                    let gj = p2_gradient_coefficients(j, g);
                    let sum: [f64; 3] = std::array::from_fn(|d| gj.iter().map(|v| v[d]).sum());
                    let stiffness: f64 = (0..4)
                        .map(|a| (0..3).map(|d| gi[a][d] * (gj[a][d] + sum[d])).sum::<f64>())
                        .sum();
                    (stiffness / 20., P2_MASS[i][j])
                };
                *value += diffusion_scale * self.coefficients[2 * e] * (g[0] * stiffness)
                    + mass_scale * self.coefficients[2 * e + 1] * (g[0] * mass);
            }
        }
        Ok((plan, values))
    }
}
