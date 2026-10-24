use super::{AmgConfig, AmgInterpolation, PointType};
use num_traits::{FromPrimitive, Zero};
use rayon::prelude::*;
use sonium_solvers_core::traits::ComplexField;
use sonium_solvers_sparse::CsrMatrix;

/// Build interpolation operator P
pub(super) fn build_interpolation<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    strong: &[Vec<usize>],
    point_types: &[PointType],
    coarse_to_fine: &[usize],
    config: &AmgConfig,
) -> CsrMatrix<T> {
    let n_fine = matrix.num_rows;
    let n_coarse = coarse_to_fine.len();

    // Build fine-to-coarse mapping
    let mut fine_to_coarse = vec![usize::MAX; n_fine];
    for (coarse_idx, &fine_idx) in coarse_to_fine.iter().enumerate() {
        fine_to_coarse[fine_idx] = coarse_idx;
    }

    // Each row of P is independent: the row-i triplets only depend on
    // matrix.row_entries(i), strong[i], point_types[*], fine_to_coarse[*]
    // — all read-only after the prelude. Build per-row triplet lists in
    // parallel and concat afterwards.
    let per_row: Vec<Vec<(usize, usize, T)>> = (0..n_fine)
        .into_par_iter()
        .map(|i| {
            interpolation_row(
                matrix,
                strong,
                point_types,
                coarse_to_fine,
                &fine_to_coarse,
                config,
                i,
            )
        })
        .collect();

    let total: usize = per_row.iter().map(|v| v.len()).sum();
    let mut triplets: Vec<(usize, usize, T)> = Vec::with_capacity(total);
    for row in per_row {
        triplets.extend(row);
    }

    CsrMatrix::from_triplets(n_fine, n_coarse, triplets)
}

/// Compute one row's worth of P triplets. Pure function of read-only inputs.
fn interpolation_row<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    strong: &[Vec<usize>],
    point_types: &[PointType],
    coarse_to_fine: &[usize],
    fine_to_coarse: &[usize],
    config: &AmgConfig,
    i: usize,
) -> Vec<(usize, usize, T)> {
    let mut row: Vec<(usize, usize, T)> = Vec::new();

    match point_types[i] {
        PointType::Coarse => {
            // C-point: identity mapping P_ij = 1 if j = coarse_index(i)
            let coarse_idx = fine_to_coarse[i];
            row.push((i, coarse_idx, T::one()));
        }
        PointType::Fine => {
            // F-point: interpolate from strong C-neighbors
            let a_ii = matrix.get(i, i);

            // Extended+i fallback: when the F-point has no C-point in its
            // strong set, use the full immediate (matrix row) neighborhood
            // as the effective strong set instead of degrading to injection.
            let fallback_neighbors: Vec<usize>;
            let strong_i: &[usize] = if config.interpolation == AmgInterpolation::ExtendedI {
                let has_strong_c = strong[i]
                    .iter()
                    .any(|&j| point_types[j] == PointType::Coarse);
                if has_strong_c {
                    &strong[i]
                } else {
                    fallback_neighbors = matrix
                        .row_entries(i)
                        .filter_map(|(j, _)| if j != i { Some(j) } else { None })
                        .collect();
                    &fallback_neighbors
                }
            } else {
                &strong[i]
            };

            // Collect strong C-neighbors
            let c_neighbors: Vec<usize> = strong_i
                .iter()
                .copied()
                .filter(|&j| point_types[j] == PointType::Coarse)
                .collect();

            if c_neighbors.is_empty() {
                // Fall back to direct diagonal interpolation if no strong C-neighbors exist.
                let mut best_c = usize::MAX;
                let mut max_val = T::Real::zero();

                for (j, val) in matrix.row_entries(i) {
                    if point_types[j] == PointType::Coarse {
                        let norm = val.norm();
                        if norm > max_val {
                            max_val = norm;
                            best_c = j;
                        }
                    }
                }

                if best_c != usize::MAX {
                    row.push((i, fine_to_coarse[best_c], T::one()));
                } else if !coarse_to_fine.is_empty() {
                    row.push((i, 0, T::one()));
                }
                return row;
            }

            // Standard interpolation weights
            match config.interpolation {
                AmgInterpolation::Direct | AmgInterpolation::Standard => {
                    let mut weights: Vec<(usize, T)> = Vec::new();

                    // Weak-connection contribution (modified classical
                    // interpolation): fold weak connections into the
                    // diagonal, `w_j = -a_ij / (a_ii + weak_sum)`.
                    // The historical rescale `1 + weak_sum / (a_ii * sum)`
                    // is algebraically similar on viable rows but turns
                    // negative — flipping the whole row — whenever weak
                    // connections dominate (`|weak_sum| > a_ii * sum`;
                    // measured: 665 sign-flipped P rows on 3D Poisson
                    // level 1). Diagonal lumping keeps M-matrix rows
                    // positive like hypre. `Direct` skips the lumping.
                    let mut weak_sum = T::zero();
                    if config.interpolation == AmgInterpolation::Standard {
                        for (j, val) in matrix.row_entries(i) {
                            if j != i && !c_neighbors.contains(&j) {
                                weak_sum += val;
                            }
                        }
                    }
                    let tol = T::Real::from_f64(1e-15).unwrap();
                    let denom = a_ii + weak_sum;
                    if denom.norm() > tol {
                        for &j in &c_neighbors {
                            let a_ij = matrix.get(i, j);
                            let w = T::zero() - a_ij * denom.inv();
                            weights.push((fine_to_coarse[j], w));
                        }
                    }

                    // Truncate small weights if configured
                    if config.trunc_factor > 0.0 {
                        let max_w = weights.iter().map(|(_, w)| w.norm()).fold(
                            T::Real::from_f64(0.0).unwrap(),
                            |a, b| {
                                if a > b { a } else { b }
                            },
                        );
                        let threshold = T::Real::from_f64(config.trunc_factor).unwrap() * max_w;
                        weights.retain(|(_, w)| w.norm() >= threshold);

                        if weights.len() > config.max_interp_elements {
                            weights.sort_by(|a, b| {
                                b.1.norm()
                                    .partial_cmp(&a.1.norm())
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            });
                            weights.truncate(config.max_interp_elements);
                        }
                    }

                    for (coarse_idx, w) in weights {
                        row.push((i, coarse_idx, w));
                    }
                }
                AmgInterpolation::Extended | AmgInterpolation::ExtendedI => {
                    let mut weights: Vec<(usize, T)> = Vec::new();
                    let tol = T::Real::from_f64(1e-15).unwrap();

                    // F-neighbors whose values are distributed through
                    // distance-2 paths (computed first: they are EXCLUDED
                    // from the lumped diagonal below).
                    let f_neighbors: Vec<usize> = strong_i
                        .iter()
                        .copied()
                        .filter(|&j| point_types[j] == PointType::Fine)
                        .collect();

                    // Weak-connection contribution lumped into the diagonal.
                    // Unlike the Standard arm (which collapses strong F
                    // connections to the diagonal and must lump them), the
                    // extended arm distributes strong F connections via
                    // distance-2 paths, so only genuinely weak connections
                    // belong here. Lumping strong F as well double-counts
                    // them and pushes P row sums above 1 (measured: 18 PCG
                    // iters on 32^3 Poisson with the over-lumped diagonal).
                    let mut weak_sum = T::zero();
                    for (j, val) in matrix.row_entries(i) {
                        if j != i && !c_neighbors.contains(&j) && !f_neighbors.contains(&j) {
                            weak_sum += val;
                        }
                    }
                    let denom = a_ii + weak_sum;
                    if denom.norm() <= tol {
                        return row;
                    }

                    // Direct C-neighbors
                    for &j in &c_neighbors {
                        let a_ij = matrix.get(i, j);
                        let w = T::zero() - a_ij * denom.inv();
                        weights.push((fine_to_coarse[j], w));
                    }

                    // F-neighbors contribute through their C-neighbors
                    // (Ruge-Stuben Eq. 4.3): F-point k's value is replaced
                    // by k's own interpolation row, so the distance-2
                    // weight is `-a_ik * a_kj / (denom * D_k)` with
                    // `D_k = sum_{m in C(k)} a_km` (a *negative* off-diagonal
                    // sum, not the positive diagonal a_kk). The previous
                    // code used `(a_ii * a_kk).inv()` with a leading minus,
                    // flipping every distance-2 weight negative.
                    // (`f_neighbors` is computed above, before the diagonal
                    // lumping, which must exclude it.)
                    for &k in &f_neighbors {
                        let a_ik = matrix.get(i, k);

                        // C-sources for k: its strong C-neighbors. Extended+i
                        // falls back to k's immediate C-neighbors when the
                        // strong set has none.
                        let c_sources: Vec<(usize, T)> = {
                            let js: Vec<usize> = {
                                let s: Vec<usize> = strong[k]
                                    .iter()
                                    .copied()
                                    .filter(|&j| point_types[j] == PointType::Coarse)
                                    .collect();
                                if s.is_empty()
                                    && config.interpolation == AmgInterpolation::ExtendedI
                                {
                                    matrix
                                        .row_entries(k)
                                        .filter_map(|(j, _)| {
                                            if point_types[j] == PointType::Coarse {
                                                Some(j)
                                            } else {
                                                None
                                            }
                                        })
                                        .collect()
                                } else {
                                    s
                                }
                            };
                            js.into_iter().map(|j| (j, matrix.get(k, j))).collect()
                        };
                        let d_k: T = c_sources
                            .iter()
                            .map(|&(_, v)| v)
                            .fold(T::zero(), |acc: T, v: T| acc + v);
                        if d_k.norm() < tol {
                            continue;
                        }

                        for (j, a_kj) in c_sources {
                            let w = T::zero() - a_ik * a_kj * (denom * d_k).inv();

                            let coarse_j = fine_to_coarse[j];
                            if let Some((_, existing)) =
                                weights.iter_mut().find(|(idx, _)| *idx == coarse_j)
                            {
                                *existing += w;
                            } else {
                                weights.push((coarse_j, w));
                            }
                        }
                    }

                    // Truncate when configured, mirroring BoomerAMG's
                    // extended+i recipe: drop weights below
                    // trunc_factor * row-max (this removes the small
                    // distance-2 stencil on fine levels while keeping the
                    // relatively larger ones on coarse levels), then enforce
                    // the count cap, then rescale the survivors to the
                    // pre-truncation row sum so constant preservation
                    // (P*1 = 1) survives the dropping.
                    if config.trunc_factor > 0.0 {
                        let sum_before: T = weights
                            .iter()
                            .map(|&(_, w)| w)
                            .fold(T::zero(), |a, b| a + b);
                        let max_w = weights.iter().map(|(_, w)| w.norm()).fold(
                            T::Real::from_f64(0.0).unwrap(),
                            |a, b| {
                                if a > b { a } else { b }
                            },
                        );
                        let threshold = T::Real::from_f64(config.trunc_factor).unwrap() * max_w;
                        weights.retain(|(_, w)| w.norm() >= threshold);
                        if weights.len() > config.max_interp_elements {
                            weights.sort_by(|a, b| {
                                b.1.norm()
                                    .partial_cmp(&a.1.norm())
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            });
                            weights.truncate(config.max_interp_elements);
                        }
                        let sum_after: T = weights
                            .iter()
                            .map(|&(_, w)| w)
                            .fold(T::zero(), |a, b| a + b);
                        if sum_after.norm() > tol && sum_before.norm() > tol {
                            let scale = sum_before * sum_after.inv();
                            for (_, w) in weights.iter_mut() {
                                *w *= scale;
                            }
                        }
                    }

                    for (coarse_idx, w) in weights {
                        row.push((i, coarse_idx, w));
                    }
                }
            }
        }
        PointType::Undecided => {}
    }

    row
}

/// Build tentative prolongation operator P_tent for Smoothed Aggregation
///
/// P_tent is an (n_fine x n_aggregates) matrix where P_tent[i, agg[i]] = 1.0
/// This is the "piecewise constant" interpolation that maps each node to its aggregate.
pub(super) fn build_tentative_prolongation<T: ComplexField>(
    n_fine: usize,
    aggregate_ids: &[usize],
    num_aggregates: usize,
) -> CsrMatrix<T> {
    let mut triplets: Vec<(usize, usize, T)> = Vec::with_capacity(n_fine);
    for (i, &agg) in aggregate_ids.iter().enumerate() {
        triplets.push((i, agg, T::one()));
    }
    CsrMatrix::from_triplets(n_fine, num_aggregates, triplets)
}

/// Smooth the tentative prolongation operator
///
/// Applies one Jacobi smoothing step to P_tent:
///   P = (I - omega * D^{-1} * A) * P_tent
///     = P_tent - omega * D^{-1} * A * P_tent
///
/// where omega = 4/(3*rho(D^{-1}A)) is chosen to damp high-frequency modes.
/// Truncation respects `max_entries` and retains a nonzero own-aggregate entry.
pub(super) fn smooth_prolongation<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    diag_inv: &[T],
    p_tent: &CsrMatrix<T>,
    omega: T::Real,
    max_entries: usize,
) -> CsrMatrix<T> {
    // Step 1: Compute A * P_tent using Gustavson's SpMM (with the AMG
    // drop tolerance, preserving the established SA numerics)
    let drop_tol = T::Real::from_f64(super::util::AMG_MATMUL_DROP_TOL).unwrap();
    let ap = matrix.matmul_with_drop(p_tent, drop_tol);

    // Step 2: Build P = P_tent - omega * D^{-1} * (A * P_tent)
    // Build row-by-row using CsrBuilder to avoid the sort in from_triplets
    let n_rows = p_tent.num_rows;
    let n_cols = p_tent.num_cols;
    let neg_omega = T::from_real(omega).neg();

    let mut builder = sonium_solvers_sparse::CsrBuilder::with_capacity(n_rows, n_cols, ap.nnz());

    // Dense workspace for accumulating row entries
    let mut row_vals = vec![T::zero(); n_cols];
    let mut active = Vec::with_capacity(32);

    for (i, &di) in diag_inv[..n_rows].iter().enumerate() {
        // Add P_tent entries
        for (j, val) in p_tent.row_entries(i) {
            row_vals[j] = val;
            active.push(j);
        }

        // Add -omega * D^{-1} * (A * P_tent) entries
        let scale = neg_omega * di;
        for (j, val) in ap.row_entries(i) {
            if row_vals[j].norm() == T::Real::from_f64(0.0).unwrap() {
                active.push(j);
            }
            row_vals[j] += scale * val;
        }

        // Sort active columns, dedup
        active.sort_unstable();
        active.dedup();

        // Collect (col, val) pairs with nonzero values
        let mut row_entries: Vec<(usize, T)> = active
            .iter()
            .filter_map(|&j| {
                let v = row_vals[j];
                if v.norm() > T::Real::from_f64(0.0).unwrap() {
                    Some((j, v))
                } else {
                    None
                }
            })
            .collect();

        // Retain the row's own aggregate when truncating. Pure magnitude
        // truncation can delete every entry of a small aggregate's column,
        // producing unsupported coarse DOFs and a singular Galerkin matrix.
        // Current callers supply one-hot tentative prolongation.
        let owner = p_tent.row_entries(i).next().map(|(column, _)| column);
        if row_entries.len() > max_entries {
            row_entries.sort_by(|a, b| {
                b.1.norm()
                    .partial_cmp(&a.1.norm())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            if let Some(position) = row_entries
                .iter()
                .position(|&(column, _)| Some(column) == owner)
                && position >= max_entries
            {
                row_entries.swap(max_entries - 1, position);
            }
            row_entries.truncate(max_entries);
            row_entries.sort_by_key(|&(j, _)| j); // restore column order
        }

        // Normalize so row sums to 1.0 (preserves constants -- key for SA convergence)
        let row_sum: T = row_entries
            .iter()
            .map(|&(_, v)| v)
            .fold(T::zero(), |a, b| a + b);
        if row_sum.norm() > T::Real::from_f64(1e-15).unwrap() {
            let inv_sum = row_sum.inv();
            for entry in &mut row_entries {
                entry.1 *= inv_sum;
            }
        }

        builder.add_row_entries(row_entries.into_iter());

        // Reset workspace
        for &j in &active {
            row_vals[j] = T::zero();
        }
        active.clear();
    }

    builder.finish()
}

// ---------------------------------------------------------------------------
// Energy-Minimizing Prolongation
// ---------------------------------------------------------------------------

/// Remove the component of `v` along the all-ones direction [1,1,...,1]/sqrt(m).
///
/// After projection, sum(v) == 0. This is used to enforce the constant-preservation
/// constraint on the CG search direction and gradient.
fn project_constraint<T: ComplexField>(v: &mut [T], m: usize) {
    if m == 0 {
        return;
    }
    // mean = sum(v) / m
    let mut sum = T::zero();
    for x in v.iter() {
        sum += *x;
    }
    let inv_m = T::from_real(T::Real::from_f64(1.0 / m as f64).unwrap());
    let mean = sum * inv_m;
    for x in v.iter_mut() {
        *x -= mean;
    }
}

/// Dense m x m matrix-vector product: y = A * x
fn matvec_dense<T: ComplexField>(m: usize, a: &[T], x: &[T], y: &mut [T]) {
    for i in 0..m {
        let mut sum = T::zero();
        for j in 0..m {
            sum += a[i * m + j] * x[j];
        }
        y[i] = sum;
    }
}

/// Conjugate dot product: sum of conj(x_i) * y_i
fn dot_local<T: ComplexField>(x: &[T], y: &[T]) -> T {
    let mut sum = T::zero();
    for (xi, yi) in x.iter().zip(y.iter()) {
        sum += xi.conj() * *yi;
    }
    sum
}

/// Compute the sparsity pattern for energy-minimizing prolongation columns.
///
/// For each column j of P_tent, grows its nonzero row-pattern by `distance`
/// applications of A's adjacency graph.
///
/// Returns `column_patterns[j]` = sorted list of row indices for column j.
pub(super) fn compute_energy_min_pattern<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    p_tent: &CsrMatrix<T>,
    distance: usize,
) -> Vec<Vec<usize>> {
    let n_cols = p_tent.num_cols;
    let n_rows = p_tent.num_rows;
    let mut column_patterns: Vec<Vec<usize>> = Vec::with_capacity(n_cols);

    // Build column-oriented view of P_tent (which rows have nonzero in each column)
    let mut col_to_rows: Vec<Vec<usize>> = vec![Vec::new(); n_cols];
    for i in 0..n_rows {
        for (j, _) in p_tent.row_entries(i) {
            col_to_rows[j].push(i);
        }
    }

    // Reusable bitmap -- allocated once, reset via touched-list between columns
    let mut in_set: Vec<bool> = vec![false; n_rows];
    // `frontier` holds the nodes whose neighbors we still need to explore
    let mut frontier: Vec<usize> = Vec::new();
    // `next_frontier` collects newly discovered nodes for the next hop
    let mut next_frontier: Vec<usize> = Vec::new();
    // `touched` tracks every index set to true so we can reset cheaply
    let mut touched: Vec<usize> = Vec::new();

    for rows_j in &col_to_rows {
        // Seed with the nonzero pattern of column j of P_tent
        frontier.clear();
        for &row in rows_j {
            if !in_set[row] {
                in_set[row] = true;
                touched.push(row);
                frontier.push(row);
            }
        }

        // Grow by `distance` hops through A's adjacency
        for _ in 0..distance {
            next_frontier.clear();
            for &i in &frontier {
                for (k, _) in matrix.row_entries(i) {
                    if !in_set[k] {
                        in_set[k] = true;
                        touched.push(k);
                        next_frontier.push(k);
                    }
                }
            }
            // Only newly discovered nodes can discover further new nodes at the
            // next hop, so swap frontier <-> next_frontier.  After the last hop
            // the full pattern is captured in `touched`.
            std::mem::swap(&mut frontier, &mut next_frontier);
        }

        // Collect sorted indices from the touched list (cheaper than scanning n_rows)
        touched.sort_unstable();
        column_patterns.push(touched.clone());

        // Reset bitmap -- only the entries we touched
        for &idx in &touched {
            in_set[idx] = false;
        }
        touched.clear();
    }

    column_patterns
}

/// Reusable workspace buffers for per-column CG solve.
///
/// Grouping these avoids the "too many arguments" pattern and makes it easy
/// to create one set of buffers per thread.
struct EnergyMinWorkspace<T: ComplexField> {
    global_to_local: Vec<usize>,
    a_local: Vec<T>,
    p: Vec<T>,
    r: Vec<T>,
    d: Vec<T>,
    ap: Vec<T>,
    ad: Vec<T>,
    out: Vec<(usize, T)>,
}

impl<T: ComplexField> EnergyMinWorkspace<T> {
    fn new(n_rows: usize, max_pattern_size: usize) -> Self {
        Self {
            global_to_local: vec![usize::MAX; n_rows],
            a_local: Vec::with_capacity(max_pattern_size * max_pattern_size),
            p: Vec::with_capacity(max_pattern_size),
            r: Vec::with_capacity(max_pattern_size),
            d: Vec::with_capacity(max_pattern_size),
            ap: Vec::with_capacity(max_pattern_size),
            ad: Vec::with_capacity(max_pattern_size),
            out: Vec::with_capacity(max_pattern_size),
        }
    }
}

/// Solve one column of the energy-minimizing prolongation via projected CG.
///
/// This is factored out so it can be called from both the sequential and
/// parallel code-paths with reusable workspace buffers.
///
/// Returns the column entries as `(global_row, value)` pairs in `ws.out`.
fn solve_energy_min_column<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    pattern: &[usize],
    p_tent_col: &[(usize, T)],
    energy_min_iterations: usize,
    ws: &mut EnergyMinWorkspace<T>,
) {
    let m = pattern.len();
    debug_assert!(m > 0);

    // --- set up global_to_local for this column ---
    for (local_idx, &global_idx) in pattern.iter().enumerate() {
        ws.global_to_local[global_idx] = local_idx;
    }

    // --- resize & zero work vectors to exactly m (or m*m for a_local) ---
    let mm = m * m;
    ws.a_local.resize(mm, T::zero());
    for v in ws.a_local[..mm].iter_mut() {
        *v = T::zero();
    }

    ws.p.resize(m, T::zero());
    for v in ws.p[..m].iter_mut() {
        *v = T::zero();
    }
    ws.r.resize(m, T::zero());
    ws.d.resize(m, T::zero());
    ws.ap.resize(m, T::zero());
    ws.ad.resize(m, T::zero());

    // --- extract local dense submatrix A_S (m x m) ---
    for (li, &gi) in pattern.iter().enumerate() {
        for (gk, val) in matrix.row_entries(gi) {
            let lk = ws.global_to_local[gk];
            if lk != usize::MAX {
                ws.a_local[li * m + lk] = val;
            }
        }
    }

    // --- initialize p from P_tent column restricted to pattern ---
    let mut constraint_val = T::zero();
    for &(row, val) in p_tent_col {
        let li = ws.global_to_local[row];
        if li != usize::MAX {
            ws.p[li] = val;
        }
        constraint_val += val;
    }

    // --- projected CG to minimize p^T A_S p subject to sum(p) = constraint_val ---
    //
    // This finds the smoothest interpolation weights within the given sparsity
    // pattern. The gradient of f(p) = (1/2) p^T A p is A*p. The negative
    // gradient is -A*p. We project it onto the constraint hyperplane
    // {v : sum(v) = 0} and run CG in this projected subspace.

    // Compute initial ap = A*p
    matvec_dense(m, &ws.a_local[..mm], &ws.p[..m], &mut ws.ap[..m]);

    // r = -A*p projected onto constraint plane
    for i in 0..m {
        ws.r[i] = T::zero() - ws.ap[i];
    }
    project_constraint(&mut ws.r[..m], m);

    // d = r (initial search direction)
    ws.d[..m].copy_from_slice(&ws.r[..m]);
    let mut r_dot_r = dot_local(&ws.r[..m], &ws.r[..m]);

    let tol = T::Real::from_f64(1e-14).unwrap();

    for _ in 0..energy_min_iterations {
        if r_dot_r.norm() < tol {
            break;
        }

        // ad = A * d
        matvec_dense(m, &ws.a_local[..mm], &ws.d[..m], &mut ws.ad[..m]);

        let d_dot_ad = dot_local(&ws.d[..m], &ws.ad[..m]);
        if d_dot_ad.norm() < tol {
            break;
        }

        let alpha = r_dot_r * d_dot_ad.inv();

        // p = p + alpha * d
        for i in 0..m {
            ws.p[i] += alpha * ws.d[i];
        }

        // r = r - alpha * A*d, then project
        for i in 0..m {
            ws.r[i] -= alpha * ws.ad[i];
        }
        project_constraint(&mut ws.r[..m], m);

        let r_dot_r_new = dot_local(&ws.r[..m], &ws.r[..m]);
        if r_dot_r_new.norm() < tol {
            break;
        }

        let beta = r_dot_r_new * r_dot_r.inv();

        // d = r + beta * d
        for i in 0..m {
            ws.d[i] = ws.r[i] + beta * ws.d[i];
        }

        r_dot_r = r_dot_r_new;
    }

    // --- final rescale: ensure sum(p) = constraint_val exactly ---
    let mut p_sum = T::zero();
    for i in 0..m {
        p_sum += ws.p[i];
    }
    if p_sum.norm() > T::Real::from_f64(1e-15).unwrap() {
        let scale = constraint_val * p_sum.inv();
        for i in 0..m {
            ws.p[i] *= scale;
        }
    }

    // --- collect nonzero entries ---
    let drop_tol = T::Real::from_f64(1e-14).unwrap();
    ws.out.clear();
    for (li, &gi) in pattern.iter().enumerate() {
        if ws.p[li].norm() > drop_tol {
            ws.out.push((gi, ws.p[li]));
        }
    }

    // --- reset global_to_local (only touched entries) ---
    for &gi in pattern {
        ws.global_to_local[gi] = usize::MAX;
    }
}

/// Build energy-minimizing prolongation operator.
///
/// For each column j of P, minimizes the A-energy (p^T A p) of the column
/// subject to the constraint that the column sums are preserved from P_tent,
/// using a projected CG iteration restricted to a local sparsity pattern.
pub(super) fn energy_minimizing_prolongation<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    p_tent: &CsrMatrix<T>,
    energy_min_iterations: usize,
    pattern_distance: usize,
) -> CsrMatrix<T> {
    let n_rows = p_tent.num_rows;
    let n_cols = p_tent.num_cols;

    // Compute sparsity patterns for each column
    let column_patterns = compute_energy_min_pattern(matrix, p_tent, pattern_distance);

    // Build column-oriented view of P_tent
    let mut p_tent_cols: Vec<Vec<(usize, T)>> = vec![Vec::new(); n_cols];
    for i in 0..n_rows {
        for (j, val) in p_tent.row_entries(i) {
            p_tent_cols[j].push((i, val));
        }
    }

    // Find max pattern size to pre-allocate buffers
    let max_m = column_patterns.iter().map(|p| p.len()).max().unwrap_or(0);

    // --- Parallel path (rayon) ---
    let col_results: Vec<Vec<(usize, T)>> = {
        (0..n_cols)
            .into_par_iter()
            .map_init(
                // Thread-local workspace initializer -- each rayon worker gets its
                // own set of buffers, avoiding any contention.
                || EnergyMinWorkspace::new(n_rows, max_m),
                |ws, j| {
                    let pattern = &column_patterns[j];
                    if pattern.is_empty() {
                        return Vec::new();
                    }
                    solve_energy_min_column(
                        matrix,
                        pattern,
                        &p_tent_cols[j],
                        energy_min_iterations,
                        ws,
                    );
                    ws.out.clone()
                },
            )
            .collect()
    };

    // --- Build CSR directly from column results ---
    //
    // Each col_results[j] contains (global_row, value) pairs.  We need the
    // CSR in row-major order.  Strategy: count per-row nnz, build row_ptrs,
    // then scatter entries using per-row write cursors.  Since we iterate
    // columns in order (j = 0..n_cols), each row's entries end up sorted
    // by column index automatically -- no per-row sort needed.

    // Count total nnz and per-row counts
    let mut row_counts = vec![0usize; n_rows];
    for col_entries in &col_results {
        for &(gi, _) in col_entries {
            row_counts[gi] += 1;
        }
    }

    // Build row_ptrs from counts
    let mut row_ptrs = vec![0usize; n_rows + 1];
    for i in 0..n_rows {
        row_ptrs[i + 1] = row_ptrs[i] + row_counts[i];
    }
    let total_nnz = row_ptrs[n_rows];

    // Scatter entries into position (using write cursors per row)
    let mut write_pos = row_ptrs[..n_rows].to_vec();
    let mut col_indices = vec![0usize; total_nnz];
    let mut values = vec![T::zero(); total_nnz];

    for (j, col_entries) in col_results.iter().enumerate() {
        for &(gi, val) in col_entries {
            let pos = write_pos[gi];
            col_indices[pos] = j;
            values[pos] = val;
            write_pos[gi] = pos + 1;
        }
    }

    let raw_p = CsrMatrix::from_raw_parts(n_rows, n_cols, row_ptrs, col_indices, values);

    // Post-process: normalize each row to sum to 1.0 (constant preservation)
    // This ensures P * ones_coarse = ones_fine, which is required for
    // preserving the constant near-nullspace vector.
    let mut builder = sonium_solvers_sparse::CsrBuilder::with_capacity(n_rows, n_cols, raw_p.nnz());
    for i in 0..n_rows {
        let mut entries: Vec<(usize, T)> = raw_p.row_entries(i).collect();
        if entries.is_empty() {
            builder.add_row_entries(std::iter::empty());
            continue;
        }
        let row_sum: T = entries
            .iter()
            .map(|&(_, v)| v)
            .fold(T::zero(), |a, b| a + b);
        if row_sum.norm() > T::Real::from_f64(1e-15).unwrap() {
            let inv_sum = row_sum.inv();
            for entry in &mut entries {
                entry.1 *= inv_sum;
            }
        }
        builder.add_row_entries(entries.into_iter());
    }
    builder.finish()
}

#[cfg(test)]
#[path = "interpolation_aggregation_tests.rs"]
mod aggregation_tests;
