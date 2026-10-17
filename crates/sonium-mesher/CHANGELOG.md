# Changelog

## Unreleased

## 0.2.0
### Fixed

- Removed committed r2factor scratch (`src/mesh/tetrahedralize.rs.bak`,
  `.r2factor/bug-reports/split-verify-failure-1785228260.md`).
- `Mesh::detect_boundaries` on an invalid mesh now logs via `log::warn`
  instead of silently clearing boundaries; the fallible
  `Mesh::try_detect_boundaries` is the single implementation and returns the
  boundary count.
- `Element::is_p2` no longer panics on malformed node counts (returns
  `false`; see `try_polynomial_order`).
- Duplicate-node spatial hashing uses explicitly saturating bucket
  coordinates alongside the existing magnitude gate, so huge
  coordinates with a tiny tolerance fail closed with `InvalidTolerance`.

### Added

- `try_layered_spherical_shell_mesh_tetrahedra` returning the new typed
  `LayeredShellError` (empty / non-increasing radii, subdivision mismatch,
  zero subdivisions) instead of panicking.
- `tests/mesher_pipeline.rs`: generate → quality → partition → extract →
  write chains for 2D/3D, including partition determinism and STL/PLY
  byte-determinism plus facet-count agreement.
- `tests/degenerate_inputs.rs`: `NaN` / empty / coplanar / duplicate /
  out-of-range inputs against every fallible entry point.

### Changed

- All rayon gates are machine-adaptive now (`mesh::ParallelPolicy`):
  `work_items >= items_per_thread × available_parallelism`, with
  `SONIUM_MESHER_PAR_THRESHOLD` absolute override and `SONIUM_SERIAL`
  serial escape hatch. Fixed constants cannot be right per machine, so
  `Mesh::PARALLEL_TOPOLOGY_THRESHOLD` and quickhull's `PARALLEL_THRESHOLD`
  are replaced by per-thread grains (`TOPOLOGY_ITEMS_PER_THREAD`,
  `QUICKHULL_ITEMS_PER_THREAD` starting points).
- Topology construction and boundary detection parallelize the merge, not
  just extraction: contiguous ranges dedup into per-shard maps concurrently
  and merge in element order, so global indices and orientations are
  identical to the serial merge (see `sharded_merge_matches_serial_merge_exactly`).
- Determinism guarantees for `partition_mesh_bisection` (stable split-axis
  sort) and `quickhull_3d` (input-order assignment) are now documented.
- Steiner insertion stops early on diminishing returns (< 1% worst-ratio
  improvement) instead of always running all 32 rounds.
- Tet mean-ratio uses `cbrt` instead of `powf(2/3)`, and `mean_ratio`
  borrows nodes into stack arrays instead of allocating a `Vec` per element.

## 0.1.4

Fixed a P1/P2 bug

## 0.1.3

### Changed

- Constrained `tetrahedralize_surface*` APIs now return structured
  `TetrahedralizationError` values instead of succeeding with an
  unconstrained fallback. `TetrahedralizationResult` includes constraint-
  recovery operation counts and geometric-validity diagnostics.
- Mesh connectivity, topology, and neighbour caches use state fingerprints in
  addition to explicit invalidation, so direct public-vector mutations cannot
  silently reuse stale derived data.

### Added

- `mesh::validate`: orientation check, duplicate-node merge, degenerate
  element removal, manifoldness report, and `repair` orchestrator.
- `mesh::quality`: per-element mean-ratio, aspect-quality, dihedral-angle
  extrema, plus edge-length / element-measure histograms and a
  `MeshQualityReport` aggregator.
- `mesh::stl` / `mesh::ply` / `mesh::gmsh_write` (feature `mshio`): binary STL
  and ASCII PLY surface writers; ASCII MSH v4.1 writer that round-trips through
  `mshio` and the existing `gmsh.rs` reader.
- `mesh::generators::convex_hull_surface`: builds a triangle surface `Mesh`
  from a 3D point cloud via the in-tree quickhull. Interior points are
  dropped automatically.
- Tet refinement now honours `TetrahedralizeConfig`: `lloyd_iterations` runs
  volume-weighted Lloyd relaxation on interior nodes; `add_interior_points` +
  `max_ratio` inserts Steiner points at the circumcentre of the worst tet
  until quality clears `max_ratio` or a safety cap is hit.
- Uniform 8-way hex refinement on `ElementType::Hexahedron` (edge midpoints
  + face centres + body centre). Replaces the previous warn-and-skip path.
- `Mesh::element_neighbors()` cache + `Mesh::invalidate_caches()` helper.
  `add_node` / `add_element` invalidate every cache through the helper rather
  than clearing fields ad hoc.
- Partitioner: `partition_mesh_bisection` (recursive coordinate bisection on
  element centroids), `refine_partition_fm` (Fiduccia-Mattheyses local search
  honouring `PartitionOptions::imbalance`). `partition_mesh` now seeds with
  RCB then refines via FM; `seed` / `objective` / `imbalance` / `refinement`
  fields all participate.
- `Partition::partition_elements` is now O(1) — backed by a cached inverse
  index that `move_element` keeps in sync.
- `mesh::tetrahedralize::validate_delaunay`: diagnostic post-pass that counts
  degenerate, inverted, and non-Delaunay tets so callers can flag in-house
  meshes that need to go through Gmsh instead.
- **Bowyer-Watson kernel upgrade**: predicates now use Shewchuk's adaptive
  `robust::insphere` / `robust::orient3d` (new `robust` dependency); tets
  carry per-face neighbour indices in a stable-slot store with a free list;
  point insertion uses a visibility walk seeded from the last insertion plus
  BFS-over-neighbours cavity discovery. Cocircular / coplanar inputs are now
  handled correctly, and asymptotic insertion cost drops from O(n²) to
  roughly O(n^{4/3}) on spatially coherent inputs. The kernel remains
  best-effort for difficult constrained geometries, but those failures now
  return typed diagnostics rather than an unconstrained mesh or a panic.

### Fixed

- Bowyer-Watson in-sphere predicate now uses the lifted-paraboloid
  determinant with orient3d sign correction. The previous orient3d-only test
  was not an in-sphere test at all.
- `Delaunay3D` now tracks the real-point count explicitly; the insertion
  loop iterates real points (formerly skipped indices 0..3 and inserted the
  super-tet vertices as if they were real). `extract_mesh` filters
  super-tet-incident tets by index against `num_real` (formerly compared
  against the wrong fixed range).
- Cavity boundary faces in Bowyer-Watson are now stored with their original
  oriented winding (sorted form remains the hash key). New tets enforce
  positive orient3d explicitly.
- `tetrahedralize_surface` no longer silently drops `surface_facets`. It
  panics when called with a non-empty facet list because constrained
  Delaunay is not implemented; use Gmsh / meshlang for surface-constrained
  meshes.
- `partition::build_element_graph`: replaced O(N²) double element scan with
  an O(elements × faces) hash over sorted-node face keys. Edges are now used
  for 2D element adjacency.
- `Mesh::element_measure` Hexahedron arm now returns the exact volume via a
  6-tet decomposition (formerly approximated by `dx·dy·dz` of three random
  edges).
- `Element::faces()` for Hexahedron now returns the six true cube faces.
  The previous implementation returned two diagonal cross-sections for
  left/right, which broke face-based adjacency and boundary detection for
  hex meshes.
- Tet refinement now picks the shortest of the three octahedron diagonals
  for the inner split (Bey shortest-diagonal rule). The fixed-diagonal split
  was the standard sliver-producing bug.

### Performance

- `Mesh::detect_boundaries` and `MeshTopology::build` parallelise the
  per-element extraction step with rayon. Merging into the global hash
  remains serial.

## 0.1.2

### Fixed

- Quickhull `find_face_with_furthest_point` passed the original `vertices` slice instead of
  `unique_vertices`, causing wrong index lookups when duplicates were removed.
- `detect_boundaries` now preserves original face winding order in `BoundaryFace::nodes` instead
  of storing sorted indices. Sorted form is used only as a hash key for deduplication.
- `compact_mesh` boundary face `element_idx` was not remapped after compaction — downstream code
  would access wrong elements or panic with out-of-bounds.
- Greedy mesh partitioner could assign elements to partition 0 even when it was in
  `used_partitions` (neighbor conflict). Now initializes `best_partition` to `None` and falls
  back to partition 0 only when all partitions are used by neighbors.
- `SubMesh::to_mesh` dimension detection checked only the first node's z-coordinate. A 3D mesh
  on the z=0 plane was misidentified as 2D. Now checks all nodes.
- Mesh refinement `refine_elements` never removed parent elements from `mesh.elements`, causing
  double coverage in assembly.

### Known Issues

- Bowyer-Watson tetrahedralisation is geometrically correct and now uses
  exact-precision predicates with neighbour-graph walking (~O(N^{4/3})), but
  it remains **unconstrained** — no boundary edge / facet recovery. Meshes
  that must preserve a given surface still have to come from Gmsh / meshlang.
