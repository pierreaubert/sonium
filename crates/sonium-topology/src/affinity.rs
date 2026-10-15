//! Best-effort worker-thread CPU affinity (Linux).
//!
//! Vendor BLAS pins its workers by default (OpenBLAS unless
//! `OPENBLAS_SET_AFFINITY=0`): without it, pool threads scatter across SMT
//! siblings and CCDs and FMA-bound bands share physical cores (dgetrf-1024
//! 8T on 64c Threadripper: 15.8ms unpinned vs 10.1ms pinned to CPUs 0-7).
//! Sonium pins the workers of every pool it owns, plus the ambient-sized
//! mirror pool behind top-level scopes, to distinct inherited-mask CPUs in
//! hierarchical pack order (see [`ordered_runs`]). Packing won over
//! CCD-spreading 2.3x for factorizations (dpotrf-1024 8T: packed 4.4-4.5
//! vs spread 9.9-11.0, interleaved x3): cross-panel L3 reuse of the
//! trailing matrix beats conflict avoidance, even though spreading wins
//! cold single-SYRK calls (0.43 -> 0.26ms). At 2048/16T scheduler
//! placement only ties packing (dgetrf full 46.5 vs 45.2, dpotrf 16.0 vs
//! 16.4 med) — the working sets exceed per-CCX L3 either way, so
//! placement stops mattering; packing stands.
//! `SONIUM_SET_AFFINITY=0` restores scheduler placement (diagnostic/A-B
//! fallback).
//!
//! ## Hierarchical packing: NUMA, then CCD, then CCX
//!
//! Worker order packs the inherited mask so every pool prefix stays as
//! local as its width allows: one NUMA node while the pool fits in it (no
//! socket crossing unless the width needs it), then CCD-adjacent L3 runs,
//! then CCX runs, with SMT siblings last so workers take distinct
//! physical cores first. A pool narrower than its node only ever sees the
//! first node bucket — on an 8-CCD box 8 workers land on one CCD and 4 on
//! one CCX with no configuration; on multi-socket Xeon/EPYC a pool
//! narrower than a node stays on its first node. Detection is sysfs-only
//! with no vendor/CPUID checks and no hardcoded CCX/CCD widths, so it
//! holds across Xeon and EPYC generations (unknown or unreadable fields
//! degrade to defaults; total sysfs failure degrades to numeric order).
//! `SONIUM_TOPO_PACK=0` restores the legacy numeric order for same-binary
//! A/B.
//!
//! Factorizations whose working set fits in L3 go further: a compact pool
//! of at most [`COMPACT_MAX_CPUS`] workers pinned to one L3 domain (head
//! of the first hierarchical run — one CCX on AMD, adjacent L3-sharing
//! cores on Xeon) keeps the trailing matrix L3-resident across panels —
//! dgetrf-1024 ~12.5ms vs ~14.6 spread, any single CCX reproduces it under
//! taskset. Drivers enter via `run_compact` (clone inside for local NUMA
//! first-touch) with an L3-fit guard; `SONIUM_COMPACT=0` forces the
//! mirror-pool funnel.
//!
//! Everything is best-effort: pin failures are ignored (a restricted
//! sandbox may refuse), non-Linux is a no-op, and the inherited mask
//! (containers/cgroups/taskset) is always respected — workers cycle within
//! it when it holds fewer CPUs than the pool. Results never change: only
//! thread placement is affected.

use std::collections::BTreeMap;
use std::sync::LazyLock;

/// Inherited CPU mask in numeric order, or `None` when affinity is
/// disabled, unavailable, or the mask read failed. Read once: the mask is
/// fixed for the process. This is the membership source for every pool;
/// worker *order* comes from [`CPU_LIST`], which packs this set
/// hierarchically (see [`ordered_runs`]).
static MASK_CPUS: LazyLock<Option<Vec<usize>>> = LazyLock::new(|| {
    if std::env::var("SONIUM_SET_AFFINITY")
        .map(|v| v == "0")
        .unwrap_or(false)
    {
        return None;
    }
    inherited_cpus()
});

/// Worker-ordered CPU list: the inherited mask packed so every prefix
/// stays as local as it fits (see [`ordered_runs`]). `pin_worker(index)`
/// pins to `CPU_LIST[index % len]`, so pools narrower than the mask
/// (tiny/small, `RAYON_NUM_THREADS`-capped mirrors) never cross a NUMA,
/// CCD, or CCX boundary they do not need. `SONIUM_TOPO_PACK=0` restores
/// the legacy numeric order (diagnostic/A-B fallback).
static CPU_LIST: LazyLock<Option<Vec<usize>>> = LazyLock::new(|| {
    if !topo_pack_on() {
        return MASK_CPUS.clone();
    }
    ORDERED_TOPO.as_ref().map(|topo| topo.flat.clone())
});

/// Whether hierarchical packing applies. `SONIUM_TOPO_PACK=0` restores
/// the legacy numeric CPU order everywhere (worker list and compact
/// sets); the spread pool is unaffected (it never used it).
fn topo_pack_on() -> bool {
    std::env::var("SONIUM_TOPO_PACK")
        .map(|v| v != "0")
        .unwrap_or(true)
}

#[cfg(target_os = "linux")]
fn inherited_cpus() -> Option<Vec<usize>> {
    // `sched_getaffinity` on pid 0 queries the calling thread; pool
    // builders run on threads that inherit the process mask, so this is
    // the mask the workers would get without pinning.
    // SAFETY: `set` is a valid zeroed `cpu_set_t`; `CPU_ISSET` only reads
    // indices below `CPU_SETSIZE`.
    unsafe {
        let mut set = std::mem::zeroed::<libc::cpu_set_t>();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) != 0 {
            return None;
        }
        let mut out = Vec::new();
        for cpu in 0..libc::CPU_SETSIZE as usize {
            if libc::CPU_ISSET(cpu, &set) {
                out.push(cpu);
            }
        }
        if out.is_empty() { None } else { Some(out) }
    }
}

#[cfg(not(target_os = "linux"))]
fn inherited_cpus() -> Option<Vec<usize>> {
    None
}

/// Number of CPUs in the inherited affinity mask, or `None` when pinning
/// is disabled or the mask is unreadable. Lets pool sizing respect
/// cpusets/masks (containers, numactl, taskset) instead of oversubscribing
/// the allowed CPUs with an ambient-sized pool.
pub fn mask_cpu_count() -> Option<usize> {
    MASK_CPUS.as_ref().map(|list| list.len())
}

/// Inherited mask in numeric order, for tests. `None` exactly when the
/// ordered list is (see [`CPU_LIST`]).
#[cfg(test)]
pub fn mask_cpus() -> Option<Vec<usize>> {
    MASK_CPUS.clone()
}

/// The CPU `pin_worker(index)` will pin to, or `None` when pinning is a
/// no-op (disabled, unavailable, or the mask read failed). Exposed for
/// tests (including consumers'); not part of the placement API.
#[doc(hidden)]
pub fn pinned_cpu(index: usize) -> Option<usize> {
    CPU_LIST.as_ref().map(|list| list[index % list.len()])
}

/// Current thread's affinity mask, for tests (including consumers').
/// Empty when unreadable.
#[doc(hidden)]
pub fn current_mask() -> Vec<usize> {
    #[cfg(target_os = "linux")]
    // SAFETY: as in `inherited_cpus`.
    unsafe {
        let mut set = std::mem::zeroed::<libc::cpu_set_t>();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) != 0 {
            return Vec::new();
        }
        let mut out = Vec::new();
        for cpu in 0..libc::CPU_SETSIZE as usize {
            if libc::CPU_ISSET(cpu, &set) {
                out.push(cpu);
            }
        }
        out
    }
    #[cfg(not(target_os = "linux"))]
    Vec::new()
}

/// Per-CPU topology identity for hierarchical packing. Every field comes
/// from stable sysfs handles (`topology/*`, `cache/*/shared_cpu_list`,
/// the `node*` membership links) present on Xeon and EPYC alike. There
/// is deliberately no vendor/CPUID check and no hardcoded CCX/CCD width:
/// unknown or unreadable fields degrade to defaults (see
/// [`ordered_runs`]), so new generations work without code changes.
#[derive(Clone, Copy, Debug)]
struct CpuTopo {
    cpu: usize,
    package: usize,
    node: usize,
    die: usize,
    cluster: usize,
    /// Identity of the CPU's L3-sharing group (its lowest member): equal
    /// for CCX-mates on AMD, socket-wide on Xeon without SNC.
    l3key: usize,
    /// Position in the SMT sibling list (0 = primary thread).
    smt: usize,
}

fn read_usize(path: &str) -> Option<usize> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// NUMA node owning `cpu`, from its `node*` membership link. `None` when
/// unreadable (callers default to node 0).
fn numa_node_of(cpu: usize) -> Option<usize> {
    let dir = std::fs::read_dir(format!("/sys/devices/system/cpu/cpu{cpu}")).ok()?;
    let mut best: Option<usize> = None;
    for entry in dir.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        let Some(suffix) = name.strip_prefix("node") else {
            continue;
        };
        if suffix.is_empty() {
            continue;
        }
        if let Ok(id) = suffix.parse::<usize>() {
            best = Some(best.map_or(id, |b| b.min(id)));
        }
    }
    best
}

/// Rank of `cpu` in its SMT sibling list: 0 for the primary thread, for
/// non-SMT cores, and when the list is unreadable.
fn smt_rank(cpu: usize) -> usize {
    let text = std::fs::read_to_string(format!(
        "/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
    ))
    .unwrap_or_default();
    parse_cpu_list(&text)
        .iter()
        .position(|&c| c == cpu)
        .unwrap_or(0)
}

fn topo_infos(mask: &[usize]) -> Vec<CpuTopo> {
    mask.iter()
        .map(|&cpu| {
            let topo = format!("/sys/devices/system/cpu/cpu{cpu}/topology");
            CpuTopo {
                cpu,
                package: read_usize(&format!("{topo}/physical_package_id")).unwrap_or(0),
                node: numa_node_of(cpu).unwrap_or(0),
                // Opportunistic CCD proxy (see `ordered_runs`): uniform
                // on kernels that do not expose dies, harmless then.
                die: read_usize(&format!("{topo}/die_id")).unwrap_or(0),
                cluster: read_usize(&format!("{topo}/cluster_id")).unwrap_or(0),
                l3key: l3_group(cpu)
                    .and_then(|g| g.first().copied())
                    .unwrap_or(cpu),
                smt: smt_rank(cpu),
            }
        })
        .collect()
}

/// NUMA buckets in `(package, node)` order; each bucket holds its
/// L3-sharing runs in `(die, cluster, lowest CPU)` order with members in
/// `(smt rank, cpu)` order. Shared core of [`ordered_runs`] (compact
/// sets) and [`ordered_flat`] (worker order).
///
/// Portability notes (Xeon and EPYC, any generation — no vendor checks,
/// no hardcoded widths):
///
/// - `die`/`cluster` are ordering-only: uniform on kernels that do not
///   expose them (no effect there); where `die` tracks the CCD, same-CCD
///   L3 runs become adjacent even if CPU numbering ever stops pairing
///   them. They can never split an L3 run: runs partition by `l3key`
///   first, so L3 contiguity holds unconditionally.
/// - A socket-wide L3 group (Xeon without SNC) is split by the node
///   buckets, so SNC/sub-NUMA machines still pack node-local.
/// - Total sysfs failure degrades to numeric order (one bucket of
///   singleton runs), which is exactly the legacy order.
fn topo_buckets(infos: &[CpuTopo]) -> Vec<Vec<Vec<&CpuTopo>>> {
    let mut buckets: BTreeMap<(usize, usize), BTreeMap<usize, Vec<&CpuTopo>>> = BTreeMap::new();
    for info in infos {
        buckets
            .entry((info.package, info.node))
            .or_default()
            .entry(info.l3key)
            .or_default()
            .push(info);
    }
    let mut out = Vec::new();
    for (_, groups) in buckets {
        let mut groups: Vec<Vec<&CpuTopo>> = groups.into_values().collect();
        for members in groups.iter_mut() {
            members.sort_by_key(|i| (i.smt, i.cpu));
        }
        // Representative key per run: minima, so one odd member cannot
        // strand its runmates (all members share these in practice).
        groups.sort_by_key(|members| {
            let die = members.iter().map(|i| i.die).min().unwrap_or(0);
            let cluster = members.iter().map(|i| i.cluster).min().unwrap_or(0);
            let first = members.iter().map(|i| i.cpu).min().unwrap_or(0);
            (die, cluster, first)
        });
        out.push(groups);
    }
    out
}

/// L3 runs in hierarchical pack order: [`topo_buckets`] flattened. Run
/// heads feed the compact sets, so a narrow pool takes one run's
/// physical cores (one CCX on AMD).
fn ordered_runs(infos: &[CpuTopo]) -> Vec<Vec<usize>> {
    topo_buckets(infos)
        .into_iter()
        .flatten()
        .map(|members| members.into_iter().map(|i| i.cpu).collect())
        .collect()
}

/// Physical cores (SMT rank 0) in packed order: buckets in
/// `(package, node)` order, runs in order, physical members only.
fn ordered_physical(infos: &[CpuTopo]) -> Vec<usize> {
    topo_buckets(infos)
        .into_iter()
        .flatten()
        .flat_map(|run| {
            run.into_iter()
                .filter(|i| i.smt == 0)
                .map(|i| i.cpu)
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Worker order over the mask: bucket-major like [`ordered_runs`], but
/// SMT-major within each bucket — every physical core of the bucket's
/// runs (in run order) before any SMT sibling. A pool narrower than its
/// node's physical cores therefore takes distinct physical cores first
/// (never SMT twins while a physical core on the node is still free),
/// and only spills to SMT past the node's physical width; it never
/// leaves the node unless the width needs it. Strict node containment
/// is deliberate (locality-first, matching the packing evidence); the
/// spread pool stays the bandwidth-first cross-node path.
fn ordered_flat(infos: &[CpuTopo]) -> Vec<usize> {
    let mut flat = Vec::with_capacity(infos.len());
    for bucket in topo_buckets(infos) {
        let mut ranks: Vec<usize> = bucket.iter().flatten().map(|i| i.smt).collect();
        ranks.sort_unstable();
        ranks.dedup();
        for rank in ranks {
            for run in &bucket {
                flat.extend(run.iter().filter(|i| i.smt == rank).map(|i| i.cpu));
            }
        }
    }
    flat
}

/// Hierarchically ordered view of the inherited mask: L3 runs in pack
/// order plus their flattening. `None` exactly when the mask is (see
/// [`MASK_CPUS`]); per-field sysfs defaults mean partial topology
/// failures degrade the order, never the presence. Read once: topology
/// is fixed for the process.
struct OrderedTopo {
    runs: Vec<Vec<usize>>,
    flat: Vec<usize>,
    /// Physical cores only (SMT rank 0) in packed order: run-major
    /// within each bucket, buckets in `(package, node)` order.
    physical: Vec<usize>,
}

static ORDERED_TOPO: LazyLock<Option<OrderedTopo>> = LazyLock::new(|| {
    let mask = MASK_CPUS.as_ref()?;
    let infos = topo_infos(mask);
    let runs = ordered_runs(&infos);
    let flat = ordered_flat(&infos);
    let physical = ordered_physical(&infos);
    Some(OrderedTopo {
        runs,
        flat,
        physical,
    })
});

/// CPUs sharing one L3 domain, for the compact factorization pool: the
/// head of the first hierarchical run (see [`ordered_runs`]) — one CCX
/// on AMD, adjacent L3-sharing cores on Xeon — capped at
/// [`COMPACT_MAX_CPUS`]. One CCX keeps the trailing matrix L3-resident
/// across panels (dgetrf-1024: any single 4-core CCX ~11.7ms vs ~14.6
/// spread over the box). `None` when affinity is disabled, the topology
/// is unreadable, or a group cannot be formed — callers fall back to the
/// mirror pool. Read once: topology is fixed for the process.
pub const COMPACT_MAX_CPUS: usize = 4;

static COMPACT_CPUS: LazyLock<Option<Vec<usize>>> = LazyLock::new(|| {
    // Manual override first (diagnostic/A-B and odd topologies):
    // `SONIUM_COMPACT_CPUS=0-7,64-67` pins the compact pool exactly.
    if let Ok(text) = std::env::var("SONIUM_COMPACT_CPUS") {
        let mut cpus = parse_cpu_list(&text);
        if let Some(list) = MASK_CPUS.as_ref() {
            cpus.retain(|c| list.contains(c));
        }
        if !cpus.is_empty() {
            return Some(cpus);
        }
    }
    let list = CPU_LIST.as_ref()?;
    if list.len() <= COMPACT_MAX_CPUS {
        return Some(list.clone());
    }
    if !topo_pack_on() {
        // Legacy path for `SONIUM_TOPO_PACK=0` A/B: first CPUs of the L3
        // group containing the first mask CPU.
        let mask = MASK_CPUS.as_ref()?;
        let group = l3_group(*mask.first()?)?;
        if group.is_empty() {
            return None;
        }
        let take = group.len().min(COMPACT_MAX_CPUS);
        return Some(group[..take].to_vec());
    }
    let runs = ORDERED_TOPO.as_ref().map(|topo| &topo.runs)?;
    let first = runs.first()?;
    if first.is_empty() {
        return None;
    }
    let take = first.len().min(COMPACT_MAX_CPUS);
    Some(first[..take].to_vec())
});

/// L3-group count for the wide compact pool: the first
/// [`COMPACT_MAX_CPUS`] CPUs of each of the first
/// [`COMPACT_WIDE_GROUPS`] L3 groups. Four stands (2026-10-02,
/// `cholphases` process-interleaved A/B, RAYON=16): 16 workers on four
/// CCXs beat 8-on-two 14/14 rounds at dpotrf-1024 (~-6%, SYRK phase
/// -11%) and tie 4/4 at dpotrf-512, while the same 16 CPUs via the
/// mirror-spawn funnel lose — the win needs install-funneling (see
/// `run_mirror`), not just width. The 8 MB factor stays L3-resident
/// across four 16 MB slices. Masks with fewer groups degrade
/// gracefully (`take` what exists; one group collapses to narrow via
/// the equality check below). `SONIUM_COMPACT_CPUS` still pins both
/// sets exactly (diagnostic/A-B and kill-switch). History: two groups
/// stood before (compute-dense rank-64 SYRK wants two L3 domains:
/// spotrf-1024 7.3 -> 5.9ms vs one CCX, while SMT twins on one CCX
/// lose); four extends the same L3-domain mechanism with width.
pub const COMPACT_WIDE_GROUPS: usize = 4;

/// CPUs for the wide compact pool: the first [`COMPACT_MAX_CPUS`] CPUs
/// of each of the first [`COMPACT_WIDE_GROUPS`] hierarchical runs (see
/// [`ordered_runs`]), so the four L3 domains stay node-local whenever
/// the node holds that many. `None` when the wide set would equal the
/// narrow set (single group or small mask: callers share the narrow
/// pool), so no duplicate pool is ever built. The manual override pins
/// both sets exactly (narrow and wide coincide there by construction).
static COMPACT_CPUS_WIDE: LazyLock<Option<Vec<usize>>> = LazyLock::new(|| {
    if std::env::var("SONIUM_COMPACT_CPUS").is_ok() {
        return None;
    }
    if !topo_pack_on() {
        // Legacy path for `SONIUM_TOPO_PACK=0` A/B: first L3 groups in
        // numeric discovery order.
        return wide_cpu_set(&l3_groups()?);
    }
    let runs = ORDERED_TOPO.as_ref().map(|topo| &topo.runs)?;
    wide_cpu_set(runs)
});

/// First [`COMPACT_MAX_CPUS`] CPUs of each of the first
/// [`COMPACT_WIDE_GROUPS`] runs/groups. `None` for fewer than two
/// groups, or when the wide set would equal the narrow set (callers
/// share the narrow pool then).
fn wide_cpu_set(groups: &[Vec<usize>]) -> Option<Vec<usize>> {
    if groups.len() < 2 {
        return None;
    }
    let mut out = Vec::new();
    for group in groups.iter().take(COMPACT_WIDE_GROUPS) {
        out.extend_from_slice(&group[..group.len().min(COMPACT_MAX_CPUS)]);
    }
    let narrow = COMPACT_CPUS.clone()?;
    if out == narrow {
        return None;
    }
    Some(out)
}

/// Worker count for the spread pool (streaming kernels): one worker per
/// L3 group on a 16-group box, so each worker owns a distinct L3 slice.
/// Sized down by the mask when fewer CPUs exist (see [`spread_cpus`]).
pub const SPREAD_THREADS: usize = 16;

/// CPUs for the spread pool, if topology allowed one to be formed: one CPU
/// per L3 group round-robin (second round takes the next CPU of each
/// group), so workers never share an L3 slice until groups run out.
/// Test-visible so pool wiring can be asserted without running a solve.
pub fn spread_cpus() -> Option<Vec<usize>> {
    SPREAD_CPUS.clone()
}

static SPREAD_CPUS: LazyLock<Option<Vec<usize>>> = LazyLock::new(|| {
    // Manual override first (diagnostic/A-B and odd topologies).
    if let Ok(text) = std::env::var("SONIUM_SPREAD_CPUS") {
        let mut cpus = parse_cpu_list(&text);
        if let Some(list) = MASK_CPUS.as_ref() {
            cpus.retain(|c| list.contains(c));
        }
        if !cpus.is_empty() {
            return Some(cpus);
        }
    }
    spread_cpu_set(SPREAD_THREADS)
});

/// Round-robin one CPU per L3 group until `width` CPUs are collected (or
/// groups run dry): each worker lands on a distinct L3 slice first, and
/// only shares once every group contributed. `None` when affinity is
/// disabled, the topology is unreadable, or no CPU could be chosen.
fn spread_cpu_set(width: usize) -> Option<Vec<usize>> {
    let groups = l3_groups()?;
    spread_from_groups(&groups, width)
}

/// Pure round-robin core of [`spread_cpu_set`], over caller-supplied
/// groups so the schedule is unit-testable without touching sysfs.
fn spread_from_groups(groups: &[Vec<usize>], width: usize) -> Option<Vec<usize>> {
    if groups.is_empty() || width == 0 {
        return None;
    }
    let mut out = Vec::new();
    let mut round = 0;
    while out.len() < width {
        let mut progressed = false;
        for g in groups.iter() {
            if out.len() >= width {
                break;
            }
            if g.is_empty() {
                continue;
            }
            let cpu = g[round % g.len()];
            if !out.contains(&cpu) {
                out.push(cpu);
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
        round += 1;
    }
    if out.is_empty() { None } else { Some(out) }
}

/// Distinct L3 groups covering the inherited mask, in numeric discovery
/// order. The spread pool round-robins over these (order-stable across
/// the packing change); compact sets use [`ordered_runs`] instead.
/// `None` when affinity is disabled or no group forms.
fn l3_groups() -> Option<Vec<Vec<usize>>> {
    let list = MASK_CPUS.as_ref()?;
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for cpu in list {
        if groups.iter().any(|g| g.contains(cpu)) {
            continue;
        }
        let group = l3_group(*cpu)?;
        if group.is_empty() {
            continue;
        }
        groups.push(group);
    }
    if groups.is_empty() {
        return None;
    }
    Some(groups)
}

/// L3-sharing CPUs of `cpu`, from sysfs (`shared_cpu_list` of the
/// level-3 cache; deepest level when no level 3 exists). Intersected with
/// the inherited mask. `None` when unreadable.
fn l3_group(cpu: usize) -> Option<Vec<usize>> {
    let base = format!("/sys/devices/system/cpu/cpu{cpu}/cache");
    let mut best: Option<(u32, Vec<usize>)> = None;
    for index in 0..8 {
        let Ok(level_text) = std::fs::read_to_string(format!("{base}/index{index}/level")) else {
            continue;
        };
        let Ok(level) = level_text.trim().parse::<u32>() else {
            continue;
        };
        let Ok(shared) = std::fs::read_to_string(format!("{base}/index{index}/shared_cpu_list"))
        else {
            continue;
        };
        let mut cpus = parse_cpu_list(&shared);
        let list = MASK_CPUS.as_ref()?;
        cpus.retain(|c| list.contains(c));
        if cpus.is_empty() {
            continue;
        }
        // Prefer level 3; otherwise the deepest level seen.
        if level == 3 {
            return Some(cpus);
        }
        if best.as_ref().map(|b| level > b.0).unwrap_or(true) {
            best = Some((level, cpus));
        }
    }
    best.map(|b| b.1)
}

/// Level-3 slice size in bytes for `cpu`'s slice, from sysfs (`size` of
/// the level-3 index). `None` when unreadable or unparsable.
fn l3_slice_bytes(cpu: usize) -> Option<usize> {
    let base = format!("/sys/devices/system/cpu/cpu{cpu}/cache");
    for index in 0..8 {
        let Ok(level_text) = std::fs::read_to_string(format!("{base}/index{index}/level")) else {
            continue;
        };
        if level_text.trim() != "3" {
            continue;
        }
        let Ok(size_text) = std::fs::read_to_string(format!("{base}/index{index}/size")) else {
            continue;
        };
        let s = size_text.trim();
        let (num, mult) = match s.chars().last()? {
            'K' => (s[..s.len() - 1].parse::<usize>().ok()?, 1024usize),
            'M' => (s[..s.len() - 1].parse::<usize>().ok()?, 1024 * 1024),
            'G' => (s[..s.len() - 1].parse::<usize>().ok()?, 1024 * 1024 * 1024),
            _ => (s.parse::<usize>().ok()?, 1),
        };
        return num.checked_mul(mult);
    }
    None
}

/// Total level-3 bytes across the machine's unique L3 groups (one slice
/// size per group, so asymmetric topologies sum correctly). Used to size
/// N-split GEMM bands (bands × A footprint past L3 self-evicts). Cached;
/// `None` when topology or sizes are unreadable (callers keep defaults).
pub fn l3_total_bytes() -> Option<usize> {
    static TOTAL: std::sync::LazyLock<Option<usize>> = std::sync::LazyLock::new(|| {
        let groups = l3_groups()?;
        let mut total = 0usize;
        for g in &groups {
            total = total.checked_add(l3_slice_bytes(*g.first()?)?)?;
        }
        Some(total)
    });
    *TOTAL
}

/// Parse a sysfs cpu list (`0-3,8-11`) into sorted CPU ids.
fn parse_cpu_list(text: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for part in text.trim().split(',') {
        let part = part.trim();
        if let Some((lo, hi)) = part.split_once('-') {
            if let (Ok(lo), Ok(hi)) = (lo.trim().parse::<usize>(), hi.trim().parse::<usize>()) {
                out.extend(lo..=hi);
            }
        } else if let Ok(cpu) = part.parse::<usize>() {
            out.push(cpu);
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// The compact CPU set for the factorization pool, if topology allowed
/// one to be formed. Test-visible so pool wiring can be asserted without
/// running a solve.
pub fn compact_cpus() -> Option<Vec<usize>> {
    COMPACT_CPUS.clone()
}

/// The wide compact set ([`COMPACT_WIDE_GROUPS`] L3 domains), if it
/// differs from the narrow set. Test-visible; see [`compact_cpus`].
pub fn compact_cpus_wide() -> Option<Vec<usize>> {
    COMPACT_CPUS_WIDE.clone()
}

/// L3 size in bytes (representative CPU 0 entry), for the compact-pool
/// fit guard. `None` when unreadable.
///
/// Cached process-wide: the value is hardware-invariant, and the serial
/// GEMM gate evaluates this per call — an uncached sysfs read there cost
/// ~40% on thin serial Schur updates (2026-09-28 merge regression).
pub fn l3_bytes() -> Option<u64> {
    static CACHED: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *CACHED.get_or_init(l3_bytes_uncached)
}

fn l3_bytes_uncached() -> Option<u64> {
    let text = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index3/size").ok()?;
    let text = text.trim();
    let (digits, mult) = if let Some(k) = text.strip_suffix('K') {
        (k, 1024u64)
    } else if let Some(m) = text.strip_suffix('M') {
        (m, 1024u64 * 1024)
    } else {
        (text, 1u64)
    };
    digits.trim().parse::<u64>().ok().map(|v| v * mult)
}

/// Pin the calling worker to the compact set. No-op when the set is
/// empty (pool is never built then, so this is unreachable in practice).
pub fn pin_compact_worker(index: usize) {
    let Some(set) = COMPACT_CPUS.as_ref() else {
        return;
    };
    if set.is_empty() {
        return;
    }
    let cpu = set[index % set.len()];
    #[cfg(target_os = "linux")]
    // SAFETY: as in `pin_worker`.
    unsafe {
        let mut mask = std::mem::zeroed::<libc::cpu_set_t>();
        libc::CPU_SET(cpu, &mut mask);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mask);
    }
    #[cfg(not(target_os = "linux"))]
    let _ = cpu;
}

/// Pin the calling worker to the spread set (one L3 slice per worker).
/// Unreachable when the set is empty (no spread pool is built then).
pub fn pin_spread_worker(index: usize) {
    let Some(set) = SPREAD_CPUS.as_ref() else {
        return;
    };
    if set.is_empty() {
        return;
    }
    let cpu = set[index % set.len()];
    #[cfg(target_os = "linux")]
    // SAFETY: as in `pin_worker`.
    unsafe {
        let mut mask = std::mem::zeroed::<libc::cpu_set_t>();
        libc::CPU_SET(cpu, &mut mask);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mask);
    }
    #[cfg(not(target_os = "linux"))]
    let _ = cpu;
}

/// Pin the calling worker to the wide compact set. Unreachable when the
/// set is empty (no wide pool is built then).
pub fn pin_compact_wide_worker(index: usize) {
    let Some(set) = COMPACT_CPUS_WIDE.as_ref() else {
        return;
    };
    if set.is_empty() {
        return;
    }
    let cpu = set[index % set.len()];
    #[cfg(target_os = "linux")]
    // SAFETY: as in `pin_worker`.
    unsafe {
        let mut mask = std::mem::zeroed::<libc::cpu_set_t>();
        libc::CPU_SET(cpu, &mut mask);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mask);
    }
    #[cfg(not(target_os = "linux"))]
    let _ = cpu;
}

/// Pin the calling worker to `CPU_LIST[index % len]` (hierarchical pack
/// order, so low indexes stay NUMA/CCD/CCX-local). No-op when affinity
/// is disabled, unavailable, or the mask read failed.
pub fn pin_worker(index: usize) {
    let Some(list) = CPU_LIST.as_ref() else {
        return;
    };
    let cpu = list[index % list.len()];
    #[cfg(target_os = "linux")]
    // SAFETY: `set` is a valid `cpu_set_t` with exactly `cpu` set; a
    // nonzero return (sandbox refusal) is ignored by design.
    unsafe {
        let mut set = std::mem::zeroed::<libc::cpu_set_t>();
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
    #[cfg(not(target_os = "linux"))]
    let _ = cpu;
}

/// Physical CPUs (SMT rank 0) in packed order — the worker budget for
/// callers that want one thread per physical core (e.g. FFT batch
/// dispatch, future ML executors) without SMT-tail spill. `None` when
/// affinity is disabled or the mask is empty of physical CPUs (a mask
/// holding only SMT threads: callers fall back to unpinned execution).
static PACKED_PHYSICAL: LazyLock<Option<Vec<usize>>> = LazyLock::new(|| {
    let mask = MASK_CPUS.as_ref()?;
    if topo_pack_on() {
        let topo = ORDERED_TOPO.as_ref()?;
        if topo.physical.is_empty() {
            return None;
        }
        return Some(topo.physical.clone());
    }
    // Legacy order: physical CPUs in numeric mask order.
    let physical: Vec<usize> = mask
        .iter()
        .copied()
        .filter(|&cpu| smt_rank(cpu) == 0)
        .collect();
    if physical.is_empty() {
        None
    } else {
        Some(physical)
    }
});

/// Worker-ordered CPU list (see [`CPU_LIST`]), or `None` when affinity
/// is disabled or unavailable.
pub fn packed_cpus() -> Option<Vec<usize>> {
    CPU_LIST.clone()
}

/// Physical CPUs in packed order (see [`PACKED_PHYSICAL`]).
pub fn physical_cpus() -> Option<Vec<usize>> {
    PACKED_PHYSICAL.clone()
}

/// Count of [`physical_cpus`], for worker-budget gates. `None` exactly
/// when [`physical_cpus`] is.
pub fn physical_cpu_count() -> Option<usize> {
    PACKED_PHYSICAL.as_ref().map(|list| list.len())
}

/// Pin the calling worker to the packed physical CPU
/// `PACKED_PHYSICAL[index % len]` (one worker per physical core, low
/// indexes NUMA/CCD/CCX-local). No-op under the same conditions as
/// [`pin_worker`].
pub fn pin_physical_worker(index: usize) {
    let Some(list) = PACKED_PHYSICAL.as_ref() else {
        return;
    };
    if list.is_empty() {
        return;
    }
    let cpu = list[index % list.len()];
    #[cfg(target_os = "linux")]
    // SAFETY: as in `pin_worker`.
    unsafe {
        let mut set = std::mem::zeroed::<libc::cpu_set_t>();
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
    #[cfg(not(target_os = "linux"))]
    let _ = cpu;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cpu_list_handles_ranges_singles_and_empty() {
        assert_eq!(
            parse_cpu_list("0-3,64-67"),
            vec![0, 1, 2, 3, 64, 65, 66, 67]
        );
        assert_eq!(parse_cpu_list("5"), vec![5]);
        assert_eq!(parse_cpu_list(""), Vec::<usize>::new());
        assert_eq!(parse_cpu_list("3-1"), Vec::<usize>::new());
        assert_eq!(parse_cpu_list("0-1,1-2"), vec![0, 1, 2]);
    }

    #[test]
    fn compact_set_is_a_small_valid_subset_when_present() {
        if let Some(set) = compact_cpus() {
            assert!(!set.is_empty());
            assert!(set.len() <= COMPACT_MAX_CPUS);
            // Unique (order is hierarchical, not numeric, so sortedness
            // is not asserted).
            let mut unique = set.clone();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), set.len());
            if let Some(list) = CPU_LIST.as_ref() {
                assert!(set.iter().all(|c| list.contains(c)));
            }
        }
    }

    /// Single-CPU topology with all-default fields; tests override via
    /// struct-update syntax.
    fn topo(cpu: usize) -> CpuTopo {
        CpuTopo {
            cpu,
            package: 0,
            node: 0,
            die: 0,
            cluster: 0,
            l3key: cpu,
            smt: 0,
        }
    }

    fn flat(infos: &[CpuTopo]) -> Vec<usize> {
        ordered_flat(infos)
    }

    #[test]
    fn hierarchical_order_packs_node_then_l3_then_smt() {
        // Two nodes x two L3 runs x (two cores + SMT twins), shuffled
        // input: node 0's bucket comes first; within a bucket every
        // physical core (run-major) precedes every SMT sibling, so a
        // pool prefix takes distinct physical cores first and never an
        // SMT twin while a physical core on the node is still free.
        let infos = vec![
            CpuTopo {
                node: 1,
                l3key: 10,
                ..topo(11)
            },
            CpuTopo {
                l3key: 0,
                smt: 1,
                ..topo(64)
            },
            CpuTopo {
                node: 1,
                l3key: 8,
                smt: 1,
                ..topo(73)
            },
            CpuTopo {
                l3key: 2,
                ..topo(3)
            },
            CpuTopo {
                node: 1,
                l3key: 8,
                ..topo(8)
            },
            CpuTopo {
                l3key: 0,
                ..topo(1)
            },
            CpuTopo {
                node: 1,
                l3key: 10,
                smt: 1,
                ..topo(74)
            },
            CpuTopo {
                l3key: 2,
                smt: 1,
                ..topo(66)
            },
            CpuTopo {
                node: 1,
                l3key: 8,
                ..topo(9)
            },
            CpuTopo {
                l3key: 0,
                smt: 1,
                ..topo(65)
            },
            CpuTopo {
                l3key: 2,
                ..topo(2)
            },
            CpuTopo {
                node: 1,
                l3key: 10,
                ..topo(10)
            },
            CpuTopo {
                l3key: 0,
                ..topo(0)
            },
            CpuTopo {
                node: 1,
                l3key: 8,
                smt: 1,
                ..topo(72)
            },
            CpuTopo {
                l3key: 2,
                smt: 1,
                ..topo(67)
            },
            CpuTopo {
                node: 1,
                l3key: 10,
                smt: 1,
                ..topo(75)
            },
        ];
        assert_eq!(
            ordered_runs(&infos),
            vec![
                vec![0, 1, 64, 65],
                vec![2, 3, 66, 67],
                vec![8, 9, 72, 73],
                vec![10, 11, 74, 75],
            ]
        );
        assert_eq!(
            flat(&infos),
            vec![0, 1, 2, 3, 64, 65, 66, 67, 8, 9, 10, 11, 72, 73, 74, 75]
        );
    }

    #[test]
    fn flat_orders_smt_last_across_runs() {
        // Single node, two runs: the flat worker order takes the second
        // run's physical cores before the first run's SMT twins (an
        // 8-wide pool gets 8 physical cores, not 4 cores + 4 twins).
        let infos = vec![
            CpuTopo {
                l3key: 0,
                smt: 1,
                ..topo(64)
            },
            CpuTopo {
                l3key: 4,
                ..topo(5)
            },
            CpuTopo {
                l3key: 0,
                ..topo(0)
            },
            CpuTopo {
                l3key: 4,
                smt: 1,
                ..topo(69)
            },
            CpuTopo {
                l3key: 4,
                ..topo(4)
            },
            CpuTopo {
                l3key: 0,
                ..topo(1)
            },
            CpuTopo {
                l3key: 0,
                smt: 1,
                ..topo(65)
            },
            CpuTopo {
                l3key: 4,
                smt: 1,
                ..topo(68)
            },
        ];
        assert_eq!(flat(&infos), vec![0, 1, 4, 5, 64, 65, 68, 69]);
    }

    #[test]
    fn hierarchical_order_splits_socket_wide_l3_by_node() {
        // Xeon SNC shape: one L3 group spanning two nodes still packs
        // node-local — the node buckets split the run.
        let infos = vec![
            CpuTopo {
                node: 1,
                l3key: 0,
                ..topo(3)
            },
            CpuTopo {
                l3key: 0,
                ..topo(1)
            },
            CpuTopo {
                node: 1,
                l3key: 0,
                ..topo(2)
            },
            CpuTopo {
                l3key: 0,
                ..topo(0)
            },
        ];
        assert_eq!(ordered_runs(&infos), vec![vec![0, 1], vec![2, 3]]);
    }

    #[test]
    fn hierarchical_order_groups_ccd_pairs_via_die() {
        // Four L3 runs whose numbering interleaves two dies: die-major
        // run order keeps same-CCD runs adjacent, and every run stays
        // contiguous (L3 contiguity never depends on `die` semantics).
        let infos = vec![
            CpuTopo {
                die: 1,
                l3key: 0,
                ..topo(0)
            },
            CpuTopo {
                die: 1,
                l3key: 0,
                ..topo(1)
            },
            CpuTopo {
                l3key: 4,
                ..topo(4)
            },
            CpuTopo {
                l3key: 4,
                ..topo(5)
            },
            CpuTopo {
                die: 1,
                l3key: 8,
                ..topo(8)
            },
            CpuTopo {
                die: 1,
                l3key: 8,
                ..topo(9)
            },
            CpuTopo {
                l3key: 12,
                ..topo(12)
            },
            CpuTopo {
                l3key: 12,
                ..topo(13)
            },
        ];
        assert_eq!(
            ordered_runs(&infos),
            vec![vec![4, 5], vec![12, 13], vec![0, 1], vec![8, 9]]
        );
    }

    #[test]
    fn hierarchical_order_degrades_to_numeric_without_topology() {
        // All defaults with singleton L3 keys (unreadable sysfs) is one
        // bucket of singleton runs: exactly numeric order.
        let infos = vec![topo(3), topo(1), topo(2), topo(0)];
        assert_eq!(flat(&infos), vec![0, 1, 2, 3]);
    }

    #[test]
    fn ordered_list_is_a_permutation_of_the_mask() {
        let mask = mask_cpus();
        let ordered = CPU_LIST.as_ref().cloned();
        let (Some(mut mask), Some(mut ordered)) = (mask, ordered) else {
            return;
        };
        mask.sort_unstable();
        ordered.sort_unstable();
        assert_eq!(mask, ordered);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn packed_prefix_stays_on_one_node_when_it_fits() {
        // Headline guarantee: a pool prefix that fits in some node stays
        // within a single node. Best-effort: skips when affinity is off,
        // the legacy order is forced, sysfs is unreadable, or no node
        // holds the probe width.
        if !topo_pack_on() {
            return;
        }
        let Some(ordered) = CPU_LIST.as_ref() else {
            return;
        };
        const WIDTH: usize = 8;
        if ordered.len() < WIDTH {
            return;
        }
        let mut nodes: Vec<Vec<usize>> = Vec::new();
        let Ok(dir) = std::fs::read_dir("/sys/devices/system/node") else {
            return;
        };
        for entry in dir.flatten() {
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            let Some(suffix) = name.strip_prefix("node") else {
                continue;
            };
            if suffix.is_empty() || suffix.parse::<usize>().is_err() {
                continue;
            }
            let Ok(text) =
                std::fs::read_to_string(format!("/sys/devices/system/node/{name}/cpulist"))
            else {
                continue;
            };
            nodes.push(parse_cpu_list(&text));
        }
        if !nodes.iter().any(|cpus| cpus.len() >= WIDTH) {
            return;
        }
        let head = &ordered[..WIDTH];
        assert!(
            nodes
                .iter()
                .any(|cpus| head.iter().all(|c| cpus.contains(c))),
            "first {WIDTH} packed CPUs cross NUMA nodes: {head:?}"
        );
    }

    #[test]
    fn l3_size_is_sane_when_readable() {
        if let Some(bytes) = l3_bytes() {
            assert!(bytes >= 1024);
        }
    }

    #[test]
    fn spread_schedule_hits_distinct_slices_first() {
        // Four L3 slices of four CPUs: the first four workers must own
        // distinct slices (one CPU from each group), the second four the
        // next CPU of each group. Synthetic groups keep this sysfs-free.
        let groups = vec![
            vec![0, 1, 2, 3],
            vec![8, 9, 10, 11],
            vec![16, 17, 18, 19],
            vec![24, 25, 26, 27],
        ];
        assert_eq!(
            spread_from_groups(&groups, 8),
            Some(vec![0, 8, 16, 24, 1, 9, 17, 25])
        );
        // Width beyond the total yields every CPU exactly once and the
        // loop still terminates (no duplicate endlessly re-added).
        assert_eq!(
            spread_from_groups(&groups, 100),
            Some((0..4).flat_map(|r| [r, 8 + r, 16 + r, 24 + r]).collect())
        );
        // Uneven + empty groups: empties skipped, sharing only by round.
        let ragged = vec![vec![], vec![0], vec![4, 5, 6]];
        assert_eq!(spread_from_groups(&ragged, 4), Some(vec![0, 4, 5, 6]));
        // Degenerate inputs yield None rather than an empty pool.
        assert_eq!(spread_from_groups(&groups, 0), None);
        assert_eq!(spread_from_groups(&[], 8), None);
        assert_eq!(spread_from_groups(&[vec![], vec![]], 8), None);
    }

    #[test]
    fn spread_set_is_unique_and_masked_when_present() {
        if std::env::var("SONIUM_SPREAD_CPUS").is_ok() {
            return;
        }
        if let Some(set) = spread_cpus() {
            assert!(!set.is_empty());
            assert!(set.len() <= SPREAD_THREADS);
            let mut unique = set.clone();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), set.len());
            if let Some(list) = CPU_LIST.as_ref() {
                assert!(set.iter().all(|c| list.contains(c)));
            }
        }
    }

    #[test]
    fn wide_set_covers_four_domains_when_present() {
        // Without the manual override (tests never set it), a wide set
        // must hold at most `COMPACT_WIDE_GROUPS` narrow-sized groups
        // from distinct L3 domains and differ from the narrow set
        // (2026-10-02: four groups; dpotrf-1024 -6% vs two, ties 512).
        if std::env::var("SONIUM_COMPACT_CPUS").is_ok() {
            return;
        }
        if let Some(wide) = compact_cpus_wide() {
            assert!(!wide.is_empty());
            assert!(wide.len() <= COMPACT_WIDE_GROUPS * COMPACT_MAX_CPUS);
            let mut unique = wide.clone();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), wide.len());
            if let Some(narrow) = compact_cpus() {
                assert_ne!(wide, narrow);
                assert!(wide.len() > narrow.len());
            }
            if let Some(list) = CPU_LIST.as_ref() {
                assert!(wide.iter().all(|c| list.contains(c)));
            }
        }
    }
}
