#!/usr/bin/env python3
"""Publish CPU/CUDA performance pages from one completed benchmark-run snapshot.

This never reads the shared Criterion cache: only the estimates captured after
individual invocations by bench_machine.py belong to the publication.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parents[1]
CELLS = {
    'blas': {'blas1_compare', 'blas2_compare', 'blas3_compare'},
    'lapack': {'lapack_compare'},
    'sparse-matrix': {'roofline_preflight', 'roofline_preflight_bench', 'sparse_x86',
                      'spmv_host_scale', 'spmv_bench', 'numa_first_touch_bench', 'two_socket_spmv_bench'},
    'dense-solvers': {'dense_lu_bench', 'dense_gemv_bench', 'low_rank_truncation_bench'},
    'sparse-solvers': {'backend_gmres_bench', 'block_gmres_accelerator', 'preconditioner_bench',
                       'solver_bench', 'sparse_lu_bench', 'multifrontal_blr_bench',
                       'selection_evidence_bench', 'petsc_compare_bench'},
}
TITLES = {'blas': 'Dense BLAS', 'lapack': 'Dense LAPACK', 'sparse-matrix': 'Sparse matrix',
          'dense-solvers': 'Dense solvers', 'sparse-solvers': 'Sparse solvers + preconditioners'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(directory):
    summary = json.loads((directory / 'summary.json').read_text())
    if not summary.get('finished_at'):
        raise ValueError('Benchmark run is still active; publication requires its terminal summary')
    if len(summary['runs']) != len(summary.get('planned_runs', summary['runs'])):
        raise ValueError('Benchmark inventory did not finish')
    records = []
    for run in summary['runs']:
        measurements = json.loads(Path(run['criterion_snapshot']).read_text())
        if run['returncode'] and measurements:
            raise ValueError('Failed benchmark cannot publish timing rows')
        if run['backend'] == 'cuda' and not run['returncode'] and not any(
                re.search(r'cuda|cublas|cusolver', item['benchmark'].get('full_id', ''), re.I)
                for item in measurements):
            raise ValueError(f'CUDA target produced no measured CUDA arm: {run["bench"]}')
        for measurement in measurements:
            value = measurement['estimates']['median']['point_estimate']
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Nonpositive or non-finite median')
            records.append(dict(backend=run['backend'], package=run['package'], bench=run['bench'],
                                **measurement))
    return summary, records


def duration(ns):
    for scale, unit in [(1e9, 's'), (1e6, 'ms'), (1e3, 'µs')]:
        if ns >= scale:
            return f'{ns / scale:.3f} {unit}'
    return f'{ns:.2f} ns'


def natural(value):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', value)]


def escape(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def tables(records):
    groups = collections.defaultdict(lambda: collections.defaultdict(dict))
    for row in records:
        identity = row['benchmark']
        group = identity.get('group_id', row['bench'])
        label = identity.get('value_str') or '—'
        variant = identity.get('function_id') or identity.get('full_id', 'measurement')
        key = row['bench'], group
        if variant in groups[key][label]:
            raise ValueError(f'Duplicate measurement identity in publication: {key}/{label}/{variant}')
        groups[key][label][variant] = row['estimates']['median']['point_estimate']
    output = []
    for (bench, group), labels in sorted(groups.items()):
        variants = sorted({variant for arms in labels.values() for variant in arms}, key=natural)
        output += [f'### `{escape(group)}`', '', f'Target: `{bench}`.', '',
                   '| shape | ' + ' | '.join(f'`{escape(v)}`' for v in variants) + ' |',
                   '|---|' + '---:|' * len(variants)]
        for label, arms in sorted(labels.items(), key=lambda item: natural(item[0])):
            output.append('| ' + escape(label) + ' | ' + ' | '.join(
                duration(arms[v]) if v in arms else '—' for v in variants) + ' |')
        output.append('')
    return '\n'.join(output)


def methodology(summary, artifact, title):
    hardware = summary['hardware']
    threads = summary['environment']
    generated = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    revision = summary['provenance']['commit'][:12]
    dirty = summary['provenance'].get('tracked_diff_sha256') not in (None, hashlib.sha256(b'').hexdigest())
    return '\n'.join([
        f'# {title}', '',
        f'_Generated {generated} from commit `{revision}{"-dirty" if dirty else ""}`._', '',
        f'_Measured {summary["started_at"]} through {summary["finished_at"]}._', '',
        '- status: measured' if all(not run['returncode'] for run in summary['runs']) else '- status: partial (failed targets retained below)',
        f'- Source commit: `{summary["provenance"]["commit"]}`; dirty source/config fingerprints are in the [run artifact]({artifact}).',
        f'- host: {hardware["system"]} {hardware["machine"]}; {hardware["cpu_threads"]} logical CPUs.',
        f'- Thread budgets: Rayon {threads.get("RAYON_NUM_THREADS")}, OpenBLAS {threads.get("OPENBLAS_NUM_THREADS")}, OMP {threads.get("OMP_NUM_THREADS")}.',
        ('- Criterion quick mode; targets execute sequentially. Times below are median estimates.' if any('--quick' in run.get('command', []) for run in summary['runs']) else '- Full harness defaults; targets execute sequentially. Times below are Criterion median estimates.'),
        '- Exact variant names are retained. A CUDA invocation also measures CPU reference arms; `sonium` by itself denotes the host CPU arm in the BLAS/LAPACK harnesses.',
        '- CUDA BLAS timings use prestaged buffers and include stream synchronization. LAPACK cuSOLVER arms include their harness upload/setup costs. Explicit `cuda_e2e` variants include transfer/allocation costs.',
        '- Thread budgets above are environment settings, not affinity limits. Harness-specific local pools can use other counts recorded in their labels and logs.',
        '- Solver variants may use different preconditioners and setup scopes. Timing success alone does not certify convergence; the legacy backend-GMRES harness does not assert every returned convergence flag.',
        '- Empty cells are unmeasured; no stale estimates or measurements from other suites are substituted.',
        f'- [Raw log and snapshot archive]({artifact.removesuffix(".json")}-logs.tar.gz).', '',
    ])


def custom_output(run):
    lines = Path(run['log']).read_text(errors='replace').splitlines()
    for index, line in enumerate(lines):
        if line.lstrip().startswith('Running benches/'):
            lines = lines[index + 1:]
            break
    return '\n'.join(lines).strip()


def render_cell(summary, records, backend, cell, artifact):
    platform = 'x86' if backend == 'cpu' else 'cuda'
    selected = [r for r in records if r['backend'] == backend and r['bench'] in CELLS[cell]]
    # Existing CUDA suite measures dense factorization, not an independent dense solve.
    if backend == 'cuda' and cell == 'dense-solvers':
        selected = [r for r in records if r['backend'] == backend and r['bench'] == 'lapack_compare']
    runs = [r for r in summary['runs'] if r['backend'] == backend and
            (r['bench'] in CELLS[cell] or (backend == 'cuda' and cell == 'dense-solvers' and r['bench'] == 'lapack_compare'))]
    title = f'{"x86-64 (Linux)" if backend == "cpu" else "CUDA (NVIDIA GPU, Linux)"} — {TITLES[cell]}'
    text = methodology(summary, artifact, title)
    if backend == 'cuda' and cell == 'dense-solvers':
        text += ('This rerun measures dense factorization through `lapack_compare`. '
                 'A fresh CUDA factor-plus-solve measurement is not supplied by this inventory.\n\n')
    failures = [r for r in runs if r['returncode']]
    if failures:
        text += 'Failed targets: ' + ', '.join(f'`{r["bench"]}` (exit {r["returncode"]})' for r in failures) + '.\n\n'
    text += '## Commands\n\n'
    for run in runs:
        import shlex
        text += '```sh\n' + shlex.join(run['command']) + '\n```\n\n'
    text += '## Results\n\n'
    text += tables(selected) if selected else 'No fresh Criterion measurements for this cell.\n'
    custom = [r for r in runs if not r.get('measurement_count') and not r['returncode']]
    if custom:
        text += '\n## Custom harness results\n\nThese harnesses use their own timing protocol; units and repetition counts follow their output.\n'
        for run in custom:
            text += f'\n### `{run["bench"]}`\n\n```text\n' + custom_output(run) + '\n```\n'
    return platform, text


def csv_table(path, backend=None):
    if not path.exists():
        return ''
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if backend:
        rows = [row for row in rows if row.get('backend', '').lower().startswith(backend)]
    if not rows:
        return ''
    columns = list(rows[0])
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join('---' for _ in columns) + ' |']
    lines += ['| ' + ' | '.join(escape(row.get(column, '')) for column in columns) + ' |' for row in rows]
    return '\n'.join(lines) + '\n'


def update_marker(path, body):
    begin, end = '<!-- BENCHMARK-RUN-BEGIN -->', '<!-- BENCHMARK-RUN-END -->'
    text = path.read_text()
    block = begin + '\n' + body.rstrip() + '\n' + end
    if begin in text:
        start, finish = text.index(begin), text.index(end) + len(end)
        text = text[:start] + block + text[finish:]
    else:
        text = text.rstrip() + '\n\n' + block + '\n'
    path.write_text(text)


def supplemental_cpu_tables(directory, cell):
    if cell not in {'dense-solvers', 'sparse-solvers'}:
        return ''
    text = ''
    metadata = directory / 'supplemental.json'
    completed = {run['name'] for run in json.loads(metadata.read_text()).get('runs', [])
                 if run.get('published') and run.get('returncode') == 0} if metadata.exists() else set()
    direct = directory / 'PERF.md'
    if 'direct' in completed and direct.exists():
        from perf_cells import split_direct_sections
        dense, sparse = split_direct_sections(direct.read_text())
        section = dense if cell == 'dense-solvers' else sparse
        if section:
            text += '\n## CPU direct factorization and RHS solve\n\nSeven timed repetitions after warm-up; this is a different scope from a factorization-only kernel. Full protocol: [PERF.md](PERF.md).\n\n' + section + '\n'
    iterative = directory / 'ITERATIVE_PERF.json'
    if cell == 'sparse-solvers' and 'iterative' in completed and iterative.exists():
        data = json.loads(iterative.read_text())
        largest = {}
        for row in data.get('runs', []):
            if row.get('scalar') != 'f64':
                continue
            key = tuple(str(row.get(field, '')) for field in ['engine', 'solver', 'preconditioner', 'matrix'])
            if key not in largest or row['n'] > largest[key]['n']:
                largest[key] = row
        text += f'\n## CPU iterative matrix\n\nThe full sweep contains {len(data.get("runs", []))} runs in [ITERATIVE_PERF.json](ITERATIVE_PERF.json). The table shows the largest f64 case per engine/solver/preconditioner/matrix, including convergence failures. It does not select the fastest result.\n\n'
        text += '| engine | solver | preconditioner | matrix | n | setup ms | solve ms | iterations | residual | converged |\n|---|---|---|---|---:|---:|---:|---:|---:|---|\n'
        def metric(value):
            return '—' if value is None else f'{value:.4g}'
        for key, row in sorted(largest.items()):
            values = [*key, row['n'], metric(row.get('setup_ms')), metric(row.get('solve_ms')),
                      row.get('iterations', '—'), metric(row.get('residual')), str(row.get('converged', 'unknown')).lower()]
            text += '| ' + ' | '.join(escape(value) for value in values) + ' |\n'
        validation = data.get('validation', {})
        text += f'\nReference validation: {validation.get("passed", 0)} passed, {validation.get("failed", 0)} failed; {validation.get("total_comparisons", 0)} comparisons. Preconditioner implementations and GMRES restart policies may differ between engines.\n'
    return text


def website_platforms(summary, records):
    """Use the website's table schema without depending on Markdown column names."""
    platforms = []
    for backend in sorted({run['backend'] for run in summary['runs']}):
        runs = [run for run in summary['runs'] if run['backend'] == backend]
        failures = [run['bench'] for run in runs if run['returncode']]
        status = f'{len(runs)} benchmark targets completed; {len(failures)} failed.'
        if failures:
            status += ' Failed targets: ' + ', '.join(failures) + '.'
        sections = [dict(heading='Run protocol and scope',
                         provenance='perf/compute-solvers-benchmarks-' + summary['started_at'][:10] + '.json',
                         blocks=[{'prose': status + ' Full harness settings and sequential execution. These timing results do not establish full backend or application coverage. FFT and full FEM/BEM application timings were not rerun in this compute/solver inventory.'}])]
        categories = {**CELLS, 'other': {row['bench'] for row in records} - set().union(*CELLS.values())}
        for cell, targets in categories.items():
            selected = [row for row in records if row['backend'] == backend and row['bench'] in targets]
            if not selected:
                continue
            groups = collections.defaultdict(lambda: collections.defaultdict(dict))
            for row in selected:
                identity = row['benchmark']
                group = row['bench'], identity.get('group_id', row['bench'])
                label = identity.get('value_str') or '—'
                variant = identity.get('function_id') or identity.get('full_id', 'measurement')
                groups[group][label][variant] = row['estimates']['median']['point_estimate']
            blocks = [{'prose': 'Criterion median wall time by exact harness variant. CPU reference arms remain CPU work even in a CUDA invocation. Setup and transfer scope follows the named harness; timing alone does not prove convergence.'}]
            for (bench, group), labels in sorted(groups.items()):
                variants = sorted({variant for arms in labels.values() for variant in arms}, key=natural)
                blocks.append({'prose': escape(group) + ' · ' + bench})
                rows = []
                for label, arms in sorted(labels.items(), key=lambda item: natural(item[0])):
                    values = [label] + [duration(arms[variant]) if variant in arms else '—' for variant in variants]
                    rows.append({'cells': values, 'classes': [None] * len(values)})
                blocks.append({'columns': ['shape', *variants], 'rows': rows})
            sections.append(dict(heading=TITLES.get(cell, 'Other compute and solver workloads'),
                                 provenance='perf/compute-solvers-benchmarks-' + summary['started_at'][:10] + '.json', blocks=blocks))
        platforms.append(dict(
            id='x86' if backend == 'cpu' else 'cuda',
            title='Linux x86-64' if backend == 'cpu' else 'NVIDIA CUDA',
            lede=f'{summary["hardware"]["cpu_threads"]} logical CPU threads · Rayon/OpenBLAS budgets {summary["environment"].get("RAYON_NUM_THREADS")}/{summary["environment"].get("OPENBLAS_NUM_THREADS")} · measured {summary["started_at"][:10]} @ {summary["provenance"]["commit"][:8]}',
            sections=sections))
    return platforms


def publish(directory, out):
    summary, records = load_run(directory)
    out.mkdir(parents=True, exist_ok=True)
    date = summary['started_at'][:10]
    artifact_name = f'compute-solvers-benchmarks-{date}.json'
    source = json.loads((directory / 'source-provenance.json').read_text())
    changed = [path for path, expected in source['source_sha256'].items()
               if not (ROOT / path).exists() or digest(ROOT / path) != expected]
    runs = []
    for run in summary['runs']:
        item = dict(run, log_sha256=digest(Path(run['log'])))
        item['log_text'] = Path(run['log']).read_text(errors='replace') if not run['measurement_count'] else None
        runs.append(item)
    archive = out / f'compute-solvers-benchmarks-{date}-logs.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        for path in sorted(directory.rglob('*')):
            if path.is_file() and path.suffix in {'.json', '.log', '.csv', '.md'}:
                bundle.add(path, arcname=path.relative_to(directory))
    artifact = dict(schema='sonium.compute-solvers-benchmarks/v1', summary={**summary, 'runs': runs},
                    source=source, source_changed_since_capture=changed,
                    logs_archive=dict(path=archive.name, sha256=digest(archive)),
                    gpu_before=(directory / 'gpu-before.csv').read_text() if (directory / 'gpu-before.csv').exists() else None,
                    gpu_after=(directory / 'gpu-after.csv').read_text() if (directory / 'gpu-after.csv').exists() else None,
                    gpu_at_cuda_start=(directory / 'gpu-at-cuda-start.csv').read_text() if (directory / 'gpu-at-cuda-start.csv').exists() else None,
                    measurements=records,
                    supplemental=json.loads((directory / 'supplemental.json').read_text()) if (directory / 'supplemental.json').exists() else None)
    artifact['primary_runner_at_launch'] = json.loads((directory / 'runner-initial.json').read_text()) if (directory / 'runner-initial.json').exists() else None
    artifact['harness_sha256'] = {str(path.relative_to(ROOT)): digest(path) for path in [
        ROOT / 'scripts/bench_machine.py', ROOT / 'scripts/publish_bench_run.py',
        ROOT / 'scripts/refresh_compute_solver_reports.py']}
    artifact['website_platforms'] = website_platforms(summary, records)
    artifact['supplemental_outputs'] = {
        run['name']: dict(sha256=digest(Path(run['output'])), text=Path(run['output']).read_text())
        for run in (artifact.get('supplemental') or {}).get('runs', []) if Path(run['output']).exists()
    }
    (out / artifact_name).write_text(json.dumps(artifact, indent=2) + '\n')
    for backend in sorted({run['backend'] for run in summary['runs']}):
        for cell in CELLS:
            platform, text = render_cell(summary, records, backend, cell, artifact_name)
            if backend == 'cpu':
                text += supplemental_cpu_tables(directory, cell)
            if cell == 'sparse-solvers':
                for name, label in [('CUDA_PRECONDITIONERS.csv', 'Preconditioner setup/apply scaling'),
                                    ('GMRES_IR_SCALING.csv', 'Mixed-precision GMRES-IR scaling')]:
                    extra = csv_table(directory / name, backend)
                    if extra:
                        text += f'\n## {label}\n\nRaw seconds and convergence flags from [{name}]({name}). Failed convergence rows remain visible and do not establish time to tolerance.\n\n' + extra
            (out / f'{platform}-{cell}.md').write_text(text)
    report = methodology(summary, artifact_name, f'Compute and solver benchmark rerun — {date}')
    report += '## Target inventory\n\n| backend | package | target | exit | fresh measurements | elapsed |\n|---|---|---|---:|---:|---:|\n'
    for run in summary['runs']:
        report += f'| {run["backend"]} | {run["package"]} | {run["bench"]} | {run["returncode"]} | {run["measurement_count"]} | {run["elapsed_s"]:.1f} s |\n'
    report += '\n## CPU Criterion estimates\n\n' + tables([r for r in records if r['backend'] == 'cpu'])
    report += '\n## CUDA invocation Criterion estimates\n\n' + tables([r for r in records if r['backend'] == 'cuda'])
    if summary.get('repair_history'):
        import shlex
        report += '\n## Corrected invocations\n\n'
        for repair in summary['repair_history']:
            report += repair['reason'] + '. The initial invocation remains in the JSON history and raw archive.\n\n'
            report += '```sh\n' + shlex.join(repair['replacement']['command']) + '\n```\n\n'
    report += '\n## Custom harness results\n\n'
    for run in summary['runs']:
        if not run['returncode'] and not run['measurement_count']:
            report += f'\n### {run["backend"]} `{run["package"]}/{run["bench"]}`\n\n```text\n' + custom_output(run) + '\n```\n'
    report += '\n## Supplemental solver reports\n\n'
    supplement = artifact.get('supplemental') or {}
    if supplement:
        report += f'Measured {supplement["started_at"]} through {supplement.get("finished_at", "still active")}.\n\n'
    for run in supplement.get('runs', []):
        report += f'- `{run["name"]}`: exit {run["returncode"]}, {run["elapsed_s"]:.1f} s; published={run["published"]}.\n'
    for name, label in [('CUDA_PRECONDITIONERS.csv', 'CPU/CUDA preconditioner scaling'),
                        ('GMRES_IR_SCALING.csv', 'CPU/CUDA mixed-precision scaling')]:
        table = csv_table(directory / name)
        if table:
            report += f'\n### {label}\n\n[{name}]({name}) retains raw seconds, iteration counts and convergence flags.\n\n' + table
    (out / f'compute-solvers-benchmarks-{date}.md').write_text(report)
    if out.resolve() == (ROOT / 'perf').resolve():
        for run in summary['runs']:
            if run['bench'] == 'selection_evidence_bench' and run['returncode'] == 0:
                table = '\n'.join(line for line in Path(run['log']).read_text().splitlines() if line.startswith('|'))
                if table:
                    selection = ROOT / 'crates/sonium-solvers/SELECTION.md'
                    previous = selection.read_text()
                    if '## Method' in previous and '## Results' in previous:
                        method = previous.split('## Method', 1)[1].split('## Results', 1)[0].strip()
                        method = method.replace('All cases below converged.', 'Convergence flags are retained in each row.')
                    else:
                        method = ('Public facade Auto/explicit-strategy solves with manufactured RHS. One warm-up and five timed solves; setup is a separate five-build median, solve time is total minus setup. Memory is a lower-bound model rather than RSS. The forced-large arms lower dispatch thresholds for contract-scale fixtures.')
                    selection.write_text(
                        f'# High-Level Solver Selection Evidence\n\nMeasured {date}; [full run provenance](../../perf/{artifact_name}).\n\n'
                        '## Method\n\n' + method + '\n\n## Results\n\n' + table + '\n')
        for run in supplement.get('runs', []):
            if run['published'] and run['name'] in {'eigen', 'preconditioners'}:
                update_marker(Path(run['destination']),
                              f'## Measurement provenance\n\nMeasured {date} on the CPU described in the '
                              f'[full run report](../../perf/compute-solvers-benchmarks-{date}.md). '
                              'Source fingerprints, thread budgets and the raw generated output are retained in the run artifact.')
            if run['published'] and run['name'] == 'direct':
                update_marker(Path(run['destination']),
                              f'## CPU and CUDA report scope\n\nThis factor-plus-solve report uses the CPU direct harness. '
                              f'[The {date} CPU/CUDA inventory](compute-solvers-benchmarks-{date}.md) also measures '
                              'CUDA factorization kernels; their setup and transfer scopes differ from this harness.')
        for path, prefix in [(ROOT / 'README.md', 'perf/'),
                             (ROOT / 'crates/sonium-compute/README.md', '../../perf/'),
                             (ROOT / 'crates/sonium-solvers/README.md', '../../perf/'),
                             (ROOT / 'perf/README.md', '')]:
            update_marker(path, f'## Latest compute and solver measurements\n\n'
                          f'The {date} CPU/CUDA rerun covers {len(summary["runs"])} benchmark targets. '
                          f'[Results and methodology]({prefix}compute-solvers-benchmarks-{date}.md) '
                          f'and [raw measurements]({prefix}{artifact_name}) retain exact commands, thread budgets and timing scopes. '
                          'Native execution, convergence, and performance coverage remain separate claims.')
        (ROOT / 'crates/sonium-solvers/PERF.md').write_text(
            '# Direct Solver Performance\n\nThe current generated report is [perf/PERF.md](../../perf/PERF.md). '
            f'The latest CPU/CUDA benchmark inventory and measurements are [here](../../perf/compute-solvers-benchmarks-{date}.md). '
            'Per-platform performance records live in [perf/](../../perf/README.md).\n')
        update_marker(ROOT / 'perf/CUDA-COVERAGE-PLAN.md',
                      f'## Latest compute/solver benchmark rerun ({date})\n\n'
                      f'[Full results](compute-solvers-benchmarks-{date}.md) and [raw evidence]({artifact_name}) '
                      'record CPU and CUDA timing scopes, exact variants and convergence limitations. '
                      'These measurements do not close unreviewed solver, BEM, FEM or application workflows.')
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'perf')
    args = parser.parse_args()
    count = publish(args.directory.resolve(), args.out_dir)
    print(f'Published {count} fresh measurements and ten CPU/CUDA cell pages.')


if __name__ == '__main__':
    main()
