#!/usr/bin/env python3
"""Finish report generation after a selected compute/solver benchmark run.

Waits for the exact existing bench_machine process when requested. Supplemental
CPU reports run sequentially after all CPU/CUDA Criterion measurements finish.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.wait_pid:
        proc = Path(f'/proc/{args.wait_pid}/cmdline')
        while proc.exists():
            command = proc.read_bytes()
            if b'scripts/bench_machine.py' not in command:
                break
            time.sleep(5)
    primary = json.loads((directory / 'summary.json').read_text())
    if not primary.get('finished_at'):
        raise SystemExit('Primary benchmark process is terminal without a completed run summary')
    env = dict(os.environ)
    for key in ['RAYON_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        env[key] = primary['environment'][key]
    env.update(DIRECT_PERF_DENSE_SIZES='128,256,512,1024,2048',
               DIRECT_PERF_SPARSE_GRIDS='16,24,32,40,64', DIRECT_PERF_REPEATS='7')
    env.pop('ITERATIVE_PERF_QUICK', None)
    # Repair earlier runner versions that let Cargo's --bench flag become a
    # custom case filter. Preserve the original invocation in the evidence.
    for index, original in enumerate(primary['runs']):
        if original['bench'] != 'spmv_host_scale' or original['backend'] != 'cpu':
            continue
        original_text = Path(original['log']).read_text()
        if '"bench":"spmv_host_scale"' in original_text:
            continue
        import bench_machine
        repair = {key: original[key] for key in ['backend', 'package', 'bench', 'command', 'manifest_dir', 'source']}
        repair['command'] = [*repair['command'], '--', 'all']
        repair_dir = directory / 'repair-spmv-host-scale'
        repair_dir.mkdir(exist_ok=True)
        status = bench_machine.execute([repair], repair_dir, env, primary['hardware'], [])
        repaired = json.loads((repair_dir / 'summary.json').read_text())['runs'][0]
        cases = [json.loads(line) for line in Path(repaired['log']).read_text().splitlines()
                 if line.startswith('{"bench":"spmv_host_scale"')]
        if status or {case['case'] for case in cases} != {'small', 'medium', 'large'}:
            raise SystemExit('Custom SpMV repair failed to time all three correctness-gated cases')
        primary.setdefault('repair_history', []).append(dict(original=original, replacement=repaired,
            reason='Cargo appended --bench was treated as a case filter; initial invocation timed no cases'))
        primary['runs'][index] = repaired
        primary['primary_finished_at'] = primary['finished_at']
        primary['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (directory / 'summary.json').write_text(json.dumps(primary, indent=2) + '\n')
    # Build the local reference from installed PETSc headers/libraries only.
    # The repository may contain a reference executable for another platform.
    petsc_flags = subprocess.run(['pkg-config', '--cflags', '--libs', 'PETSc'], capture_output=True, text=True)
    vendor = dict(available=False)
    if petsc_flags.returncode == 0 and shutil.which('mpicc'):
        reference = directory / 'petsc_reference'
        command = ['mpicc', '-O2', '-o', str(reference),
                   str(ROOT / 'crates/sonium-solvers/benches/ffi/petsc_reference.c'),
                   *shlex.split(petsc_flags.stdout)]
        with (directory / 'petsc-build.log').open('w') as stream:
            built = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        vendor = dict(available=built.returncode == 0, command=command, returncode=built.returncode)
        if vendor['available']:
            env['ITERATIVE_PERF_PETSC'] = str(reference)
            vendor['path'] = str(reference)
    jobs = [
        ('direct', ['bash', 'scripts/direct_solver_perf.sh', str(directory / 'PERF.md')], 'PERF.md', ROOT / 'perf/PERF.md'),
        ('iterative', ['bash', 'scripts/iterative_solver_perf.sh', str(directory / 'ITERATIVE_PERF.json')], 'ITERATIVE_PERF.json', ROOT / 'perf/ITERATIVE_PERF.json'),
        ('eigen', ['bash', 'scripts/eigen_breakdown.sh', str(directory / 'EIGEN_BREAKDOWN.md')], 'EIGEN_BREAKDOWN.md', ROOT / 'crates/sonium-solvers/EIGEN_BREAKDOWN.md'),
        ('preconditioners', ['bash', 'scripts/precond_tradeoff.sh', str(directory / 'PRECOND_TRADEOFF.md')], 'PRECOND_TRADEOFF.md', ROOT / 'crates/sonium-solvers/PRECOND_TRADEOFF.md'),
        ('cuda-preconditioners', ['cargo', 'run', '--release', '-p', 'sonium-solvers', '--features', 'cli,cuda', '--bin', 'precond-cuda-scaling', '--', '--csv', str(directory / 'CUDA_PRECONDITIONERS.csv')], 'CUDA_PRECONDITIONERS.csv', ROOT / 'perf/CUDA_PRECONDITIONERS.csv'),
        ('gmres-ir', ['cargo', 'run', '--release', '-p', 'sonium-solvers', '--features', 'cli,cuda', '--bin', 'gmres-ir-scaling', '--', '--cpu-size-limit', '0', '--csv', str(directory / 'GMRES_IR_SCALING.csv')], 'GMRES_IR_SCALING.csv', ROOT / 'perf/GMRES_IR_SCALING.csv'),
    ]
    summary = dict(started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), petsc_reference=vendor,
                   environment={key: env.get(key) for key in ['RAYON_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'ITERATIVE_PERF_PETSC', 'DIRECT_PERF_REPEATS', 'DIRECT_PERF_DENSE_SIZES', 'DIRECT_PERF_SPARSE_GRIDS']}, runs=[])
    for name, command, output, destination in jobs:
        print(f'Starting supplemental {name} report', flush=True)
        start = time.monotonic()
        log = directory / f'supplemental-{name}.log'
        with log.open('w') as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        generated = directory / output
        success = result.returncode == 0 and generated.exists()
        if success:
            shutil.copyfile(generated, destination)
        record = dict(name=name, command=command, returncode=result.returncode,
                      output=str(generated), published=success, destination=str(destination),
                      elapsed_s=time.monotonic()-start, log=str(log))
        summary['runs'].append(record)
        (directory / 'supplemental.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(f'{name}: {"PASS" if success else "FAIL"} ({record["elapsed_s"]:.1f}s)', flush=True)
    summary['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (directory / 'supplemental.json').write_text(json.dumps(summary, indent=2) + '\n')
    result = subprocess.run(['python3', 'scripts/publish_bench_run.py', str(directory)], cwd=ROOT, env=env)
    return int(result.returncode != 0 or any(not row['published'] for row in summary['runs'])
               or any(row['returncode'] for row in primary['runs']))


if __name__ == '__main__':
    raise SystemExit(main())
