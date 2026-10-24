#!/usr/bin/env python3
"""Run native workspace benchmarks, then available CUDA benchmark arms."""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
CUDA_GATE = re.compile(r'feature\s*=\s*"cuda"')


def library(name):
    """Load a library through the same runtime search path as the benchmarks."""
    path = ctypes.util.find_library(name)
    if not path:
        raise OSError(f'{name}: shared library not found')
    return ctypes.CDLL(path)


def capabilities():
    info = dict(system=platform.system(), machine=platform.machine(),
                cpu_threads=os.cpu_count() or 1, openblas=False, cuda=False)
    try:
        library('openblas')
        info['openblas'] = True
    except OSError:
        pass
    try:
        driver = library('cuda')
        driver.cuInit.argtypes = [ctypes.c_uint]
        driver.cuInit.restype = ctypes.c_int
        driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        driver.cuDeviceGetCount.restype = ctypes.c_int
        count = ctypes.c_int()
        status = driver.cuInit(0)
        if status:
            raise OSError(f'CUDA driver initialization failed (code {status})')
        if driver.cuDeviceGetCount(ctypes.byref(count)) or count.value < 1:
            raise OSError('No visible CUDA device')
        for name in ('cublas', 'cusparse', 'cusolver', 'nvrtc'):
            library(name)
        info.update(cuda=True, cuda_devices=count.value)
    except (OSError, AttributeError) as error:
        info['cuda_reason'] = str(error)
    return info


def inventory(cargo):
    result = subprocess.run([cargo, 'metadata', '--locked', '--no-deps', '--format-version', '1'],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def plan(metadata, hardware, cargo, selected='auto', packages=(), benches=(), quick=False, filt=None):
    runs, skipped = [], []
    members = set(metadata['workspace_members'])
    for backend in ('cpu', 'cuda'):
        if selected not in ('auto', backend) or (backend == 'cuda' and not hardware['cuda']):
            continue
        for package in metadata['packages']:
            if package['id'] not in members or (packages and package['name'] not in packages):
                continue
            for target in package['targets']:
                if 'bench' not in target['kind'] or (benches and target['name'] not in benches):
                    continue
                label = dict(backend=backend, package=package['name'], bench=target['name'])
                # Cargo targets may legally compile to empty platform stubs.
                if target['name'] == 'fft_vs_accelerate' and hardware['system'] != 'Darwin':
                    if backend == 'cpu':
                        skipped.append(dict(**label, reason='Apple Accelerate requires macOS'))
                    continue
                if target['name'] == 'sparse_x86' and hardware['machine'].lower() not in ('x86_64', 'amd64'):
                    if backend == 'cpu':
                        skipped.append(dict(**label, reason='x86-specific benchmark'))
                    continue
                source = Path(target['src_path']).read_text()
                features = set(target.get('required-features', []))
                if backend == 'cpu' and 'cuda' in features:
                    continue
                if 'metal' in features and hardware['system'] != 'Darwin':
                    skipped.append(dict(**label, reason='Metal requires macOS'))
                    continue
                if backend == 'cuda':
                    if 'cuda' not in package['features'] or not ('cuda' in features or CUDA_GATE.search(source)):
                        continue
                    features.add('cuda')
                if hardware['openblas'] and 'bench-openblas' in package['features']:
                    features.add('bench-openblas')
                command = [cargo, 'bench', '--locked', '-p', package['name'], '--bench', target['name']]
                if features:
                    command += ['--features', ','.join(sorted(features))]
                if 'criterion_main!' in source:
                    # Separate retained CPU/CUDA baselines despite shared benchmark names.
                    command += ['--', '--save-baseline', backend]
                    if quick:
                        command.append('--quick')
                    if filt:
                        command.append(filt)
                elif target['name'] == 'spmv_host_scale':
                    # Cargo appends --bench; this custom main reads argv[1] as a case filter.
                    command += ['--', 'all']
                runs.append(dict(**label, command=command, manifest_dir=str(Path(package.get('manifest_path', ROOT / 'Cargo.toml')).parent), source=str(target['src_path'])))
    return runs, skipped


def fresh_criterion(run, started_ns):
    """Snapshot only baseline estimates written by this exact benchmark invocation."""
    target = Path(os.environ.get('CARGO_TARGET_DIR', ROOT / 'target'))
    if not target.is_absolute():
        target = ROOT / target
    roots = {target / 'criterion', Path(run.get('manifest_dir', ROOT)) / 'target/criterion'}
    if os.environ.get('CRITERION_HOME'):
        roots.add(Path(os.environ['CRITERION_HOME']))
    measurements = []
    for root in roots:
        if not root.exists():
            continue
        for estimate in sorted(root.rglob(f"{run['backend']}/estimates.json")):
            if estimate.stat().st_mtime_ns < started_ns:
                continue
            benchmark = estimate.with_name('benchmark.json')
            if not benchmark.exists():
                continue
            identity = json.loads(benchmark.read_text())
            measurements.append(dict(
                benchmark=identity,
                estimates=json.loads(estimate.read_text()),
                path=str(estimate),
            ))
    return measurements


def revision():
    """Record source provenance without changing the current checkout."""
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    diff = subprocess.run(['git', 'diff', 'HEAD', '--binary'], cwd=ROOT, capture_output=True)
    return dict(commit=result.stdout.strip(), tracked_diff_sha256=hashlib.sha256(diff.stdout).hexdigest())


def execute(runs, directory, env, hardware, skipped):
    summary = dict(hardware=hardware, skipped=skipped, runs=[], planned_runs=runs,
                   provenance=revision(), started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   environment={key: env.get(key) for key in ('RAYON_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'SONIUM_REQUIRE_CUDA')})
    path = directory / 'summary.json'
    path.write_text(json.dumps(summary, indent=2) + '\n')
    for index, run in enumerate(runs, 1):
        log = directory / f"{index:02d}-{run['backend']}-{run['package']}-{run['bench']}.log"
        command = shlex.join(run['command'])
        print(f'[{index}/{len(runs)}] {command}\n  log: {log}', flush=True)
        started_ns = time.time_ns()
        start = time.monotonic()
        with log.open('w') as stream:
            stream.write(command + '\n\n')
            stream.flush()
            result = subprocess.run(run['command'], cwd=ROOT, env=env,
                                    stdout=stream, stderr=subprocess.STDOUT)
        measurements = fresh_criterion(run, started_ns) if result.returncode == 0 else []
        snapshot = log.with_suffix('.criterion.json')
        snapshot.write_text(json.dumps(measurements, indent=2) + '\n')
        record = dict(run, returncode=result.returncode, elapsed_s=time.monotonic() - start, log=str(log),
                      criterion_snapshot=str(snapshot), measurement_count=len(measurements))
        summary['runs'].append(record)
        path.write_text(json.dumps(summary, indent=2) + '\n')
        print('  PASS' if result.returncode == 0 else f'  FAIL ({result.returncode}); see {log}', flush=True)
    summary['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    path.write_text(json.dumps(summary, indent=2) + '\n')
    failures = [run for run in summary['runs'] if run['returncode']]
    print(f"Benchmark summary: {len(runs) - len(failures)} passed, {len(failures)} failed, {len(skipped)} skipped")
    print(f'Results: {path}')
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--dry-run', action='store_true', help='Detect hardware and list commands without benchmarking')
    parser.add_argument('--package', action='append', default=[], help='Restrict to a package (repeatable)')
    parser.add_argument('--bench', action='append', default=[], help='Restrict to a benchmark target (repeatable)')
    parser.add_argument('--quick', action='store_true', help='Use Criterion quick mode; custom harnesses keep their normal workload')
    parser.add_argument('--filter', help='Criterion benchmark-name filter')
    args = parser.parse_args()
    hardware = capabilities()
    print(f"Host: {hardware['system']} {hardware['machine']}; CUDA: " +
          ('available' if hardware['cuda'] else f"unavailable ({hardware['cuda_reason']})"), flush=True)
    if args.backend == 'cuda' and not hardware['cuda']:
        parser.error('The CUDA backend was requested but is unavailable')
    metadata = inventory(os.environ.get('SONIUM_CARGO', 'cargo'))
    known = {p['name'] for p in metadata['packages']}
    unknown = set(args.package) - known
    if unknown:
        parser.error(f'Unknown packages: {sorted(unknown)}')
    runs, skipped = plan(metadata, hardware, os.environ.get('SONIUM_CARGO', 'cargo'), args.backend,
                         args.package, args.bench, args.quick or os.environ.get('PERF_QUICK') == '1', args.filter)
    for skip in skipped:
        print(f"SKIP {skip['package']}/{skip['bench']}: {skip['reason']}")
    if not runs:
        parser.error('No runnable benchmarks match this selection')
    print(f"Plan: {sum(r['backend'] == 'cpu' for r in runs)} CPU targets, "
          f"{sum(r['backend'] == 'cuda' for r in runs)} CUDA targets", flush=True)
    if args.dry_run:
        for run in runs:
            print(shlex.join(run['command']))
        return 0
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S-')
    base = ROOT / 'target/bench-runs'
    base.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=stamp, dir=base))
    env = dict(os.environ)
    threads = str(min(16, hardware['cpu_threads']))
    for variable in ('RAYON_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env.setdefault(variable, threads)
    return execute(runs, directory, env, hardware, skipped)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('Benchmark run interrupted; completed logs remain in target/bench-runs/', file=sys.stderr)
        sys.exit(130)
