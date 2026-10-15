#!/usr/bin/env python3
"""Regenerate website galleries, QA pages, native reference data, and GPUI bundles."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

# Two-root layout: this script lives in the open-source `sonium/` submodule,
# while the demos and website live in the enclosing `all_of_sonium` checkout.
SONIUM = Path(__file__).resolve().parents[1]
ROOT = SONIUM.parent
# NOTE: SONIUM_CARGO, not CARGO. CARGO is reserved by the cargo ecosystem
# (cargo sets it for children; wrappers like mbx read it as the cargo
# binary), so exporting CARGO=mbx makes mbx invoke itself and hang.
CARGO = os.environ.get('SONIUM_CARGO', 'cargo')


def run(*command, cwd=SONIUM, env=None):
    print('+', ' '.join(map(str, command)), flush=True)
    if env is None:
        env = dict(os.environ, CARGO_TARGET_DIR=str(ROOT / 'target'))
    subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-galleries', action='store_true', help='Reuse the existing gallery during UI development')
    args = parser.parse_args()
    env = os.environ.copy()
    lock = tomllib.loads((ROOT / 'apps/sim-demo/Cargo.lock').read_text())
    version = next(p['version'] for p in lock['package'] if p['name'] == 'wasm-bindgen')
    speaker_lock = tomllib.loads((ROOT / 'apps/speaker-demo/Cargo.lock').read_text())
    speaker_version = next(p['version'] for p in speaker_lock['package'] if p['name'] == 'wasm-bindgen')
    if speaker_version != version:
        raise RuntimeError('The two demo workspaces must use the same wasm-bindgen version')

    def matches(tool):
        return tool and subprocess.check_output([str(tool), '--version'], text=True).strip() == f'wasm-bindgen {version}'

    bindgen = env.get('WASM_BINDGEN') or shutil.which('wasm-bindgen')
    if not matches(bindgen):
        if env.get('WASM_BINDGEN'):
            raise RuntimeError(f'WASM_BINDGEN must point to wasm-bindgen {version}')
        tool = ROOT / 'target/demo-tools/bin/wasm-bindgen'
        if not tool.exists() or not matches(tool):
            run(CARGO, 'install', 'wasm-bindgen-cli', '--version', version, '--locked', '--force', '--root', tool.parents[1])
        bindgen = str(tool)
    env['WASM_BINDGEN'] = bindgen
    if not args.skip_galleries:
        run(sys.executable, SONIUM / 'scripts/website_gallery.py', '--work-dir', ROOT / 'target/website-regenerate',
            '--out-dir', ROOT / 'website/public/gallery')
    run(sys.executable, ROOT / 'website/scripts/generate_qa_pages.py')
    run(sys.executable, ROOT / 'website/scripts/generate_perf_data.py')
    run(CARGO, 'build', '--release', '-p', 'sonium-sim-speaker', '--example', 'speaker_lab_worker')
    run(CARGO, 'build', '--release', '-p', 'sonium-demo-workers', '--bin', 'sonium-demo-workers')
    run(sys.executable, ROOT / 'apps/sim-demo/prepare_studies.py', '--worker', ROOT / 'target/release/sonium-demo-workers')
    run(CARGO, 'run', '--release', '-p', 'sonium-sim-speaker', '--example', 'export_demo', '--',
        ROOT / 'apps/speaker-demo/assets/study.json')
    for demo in ['speaker', 'sim']:
        demo_env = dict(env, CARGO_TARGET_DIR=str(ROOT / f'apps/{demo}-demo/target'))
        run('bash', ROOT / f'apps/{demo}-demo/build.sh', env=demo_env)
    run(sys.executable, ROOT / 'website/scripts/package_demos.py', '--worker', ROOT / 'target/release/sonium-demo-workers')
    run('npm', 'ci', cwd=ROOT / 'website')
    run('npm', 'run', 'build', cwd=ROOT / 'website')
    print('Website and six GPUI demos regenerated in website/dist/\nServe with just website-serve', flush=True)


if __name__ == '__main__':
    main()
