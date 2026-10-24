"""Hardware selection, full target coverage, and failure reporting contracts."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch
import bench_machine as bench


class MachineBenchTests(unittest.TestCase):
    def test_workspace_plan_covers_cpu_and_cuda_without_platform_stubs(self):
        with tempfile.TemporaryDirectory() as tmp:
            sources = {}
            for name, source in [('cpu', 'criterion_main!(benches);'),
                                 ('gpu', '#[cfg(feature = "cuda")] fn gpu() {}\ncriterion_main!(benches);'),
                                 ('fft_vs_accelerate', 'criterion_main!(benches);')]:
                path = Path(tmp) / (name + '.rs')
                path.write_text(source)
                sources[name] = dict(name=name, src_path=str(path), kind=['bench'])
            metadata = dict(workspace_members=['member'], packages=[dict(
                id='member', name='package', features={'cuda': [], 'bench-openblas': []},
                targets=list(sources.values()))])
            hardware = dict(system='Linux', machine='x86_64', cuda=True, openblas=True)
            runs, skipped = bench.plan(metadata, hardware, 'cargo', quick=True)
            self.assertEqual([(r['backend'], r['bench']) for r in runs],
                             [('cpu', 'cpu'), ('cpu', 'gpu'), ('cuda', 'gpu')])
            self.assertEqual(len(skipped), 1)
            self.assertIn('bench-openblas,cuda', runs[-1]['command'])
            self.assertIn('--quick', runs[0]['command'])
            hardware['cuda'] = False
            runs, _ = bench.plan(metadata, hardware, 'cargo')
            self.assertEqual(len(runs), 2)
            self.assertTrue(all(r['backend'] == 'cpu' for r in runs))

    def test_required_cuda_target_does_not_need_redundant_source_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'required.rs'
            source.write_text('criterion_main!(benches);')
            metadata = dict(workspace_members=['p'], packages=[dict(
                id='p', name='p', features={'cuda': []}, targets=[dict(
                    name='required', src_path=str(source), kind=['bench'],
                    **{'required-features': ['cuda']})])])
            hardware = dict(system='Linux', machine='x86_64', cuda=True, openblas=False)
            runs, _ = bench.plan(metadata, hardware, 'cargo')
            self.assertEqual([(r['backend'], r['bench']) for r in runs], [('cuda', 'required')])

    def test_snapshots_exclude_stale_and_other_backend_estimates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'target/criterion/group'
            for label, stamp in [('fresh/cpu', 200), ('stale/cpu', 50), ('other/cuda', 200)]:
                directory = root / label
                directory.mkdir(parents=True)
                estimate = directory / 'estimates.json'
                estimate.write_text('{"median": {"point_estimate": 123}}')
                os.utime(estimate, ns=(stamp, stamp))
                (directory / 'benchmark.json').write_text('{"full_id": "group/' + label + '"}')
            with patch.dict(os.environ, {'CARGO_TARGET_DIR': str(Path(tmp) / 'target')}, clear=True):
                measurements = bench.fresh_criterion(dict(backend='cpu', manifest_dir=tmp), 100)
            self.assertEqual(len(measurements), 1)
            self.assertEqual(measurements[0]['benchmark']['full_id'], 'group/fresh/cpu')

    def test_custom_spmv_filter_is_not_cargos_implicit_bench_argument(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'spmv_host_scale.rs'
            source.write_text('fn main() { let filter = std::env::args().nth(1); }')
            metadata = dict(workspace_members=['p'], packages=[dict(id='p', name='p', features={},
                targets=[dict(name='spmv_host_scale', src_path=str(source), kind=['bench'])])])
            hardware = dict(system='Linux', machine='x86_64', cuda=False, openblas=False)
            runs, _ = bench.plan(metadata, hardware, 'cargo')
            self.assertEqual(runs[0]['command'][-2:], ['--', 'all'])

    def test_cuda_probe_requires_live_device_and_vendor_libraries(self):
        driver = Mock()
        driver.cuInit.return_value = 0
        def count(pointer):
            pointer._obj.value = 1
            return 0
        driver.cuDeviceGetCount.side_effect = count
        with patch.object(bench, 'library', return_value=driver):
            self.assertTrue(bench.capabilities()['cuda'])
        driver.cuInit.return_value = 100
        with patch.object(bench, 'library', return_value=driver):
            capabilities = bench.capabilities()
            self.assertFalse(capabilities['cuda'])
            self.assertIn('100', capabilities['cuda_reason'])
        driver.cuInit.return_value = 0
        def load(name):
            if name == 'cusolver':
                raise OSError('missing cuSOLVER')
            return driver
        with patch.object(bench, 'library', side_effect=load):
            self.assertFalse(bench.capabilities()['cuda'])

    def test_failure_does_not_prevent_remaining_benchmarks(self):
        runs = [dict(backend=backend, package='package', bench='bench', command=['cargo', 'bench'])
                for backend in ['cpu', 'cuda']]
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
                patch.object(bench, 'revision', return_value={}), \
                patch.object(bench, 'fresh_criterion', return_value=[]), \
                patch.object(bench.subprocess, 'run', side_effect=[
                    subprocess.CompletedProcess([], 1), subprocess.CompletedProcess([], 0)]) as execute:
            status = bench.execute(runs, Path(tmp), {}, {}, [])
            self.assertEqual(status, 1)
            self.assertEqual(execute.call_count, 2)
            summary = json.loads((Path(tmp) / 'summary.json').read_text())
            self.assertEqual([r['returncode'] for r in summary['runs']], [1, 0])
            self.assertTrue(all(Path(r['log']).exists() for r in summary['runs']))


if __name__ == '__main__':
    unittest.main()
