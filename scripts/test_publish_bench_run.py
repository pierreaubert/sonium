"""Publication must preserve timing provenance and reject incomplete evidence."""
import json
from pathlib import Path
import tempfile
import unittest

import publish_bench_run as publish


class PublicationTests(unittest.TestCase):
    def test_failed_supplemental_output_is_not_published_as_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'ITERATIVE_PERF.json').write_text(json.dumps({'runs': [
                dict(scalar='f64', engine='failed-partial', n=123)]}))
            metadata = directory / 'supplemental.json'
            metadata.write_text(json.dumps({'runs': [dict(name='iterative', published=False, returncode=1)]}))
            self.assertEqual(publish.supplemental_cpu_tables(directory, 'sparse-solvers'), '')
            metadata.write_text(json.dumps({'runs': [dict(name='iterative', published=True, returncode=0)]}))
            self.assertIn('failed-partial', publish.supplemental_cpu_tables(directory, 'sparse-solvers'))

    def test_active_or_truncated_run_is_not_publishable(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / 'summary.json').write_text(json.dumps({'runs': []}))
            with self.assertRaisesRegex(ValueError, 'still active'):
                publish.load_run(directory)
            (directory / 'summary.json').write_text(json.dumps({
                'finished_at': '2026-10-24', 'runs': [], 'planned_runs': [{}]}))
            with self.assertRaisesRegex(ValueError, 'did not finish'):
                publish.load_run(directory)

    def test_failed_target_cannot_supply_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            snapshot = directory / 'snapshot.json'
            snapshot.write_text(json.dumps([{'estimates': {'median': {'point_estimate': 10}}}]))
            (directory / 'summary.json').write_text(json.dumps({
                'finished_at': '2026-10-24', 'runs': [{'returncode': 1, 'criterion_snapshot': str(snapshot)}]}))
            with self.assertRaisesRegex(ValueError, 'Failed benchmark'):
                publish.load_run(directory)

    def test_cuda_feature_build_without_native_measurement_is_not_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            snapshot = directory / 'snapshot.json'
            snapshot.write_text(json.dumps([dict(benchmark={'full_id': 'gemm/cpu/64'},
                                                    estimates={'median': {'point_estimate': 10}})]))
            (directory / 'summary.json').write_text(json.dumps({
                'finished_at': '2026-10-24', 'runs': [dict(returncode=0, backend='cuda',
                    bench='gpu', criterion_snapshot=str(snapshot))]}))
            with self.assertRaisesRegex(ValueError, 'no measured CUDA arm'):
                publish.load_run(directory)

    def test_completed_run_publishes_backend_specific_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            runs = []
            for backend, variant, value in [('cpu', 'sonium', 20), ('cuda', 'cublas', 30)]:
                snapshot = directory / f'{backend}.json'
                snapshot.write_text(json.dumps([dict(
                    benchmark=dict(group_id='gemm_f64', function_id=variant, value_str='64', full_id=f'gemm_f64/{variant}/64'),
                    estimates={'median': {'point_estimate': value}})]))
                log = directory / f'{backend}.log'
                log.write_text('measured')
                runs.append(dict(backend=backend, package='p', bench='blas3_compare', returncode=0,
                                 criterion_snapshot=str(snapshot), measurement_count=1, command=['cargo'],
                                 elapsed_s=1, log=str(log)))
            summary = dict(started_at='2026-10-24', finished_at='2026-10-24', provenance={'commit': 'abc'},
                           hardware={'system': 'Linux', 'machine': 'x86_64', 'cpu_threads': 128},
                           environment={}, runs=runs, planned_runs=runs)
            (directory / 'summary.json').write_text(json.dumps(summary))
            (directory / 'source-provenance.json').write_text('{"source_sha256": {}}')
            out = directory / 'published'
            self.assertEqual(publish.publish(directory, out), 2)
            self.assertIn('20.00 ns', (out / 'x86-blas.md').read_text())
            self.assertNotIn('30.00 ns', (out / 'x86-blas.md').read_text())
            self.assertIn('30.00 ns', (out / 'cuda-blas.md').read_text())
            artifact = json.loads((out / 'compute-solvers-benchmarks-2026-10-24.json').read_text())
            self.assertEqual(len(artifact['measurements']), 2)
            self.assertFalse(artifact['source_changed_since_capture'])
            self.assertEqual({row['id'] for row in artifact['website_platforms']}, {'x86', 'cuda'})
            for platform in artifact['website_platforms']:
                for section in platform['sections']:
                    for block in section['blocks']:
                        if 'columns' in block:
                            self.assertTrue(all(len(row['cells']) == len(block['columns']) == len(row['classes']) for row in block['rows']))

    def test_cuda_page_uses_only_cuda_invocation_rows(self):
        summary = dict(started_at='2026-10-24', finished_at='2026-10-24',
                       provenance={'commit': 'abc'}, hardware={'system': 'Linux', 'machine': 'x86_64', 'cpu_threads': 128},
                       environment={}, runs=[])
        identity = dict(group_id='gemm_f64', function_id='sonium', value_str='64', full_id='gemm_f64/sonium/64')
        rows = [dict(backend=backend, bench='blas3_compare', benchmark=identity,
                     estimates={'median': {'point_estimate': value}})
                for backend, value in [('cpu', 20), ('cuda', 30)]]
        _, page = publish.render_cell(summary, rows, 'cuda', 'blas', 'artifact.json')
        self.assertIn('30.00 ns', page)
        self.assertNotIn('20.00 ns', page)
        self.assertIn('host CPU arm', page)


if __name__ == '__main__':
    unittest.main()
