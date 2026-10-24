"""Fail-closed checks for the bounded reviewed CUDA workflow denominator."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from check_cuda_workflow_registry import REGISTRY, ROOT, source_block, validate


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.document = json.loads((ROOT / REGISTRY).read_text())

    def test_current_review(self):
        self.assertEqual(validate(ROOT, self.document), (40, 40))

    def test_missing_and_duplicate_rows(self):
        for mutate in (lambda rows: rows.pop(), lambda rows: rows.append(copy.deepcopy(rows[0]))):
            document = copy.deepcopy(self.document)
            mutate(document['workflows'])
            with self.assertRaises(ValueError):
                validate(ROOT, document)

    def test_dispatch_must_agree_with_source(self):
        row = self.document['workflows'][0]
        row['dispatch'] = 'explicit_rejection'
        with self.assertRaisesRegex(ValueError, 'dispatch disagrees'):
            validate(ROOT, self.document)

    def test_stale_source_and_missing_evidence(self):
        document = copy.deepcopy(self.document)
        document['reviewed_source_sha256'][next(iter(document['reviewed_source_sha256']))] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'review stale'):
            validate(ROOT, document)
        document = copy.deepcopy(self.document)
        document['evidence'][0]['path'] = 'perf/nonexistent-workflow-proof.json'
        with self.assertRaisesRegex(ValueError, 'missing or nonlocal'):
            validate(ROOT, document)

    def test_completion_claim_is_not_derived_from_dispatch(self):
        self.document['workflows'][0]['complete'] = True
        with self.assertRaisesRegex(ValueError, 'no fully closed'):
            validate(ROOT, self.document)

    def test_native_claim_requires_declared_evidence(self):
        row = next(r for r in self.document['workflows'] if r['dispatch'] == 'retained_cuda')
        row['current_native_evidence'] = 'perf/undeclared-evidence.json'
        with self.assertRaisesRegex(ValueError, 'declared evidence'):
            validate(ROOT, self.document)

    def test_balanced_source_ignores_comment_and_string_braces(self):
        result = source_block('fn validate_strategy() { /* } */ let s = "}"; if true { ok() } }',
                              r'fn\s+validate_strategy\b')
        self.assertIn('if true { ok() }', result)

    def test_added_strategy_requires_new_rows(self):
        # A fixture copied from the real review lets source drift be explicitly
        # re-reviewed while checking that a newly added enum variant cannot hide.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in self.document['reviewed_source_sha256']:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((ROOT / name).read_bytes())
            # This test isolates denominator validation, not native evidence.
            for row in self.document['workflows']:
                if row['dispatch'] == 'retained_cuda':
                    row['correctness'] = 'pending_current_native_run'
            enum_path = root / self.document['strategy_enum']['path']
            text = enum_path.read_text().replace('pub enum LinearSolverType {',
                                                 'pub enum LinearSolverType { NewStrategy,', 1)
            enum_path.write_text(text)
            import hashlib
            self.document['reviewed_source_sha256'][self.document['strategy_enum']['path']] = hashlib.sha256(enum_path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, 'denominator mismatch'):
                validate(root, self.document)


if __name__ == '__main__':
    unittest.main()
