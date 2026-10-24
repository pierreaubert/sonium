"""Tests for source discovery completeness and false public-declaration hints."""

import json
from pathlib import Path
import tempfile
import unittest

from cuda_workflow_inventory import CORE, DECLARATION, build_inventory, changed_units, encoded, mask_noncode


class InventoryTests(unittest.TestCase):
    def test_comments_strings_chars_and_lifetimes(self):
        source = '''// pub fn comment() {}
/* outer /* pub fn nested() {} */ pub fn hidden() {} */
const TEXT: &str = r##"pub fn raw() {} /*"##;
const BYTES: &[u8] = br#"pub fn bytes() {}"#;
const NORMAL: &str = "escaped \\" pub fn string() {}";
pub fn real<'a>(x: &'a str) { let c = '"'; let q = '\\''; }
pub(crate) fn internal() {}
pub async unsafe fn r#await() {}
'''
        masked = mask_noncode(source)
        self.assertEqual(len(masked), len(source))
        self.assertEqual([i for i, c in enumerate(masked) if c == '\n'],
                         [i for i, c in enumerate(source) if c == '\n'])
        self.assertEqual([m[3] for m in DECLARATION.finditer(masked)], ['real', 'await'])

    def test_complete_sources_new_packages_and_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def package(path, name):
                folder = root / path
                (folder / 'src/private').mkdir(parents=True)
                (folder / 'Cargo.toml').write_text(f'[package]\nname = "{name}"\n')
                (folder / 'src/lib.rs').write_text('pub fn solve() {}\n')
                (folder / 'src/private/tests.rs').write_text('fn no_public_hints() {}\n')
                return folder
            for name in CORE:
                package('crates/' + name, name)
            package('crates/sonium-solvers/crates/new-solver', 'new-solver')
            app = package('crates/sonium-sim-new', 'sonium-sim-new')
            original = build_inventory(root)
            self.assertEqual(len(original['packages']), 6)
            self.assertEqual(len(original['files']), 12)
            self.assertTrue(all(f['workflow_classification'] == 'unreviewed' for f in original['files']))
            self.assertEqual(encoded(original), encoded(build_inventory(root)))
            (app / 'src/lib.rs').write_text('pub fn solve() { /* changed body */ }\n')
            self.assertNotEqual(encoded(original), encoded(build_inventory(root)))
            self.assertEqual(changed_units(original, build_inventory(root))['changed'],
                             ['crates/sonium-sim-new/src/lib.rs'])
            (app / 'src/new.rs').write_text('pub fn added() {}\n')
            added = build_inventory(root)
            self.assertEqual(len(added['files']), 13)
            self.assertEqual(changed_units(original, added)['added'],
                             ['crates/sonium-sim-new/src/new.rs'])
            (app / 'src/new.rs').unlink()
            self.assertEqual(len(build_inventory(root)['files']), 12)
            self.assertEqual(changed_units(added, build_inventory(root))['removed'],
                             ['crates/sonium-sim-new/src/new.rs'])
            self.assertEqual(json.loads(encoded(original)), original)


if __name__ == '__main__':
    unittest.main()
