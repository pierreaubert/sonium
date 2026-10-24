#!/usr/bin/env python3
"""Inventory source review units for CUDA coverage; never infer execution coverage.

Run with --write to refresh perf/cuda-source-inventory.json, or --check to fail
on source drift. No Rust build, graph index update or CUDA device is required.
Public declarations are lexical review hints, not a Rust export resolver. All
source files remain in the inventory, including macro-generated APIs, private
modules and tests, so a missing hint cannot silently remove a review unit.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("perf/cuda-source-inventory.json")
CORE = ("sonium-compute", "sonium-solvers", "sonium-bem", "sonium-fem")
DECLARATION = re.compile(
    r"\bpub\s+(?:(?:async|unsafe|const)\s+)*(fn|struct|enum|trait|type|mod)\s+"
    r"(r#)?([A-Za-z_][A-Za-z_0-9]*)"
)
RAW_STRING = re.compile(r'(?:br|cr|r)(#*)"')
CHARACTER = re.compile(r"'(?:\\(?:u\{[0-9A-Fa-f_]+\}|x[0-9A-Fa-f]{2}|.)|[^'\\\n])'")


def mask_noncode(source):
    """Mask comments and strings while preserving character positions and newlines."""
    chars = list(source)
    index = 0

    def erase(start, end):
        for pos in range(start, end):
            if chars[pos] != "\n":
                chars[pos] = " "

    while index < len(source):
        start = index
        if source.startswith("//", index):
            index = source.find("\n", index)
            if index < 0:
                index = len(source)
        elif source.startswith("/*", index):
            depth = 1
            index += 2
            while index < len(source) and depth:
                if source.startswith("/*", index):
                    depth += 1
                    index += 2
                elif source.startswith("*/", index):
                    depth -= 1
                    index += 2
                else:
                    index += 1
        else:
            raw = RAW_STRING.match(source, index)
            if raw:
                terminator = '"' + raw.group(1)
                end = source.find(terminator, raw.end())
                index = len(source) if end < 0 else end + len(terminator)
            elif source[index] == '"':
                index += 1
                while index < len(source):
                    if source[index] == "\\":
                        index += 2
                    elif source[index] == '"':
                        index += 1
                        break
                    else:
                        index += 1
                index = min(index, len(source))
            elif source[index] == "'":
                # Preserve lifetimes; mask only complete character literals.
                char = CHARACTER.match(source, index)
                if char:
                    index = char.end()
                else:
                    index += 1
                    continue
            else:
                index += 1
                continue
        erase(start, index)
    return "".join(chars)


def package_manifests(root):
    manifests = [root / "crates" / name / "Cargo.toml" for name in CORE]
    manifests += sorted((root / "crates/sonium-solvers/crates").glob("*/Cargo.toml"))
    # Application producers are discovery scope even though the four core
    # components own numerical implementations. Do not omit newly added apps.
    manifests += sorted((root / "crates").glob("sonium-sim-*/Cargo.toml"))
    return manifests


def build_inventory(root):
    packages = []
    files = []
    for manifest in package_manifests(root):
        metadata = tomllib.loads(manifest.read_text())
        name = metadata["package"]["name"]
        directory = manifest.parent
        sources = sorted((directory / "src").rglob("*.rs"))
        if not sources:
            raise ValueError(f"no source files for {name}")
        packages.append({
            "name": name,
            "manifest": manifest.relative_to(root).as_posix(),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "source_file_count": len(sources),
        })
        for source in sources:
            raw = source.read_bytes()
            text = raw.decode("utf-8")
            local = source.relative_to(directory / "src")
            hints = [{"kind": match[1], "name": match[3],
                      "line": text.count("\n", 0, match.start()) + 1}
                     for match in DECLARATION.finditer(mask_noncode(text))]
            group = name
            if name == "sonium-fem" and local.parts[0] == "physics":
                group += "/" + "/".join(local.parts[:-1])
            files.append({
                "path": source.relative_to(root).as_posix(),
                "package": name,
                "review_group": group,
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "public_declaration_hints": hints,
                "workflow_classification": "unreviewed",
            })
    return {
        "schema_version": 1,
        "purpose": "Source review denominator, not a count of workflows or CUDA coverage.",
        "discovery": {
            "core_packages": list(CORE),
            "solver_packages": "every crates/sonium-solvers/crates/*/Cargo.toml",
            "application_packages": "every crates/sonium-sim-*/Cargo.toml",
            "sources": "all src/**/*.rs, regardless of cfg, visibility or test status",
            "hints": "lexical unrestricted pub declarations; exports, macros and trait implementations require review",
            "excluded": "examples, benches, external workspaces and non-Rust source; audit separately",
        },
        "workflow_review_requirements": [
            "Resolve public CPU entrypoints and aliases; exclude helper/test units with rationale.",
            "Split physics, space/order/topology, scalar, boundary/constraint and action variants.",
            "Record solver/preconditioner, single/block RHS, update and output variants.",
            "Track setup, solve, numeric update and output execution separately.",
            "Attach dispatch, native correctness, residency, recovery and performance evidence separately.",
            "Unsupported CUDA requests remain open; source presence and passing CPU tests do not close them.",
        ],
        "packages": packages,
        "files": files,
    }


def encoded(inventory):
    return json.dumps(inventory, indent=2, ensure_ascii=False) + "\n"


def changed_units(before, after):
    """Return added, removed and changed files/manifests without hiding deletions."""
    def units(document):
        result = {row["path"]: row for row in document.get("files", [])}
        result.update({row["manifest"]: row for row in document.get("packages", [])})
        return result
    previous, current = units(before), units(after)
    return {
        "added": sorted(current.keys() - previous.keys()),
        "removed": sorted(previous.keys() - current.keys()),
        "changed": sorted(path for path in current.keys() & previous.keys()
                          if previous[path] != current[path]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    current = encoded(build_inventory(ROOT))
    target = ROOT / OUTPUT
    if args.write:
        target.write_text(current)
    elif not target.exists() or target.read_text() != current:
        print(f"Stale source inventory: run {Path(__file__).name} --write and review changed units.", file=sys.stderr)
        if target.exists():
            try:
                changes = changed_units(json.loads(target.read_text()), json.loads(current))
                for kind, paths in changes.items():
                    print(f"{kind}: {len(paths)}", file=sys.stderr)
                    for path in paths[:20]:
                        print(f"  {path}", file=sys.stderr)
            except (ValueError, KeyError, TypeError):
                print("Previous inventory is malformed; regenerate it.", file=sys.stderr)
        return 1
    inventory = json.loads(current)
    print(f"{len(inventory['packages'])} packages, {len(inventory['files'])} source review units; workflow coverage unclassified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
