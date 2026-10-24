#!/usr/bin/env python3
"""Validate reviewed facade rows against current source; no coverage percentage.

This checks the initial single-forward strategy/scalar subset. It neither
executes GPU tests nor certifies arbitrary preconditioner callbacks or producers.
"""

import hashlib
import json
from pathlib import Path
import re
import sys

from cuda_workflow_inventory import ROOT, mask_noncode


REGISTRY = Path("perf/cuda-workflow-registry.json")
SESSION = "crates/sonium-solvers/src/solve/cuda_session.rs"
SCALARS = {"f32", "f64", "Complex32", "Complex64"}


def source_block(source, declaration):
    """Read a balanced declaration body after masking comments and strings."""
    text = mask_noncode(source)
    match = re.search(declaration, text)
    if not match:
        raise ValueError(f"missing source declaration: {declaration}")
    start = text.index("{", match.end())
    depth = 1
    for end in range(start + 1, len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            return text[start + 1:end]
    raise ValueError("unbalanced source declaration")


def local_file(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"missing or nonlocal registry reference: {value}")
    return path


def validate(root, document):
    if document.get("schema_version") != 1:
        raise ValueError("unsupported registry schema")
    if set(document["scalars"]) != SCALARS or len(document["scalars"]) != 4:
        raise ValueError("the four principal scalars must each occur once")
    for path, expected in document["reviewed_source_sha256"].items():
        actual = hashlib.sha256(local_file(root, path).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"review stale: {path}; inspect changes before updating the hash")
    enum = document["strategy_enum"]
    if enum["path"] not in document["reviewed_source_sha256"] or SESSION not in document["reviewed_source_sha256"]:
        raise ValueError("dispatch and enum sources must have review hashes")
    body = source_block(local_file(root, enum["path"]).read_text(),
                        r"\bpub\s+enum\s+" + re.escape(enum["name"]) + r"\b")
    # Fail closed if this evolves from unit variants into data-carrying variants.
    if not re.fullmatch(r"\s*(?:[A-Za-z_]\w*\s*,\s*)+", body):
        raise ValueError("strategy enum syntax changed; update the reviewed extractor")
    strategies = set(re.findall(r"([A-Za-z_]\w*)\s*,", body))
    admission = source_block(local_file(root, SESSION).read_text(), r"\bfn\s+validate_strategy\b")
    admitted = set(re.findall(r"LinearSolverType::([A-Za-z_]\w*)", admission))
    if not admitted or not admitted <= strategies:
        raise ValueError("strategy admission extraction failed")
    expected = {(strategy, scalar) for strategy in strategies for scalar in SCALARS}
    seen, identifiers = set(), set()
    for row in document["workflows"]:
        key = row["strategy"], row["scalar"]
        if key in seen or row["id"] in identifiers:
            raise ValueError(f"duplicate workflow: {row['id']}")
        seen.add(key)
        identifiers.add(row["id"])
        if row["action"] != "forward" or row["rhs"] != "single":
            raise ValueError("this registry subset covers only single forward solves")
        supported = row["strategy"] in admitted
        if row["dispatch"] != ("retained_cuda" if supported else "explicit_rejection"):
            raise ValueError(f"dispatch disagrees with source: {row['id']}")
        if set(row["phases"]) != {"setup", "solve", "numeric_update", "output"}:
            raise ValueError("each workflow must describe all four execution phases")
        if not supported and any(value != "unavailable" for value in row["phases"].values()):
            raise ValueError("rejected workflows cannot claim available execution phases")
        if row["complete"] is not False or not row["remaining"]:
            raise ValueError("this initial review has no fully closed workflows")
        for field in ("cpu_entrypoint", "cuda_entrypoint", "matrix", "constraints",
                      "assumption", "correctness", "residency", "performance"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"missing workflow dimension: {field}")
        if row["correctness"] == "bounded_native_tests_passed":
            evidence_path = row.get("current_native_evidence")
            if evidence_path not in {item["path"] for item in document["evidence"]}:
                raise ValueError("native test claim needs a declared evidence artifact")
            evidence = json.loads(local_file(root, evidence_path).read_text())
            if evidence.get("source_sha256") != document["reviewed_source_sha256"]:
                raise ValueError("native evidence does not match reviewed source anchors")
            if not re.search(r"test result: ok\. [1-9]\d* passed; 0 failed", evidence.get("result", "")):
                raise ValueError("native evidence has no successful test result")
    if seen != expected:
        raise ValueError(f"strategy/scalar denominator mismatch: missing={sorted(expected-seen)}, extra={sorted(seen-expected)}")
    local_file(root, document["source_inventory"])
    for evidence in document["evidence"]:
        json.loads(local_file(root, evidence["path"]).read_text())
        if not evidence["use"].strip():
            raise ValueError("evidence must state its bounded use")
    if not document["unreviewed_scope"]:
        raise ValueError("initial subset cannot erase the remaining review scope")
    return len(seen), len(admitted) * len(SCALARS)


def main():
    try:
        count, admitted = validate(ROOT, json.loads((ROOT / REGISTRY).read_text()))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"CUDA workflow review invalid: {error}", file=sys.stderr)
        return 1
    print(f"Validated {count} reviewed facade rows: {admitted} retained-dispatch, {count-admitted} rejected; zero complete. Remaining scope is unreviewed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
