#!/usr/bin/env python3
"""
Run lightweight Colab readiness smoke tests for enabled demo notebooks.

This executes a temporary notebook prefix consisting of:
- a prelude cell that fakes ``google.colab``
- the notebook's Colab bootstrap cell (with shell installs rewritten to no-ops)
- up to a few smoke-safe import/setup code cells after the bootstrap cell

The goal is to catch runtime import/path/setup regressions in CI without
executing entire notebooks or re-running heavyweight package installs.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import tempfile
from pathlib import Path

from scripts.check_colab_notebooks import (
    DEFAULT_MANIFEST,
    cell_source_text,
    is_bootstrap_cell,
    load_json,
    python_source_for_ast,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
def make_prelude_cell(notebook_path: Path) -> dict:
    notebook_dir = notebook_path.parent.resolve().as_posix()
    repo_root = REPO_ROOT.resolve().as_posix()
    source = f"""import os
import sys
import types

google = sys.modules.get("google")
if google is None:
    google = types.ModuleType("google")
    sys.modules["google"] = google

colab = types.ModuleType("google.colab")
google.colab = colab
sys.modules["google.colab"] = colab

os.environ["QMC_COLAB_SMOKE"] = "1"
os.environ["QMC_COLAB_SMOKE_REPO_ROOT"] = r"{repo_root}"
os.environ["QMC_COLAB_SMOKE_NOTEBOOK_DIR"] = r"{notebook_dir}"
"""
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": "smoke-prelude",
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def rewrite_shell_magics(source: str) -> str:
    rewritten_lines: list[str] = []
    for line in source.splitlines(keepends=True):
        stripped = line.lstrip()
        indent = line[: len(line) - len(stripped)]
        if stripped.startswith("!"):
            command = stripped[1:].rstrip()
            rewritten_lines.append(f"{indent}print({command!r})\n")
        else:
            rewritten_lines.append(line)
    return "".join(rewritten_lines)


def rewrite_bootstrap_source(source: str) -> str:
    repo_root = REPO_ROOT.resolve().as_posix()
    source = source.replace('"/content/QMCSoftware"', repr(repo_root))
    source = source.replace("'/content/QMCSoftware'", repr(repo_root))
    return rewrite_shell_magics(source)


def assignment_target_is_smoke_safe(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(assignment_target_is_smoke_safe(elt) for elt in node.elts)
    if isinstance(node, ast.Attribute):
        return expression_is_smoke_safe(node.value)
    if isinstance(node, ast.Subscript):
        return expression_is_smoke_safe(node.value) and expression_is_smoke_safe(node.slice)
    return False


def expression_is_smoke_safe(node: ast.AST | None) -> bool:
    if node is None:
        return True
    if isinstance(node, (ast.Constant, ast.Name)):
        return True
    if isinstance(node, ast.Attribute):
        return expression_is_smoke_safe(node.value)
    if isinstance(node, ast.Tuple):
        return all(expression_is_smoke_safe(elt) for elt in node.elts)
    if isinstance(node, ast.List):
        return all(expression_is_smoke_safe(elt) for elt in node.elts)
    if isinstance(node, ast.Set):
        return all(expression_is_smoke_safe(elt) for elt in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            expression_is_smoke_safe(key) and expression_is_smoke_safe(value)
            for key, value in zip(node.keys, node.values)
        )
    if isinstance(node, ast.UnaryOp):
        return expression_is_smoke_safe(node.operand)
    if isinstance(node, ast.BinOp):
        return expression_is_smoke_safe(node.left) and expression_is_smoke_safe(node.right)
    if isinstance(node, ast.BoolOp):
        return all(expression_is_smoke_safe(value) for value in node.values)
    if isinstance(node, ast.Compare):
        return expression_is_smoke_safe(node.left) and all(
            expression_is_smoke_safe(comparator) for comparator in node.comparators
        )
    if isinstance(node, ast.Subscript):
        return expression_is_smoke_safe(node.value) and expression_is_smoke_safe(node.slice)
    if isinstance(node, ast.Slice):
        return (
            expression_is_smoke_safe(node.lower)
            and expression_is_smoke_safe(node.upper)
            and expression_is_smoke_safe(node.step)
        )
    if isinstance(node, ast.IfExp):
        return (
            expression_is_smoke_safe(node.test)
            and expression_is_smoke_safe(node.body)
            and expression_is_smoke_safe(node.orelse)
        )
    if isinstance(node, ast.JoinedStr):
        return all(
            expression_is_smoke_safe(value.value)
            if isinstance(value, ast.FormattedValue)
            else expression_is_smoke_safe(value)
            for value in node.values
        )
    if isinstance(node, ast.FormattedValue):
        return expression_is_smoke_safe(node.value)
    return False


def statement_is_smoke_safe(node: ast.stmt) -> bool:
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Pass)):
        return True
    if isinstance(node, ast.Assign):
        return all(assignment_target_is_smoke_safe(target) for target in node.targets) and expression_is_smoke_safe(node.value)
    if isinstance(node, ast.AnnAssign):
        return assignment_target_is_smoke_safe(node.target) and expression_is_smoke_safe(node.value)
    if isinstance(node, ast.AugAssign):
        return assignment_target_is_smoke_safe(node.target) and expression_is_smoke_safe(node.value)
    if isinstance(node, ast.Expr):
        return expression_is_smoke_safe(node.value)
    if isinstance(node, ast.Try):
        return (
            all(statement_is_smoke_safe(stmt) for stmt in node.body)
            and all(
                expression_is_smoke_safe(handler.type)
                and all(statement_is_smoke_safe(stmt) for stmt in handler.body)
                for handler in node.handlers
            )
            and all(statement_is_smoke_safe(stmt) for stmt in node.orelse)
            and all(statement_is_smoke_safe(stmt) for stmt in node.finalbody)
        )
    if isinstance(node, ast.If):
        return (
            expression_is_smoke_safe(node.test)
            and all(statement_is_smoke_safe(stmt) for stmt in node.body)
            and all(statement_is_smoke_safe(stmt) for stmt in node.orelse)
        )
    return False


def is_smoke_safe_code_cell(source: str) -> bool:
    cleaned = python_source_for_ast(source)
    if not cleaned.strip():
        return True
    try:
        tree = ast.parse(cleaned)
    except SyntaxError:
        return False
    return all(statement_is_smoke_safe(statement) for statement in tree.body)


def build_smoke_notebook(notebook_path: Path, cells_after_bootstrap: int) -> tuple[dict, list[int | None]]:
    with notebook_path.open(encoding="utf-8") as handle:
        original_nb = json.load(handle)

    bootstrap_idx = next(
        (idx for idx, cell in enumerate(original_nb["cells"]) if is_bootstrap_cell(cell)),
        None,
    )
    if bootstrap_idx is None:
        raise RuntimeError(f"{notebook_path.as_posix()}: missing Colab bootstrap cell")

    stop_idx = bootstrap_idx
    safe_code_cells = 0
    for idx in range(bootstrap_idx + 1, len(original_nb["cells"])):
        cell = original_nb["cells"][idx]
        if cell.get("cell_type") != "code":
            stop_idx = idx
            continue
        if safe_code_cells >= cells_after_bootstrap:
            break
        if not is_smoke_safe_code_cell(cell_source_text(cell)):
            break
        safe_code_cells += 1
        stop_idx = idx

    temp_cells = [make_prelude_cell(notebook_path)]
    source_indices: list[int | None] = [None]

    for idx, cell in enumerate(original_nb["cells"][: stop_idx + 1]):
        cloned = copy.deepcopy(cell)
        cloned["id"] = f"smoke-{idx}"
        if cloned.get("cell_type") == "code":
            cloned["source"] = rewrite_bootstrap_source(cell_source_text(cloned))
            cloned["execution_count"] = None
            cloned["outputs"] = []
        temp_cells.append(cloned)
        source_indices.append(idx)

    smoke_nb = {
        "cells": temp_cells,
        "metadata": copy.deepcopy(original_nb.get("metadata", {})),
        "nbformat": original_nb.get("nbformat", 4),
        "nbformat_minor": original_nb.get("nbformat_minor", 5),
    }
    return smoke_nb, source_indices


def execute_smoke_notebook(notebook_path: Path, smoke_nb: dict, source_indices: list[int | None], timeout: int) -> None:
    try:
        from testbook import testbook
    except ImportError as exc:  # pragma: no cover - depends on the local environment
        raise RuntimeError(
            "testbook is required for Colab smoke tests. "
            "Install test dependencies, e.g. `pip install -e .[test]`."
        ) from exc

    notebook_dir = notebook_path.parent.resolve()
    with tempfile.NamedTemporaryFile(
        suffix=".ipynb",
        prefix=".tmp_colab_smoke_",
        dir=notebook_dir,
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)

    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(smoke_nb, handle)

        original_cwd = Path.cwd()
        try:
            os.chdir(notebook_dir)
            with testbook(temp_path.as_posix(), timeout=timeout, execute=False) as tb:
                for temp_idx, cell in enumerate(tb.cells):
                    if cell.cell_type != "code":
                        continue
                    try:
                        tb.execute_cell(temp_idx)
                    except Exception as exc:  # noqa: BLE001
                        source_idx = source_indices[temp_idx]
                        if source_idx is None:
                            location = "prelude cell"
                        else:
                            location = f"notebook cell {source_idx + 1}"
                        raise RuntimeError(
                            f"{notebook_path.relative_to(REPO_ROOT).as_posix()}: Colab smoke failed in {location}: {exc}"
                        ) from exc
        finally:
            os.chdir(original_cwd)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lightweight Colab readiness smoke tests for enabled notebooks."
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help=f"Path to the Colab manifest (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--cells-after-bootstrap",
        type=int,
        default=2,
        help="Maximum number of smoke-safe code cells to execute after the bootstrap cell (default: 2)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-notebook execution timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--notebook",
        action="append",
        default=[],
        help="Optional manifest-relative notebook path to smoke test. May be passed multiple times.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)
    enabled = list(manifest.get("enabled", []))

    if args.notebook:
        enabled = [path for path in enabled if path in set(args.notebook)]

    if not enabled:
        print("No enabled notebooks selected for Colab smoke tests.")
        return 0

    for notebook_rel in enabled:
        notebook_path = (REPO_ROOT / notebook_rel).resolve()
        print(f"Smoke testing {notebook_rel}")
        smoke_nb, source_indices = build_smoke_notebook(
            notebook_path, args.cells_after_bootstrap
        )
        execute_smoke_notebook(
            notebook_path,
            smoke_nb,
            source_indices,
            timeout=args.timeout,
        )

    print(f"Colab smoke tests passed: {len(enabled)} notebook(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
