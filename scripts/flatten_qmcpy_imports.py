#!/usr/bin/env python3
"""Flatten public absolute QMCPy imports to use the top-level package.

Adjacent public ``from qmcpy import ...`` statements in the same scope are
combined, deduplicated, and ordered alphabetically.  The combined import stays
on one line when it fits within 88 characters; otherwise it uses a
parenthesized block with one name per line.  A blank line, comment, different
statement, or change in indentation ends a group.  Authors can therefore keep
intentional semantic groups (for example, true measures and integrands) by
separating and, when useful, labeling those groups themselves.

Private module paths and private imported names are left unchanged.  They stay
as separate statements and end any adjacent public-import group.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable, Iterator
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tokenize


SUPPORTED_SUFFIXES = {".ipynb", ".md", ".py", ".pyi", ".rst", ".txt"}
SKIPPED_DIRECTORIES = {
    ".git",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".pdm-build",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "artifacts",
    "build",
    "dist",
    "htmlcov",
    "node_modules",
    "site",
    "venv",
}

# Matching bytes avoids changing line endings or reformatting notebook JSON.
QMCPY_IMPORT_RE = re.compile(
    rb"\bfrom(?P<after_from>[ \t]+)qmcpy"
    rb"(?P<module_path>(?:\.[A-Za-z_][A-Za-z0-9_]*)+)"
    rb"(?P<before_import>[ \t]+)import\b"
    rb"(?P<imported>[ \t]*(?:\([^)]*\)|[^\r\n]*))"
)
TEXT_STAR_IMPORT_RE = re.compile(
    rb"^(?P<indent>[ \t]*)from[ \t]+qmcpy[ \t]+import[ \t]+\*[ \t]*"
    rb"(?:\r\n|\n|\r)?$"
)
NOTEBOOK_STAR_IMPORT_RE = re.compile(
    rb'^(?P<json_indent>[ \t]*)"(?P<code_indent>(?:[ \t]|\\t)*)'
    rb"from[ \t]+qmcpy[ \t]+import[ \t]+\*(?:\\r)?(?:\\n)?\""
    rb",?[ \t]*(?:\r\n|\n|\r)?$"
)
STAR_IMPORT_LITERAL = b"from qmcpy import *"
TEXT_NAMED_IMPORT_START_RE = re.compile(
    rb"^(?P<indent>[ \t]*)from[ \t]+qmcpy[ \t]+import[ \t]+"
    rb"(?P<imported>[^\r\n]*?)[ \t]*(?:\r\n|\n|\r)?$"
)
NOTEBOOK_SOURCE_LINE_RE = re.compile(
    rb'^(?P<json_indent>[ \t]*)(?P<string>"(?:[^"\\]|\\.)*")'
    rb"(?P<comma>,?)(?P<trailing>[ \t]*)(?P<ending>\r\n|\n|\r)?$"
)
NOTEBOOK_SOURCE_FIELD_RE = re.compile(
    rb'^[ \t]*"source"[ \t]*:[ \t]*(?P<value>.*?)[ \t]*(?:\r\n|\n|\r)?$'
)
# Bare (already top-level) single-line "from qmcpy import ..." statements, so
# stale comma spacing can be cleaned up even when there's no module path to
# flatten. Anchored to line start, so notebook JSON lines (which have a
# leading quote before "from") never match.
NAMED_IMPORT_LINE_RE = re.compile(
    rb"^(?P<prefix>[ \t]*from[ \t]+qmcpy[ \t]+import)(?P<imported>[ \t][^\r\n]*?)"
    rb"[ \t]*(?:\r\n|\n|\r)?$"
)
COMMA_SPACING_RE = re.compile(rb",(?=\S)")

# A line only looks like an IPython magic/shell escape when '%'/'%%' is
# immediately followed by a name (e.g. "%matplotlib"); "% (x, y)" is a
# modulo-operator continuation and must be left alone.
MAGIC_LINE_RE = re.compile(r"^[ \t]*(?:%{1,2}[A-Za-z_]|!|\?)")


def _python_protected_lines(content: bytes) -> set[int] | None:
    """Return 1-based line numbers covered by Python string/comment tokens."""

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return None

    protected: set[int] = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            tok_name = tokenize.tok_name.get(tok.type, "")
            if tok.type != tokenize.STRING and tok_name not in {
                "COMMENT",
                "FSTRING_START",
                "FSTRING_MIDDLE",
                "FSTRING_END",
            }:
                continue
            start_row = tok.start[0]
            end_row = tok.end[0]
            protected.update(range(start_row, end_row + 1))
    except (SyntaxError, IndentationError, tokenize.TokenError):
        return None

    return protected


def _notebook_code_source_lines(content: bytes) -> set[int] | None:
    """Return physical line indexes belonging to code-cell source arrays."""

    try:
        notebook = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        return None

    cells = notebook["cells"]
    code_lines: set[int] = set()
    cell_index = 0
    lines = content.splitlines(keepends=True)
    line_index = 0
    while line_index < len(lines) and cell_index < len(cells):
        field_match = NOTEBOOK_SOURCE_FIELD_RE.match(lines[line_index])
        if field_match is None:
            line_index += 1
            continue

        cell = cells[cell_index]
        if not isinstance(cell, dict):
            return None
        expected_source = cell.get("source", "")
        value = field_match.group("value").rstrip().rstrip(b",").rstrip()

        if value != b"[":
            try:
                inline_source = json.loads(value)
            except (json.JSONDecodeError, UnicodeDecodeError):
                line_index += 1
                continue
            if inline_source == expected_source:
                cell_index += 1
            line_index += 1
            continue

        source_lines: list[str] = []
        source_line_indexes: list[int] = []
        stop = line_index + 1
        valid_array = True
        while stop < len(lines) and not lines[stop].lstrip().startswith(b"]"):
            source_match = NOTEBOOK_SOURCE_LINE_RE.match(lines[stop])
            if source_match is None:
                valid_array = False
                break
            try:
                source_item = json.loads(source_match.group("string"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                valid_array = False
                break
            if not isinstance(source_item, str):
                valid_array = False
                break
            source_lines.append(source_item)
            source_line_indexes.append(stop)
            stop += 1

        if stop >= len(lines):
            return None
        if valid_array and source_lines == expected_source:
            if cell.get("cell_type") == "code":
                protected_rows = _python_protected_lines(
                    "".join(source_lines).encode("utf-8")
                )
                if protected_rows is not None:
                    source_row = 1
                    for source_item, source_line_index in zip(
                        source_lines, source_line_indexes
                    ):
                        newline_count = source_item.count("\n")
                        is_single_line = newline_count == 0 or (
                            newline_count == 1 and source_item.endswith("\n")
                        )
                        if is_single_line and source_row not in protected_rows:
                            code_lines.add(source_line_index)
                        source_row += newline_count
            cell_index += 1
        line_index = stop + 1

    if cell_index != len(cells):
        return None
    return code_lines


def _flatten_nested_import_match(
    match: re.Match[bytes], public_names: frozenset[str] | None
) -> bytes | None:
    """Return a flattened import match, or None when it is not safe to rewrite."""

    if public_names is None:
        return None
    module_segments = match.group("module_path").lstrip(b".").split(b".")
    if module_segments[0] == b"util" or any(
        segment.startswith(b"_") for segment in module_segments
    ):
        return None

    try:
        tree = ast.parse(match.group(0).decode("utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.ImportFrom):
        return None

    statement = tree.body[0]
    expected_module = "qmcpy" + match.group("module_path").decode("ascii")
    if statement.level or statement.module != expected_module:
        return None
    if any(
        alias.name.startswith("_")
        or (alias.asname is not None and alias.asname.startswith("_"))
        for alias in statement.names
    ):
        return None

    imported_names = {alias.name for alias in statement.names}
    if "*" not in imported_names and not imported_names <= public_names:
        return None

    return (
        b"from"
        + match.group("after_from")
        + b"qmcpy"
        + match.group("before_import")
        + b"import"
        + match.group("imported")
    )


def _flatten_notebook_nested_imports(
    content: bytes,
    public_names: frozenset[str] | None,
    code_lines: set[int],
) -> tuple[bytes, int]:
    """Flatten nested imports only in source lines from notebook code cells."""

    output: list[bytes] = []
    change_count = 0
    for line_index, line in enumerate(content.splitlines(keepends=True)):
        line_match = NOTEBOOK_SOURCE_LINE_RE.match(line)
        if line_index not in code_lines or line_match is None:
            output.append(line)
            continue

        try:
            source = json.loads(line_match.group("string"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            output.append(line)
            continue
        if not isinstance(source, str):
            output.append(line)
            continue

        source_bytes = source.encode("utf-8")
        protected_lines = _python_protected_lines(source_bytes)
        if protected_lines is None:
            output.append(line)
            continue

        line_changes = 0

        def replace(match: re.Match[bytes]) -> bytes:
            nonlocal line_changes
            source_line = source_bytes.count(b"\n", 0, match.start()) + 1
            if source_line in protected_lines:
                return match.group(0)
            replacement = _flatten_nested_import_match(match, public_names)
            if replacement is None:
                return match.group(0)
            line_changes += 1
            return replacement

        updated_source = QMCPY_IMPORT_RE.sub(replace, source_bytes)
        if not line_changes:
            output.append(line)
            continue

        output.append(
            line_match.group("json_indent")
            + json.dumps(updated_source.decode("utf-8")).encode()
            + line_match.group("comma")
            + line_match.group("trailing")
            + (line_match.group("ending") or b"")
        )
        change_count += line_changes

    return b"".join(output), change_count


def _star_import_context(line: bytes) -> tuple[bytes, bytes] | None:
    text_match = TEXT_STAR_IMPORT_RE.match(line)
    if text_match:
        return b"text", text_match.group("indent")

    notebook_match = NOTEBOOK_STAR_IMPORT_RE.match(line)
    if notebook_match:
        indentation = (
            notebook_match.group("json_indent")
            + b'"'
            + notebook_match.group("code_indent")
        )
        return b"notebook", indentation

    return None


def _deduplicate_adjacent_star_imports(
    content: bytes, eligible_lines: set[int] | None = None
) -> tuple[bytes, int]:
    """Keep the last line in each run of same-scope QMCPy star imports."""

    output: list[bytes] = []
    previous_context: tuple[bytes, bytes] | None = None
    removed_count = 0

    for line_index, line in enumerate(content.splitlines(keepends=True)):
        context = (
            _star_import_context(line)
            if eligible_lines is None or line_index in eligible_lines
            else None
        )
        if context is not None and context == previous_context:
            # Keeping the last line preserves JSON comma and newline placement.
            output[-1] = line
            removed_count += 1
        else:
            output.append(line)
        previous_context = context

    return b"".join(output), removed_count


def _load_qmcpy_public_names(repository_root: Path) -> frozenset[str] | None:
    """Return qmcpy public names using an optional-dependency-free import context."""

    blocklist = (
        "torch",
        "gpytorch",
        "pyg_lib",
        "torch_geometric",
        "torch_cluster",
        "torch_scatter",
        "torch_sparse",
        "torch_spline_conv",
    )
    probe = r"""
import builtins
import json
import sys

repository_root = sys.argv[1]
blocked_roots = set(sys.argv[2:])
real_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split('.', 1)[0]
    if root in blocked_roots:
        raise ModuleNotFoundError('blocked optional dependency', name=root)
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
if repository_root not in sys.path:
    sys.path.insert(0, repository_root)

import qmcpy
print(json.dumps(sorted(name for name in qmcpy.__dict__ if not name.startswith('_'))))
"""

    result = subprocess.run(
        [sys.executable, "-c", probe, str(repository_root), *blocklist],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        names = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return frozenset(name for name in names if isinstance(name, str))


def _names_needing_import(source: str, public_names: frozenset[str]) -> set[str] | None:
    """Return public qmcpy names referenced but not otherwise bound in `source`.

    Returns None if `source` isn't parseable Python. The scan is file-wide
    rather than scope-aware, so a name bound anywhere (even in an unrelated
    scope) is treated as locally defined and excluded, which is the safe
    direction to err in.
    """

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    loaded: set[str] = set()
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (loaded if isinstance(node.ctx, ast.Load) else bound).add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.alias):
            bound.add((node.asname or node.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)

    return (loaded & public_names) - bound


def _line_ending(line: bytes) -> bytes:
    for ending in (b"\r\n", b"\n", b"\r"):
        if line.endswith(ending):
            return ending
    return b""


def _format_expanded_import(indent: bytes, names: Iterable[str], ending: bytes) -> bytes:
    sorted_names = sorted(
        dict.fromkeys(names),
        key=lambda name: tuple(part.casefold() for part in name.partition(" as ")),
    )
    indent_text = indent.decode()
    single_line = f"{indent_text}from qmcpy import {', '.join(sorted_names)}"
    if len(single_line) <= 88:
        return single_line.encode() + ending

    separator = ending or b"\n"
    body_lines = [f"{indent_text}from qmcpy import ("]
    body_lines.extend(f"{indent_text}    {name}," for name in sorted_names)
    body_lines.append(f"{indent_text})")
    return separator.join(line.encode() for line in body_lines) + ending


def _parse_named_import_statement(statement_bytes: bytes, indent: bytes):
    """Return imported names for a safe top-level QMCPy import statement."""

    if b"#" in statement_bytes:
        return None

    dedented = b"".join(
        line[len(indent) :] if line.startswith(indent) else line
        for line in statement_bytes.splitlines(keepends=True)
    )
    try:
        tree = ast.parse(dedented.decode("utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.ImportFrom):
        return None
    statement = tree.body[0]
    if statement.level or statement.module != "qmcpy":
        return None

    names = []
    for alias in statement.names:
        if alias.name == "*" or alias.name.startswith("_"):
            return None
        names.append(
            alias.name if alias.asname is None else f"{alias.name} as {alias.asname}"
        )
    return names


def _parse_text_named_import(lines: list[bytes], start: int):
    """Parse one safe single-line or parenthesized QMCPy import."""

    match = TEXT_NAMED_IMPORT_START_RE.match(lines[start])
    if match is None:
        return None

    imported = match.group("imported").lstrip()
    stop = start + 1
    if imported.startswith(b"("):
        depth = lines[start].count(b"(") - lines[start].count(b")")
        while depth > 0 and stop < len(lines):
            depth += lines[stop].count(b"(") - lines[stop].count(b")")
            stop += 1
        if depth != 0:
            return None

    original = b"".join(lines[start:stop])
    indent = match.group("indent")
    names = _parse_named_import_statement(original, indent)
    if names is None:
        return None
    return indent, names, _line_ending(lines[stop - 1]), original, stop


def _combine_text_named_imports(
    content: bytes, eligible_lines: set[int] | None = None
) -> tuple[bytes, int]:
    """Combine adjacent, same-scope public QMCPy imports in text files."""

    lines = content.splitlines(keepends=True)
    output: list[bytes] = []
    run: list[tuple[bytes, list[str], bytes, bytes]] = []
    change_count = 0

    def flush() -> None:
        nonlocal change_count
        if not run:
            return
        indent = run[0][0]
        names = [name for _, imported, _, _ in run for name in imported]
        combined = _format_expanded_import(indent, names, run[-1][2])
        original = b"".join(item[3] for item in run)
        output.append(combined)
        change_count += int(combined != original)
        run.clear()

    index = 0
    while index < len(lines):
        if eligible_lines is not None and index not in eligible_lines:
            flush()
            output.append(lines[index])
            index += 1
            continue
        parsed = _parse_text_named_import(lines, index)
        if parsed is None:
            flush()
            output.append(lines[index])
            index += 1
            continue
        indent, names, ending, original, stop = parsed
        if run and run[-1][0] != indent:
            flush()
        run.append((indent, names, ending, original))
        index = stop
    flush()
    return b"".join(output), change_count


def _parse_notebook_named_import(line: bytes):
    match = NOTEBOOK_SOURCE_LINE_RE.match(line)
    if match is None:
        return None
    try:
        source = json.loads(match.group("string"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(source, str):
        return None
    source_bytes = source.encode("utf-8")
    source_lines = source_bytes.splitlines(keepends=True)
    if not source_lines:
        return None
    parsed = _parse_text_named_import(source_lines, 0)
    if parsed is None or parsed[4] != len(source_lines):
        return None
    code_indent, names, code_ending, _, _ = parsed
    context = (match.group("json_indent"), code_indent)
    return context, names, code_ending, match


def _combine_notebook_named_imports(
    content: bytes, code_lines: set[int]
) -> tuple[bytes, int]:
    """Combine adjacent public QMCPy imports in notebook source arrays."""

    output: list[bytes] = []
    run = []
    change_count = 0

    def flush() -> None:
        nonlocal change_count
        if not run:
            return
        context = run[0][0]
        names = [name for _, imported, _, _, _ in run for name in imported]
        code = _format_expanded_import(context[1], names, run[-1][2]).decode()
        last_match = run[-1][3]
        combined = (
            context[0]
            + json.dumps(code).encode()
            + last_match.group("comma")
            + last_match.group("trailing")
            + (last_match.group("ending") or b"")
        )
        original = b"".join(item[4] for item in run)
        output.append(combined)
        change_count += int(combined != original)
        run.clear()

    for line_index, line in enumerate(content.splitlines(keepends=True)):
        parsed = (
            _parse_notebook_named_import(line)
            if line_index in code_lines
            else None
        )
        if parsed is None:
            flush()
            output.append(line)
            continue
        context, names, code_ending, match = parsed
        if run and run[-1][0] != context:
            flush()
        run.append((context, names, code_ending, match, line))
    flush()
    return b"".join(output), change_count


def _expand_text_star_imports(content: bytes, public_names: frozenset[str]) -> tuple[bytes, int]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return content, 0

    needed = _names_needing_import(text, public_names)
    if not needed:
        return content, 0

    output: list[bytes] = []
    change_count = 0
    for line in content.splitlines(keepends=True):
        match = TEXT_STAR_IMPORT_RE.match(line)
        if match is None:
            output.append(line)
            continue
        output.append(
            _format_expanded_import(match.group("indent"), needed, _line_ending(line))
        )
        change_count += 1

    return b"".join(output), change_count


def _sanitize_magic_lines(source: str) -> str:
    """Blank out IPython magic/shell-escape lines so the cell can be parsed as Python."""

    return "".join(
        "pass\n" if MAGIC_LINE_RE.match(line) else line
        for line in source.splitlines(keepends=True)
    )


def _notebook_python_source(content: bytes) -> str | None:
    """Concatenate a notebook's code cells into one pseudo-module for analysis."""

    try:
        notebook = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(notebook, dict):
        return None
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        return None

    sources = []
    for cell in cells:
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if isinstance(source, str) and source:
            sources.append(_sanitize_magic_lines(source))

    return "\n\n".join(sources)


def _expand_notebook_star_imports(content: bytes, public_names: frozenset[str]) -> tuple[bytes, int]:
    combined_source = _notebook_python_source(content)
    if combined_source is None:
        return content, 0

    needed = _names_needing_import(combined_source, public_names)
    if not needed:
        return content, 0

    import_text = ("from qmcpy import " + ", ".join(sorted(needed))).encode()

    output: list[bytes] = []
    change_count = 0
    for line in content.splitlines(keepends=True):
        if NOTEBOOK_STAR_IMPORT_RE.match(line) is None:
            output.append(line)
            continue
        output.append(line.replace(STAR_IMPORT_LITERAL, import_text, 1))
        change_count += 1

    return b"".join(output), change_count


def _normalize_comma_spacing(
    content: bytes, eligible_lines: set[int] | None = None
) -> tuple[bytes, int]:
    """Ensure a space follows each comma in single-line qmcpy import statements."""

    output: list[bytes] = []
    change_count = 0
    for line_index, line in enumerate(content.splitlines(keepends=True)):
        if eligible_lines is not None and line_index not in eligible_lines:
            output.append(line)
            continue
        match = NAMED_IMPORT_LINE_RE.match(line)
        if match is None:
            output.append(line)
            continue

        imported = match.group("imported")
        normalized = COMMA_SPACING_RE.sub(b", ", imported)
        if normalized == imported:
            output.append(line)
            continue

        change_count += 1
        output.append(match.group("prefix") + normalized + _line_ending(line))

    return b"".join(output), change_count


def flatten_imports(
    content: bytes,
    public_names: frozenset[str] | None = None,
    *,
    protect_python: bool = True,
) -> tuple[bytes, int]:
    """Flatten, combine, alphabetize, and deduplicate public imports.

    `public_names` is qmcpy's public API surface (see `_load_qmcpy_public_names`).
    When it's None, nested imports are left unchanged and existing top-level
    star imports are deduplicated but left unexpanded. `protect_python` should
    be true for Python files so strings and comments are never rewritten.
    """

    change_count = 0
    notebook_code_lines = _notebook_code_source_lines(content)
    is_notebook = notebook_code_lines is not None
    protected_lines = (
        None
        if is_notebook
        else _python_protected_lines(content) if protect_python else set()
    )

    def eligible_text_lines(current: bytes) -> set[int] | None:
        if not protect_python:
            return None
        current_protected = _python_protected_lines(current)
        if current_protected is None:
            return set()
        return {
            index
            for index, _ in enumerate(current.splitlines(keepends=True))
            if index + 1 not in current_protected
        }

    def _line_number(position: int) -> int:
        return content.count(b"\n", 0, position) + 1

    def replace(match: re.Match[bytes]) -> bytes:
        nonlocal change_count
        if protected_lines is None or _line_number(match.start()) in protected_lines:
            return match.group(0)
        replacement = _flatten_nested_import_match(match, public_names)
        if replacement is None:
            return match.group(0)
        change_count += 1
        return replacement

    if is_notebook:
        assert notebook_code_lines is not None
        updated, nested_count = _flatten_notebook_nested_imports(
            content, public_names, notebook_code_lines
        )
        change_count += nested_count
        notebook_code_lines = _notebook_code_source_lines(updated) or set()
        updated, duplicate_count = _deduplicate_adjacent_star_imports(
            updated, notebook_code_lines
        )
    else:
        updated = QMCPY_IMPORT_RE.sub(replace, content)
        updated, duplicate_count = _deduplicate_adjacent_star_imports(
            updated, eligible_text_lines(updated)
        )
    change_count += duplicate_count

    # Star-import expansion is intentionally disabled because the current
    # file-wide name analysis is not scope/order-aware and can change semantics.

    if is_notebook:
        notebook_code_lines = _notebook_code_source_lines(updated) or set()
        updated, combine_count = _combine_notebook_named_imports(
            updated, notebook_code_lines
        )
    else:
        updated, combine_count = _combine_text_named_imports(
            updated, eligible_text_lines(updated)
        )
    change_count += combine_count

    if not is_notebook:
        updated, comma_count = _normalize_comma_spacing(
            updated, eligible_text_lines(updated)
        )
        change_count += comma_count

    return updated, change_count


def _is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_SUFFIXES


def iter_target_files(paths: Iterable[Path]) -> Iterator[Path]:
    """Yield supported files under paths, pruning generated and cache directories."""

    seen: set[Path] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)

        if path.is_file():
            if not _is_supported(path):
                raise ValueError(f"unsupported file type: {path}")
            candidates = [path]
        else:
            candidates = []
            for root, directory_names, file_names in os.walk(path):
                directory_names[:] = sorted(
                    name
                    for name in directory_names
                    if name not in SKIPPED_DIRECTORIES
                    and not name.endswith((".egg-info", ".dist-info"))
                )
                root_path = Path(root)
                candidates.extend(
                    root_path / name
                    for name in sorted(file_names)
                    if _is_supported(Path(name))
                )

        for candidate in candidates:
            if candidate.is_symlink():
                continue
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield candidate


def _display_path(path: Path, base: Path) -> Path:
    try:
        return path.resolve().relative_to(base)
    except ValueError:
        return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report files that need changes without rewriting them",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="files or directories to process (default: repository root)",
    )
    args = parser.parse_args(argv)

    repository_root = Path(__file__).resolve().parent.parent
    paths = args.paths or [repository_root]

    try:
        targets = sorted(iter_target_files(paths), key=str)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    public_names = _load_qmcpy_public_names(repository_root)
    if public_names is None:
        print(
            "warning: qmcpy is not importable; nested imports will be left "
            "unchanged and top-level star imports will not be expanded",
            file=sys.stderr,
        )

    changed = []  # list of (display_path, import_count)
    changed_imports = 0
    for path in targets:
        original = path.read_bytes()
        updated, count = flatten_imports(
            original,
            public_names,
            protect_python=path.suffix.lower() in {".py", ".pyi"},
        )
        if not count:
            continue

        changed.append((_display_path(path, repository_root), count))
        changed_imports += count
        if not args.check:
            path.write_bytes(updated)

    action = "would update" if args.check else "updated"
    if changed:
        file_label = "file" if len(changed) == 1 else "files"
        import_label = "import" if changed_imports == 1 else "imports"
        print(
            f"qmcpy imports {action}: {len(changed)} {file_label}, "
            f"{changed_imports} {import_label}:"
        )
        for display_path, count in sorted(changed):
            per = "import" if count == 1 else "imports"
            print(f"  {display_path} ({count} {per})")
    else:
        print(f"  qmcpy imports {action}: 0 files")

    return int(args.check and bool(changed))


if __name__ == "__main__":
    raise SystemExit(main())
