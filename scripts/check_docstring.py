#!/usr/bin/env python3
"""Check that public docstrings under ``qmcpy/`` follow Google style.

A *public* object is a module, or a class / function / method whose name does
not start with ``_``.  Only module-level functions/classes and the methods of
public classes are inspected (helpers nested inside functions are skipped).
For every public docstring this script flags:

* ``missing``                  -- public class / function / method has no
                                  docstring (suppressed by ``--skip-missing``)
* ``numpy-section``            -- a section written NumPy-style (``Returns``
                                  followed by a ``-----`` underline) instead of
                                  Google style (``Returns:``)
* ``no-blank-before-section``  -- a Google section header (``Args:``,
                                  ``Returns:``, ``Raises:``, ...) is not preceded
                                  by a blank line
* ``malformed-section-header`` -- a section word on its own line that is not the
                                  canonical ``Name:`` form (missing colon, a
                                  stray space before the colon, ...)

Usage:
    python scripts/check_docstring.py [PATH ...] [--strict] [--quiet] [--skip-missing]

PATH defaults to ``qmcpy``.  Informational by default (exit 0); ``--strict``
makes the exit code non-zero when anything is flagged, so it can gate CI
(``STRICT=--strict make check_docstring``).
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

# Canonical Google section headers, written as ``Name:`` on their own line.
GOOGLE_SECTIONS = {
    "Args", "Arguments", "Attributes", "Example", "Examples", "Keyword Args",
    "Note", "Notes", "Raises", "References", "Return", "Returns", "See Also",
    "Todo", "Warning", "Warnings", "Warns", "Yield", "Yields",
}
# Section words that, followed by a dashed underline, mean the docstring is
# using NumPy style rather than Google style.
NUMPY_SECTIONS = {
    "Parameters", "Other Parameters", "Returns", "Raises", "Yields",
    "Attributes", "Notes", "Examples", "See Also", "References", "Warns",
    "Warnings", "Methods",
}
_DASHES = re.compile(r"^-{3,}$")
# Canonical header: capitalised word(s), a single colon, nothing else.
_HEADER = re.compile(r"^([A-Z][A-Za-z]*(?: [A-Z][A-Za-z]*)*):$")


def _iter_public(tree):
    """Yield ``(node, kind)`` for the module plus its public API objects."""
    yield tree, "module"
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                yield node, "function"
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            yield node, "class"
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                        and not sub.name.startswith("_"):
                    yield sub, "method"


def _doc_node(node):
    """Return the string-literal node holding ``node``'s docstring, or None."""
    body = getattr(node, "body", None)
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        return body[0].value
    return None


def check_file(path, skip_missing=False):
    """Return a list of ``(lineno, category, detail)`` findings for one file."""
    findings = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node, kind in _iter_public(tree):
        dnode = _doc_node(node)
        if dnode is None:
            if kind != "module" and not skip_missing:
                findings.append((
                    getattr(node, "lineno", 1), "missing",
                    f"public {kind} `{getattr(node, 'name', path.stem)}` has no docstring",
                ))
            continue
        lines = dnode.value.split("\n")
        for i, raw in enumerate(lines):
            s = raw.strip()
            if not s:
                continue
            word = s[:-1].strip() if s.endswith(":") else s
            nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if word in NUMPY_SECTIONS and _DASHES.match(nxt):
                findings.append((
                    dnode.lineno + i, "numpy-section",
                    f"`{word}` written NumPy-style; use Google `{word}:`",
                ))
                continue
            m = _HEADER.match(s)
            if m and m.group(1) in GOOGLE_SECTIONS:
                if i > 0 and lines[i - 1].strip() != "":
                    findings.append((
                        dnode.lineno + i, "no-blank-before-section",
                        f"add a blank line before `{s}`",
                    ))
            elif word in GOOGLE_SECTIONS and not _DASHES.match(nxt):
                findings.append((
                    dnode.lineno + i, "malformed-section-header",
                    f"`{s}` is not the canonical `{word}:` form",
                ))
    return findings


def main(argv):
    strict = "--strict" in argv
    quiet = "--quiet" in argv
    skip_missing = "--skip-missing" in argv
    paths = [a for a in argv if not a.startswith("-")] or ["qmcpy"]

    files = []
    for p in map(Path, paths):
        files.extend(sorted(p.rglob("*.py")) if p.is_dir() else [p])
    if not files:
        print(f"no *.py files under {', '.join(paths)}", file=sys.stderr)
        return 1

    total = 0
    by_cat = {}
    for f in files:
        try:
            findings = check_file(f, skip_missing=skip_missing)
        except SyntaxError as exc:
            print(f"{f.as_posix()}: skipped (syntax error: {exc})", file=sys.stderr)
            continue
        for lineno, cat, detail in findings:
            by_cat[cat] = by_cat.get(cat, 0) + 1
            total += 1
            if not quiet:
                print(f"{f.as_posix()}:{lineno}: {cat}: {detail}")

    if not quiet:
        print()
    if total == 0:
        print(f"OK: {len(files)} file(s) scanned, public docstrings are Google style")
    else:
        summary = ", ".join(f"{v} {k}" for k, v in sorted(by_cat.items()))
        print(f"{total} issue(s) across {len(files)} file(s) scanned: {summary}")

    return 1 if (strict and total) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
