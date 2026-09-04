#!/usr/bin/env python3
"""Check that public docstrings under ``qmcpy/`` follow Google style.

A *public* object is a module, a class / function / method whose name does not
start with ``_``, or the ``__init__`` of a public class (QMCPy documents
constructor arguments in ``__init__``'s own docstring).  Only module-level
functions/classes and the methods of public classes are inspected (helpers
nested inside functions are skipped).  For every such docstring this script
flags:

* ``missing``                  -- public class / function / method has no
                                  docstring (suppressed by ``--skip-missing``)
* ``missing-summary``          -- the docstring opens straight with a section
                                  header (``Args:``, ``Returns:``, ...) instead
                                  of a one-line summary.  This is the common
                                  cause of pydoclint's opaque ``DOC001``.
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
    python scripts/check_docstring.py [PATH ...] [--strict] [--quiet]
                                      [--skip-missing] [--diff [REF]]

PATH defaults to ``qmcpy``.  Informational by default (exit 0); ``--strict``
makes the exit code non-zero when anything is flagged, so it can gate CI
(``STRICT=--strict make check_docstring``).  ``--diff [REF]`` (REF defaults to
``develop``) prints a second summary restricted to the scanned files that
changed relative to REF -- committed on the branch, modified in the working
tree, or untracked.
"""
from __future__ import annotations

import ast
import re
import subprocess
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
                if not isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if not sub.name.startswith("_"):
                    yield sub, "method"
                elif sub.name == "__init__":
                    yield sub, "constructor"


def _doc_node(node):
    """Return the string-literal node holding ``node``'s docstring, or None."""
    body = getattr(node, "body", None)
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        return body[0].value
    return None


def _is_section_word(s):
    """Return the section word if ``s`` is a lone Google/NumPy section header."""
    word = s[:-1].strip() if s.endswith(":") else s
    if word in GOOGLE_SECTIONS or word in NUMPY_SECTIONS:
        return word
    return None


def check_file(path, skip_missing=False):
    """Return a list of ``(lineno, category, detail)`` findings for one file."""
    findings = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node, kind in _iter_public(tree):
        dnode = _doc_node(node)
        if dnode is None:
            # A bare "no docstring" finding is noise for __init__ (pydoclint
            # owns constructor-argument coverage) and meaningless for a module.
            if kind not in ("module", "constructor") and not skip_missing:
                findings.append((
                    getattr(node, "lineno", 1), "missing",
                    f"public {kind} `{getattr(node, 'name', path.stem)}` has no docstring",
                ))
            continue
        lines = dnode.value.split("\n")
        first_nonblank = next((j for j, ln in enumerate(lines) if ln.strip()), None)
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
                if i == first_nonblank and kind != "module":
                    findings.append((
                        dnode.lineno + i, "missing-summary",
                        f"docstring opens with `{s}`; add a one-line summary first",
                    ))
                elif i > 0 and lines[i - 1].strip() != "":
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


def _changed_files(ref):
    """Return resolved paths of *.py files that changed relative to ``ref``.

    Union of files committed on the branch (``ref...HEAD``), files modified in
    the working tree, and untracked files.  Raises ``RuntimeError`` if git is
    unavailable or ``ref`` cannot be resolved.
    """
    commands = (
        ["git", "diff", "--name-only", "--diff-filter=ACMR", f"{ref}...HEAD"],
        ["git", "diff", "--name-only", "--diff-filter=ACMR", "HEAD"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    )
    names = set()
    for cmd in commands:
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, check=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"`{' '.join(cmd)}` failed: {exc}") from exc
        names.update(n for n in out.splitlines() if n.endswith(".py"))
    return {Path(n).resolve() for n in names}


def _summary(total, n_files, by_cat, label):
    """Format one summary line."""
    if total == 0:
        return f"{label}: no issues in {n_files} file(s)"
    breakdown = ", ".join(f"{v} {k}" for k, v in sorted(by_cat.items()))
    return f"{label}: {total} issue(s) across {n_files} file(s): {breakdown}"


def _parse_diff_flag(argv):
    """Pull ``--diff [REF]`` out of ``argv``; return (remaining_argv, ref|None)."""
    args, ref, i = [], None, 0
    while i < len(argv):
        a = argv[i]
        if a == "--diff":
            nxt = argv[i + 1] if i + 1 < len(argv) else ""
            if nxt and not nxt.startswith("-"):
                ref, i = nxt, i + 2
            else:
                ref, i = "develop", i + 1
            continue
        if a.startswith("--diff="):
            ref = a.split("=", 1)[1] or "develop"
            i += 1
            continue
        args.append(a)
        i += 1
    return args, ref


def main(argv):
    argv, diff_ref = _parse_diff_flag(list(argv))
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
    per_file = {}
    for f in files:
        try:
            findings = check_file(f, skip_missing=skip_missing)
        except SyntaxError as exc:
            print(f"{f.as_posix()}: skipped (syntax error: {exc})", file=sys.stderr)
            continue
        per_file[f] = findings
        for lineno, cat, detail in findings:
            by_cat[cat] = by_cat.get(cat, 0) + 1
            total += 1
            if not quiet:
                print(f"{f.as_posix()}:{lineno}: {cat}: {detail}")

    if not quiet:
        print()
    print(_summary(total, len(files), by_cat, f"{len(files)} file(s) scanned"))

    if diff_ref is not None:
        try:
            changed = _changed_files(diff_ref)
        except RuntimeError as exc:
            print(f"--diff {diff_ref}: skipped ({exc})", file=sys.stderr)
        else:
            sub_cat, sub_total, sub_files = {}, 0, 0
            for f, findings in per_file.items():
                if f.resolve() not in changed:
                    continue
                sub_files += 1
                for _, cat, _ in findings:
                    sub_cat[cat] = sub_cat.get(cat, 0) + 1
                    sub_total += 1
            print(_summary(sub_total, sub_files, sub_cat, f"changed vs {diff_ref}"))

    return 1 if (strict and total) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
