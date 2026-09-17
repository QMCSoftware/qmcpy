#!/usr/bin/env python3
"""Check (and fix) that a Google-style docstring section's body is indented
deeper than its own header line.

Google-style sections (``Args:``, ``Returns:``, ``Examples:``, ...) delimit
their body by indentation, not by a closing marker. ``mkdocstrings``' Google
docstring parser (via ``griffe``) relies on that indentation to know where a
section's content starts and ends -- a body written at the SAME indent as its
header (instead of one level deeper) is not recognised as belonging to the
section, which is why e.g. an ``Examples:`` section's ``>>>`` lines can fail
to render as a doctest block in the built HTML even though they look fine as
plain text/in an IDE.

Wrong (body at the same indent as the header):

    Examples:
    >>> f(2)
    4

Right (body indented one level deeper):

    Examples:
        >>> f(2)
        4

A section's body is scanned one "chunk" at a time, where a chunk is a maximal
run of consecutive non-blank lines (chunks are separated by blank lines --
typically distinct paragraphs, bullet points, or doctest blocks). Only a
chunk whose OWN least-indented line is at or below the header's indentation
is flagged/fixed; a chunk that is already indented deeper than the header is
left alone even if a sibling chunk in the same section is broken. This
matters because some sections (e.g. a ``Notes:`` block with several
paragraphs at different, individually-correct depths) are not uniformly
indented to begin with.

``References:`` is exempt from this check: mkdocstrings/griffe does not
require a references/bibliography block's body to be indented deeper than
the header to render correctly, so a ``References:`` body written at the
same indent as its own header is fine and is never flagged.

What is checked (informational by default; ``--strict`` fails the build):

* ``section-not-indented`` -- a chunk within a (non-``References``) Google
  section's body has at least one non-blank line at or below the header's own
  indentation. The body runs from the line after the header to the next
  Google section header (regardless of ITS indentation -- sections are
  siblings, never nested one inside another, even if a docstring
  inconsistently writes one deeper than another), the docstring's closing
  ``\"\"\"``/``'''``, or EOF.

What ``--fix`` does: shifts every line in a flagged chunk right by a constant
number of spaces, just enough to bring its least-indented line to the
header's indent plus 4. This is a uniform shift -- it does not change any
line's indentation RELATIVE to the others in the chunk, so an already-nested
continuation line (e.g. a wrapped array repr or a ``...`` doctest
continuation) keeps its own relative offset. The docstring's closing quote
line is never shifted. A chunk that is already indented deeper than its
header (even if inconsistently deeper than intended) is left untouched, and
so is every chunk of a ``References:`` section.

Usage:
    python scripts/check_docstring_indent.py [PATH ...] [--strict] [--fix]
                                             [--quiet] [--diff [REF]]

PATH defaults to ``qmcpy``. ``--diff [REF]`` (REF defaults to ``develop``)
restricts scanning/fixing to files that changed relative to REF -- committed
on the branch, modified in the working tree, or untracked.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Kept in sync with check_docstring.py's GOOGLE_SECTIONS.
GOOGLE_SECTIONS = {
    "Args", "Arguments", "Attributes", "Example", "Examples", "Keyword Args",
    "Note", "Notes", "Raises", "References", "Return", "Returns", "See Also",
    "Todo", "Warning", "Warnings", "Warns", "Yield", "Yields",
}
# Sections whose body is exempt from the "must be indented deeper than the
# header" requirement -- mkdocstrings/griffe renders these fine either way.
NO_INDENT_REQUIRED = {"References"}
_HEADER = re.compile(
    r"^(\s*)\*{0,2}(" + "|".join(sorted(GOOGLE_SECTIONS, key=len, reverse=True))
    + r"):\*{0,2}\s*$"
)


def _iter_chunks(lines, start, end):
    """Yield (chunk_start, chunk_end) for each maximal run of consecutive
    non-blank lines within lines[start:end) (exclusive end)."""
    i = start
    while i < end:
        if not lines[i].strip():
            i += 1
            continue
        j = i
        while j < end and lines[j].strip():
            j += 1
        yield i, j
        i = j


def _find_flagged_chunks(lines):
    """Yield (header_lineno, header_indent, chunk_start, chunk_end) for every
    chunk (a maximal run of consecutive non-blank lines) within a Google
    section's body whose least-indented line is not deeper than the
    section's own header. `References:` sections are never flagged.

    `chunk_end` is exclusive; `lines` is 0-indexed (as from `str.split("\\n")`).
    """
    n = len(lines)
    i = 0
    while i < n:
        m = _HEADER.match(lines[i])
        if m is None:
            i += 1
            continue
        header_indent = len(m.group(1))
        section_name = m.group(2)
        j = i + 1
        while j < n:
            stripped = lines[j].strip()
            if stripped in ('"""', "'''"):
                break
            if _HEADER.match(lines[j]) is not None:
                # Any subsequent canonical section header ends this section's
                # body, regardless of its own indent -- Google-style sections
                # are siblings, never nested one inside another, so a header
                # at a DEEPER indent than the current one (a pre-existing
                # inconsistency in some docstrings) must still end the scan
                # here rather than being swallowed as more body content.
                break
            j += 1
        body_start, body_end = i + 1, j
        if section_name not in NO_INDENT_REQUIRED:
            for chunk_start, chunk_end in _iter_chunks(lines, body_start, body_end):
                indents = [
                    len(lines[k]) - len(lines[k].lstrip())
                    for k in range(chunk_start, chunk_end)
                ]
                if min(indents) <= header_indent:
                    yield i + 1, header_indent, chunk_start, chunk_end
        i = j


def check_file(path):
    """Return a list of (lineno, category, detail) findings for one file."""
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    findings = []
    for header_lineno, header_indent, chunk_start, chunk_end in _find_flagged_chunks(lines):
        findings.append((
            header_lineno, "section-not-indented",
            f"chunk (lines {chunk_start + 1}-{chunk_end}) is not indented "
            f"deeper than its section header (indent {header_indent})",
        ))
    return findings


def fix_file(path):
    """Apply the uniform-shift fix in place; return True if the file changed."""
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    changed = False
    # Re-scan after each fix since line content (not count) changes; indices
    # into `lines` stay valid because a shift never adds/removes lines.
    while True:
        flagged = list(_find_flagged_chunks(lines))
        if not flagged:
            break
        header_lineno, header_indent, chunk_start, chunk_end = flagged[0]
        indents = [
            len(lines[k]) - len(lines[k].lstrip())
            for k in range(chunk_start, chunk_end)
        ]
        shift = (header_indent + 4) - min(indents)
        for k in range(chunk_start, chunk_end):
            lines[k] = " " * shift + lines[k]
        changed = True
    if changed:
        path.write_text("\n".join(lines), encoding="utf-8")
    return changed


def _iter_python_files(root, paths):
    if not paths:
        yield from sorted((root / "qmcpy").rglob("*.py"))
        return
    for p in map(Path, paths):
        p = p.resolve()
        if p.is_dir():
            yield from sorted(p.rglob("*.py"))
        elif p.suffix == ".py":
            yield p


def _changed_files(ref):
    """Return resolved paths of *.py files that changed relative to `ref`."""
    commands = (
        ["git", "diff", "--name-only", "--diff-filter=ACMR", f"{ref}...HEAD"],
        ["git", "diff", "--name-only", "--diff-filter=ACMR", "HEAD"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    )
    names = set()
    for cmd in commands:
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, check=True, cwd=REPO_ROOT,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"`{' '.join(cmd)}` failed: {exc}") from exc
        names.update(n for n in out.splitlines() if n.endswith(".py"))
    return {(REPO_ROOT / n).resolve() for n in names}


def _parse_diff_flag(argv):
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


def _summary(total, n_files, by_cat, label):
    if total == 0:
        return f"{label}: no issues in {n_files} file(s)"
    breakdown = ", ".join(f"{v} {k}" for k, v in sorted(by_cat.items()))
    return f"{label}: {total} issue(s) across {n_files} file(s): {breakdown}"


def main(argv):
    argv, diff_ref = _parse_diff_flag(list(argv))
    strict = "--strict" in argv
    quiet = "--quiet" in argv
    do_fix = "--fix" in argv
    paths = [a for a in argv if not a.startswith("-")]
    root = REPO_ROOT

    files = list(_iter_python_files(root, paths))
    if diff_ref is not None:
        try:
            changed = _changed_files(diff_ref)
        except RuntimeError as exc:
            print(f"--diff {diff_ref}: skipped ({exc})", file=sys.stderr)
        else:
            files = [f for f in files if f.resolve() in changed]

    total = 0
    by_cat = {}
    per_file = {}
    n_fixed = 0
    for f in files:
        try:
            if do_fix and fix_file(f):
                n_fixed += 1
            findings = check_file(f)
        except UnicodeDecodeError as exc:
            print(f"{f.as_posix()}: skipped ({exc})", file=sys.stderr)
            continue
        if findings:
            per_file[f] = findings
            for _, cat, _ in findings:
                by_cat[cat] = by_cat.get(cat, 0) + 1
                total += 1

    if total and not quiet:
        print()
        for f, findings in per_file.items():
            for lineno, cat, detail in findings:
                print(f"  - {f.as_posix()}:{lineno}: {cat}: {detail}")
    if do_fix:
        print(f"  - fixed {n_fixed} file(s) (re-indented section bodies)")
    print("  - " + _summary(total, len(files), by_cat, f"{len(files)} file(s) scanned"))

    if total == 0:
        print(f"clean  (0 of {len(files)} files)")
    else:
        prefix = "ERROR" if (strict and total) else "WARNING"
        print(f"{prefix}: {len(per_file)} file(s) with issues  ({len(per_file)} of {len(files)} files)")
    return 1 if (strict and total) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
