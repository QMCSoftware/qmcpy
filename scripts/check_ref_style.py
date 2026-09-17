#!/usr/bin/env python3
"""Check (and, for unambiguous cases, fix) that bibliography/"References"
sections across ``qmcpy/`` docstrings, ``*.md`` files, and ``demos/**/*.ipynb``
notebooks follow this project's house citation convention: IEEE-style numbered
brackets (``[1]``, ``[2]``, ...) in citation order, one entry per number.

Scope, and what is deliberately excluded:

* ``qmcpy/**/*.py`` docstrings -- the ``References:`` section recognised by
  ``scripts/check_docstring.py``.
* Root ``*.md`` files and source ``docs/*.md`` files -- but NOT the files
  ``make copydocs`` generates as build-artifact copies (``docs/README.md``,
  ``docs/AGENTS.md``, ``docs/CONTRIBUTING.md``, ``docs/community.md``,
  ``docs/demos/``, ``docs/paper/``); checking a generated copy would just
  duplicate (or, once it drifts, contradict) findings from its source.
* ``demos/**/*.ipynb`` markdown cells.
* ``paper/paper.md`` and every ``*.bib`` file are OUT OF SCOPE on purpose:
  that paper is built by Pandoc from BibTeX keys (``[@key]``) resolved
  against ``paper/paper.bib`` under a citation-style-language template, a
  completely different (and already-correct) mechanism. Rewriting its prose
  citations here would not match what Pandoc actually renders.

What is checked (informational by default; ``--strict`` fails the build):

* ``header-not-colon`` (docstrings only) -- ``References`` / ``**References**``
  instead of the canonical ``References:`` every other Google-style section
  uses (see ``docs/good_practices.md``).
* ``non-bracket-marker`` -- an entry numbered some way other than a plain
  ``[N]`` (``$[N]$``, ``N.``, an unnumbered bullet, an HTML anchor, ...).
* ``numbering-not-sequential`` -- a section's bracket/dot numbers are not
  exactly ``1, 2, 3, ...`` in the order they appear.
* ``missing-year`` -- an entry with no 4-digit year anywhere in its text.

What ``--fix`` actually rewrites (only the unambiguous, purely presentational
cases; everything else above is reported but left for a human, since blindly
renumbering or reformatting free-text citations risks breaking a
cross-reference elsewhere in the same document or silently mangling content
this script cannot reliably parse):

* ``**References**`` / ``References`` (bare header line) -> ``References:``
  in docstrings.
* ``$[N]$`` -> ``[N]`` in markdown / notebooks (strips a LaTeX math-mode
  wrapper some notebooks used around an otherwise-correct bracket number).

Usage:
    python scripts/check_ref_style.py [PATH ...] [--strict] [--fix] [--quiet]
                                      [--diff [REF]]

PATH defaults to the whole repository. ``--fix`` rewrites files in place and
then reports what remains; combine with ``--strict`` to fail if anything is
still flagged after fixing. ``--diff [REF]`` (REF defaults to ``develop``)
restricts scanning/fixing to files that changed relative to REF -- committed
on the branch, modified in the working tree, or untracked.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Build-artifact copies made by `make copydocs` -- never scan these directly,
# their source (root *.md or demos/) is already in scope.
_GENERATED_MD = {
    REPO_ROOT / "docs" / "README.md",
    REPO_ROOT / "docs" / "AGENTS.md",
    REPO_ROOT / "docs" / "CONTRIBUTING.md",
    REPO_ROOT / "docs" / "community.md",
}
_GENERATED_DIRS = (REPO_ROOT / "docs" / "demos", REPO_ROOT / "docs" / "paper")

# A docstring/markdown/notebook line that *is* a References-type heading.
_DOCSTRING_HEADER = re.compile(r"^\*{0,2}References:?\*{0,2}\s*$")
_MD_HEADING = re.compile(r"^#{1,6}\s.*\bReferences\b", re.IGNORECASE)
_MD_BOLD_HEADING = re.compile(r"^\*\*[^*]*\bReferences\b[^*]*\*\*\s*$", re.IGNORECASE)
# Any other canonical Google docstring section that would end a References
# block (kept in sync with check_docstring.py's GOOGLE_SECTIONS).
_NEXT_DOCSTRING_SECTION = re.compile(
    r"^\*{0,2}(Args|Arguments|Attributes|Example|Examples|Keyword Args|Note|"
    r"Notes|Raises|Return|Returns|See Also|Todo|Warning|Warnings|Warns|Yield|"
    r"Yields)\*{0,2}:?\s*$"
)

# Entry markers, tried in this order. Each yields (kind, number|None, rest).
_MARKERS = [
    ("dollar-bracket", re.compile(r"^\$\[(\d+)\]\$\s*(.*)$")),
    ("bracket", re.compile(r"^\[(\d+)\]\s*(.*)$")),
    ("dot", re.compile(r"^(\d+)\.\s+(.*)$")),
    ("anchor", re.compile(r'^<a\s+id="[^"]*">\s*</a>\s*(.*)$')),
    ("bullet", re.compile(r"^[-*]\s+(.*)$")),
]
_YEAR = re.compile(r"(?:19|20)\d{2}")


def _classify_marker(line):
    """Return (kind, number_or_None, rest_of_line) if `line` starts an entry."""
    for kind, pattern in _MARKERS:
        m = pattern.match(line)
        if m:
            groups = m.groups()
            number = int(groups[0]) if kind != "anchor" and kind != "bullet" else None
            rest = groups[-1]
            return kind, number, rest
    return None


class Section:
    """One detected References section: its header and parsed entries."""

    def __init__(self, header_lineno, header_text, is_docstring):
        self.header_lineno = header_lineno
        self.header_text = header_text
        self.is_docstring = is_docstring
        self.entries = []  # list of [lineno, kind, number, text] (mutable)

    def add(self, lineno, kind, number, text):
        self.entries.append([lineno, kind, number, text])

    def extend_last(self, text):
        """Append a continuation line to the most recently added entry."""
        if self.entries:
            self.entries[-1][3] = f"{self.entries[-1][3]} {text}".strip()


def _find_sections_in_lines(lines, is_docstring):
    """Split `lines` (1-indexed access via enumerate below) into Sections.

    A section runs from a References-type header to the next header of the
    same kind (docstring: any canonical section; markdown: any heading line
    at or above the References heading's own level, approximated here by any
    line starting with '#') or end of text.
    """
    sections = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip("\n")
        is_header = (
            _DOCSTRING_HEADER.match(line.strip())
            if is_docstring
            else (_MD_HEADING.match(line.strip()) or _MD_BOLD_HEADING.match(line.strip()))
        )
        if not is_header:
            i += 1
            continue
        section = Section(i + 1, line.strip(), is_docstring)
        j = i + 1
        entry_open = False  # True right after a marker line, until a blank line
        while j < n:
            body_line = lines[j].rstrip("\n")
            stripped = body_line.strip()
            if is_docstring and _NEXT_DOCSTRING_SECTION.match(stripped):
                break
            if is_docstring and stripped in ('"""', "'''"):
                # End of the docstring itself -- stop before folding the
                # closing triple-quote into the last entry's text.
                break
            if not is_docstring and stripped.startswith("#"):
                break
            if not stripped:
                entry_open = False
                j += 1
                continue
            classified = _classify_marker(stripped)
            if classified is not None:
                kind, number, rest = classified
                # An anchor-only line (`<a id="ref1"></a>`) is usually
                # immediately followed by the real `[N] ...` marker line;
                # fold it into that next entry instead of double-counting.
                if kind == "anchor" and not rest:
                    j += 1
                    continue
                section.add(j + 1, kind, number, rest)
                entry_open = True
            elif entry_open:
                # A continuation line of the entry above (e.g. title/venue/year
                # on their own lines) -- fold into that entry's text so
                # `missing-year` sees the whole citation, not just its first line.
                section.extend_last(stripped)
            j += 1
        sections.append(section)
        i = j
    return sections


def _check_section(path, section):
    """Return a list of (lineno, category, detail) findings for one section."""
    findings = []
    if section.is_docstring and ":" not in section.header_text:
        findings.append((
            section.header_lineno, "header-not-colon",
            f"`{section.header_text}` should be `References:` "
            "(matches every other Google-style section header)",
        ))
    numbered_kinds = {"bracket", "dollar-bracket", "dot"}
    seen_numbers = []
    for lineno, kind, number, text in section.entries:
        if kind != "bracket":
            findings.append((
                lineno, "non-bracket-marker",
                f"entry marker is {kind!r}, not a plain `[N]`: {text[:60]!r}",
            ))
        if kind in numbered_kinds and number is not None:
            seen_numbers.append((lineno, number))
        if not _YEAR.search(text):
            findings.append((
                lineno, "missing-year",
                f"no 4-digit year found in entry: {text[:60]!r}",
            ))
    expected = list(range(1, len(seen_numbers) + 1))
    actual = [num for _, num in seen_numbers]
    if seen_numbers and actual != expected:
        findings.append((
            seen_numbers[0][0], "numbering-not-sequential",
            f"entries numbered {actual} in a References section with "
            f"{len(seen_numbers)} entries; expected {expected}",
        ))
    return findings


def _fix_lines(lines, is_docstring):
    """Apply only the unambiguous fixes; return (new_lines, changed)."""
    changed = False
    new_lines = []
    for line in lines:
        stripped = line.rstrip("\n")
        newline_suffix = line[len(stripped):]
        fixed = stripped
        if is_docstring and _DOCSTRING_HEADER.match(fixed.strip()) and ":" not in fixed.strip():
            leading_ws = fixed[: len(fixed) - len(fixed.lstrip())]
            body = fixed.strip()
            if body.startswith("**") and body.endswith("**"):
                # `**References**` -> `**References:**` (colon inside the bold markers)
                fixed = f"{leading_ws}**{body[2:-2]}:**"
            else:
                fixed = f"{leading_ws}{body}:"
        fixed, n = re.subn(r"\$\[(\d+)\]\$", r"[\1]", fixed)
        if fixed != stripped:
            changed = True
        new_lines.append(fixed + newline_suffix)
    return new_lines, changed


# --------------------------------------------------------------------------
# File discovery
# --------------------------------------------------------------------------

def _iter_python_files(root):
    yield from sorted((root / "qmcpy").rglob("*.py"))


def _iter_markdown_files(root):
    for p in sorted(root.glob("*.md")):
        yield p
    for p in sorted((root / "docs").rglob("*.md")):
        if p in _GENERATED_MD or any(str(p).startswith(str(d)) for d in _GENERATED_DIRS):
            continue
        yield p


def _iter_notebook_files(root):
    yield from sorted((root / "demos").rglob("*.ipynb"))


# --------------------------------------------------------------------------
# Per-file-type check/fix
# --------------------------------------------------------------------------

def check_python_file(path):
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    findings = []
    for section in _find_sections_in_lines(lines, is_docstring=True):
        findings.extend(_check_section(path, section))
    return findings


def fix_python_file(path):
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    new_lines, changed = _fix_lines(lines, is_docstring=True)
    if changed:
        path.write_text("".join(new_lines), encoding="utf-8")
    return changed


def check_markdown_file(path):
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    findings = []
    for section in _find_sections_in_lines(lines, is_docstring=False):
        findings.extend(_check_section(path, section))
    return findings


def fix_markdown_file(path):
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    new_lines, changed = _fix_lines(lines, is_docstring=False)
    if changed:
        path.write_text("".join(new_lines), encoding="utf-8")
    return changed


def _notebook_markdown_cells(notebook):
    for idx, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") == "markdown":
            yield idx, cell


def check_notebook_file(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    findings = []
    for idx, cell in _notebook_markdown_cells(notebook):
        lines = cell.get("source", [])
        for section in _find_sections_in_lines(lines, is_docstring=False):
            for lineno, cat, detail in _check_section(path, section):
                findings.append((f"cell {idx}, line {lineno}", cat, detail))
    return findings


def fix_notebook_file(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for _, cell in _notebook_markdown_cells(notebook):
        lines = cell.get("source", [])
        new_lines, cell_changed = _fix_lines(lines, is_docstring=False)
        if cell_changed:
            cell["source"] = new_lines
            changed = True
    if changed:
        # ensure_ascii=False: keep existing unicode bytes as-is (matches how
        # this repo's notebooks are already saved) instead of rewriting every
        # non-ASCII character in the file into a \uXXXX escape, which would
        # turn a one-line fix into a repo-wide encoding diff.
        path.write_text(
            json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return changed


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def _display_path(f, root):
    """Path for printing: relative to `root` when possible, absolute otherwise."""
    try:
        return f.relative_to(root).as_posix()
    except ValueError:
        return f.as_posix()


def _changed_files(ref):
    """Return resolved paths of files that changed relative to `ref`.

    Union of files committed on the branch (``ref...HEAD``), files modified
    in the working tree, and untracked files. Raises ``RuntimeError`` if git
    is unavailable or ``ref`` cannot be resolved.
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
                cmd, capture_output=True, text=True, check=True, cwd=REPO_ROOT,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"`{' '.join(cmd)}` failed: {exc}") from exc
        names.update(out.splitlines())
    return {(REPO_ROOT / n).resolve() for n in names}


def _parse_diff_flag(argv):
    """Pull ``--diff [REF]`` out of `argv`; return (remaining_argv, ref|None)."""
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

    if paths:
        py_files, md_files, nb_files = [], [], []
        for p in map(Path, paths):
            p = p.resolve()
            if p.suffix == ".py":
                py_files.append(p)
            elif p.suffix == ".md":
                md_files.append(p)
            elif p.suffix == ".ipynb":
                nb_files.append(p)
            elif p.is_dir():
                py_files.extend(sorted(p.rglob("*.py")))
                md_files.extend(sorted(p.rglob("*.md")))
                nb_files.extend(sorted(p.rglob("*.ipynb")))
    else:
        py_files = list(_iter_python_files(root))
        md_files = list(_iter_markdown_files(root))
        nb_files = list(_iter_notebook_files(root))

    if diff_ref is not None:
        try:
            changed = _changed_files(diff_ref)
        except RuntimeError as exc:
            print(f"--diff {diff_ref}: skipped ({exc})", file=sys.stderr)
        else:
            py_files = [f for f in py_files if f.resolve() in changed]
            md_files = [f for f in md_files if f.resolve() in changed]
            nb_files = [f for f in nb_files if f.resolve() in changed]

    checkers = (
        (py_files, check_python_file, fix_python_file),
        (md_files, check_markdown_file, fix_markdown_file),
        (nb_files, check_notebook_file, fix_notebook_file),
    )

    total = 0
    by_cat = {}
    per_file = {}
    n_fixed = 0
    for files, check_fn, fix_fn in checkers:
        for f in files:
            try:
                if do_fix:
                    if fix_fn(f):
                        n_fixed += 1
                findings = check_fn(f)
            except (SyntaxError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                print(f"{_display_path(f, root)}: skipped ({exc})", file=sys.stderr)
                continue
            if findings:
                per_file[f] = findings
                for _, cat, _ in findings:
                    by_cat[cat] = by_cat.get(cat, 0) + 1
                    total += 1

    n_files = len(py_files) + len(md_files) + len(nb_files)
    if total and not quiet:
        print()
        for f, findings in per_file.items():
            for lineno, cat, detail in findings:
                print(f"  - {_display_path(f, root)}:{lineno}: {cat}: {detail}")
    if do_fix:
        print(f"  - fixed {n_fixed} file(s) (header colon / `$[N]$` -> `[N]`)")
    print("  - " + _summary(total, n_files, by_cat, f"{n_files} file(s) scanned"))

    if total == 0:
        print(f"clean  (0 of {n_files} files)")
    else:
        prefix = "ERROR" if (strict and total) else "WARNING"
        print(f"{prefix}: {len(per_file)} file(s) with issues  ({len(per_file)} of {n_files} files)")
    return 1 if (strict and total) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
