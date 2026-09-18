#!/usr/bin/env python3
"""Check that LaTeX math commands in docstrings/Markdown/notebooks are
actually wrapped in ``$...$`` / ``$$...$$`` math-mode delimiters.

A raw LaTeX command like ``\\boldsymbol{x}`` or ``\\in`` typed outside a
``$...$``/``$$...$$`` span is not math to a Markdown renderer (mkdocstrings,
GitHub, Jupyter) -- it prints as the literal, garbled command text in the
built HTML/rendered notebook. This script flags exactly that: a backslash
command from a fixed allowlist of common math commands (Greek letters,
``\\boldsymbol``, ``\\int``, ``\\sum``, ``\\in``, ...) found outside any
math-mode span.

This is deliberately narrow. Whether a plain-English phrase like "x in
(0,1)^d" *should* be math is a judgment call (see the PR #613 review) --
automating that would flag far too much prose and be routinely ignored. A
bare LaTeX command leaking outside `$...$` is not a judgment call: it is
mechanically wrong wherever it appears, which is exactly the kind of
unambiguous check this repo's other check_*.py scripts focus on.

Scope: every string literal in ``qmcpy/**/*.py`` (this includes every
docstring; a ``#`` comment is deliberately excluded, since it is never
rendered as documentation and can legitimately hold informal math-ish
shorthand), root ``*.md`` and
source ``docs/*.md`` files (excluding ``make copydocs`` build-artifact
copies, same exclusion list as check_ref_style.py), and ``demos/**/*.ipynb``
markdown cells. Fenced ```` ``` ```` code blocks in Markdown/notebooks are
skipped (a code sample legitimately printing a raw LaTeX string is not this
bug).

There is no ``--fix``: deciding exactly what span to wrap in ``$...$`` is a
judgment call this script does not attempt to automate (see PRODUCT_MEASURE
and SCIPY_WRAPPER fixes made by hand in the same review that prompted this
tool). Informational by default; ``--strict`` fails the build.

Usage:
    python scripts/check_latex_math.py [PATH ...] [--strict] [--quiet]
                                       [--diff [REF]]

PATH defaults to the whole repository. ``--diff [REF]`` (REF defaults to
``develop``) restricts scanning to files that changed relative to REF --
committed on the branch, modified in the working tree, or untracked.
"""
from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import tokenize
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Build-artifact copies made by `make copydocs` -- never scan these directly,
# their source (root *.md or demos/) is already in scope. Kept in sync with
# check_ref_style.py's identical exclusion list.
_GENERATED_MD = {
    REPO_ROOT / "docs" / "README.md",
    REPO_ROOT / "docs" / "AGENTS.md",
    REPO_ROOT / "docs" / "CONTRIBUTING.md",
    REPO_ROOT / "docs" / "community.md",
}
_GENERATED_DIRS = (REPO_ROOT / "docs" / "demos", REPO_ROOT / "docs" / "paper")

# Common math-mode commands. Deliberately a fixed allowlist, not "any
# backslash-letters" -- that would also match Windows paths (\Users\...),
# regex escapes (\d, \s), and other legitimate non-math backslashes.
_LATEX_COMMANDS = {
    "boldsymbol", "mathbb", "mathcal", "mathrm", "mathbf", "operatorname", "text",
    "frac", "sqrt", "sum", "prod", "int", "iint", "oint", "lim", "log", "exp",
    "det", "bmod", "pmod", "left", "right", "qquad", "quad", "sim", "propto",
    "approx", "equiv", "neq", "geq", "leq", "ll", "gg", "in", "notin", "ni",
    "subset", "subseteq", "supset", "supseteq", "cup", "cap", "setminus",
    "emptyset", "varnothing", "forall", "exists", "infty", "partial", "nabla",
    "times", "cdot", "cdots", "ldots", "dots", "vdots", "ddots", "to",
    "rightarrow", "leftarrow", "Rightarrow", "Leftrightarrow", "mapsto",
    "binom", "perp", "parallel", "otimes", "oplus", "circ", "pm", "mp",
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta",
    "theta", "vartheta", "iota", "kappa", "lambda", "mu", "nu", "xi", "pi",
    "rho", "varrho", "sigma", "varsigma", "tau", "upsilon", "phi", "varphi",
    "chi", "psi", "omega", "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi",
    "Sigma", "Upsilon", "Phi", "Psi", "Omega",
}
_BARE_COMMAND = re.compile(r"\\([A-Za-z]+)")
_BEGIN_ENV = re.compile(r"\\begin\{[A-Za-z*]+\}")
_END_ENV = re.compile(r"\\end\{[A-Za-z*]+\}")


def _strip_math(lines):
    """Yield (lineno, text) with every math-mode span AND every backtick
    code span removed: $...$, $$...$$, \\begin{...}...\\end{...} LaTeX
    environments, `` `...` ``, and ``` ``...`` ``` (this codebase's
    notebooks/docstrings use all of these).

    Code spans are exempt for the same reason math spans are matched: text
    like "a bare `\\boldsymbol` command" names the command literally, as
    code, precisely to talk ABOUT it without invoking math rendering -- it
    is not a rendering mistake the way an unwrapped command elsewhere would
    be, and this tool's own documentation (see check_latex_math_changed's
    notebook demo) does exactly this.

    A character-level state machine (not a regex) so a span that opens on
    one line and closes on a later one (common for all these conventions
    here) is tracked correctly across the whole `lines` sequence, not just
    within a single line. Environment names are not required to match
    between \\begin and \\end -- this tool only needs to know whether a
    span of text is inside SOME math or code environment, not validate the
    LaTeX/Markdown.
    """
    state = None  # None, "$", "$$", "env", "`", or "``"
    for i, raw in enumerate(lines):
        out = []
        j, n = 0, len(raw)
        while j < n:
            if state is None:
                if raw[j:j + 2] == "$$":
                    state, j = "$$", j + 2
                elif raw[j] == "$":
                    state, j = "$", j + 1
                elif raw[j:j + 2] == "``":
                    state, j = "``", j + 2
                elif raw[j] == "`":
                    state, j = "`", j + 1
                elif (m := _BEGIN_ENV.match(raw, j)):
                    state, j = "env", m.end()
                else:
                    out.append(raw[j])
                    j += 1
            elif state == "$$":
                if raw[j:j + 2] == "$$":
                    state, j = None, j + 2
                else:
                    j += 1
            elif state == "``":
                if raw[j:j + 2] == "``":
                    state, j = None, j + 2
                else:
                    j += 1
            elif state == "env":
                if (m := _END_ENV.match(raw, j)):
                    state, j = None, m.end()
                else:
                    j += 1
            elif state == "`":
                if raw[j] == "`":
                    state, j = None, j + 1
                else:
                    j += 1
            else:  # state == "$"
                if raw[j] == "$":
                    state, j = None, j + 1
                else:
                    j += 1
        yield i, "".join(out)


def _find_bare_commands(lines, skip_fences=False):
    """Yield (lineno, command) for each disallowed bare LaTeX command."""
    if not skip_fences:
        source = lines
    else:
        source = []
        in_fence = False
        for line in lines:
            if line.strip().startswith("```"):
                in_fence = not in_fence
                source.append("")
                continue
            source.append("" if in_fence else line)
    for lineno, text in _strip_math(source):
        for m in _BARE_COMMAND.finditer(text):
            if m.group(1) in _LATEX_COMMANDS:
                yield lineno, m.group(1)


def check_python_file(path):
    """Check only string-literal tokens (this includes every docstring) --
    NOT the whole file's raw text. A `#` comment can legitimately contain
    informal math-ish shorthand (e.g. `# |\\Psi1/\\lambda(...)`) that is
    never rendered as documentation, so it must not be flagged here."""
    source = path.read_text(encoding="utf-8")
    findings = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError) as exc:
        print(f"{path.as_posix()}: skipped (tokenize error: {exc})", file=sys.stderr)
        return findings
    for tok in tokens:
        if tok.type != tokenize.STRING:
            continue
        base_lineno = tok.start[0]
        string_lines = tok.string.split("\n")
        for lineno, cmd in _find_bare_commands(string_lines):
            findings.append((
                base_lineno + lineno, "bare-latex-command",
                f"\\{cmd} used outside $...$/$$...$$",
            ))
    return findings


def check_markdown_file(path):
    lines = path.read_text(encoding="utf-8").split("\n")
    return [
        (lineno + 1, "bare-latex-command", f"\\{cmd} used outside $...$/$$...$$")
        for lineno, cmd in _find_bare_commands(lines, skip_fences=True)
    ]


def check_notebook_file(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    findings = []
    for idx, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "markdown":
            continue
        lines = [line.rstrip("\n") for line in cell.get("source", [])]
        for lineno, cmd in _find_bare_commands(lines, skip_fences=True):
            findings.append((
                f"cell {idx}, line {lineno + 1}", "bare-latex-command",
                f"\\{cmd} used outside $...$/$$...$$",
            ))
    return findings


# --------------------------------------------------------------------------
# File discovery -- same scope/exclusions as check_ref_style.py.
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


def _changed_files(ref):
    """Return resolved paths of files that changed relative to `ref`."""
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
        names.update(n for n in out.splitlines())
    return {(REPO_ROOT / n).resolve() for n in names}


def _display_path(f, root):
    """Path for printing: relative to `root` when possible, else relative to
    the current directory (e.g. a file outside `root` entirely, such as a
    scratch file under the OS's temp directory), else absolute."""
    for base in (root, Path.cwd()):
        try:
            return f.relative_to(base).as_posix()
        except ValueError:
            continue
    return f.as_posix()


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


def main(argv):
    argv, diff_ref = _parse_diff_flag(list(argv))
    strict = "--strict" in argv
    quiet = "--quiet" in argv
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

    total = 0
    per_file = {}
    for f in py_files:
        findings = check_python_file(f)
        if findings:
            per_file[f] = findings
            total += len(findings)
    for f in md_files:
        findings = check_markdown_file(f)
        if findings:
            per_file[f] = findings
            total += len(findings)
    for f in nb_files:
        findings = check_notebook_file(f)
        if findings:
            per_file[f] = findings
            total += len(findings)

    n_files = len(py_files) + len(md_files) + len(nb_files)
    if total and not quiet:
        print()
        for f, findings in per_file.items():
            for lineno, cat, detail in findings:
                print(f"  - {_display_path(f, root)}:{lineno}: {cat}: {detail}")
    print(f"  - {n_files} file(s) scanned: "
          + (f"{total} issue(s) across {len(per_file)} file(s)" if total else f"no issues in {n_files} file(s)"))

    if total == 0:
        print(f"clean  (0 of {n_files} files)")
    else:
        prefix = "ERROR" if strict else "WARNING"
        print(f"{prefix}: {len(per_file)} file(s) with issues  ({len(per_file)} of {n_files} files)")
    return 1 if (strict and total) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
