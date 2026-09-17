#!/usr/bin/env python3
"""Ratchet gate for the informational docstring/annotation checks.

`check_docstring`, `pydoclint`, and `annotate_public_api_types` are
informational today (see F9/F10 in the PR #613 review) because fixing every
existing violation before enabling them as hard gates is a large, separate
undertaking. This script tracks each check's full-tree violation count in
`scripts/baseline_counts.json` and fails only if a count *increases* --
new violations are blocked; the existing backlog is not required to be
cleared just to land an unrelated change.

Usage:
    python scripts/check_baseline.py            # compare against the baseline
    python scripts/check_baseline.py --update    # write current counts as the new baseline
    python scripts/check_baseline.py --diff [REF]  # also gate on the PR's own diff

`--update` is for a change that intentionally reduces (or, with justification
in the PR description, increases) one of these counts.

The whole-repo ratchet above only catches a *net* increase: a PR that fixes N
pre-existing issues elsewhere while introducing M<N new ones in its own files
reports "improved" and passes. `--diff [REF]` (REF defaults to `develop`) adds
a second, independent check: for just the `qmcpy/*.py` files that changed
relative to REF, each tool is run against both that ref's content and the
current worktree's content, and it fails if a file's own change made any
count go up -- independent of the whole-repo trend.

A mid-migration branch (e.g. adding type hints to signatures across many
files) can make a count spike well above the committed baseline before it
comes back down -- pydoclint's DOC105/106/107 cross-check every arg's
signature type against its docstring type, so partially-applied hints
surface more mismatches than having no hints at all. That is expected, not
a bug in this script: `make check` will keep reporting "REGRESSED" for that
check until the migration is complete and `--update` is run to record the
new, lower count.
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = Path(__file__).resolve().parent / "baseline_counts.json"

def _check_docstring_by_file(output):
    """One finding line is "  - <path>:<line>: <category>: <detail>" --
    group by path, keeping "<category>: <detail>" (path/line stripped) as
    that finding's identity, so a line-number shift from an unrelated edit
    isn't miscounted as new, but a genuinely different finding is."""
    by_file = {}
    for path, detail in re.findall(r"^\s*-\s+(\S+):\d+:\s+(.+)$", output, re.M):
        by_file.setdefault(path, set()).add(detail)
    return by_file


def _pydoclint_by_file(output):
    """pydoclint prints a bare file-path header line, then indented
    "<line>: DOC###: <message>" lines for that file -- track the most
    recent header line to attribute each violation to its file."""
    by_file = {}
    current = None
    for line in output.splitlines():
        m = re.match(r"^\s*\d+: (DOC\d+:.*)$", line)
        if m:
            if current is not None:
                by_file.setdefault(current, set()).add(m.group(1))
            continue
        stripped = line.strip()
        if stripped and not line[:1].isspace():
            current = stripped
    return by_file


def _unsafe_annotations_by_file(output):
    by_file = {}
    for path, detail in re.findall(r"^(\S+):\d+: (unsafe existing annotation .*)$", output, re.M):
        by_file.setdefault(path, set()).add(detail)
    return by_file


CHECKS = {
    "check_docstring": {
        "cmd": [sys.executable, "scripts/check_docstring.py", "qmcpy"],
        "files_cmd": lambda files: [sys.executable, "scripts/check_docstring.py", *files],
        # check_docstring.py's summary line reads either "N issue(s) across
        # M file(s)" or, once N reaches zero, "no issues in M file(s)" --
        # match both so the ratchet keeps working after a check is fully fixed.
        "pattern": re.compile(
            r"^\s*(?:- )?\d+ file\(s\) scanned: "
            r"(?:(\d+) issue\(s\) across|no issues in) \d+ file\(s\)",
            re.M,
        ),
        "by_file": _check_docstring_by_file,
    },
    "pydoclint": {
        "cmd": ["pydoclint", "-q", "qmcpy"],
        "files_cmd": lambda files: ["pydoclint", "-q", *files],
        "line_pattern": re.compile(r"^\s*\d+: DOC\d+:", re.M),
        "by_file": _pydoclint_by_file,
    },
    "unsafe_annotations": {
        "cmd": [sys.executable, "-m", "scripts.annotate_public_api_types", "--check", "--root", "qmcpy"],
        "files_cmd": lambda files: [sys.executable, "-m", "scripts.annotate_public_api_types", "--check", *files],
        "pattern": re.compile(r"(\d+) unsafe existing annotation\(s\)"),
        "by_file": _unsafe_annotations_by_file,
    },
}


def run_check(spec, cmd=None):
    result = subprocess.run(cmd or spec["cmd"], capture_output=True, text=True, cwd=REPO_ROOT)
    output = result.stdout + result.stderr
    if "line_pattern" in spec:
        return len(spec["line_pattern"].findall(output))
    match = spec["pattern"].search(output)
    if match is None:
        raise RuntimeError(f"could not parse a count from output of {cmd or spec['cmd']}")
    return int(match.group(1) or 0)


def _violations_by_file(spec, cmd):
    """Run `cmd` once and return {file: {violation identity, ...}} per
    `spec["by_file"]` -- one subprocess call covers every file in `cmd`,
    rather than one call per file."""
    if not cmd:
        return {}
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    return spec["by_file"](result.stdout + result.stderr)


def _changed_python_files(ref):
    """Return repo-relative `qmcpy/*.py` paths that changed relative to `ref`."""
    commands = (
        ["git", "diff", "--name-only", "--diff-filter=ACMR", f"{ref}...HEAD"],
        ["git", "diff", "--name-only", "--diff-filter=ACMR", "HEAD"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    )
    names = set()
    for cmd in commands:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=REPO_ROOT).stdout
        names.update(n for n in out.splitlines() if n.startswith("qmcpy/") and n.endswith(".py"))
    return sorted(names)


def _materialize_at_ref(ref, files, dest_root):
    """Write each file's `ref` content under `dest_root`, mirroring its repo
    path. Returns {file: dest_path|None}; None means the file has no content
    at `ref` (added since), so its base violation count is implicitly 0."""
    base_paths = {}
    for f in files:
        result = subprocess.run(["git", "show", f"{ref}:{f}"], capture_output=True, text=True, cwd=REPO_ROOT)
        if result.returncode != 0:
            base_paths[f] = None
            continue
        dest = dest_root / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(result.stdout, encoding="utf-8")
        base_paths[f] = str(dest)
    return base_paths


def diff_regressions(ref):
    """Compare each CHECKS entry's violation IDENTITIES file by file,
    restricted to the `qmcpy/*.py` files that changed relative to `ref`,
    between that ref's content and the current worktree.

    Deliberately per-file and per-violation-identity rather than a single
    aggregate count, for two independent reasons a plain count can hide:
    (1) an aggregate total could mask a regression in one changed file
    behind an improvement in another -- the same "robbed Peter to pay Paul"
    blind spot this check exists to close, just rescoped from the whole
    repo down to the diff; (2) even scoped to one file, a tied count (one
    pre-existing violation fixed, one different violation introduced) would
    look like "no change" -- comparing the actual set of violations catches
    that a genuinely new one appeared, independent of what else changed.

    Returns (regressed, details): regressed is a list of (check_name, file)
    pairs with at least one violation at HEAD absent at `ref`;
    details[(check_name, file)] = (base_violations, head_violations,
    new_violations), each a set of identity strings.
    """
    files = _changed_python_files(ref)
    if not files:
        return [], {}
    regressed = []
    details = {}
    with tempfile.TemporaryDirectory() as tmp:
        base_paths = _materialize_at_ref(ref, files, Path(tmp))
        head_abs = {f: str(REPO_ROOT / f) for f in files}
        head_file_list = list(head_abs.values())
        base_file_list = [p for p in base_paths.values() if p is not None]
        # One subprocess call per check for ALL changed files together (not
        # one call per file) -- for a ~90-file diff this is the difference
        # between ~6 subprocess spawns and ~550, since each of the 3
        # underlying tools has real interpreter/import startup cost.
        for name, spec in CHECKS.items():
            head_by_file = _violations_by_file(spec, spec["files_cmd"](head_file_list)) if head_file_list else {}
            base_by_file = _violations_by_file(spec, spec["files_cmd"](base_file_list)) if base_file_list else {}
            for f in files:
                head_violations = head_by_file.get(head_abs[f], set())
                base_path = base_paths[f]
                base_violations = base_by_file.get(base_path, set()) if base_path else set()
                new_violations = head_violations - base_violations
                details[(name, f)] = (base_violations, head_violations, new_violations)
                if new_violations:
                    regressed.append((name, f))
    return regressed, details


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
    update = "--update" in argv
    baseline = json.loads(BASELINE_PATH.read_text()) if BASELINE_PATH.exists() else {}

    print(f"whole-repo issue counts vs. the committed baseline ({BASELINE_PATH.name}):")
    current = {}
    regressed = []
    for name, spec in CHECKS.items():
        count = run_check(spec)
        current[name] = count
        base = baseline.get(name)
        if base is None:
            status = "no baseline recorded yet"
        elif count > base:
            status = f"UP from {base} -- {count - base} new issue(s) since the baseline"
            regressed.append(name)
        elif count < base:
            status = f"DOWN from {base} -- {base - count} issue(s) fixed since the baseline"
        else:
            status = f"same as baseline ({base})"
        print(f"  - {name}: {count} issue(s) now  ({status})")
        if base is not None and count > base:
            print(f"      run `{' '.join(spec['cmd'])}` to see which")

    if update:
        BASELINE_PATH.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote new baseline to {BASELINE_PATH.relative_to(REPO_ROOT)}")
        return 0

    diff_regressed = []
    if diff_ref is not None:
        diff_regressed, diff_details = diff_regressions(diff_ref)
        n_pairs = len(diff_details)
        print(f"\nchecked {n_pairs} check/file combination(s) in the qmcpy/*.py files you "
              f"changed (vs {diff_ref}) for a violation that wasn't already there:")
        for (name, f), (_base, _head, new) in diff_details.items():
            for violation in sorted(new):
                print(f"  - NEW in {f} ({name}): {violation}")
        if not diff_regressed:
            print("  - none found")

    # One final verdict, not a separate "clean"/"ERROR" per section -- a
    # whole-repo count regression and a diff-scoped new violation both feed
    # into the SAME pass/fail decision for this one command.
    print()
    if not regressed and not diff_regressed:
        print("clean  (no new violations, whole-repo or in files you changed)")
        return 0
    problems = []
    if regressed:
        problems.append(f"{', '.join(regressed)} increased above baseline")
    if diff_regressed:
        problems.append(f"{len(diff_regressed)} new violation(s) in files you changed (see above)")
    print(f"ERROR: {'; '.join(problems)}.")
    print("Fix the violation(s) above, or if the increase is intentional and justified in "
          "the PR description, run `python scripts/check_baseline.py --update` and commit "
          "the updated baseline file.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
