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

`--update` is for a change that intentionally reduces (or, with justification
in the PR description, increases) one of these counts.

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
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = Path(__file__).resolve().parent / "baseline_counts.json"

CHECKS = {
    "check_docstring": {
        "cmd": [sys.executable, "scripts/check_docstring.py", "qmcpy"],
        # check_docstring.py's summary line reads either "N issue(s) across
        # M file(s)" or, once N reaches zero, "no issues in M file(s)" --
        # match both so the ratchet keeps working after a check is fully fixed.
        "pattern": re.compile(
            r"^\s*(?:- )?\d+ file\(s\) scanned: "
            r"(?:(\d+) issue\(s\) across|no issues in) \d+ file\(s\)",
            re.M,
        ),
    },
    "pydoclint": {
        "cmd": ["pydoclint", "-q", "qmcpy"],
        "line_pattern": re.compile(r"^\s*\d+: DOC\d+:", re.M),
    },
    "unsafe_annotations": {
        "cmd": [sys.executable, "-m", "scripts.annotate_public_api_types", "--check", "--root", "qmcpy"],
        "pattern": re.compile(r"(\d+) unsafe existing annotation\(s\)"),
    },
}


def run_check(spec):
    result = subprocess.run(spec["cmd"], capture_output=True, text=True, cwd=REPO_ROOT)
    output = result.stdout + result.stderr
    if "line_pattern" in spec:
        return len(spec["line_pattern"].findall(output))
    match = spec["pattern"].search(output)
    if match is None:
        raise RuntimeError(f"could not parse a count from output of {spec['cmd']}")
    return int(match.group(1) or 0)


def main(argv):
    update = "--update" in argv
    baseline = json.loads(BASELINE_PATH.read_text()) if BASELINE_PATH.exists() else {}

    current = {}
    regressed = []
    for name, spec in CHECKS.items():
        count = run_check(spec)
        current[name] = count
        base = baseline.get(name)
        if base is None:
            status = "no baseline yet"
        elif count > base:
            status = f"REGRESSED from {base}"
            regressed.append(name)
        elif count < base:
            status = f"improved from {base}"
        else:
            status = "unchanged"
        print(f"  - {name}: {count} ({status})")

    if not regressed:
        print(f"clean  (0 of {len(CHECKS)} counts)")
    else:
        print(f"ERROR: {len(regressed)} regressed  ({len(regressed)} of {len(CHECKS)} counts)")

    if update:
        BASELINE_PATH.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote new baseline to {BASELINE_PATH.relative_to(REPO_ROOT)}")
        return 0

    if regressed:
        print(
            f"\nRegression in: {', '.join(regressed)}. Fix the new violations, "
            "or if the increase is intentional and justified in the PR "
            "description, run `python scripts/check_baseline.py --update` "
            "and commit the updated baseline file.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
