#!/usr/bin/env python3
"""Nudge check: does each demo notebook changed relative to develop end with
a References section?

This is deliberately not a style check (see check_ref_style.py for that) and
never fails the build -- a demo genuinely has nothing to cite is not an
error. It only prints a prompt for changed notebooks with no References
heading near the end, so an author can decide whether to add one.

Usage:
    python scripts/check_demo_references.py [--diff REF]

REF defaults to develop.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_ref_style import _MD_BOLD_HEADING, _MD_HEADING, _changed_files, _display_path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMOS_DIR = REPO_ROOT / "demos"
PROMPT = "Any references worth citing? Consider adding them at the end of the notebook."


def _has_references_near_end(path, tail=5):
    """True if a References-type markdown heading appears in the last
    `tail` cells (generous, since a short closing note or output cell may
    sit between the real content and the References section)."""
    cells = json.loads(path.read_text(encoding="utf-8")).get("cells", [])
    for cell in cells[-tail:]:
        if cell.get("cell_type") != "markdown":
            continue
        for line in cell.get("source", []):
            stripped = line.strip()
            if _MD_HEADING.match(stripped) or _MD_BOLD_HEADING.match(stripped):
                return True
    return False


def _parse_diff_flag(argv):
    ref = "develop"
    if "--diff" in argv:
        i = argv.index("--diff")
        if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
            ref = argv[i + 1]
    return ref


def main(argv):
    ref = _parse_diff_flag(argv)
    try:
        changed = _changed_files(ref)
    except RuntimeError as exc:
        print(f"--diff {ref}: skipped ({exc})", file=sys.stderr)
        return 0

    demo_notebooks = sorted(p for p in changed if p.suffix == ".ipynb" and DEMOS_DIR in p.parents)
    missing = [p for p in demo_notebooks if not _has_references_near_end(p)]

    if missing:
        print(f"  - {PROMPT}")
        for p in missing:
            print(f"      - {_display_path(p, REPO_ROOT)}")
    elif demo_notebooks:
        print(f"  - clean: all {len(demo_notebooks)} changed demo(s) (vs {ref}) already have a References section")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
