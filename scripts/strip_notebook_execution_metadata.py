#!/usr/bin/env python3
"""Strip per-cell execution-timing metadata from notebooks.

`nbconvert`/`nbclient`'s `ExecutePreprocessor` records IOPub/shell timestamps
(`cell.metadata.execution`) by default on every run, which just adds diff
noise on every re-execution -- the notebook behaves identically either way.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import nbformat


def iter_notebooks(paths: list[str]) -> list[Path]:
    """Collect the tracked and untracked notebooks under the given paths.

    Args:
        paths (list[str]): Files or directories to restrict the search to.

    Returns:
        list[Path]: Sorted `.ipynb` files.
    """
    command = ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *paths]
    output = subprocess.run(command, check=True, capture_output=True).stdout
    return sorted(Path(raw) for raw in output.decode().split("\0") if raw.endswith(".ipynb"))


def strip_execution_metadata(path: Path) -> bool:
    """Strip `metadata.execution` from every cell of one notebook, in place.

    Args:
        path (Path): Notebook to process.

    Returns:
        bool: Whether the file changed.
    """
    nb = nbformat.read(path, as_version=nbformat.NO_CONVERT)
    changed = False
    for cell in nb.cells:
        if cell.get("metadata", {}).pop("execution", None) is not None:
            changed = True
    if changed:
        nbformat.write(nb, path)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="tracked notebooks or directories to process")
    args = parser.parse_args()

    scanned = iter_notebooks(args.paths)
    changed = sorted(path for path in scanned if strip_execution_metadata(path))
    if changed:
        print(f"  - stripped execution metadata from {len(changed)} file(s):")
        for path in changed:
            print(f"    - {path}")
    print(f"{len(changed)} changed  ({len(changed)} of {len(scanned)} files)"
          if changed else f"clean  (0 of {len(scanned)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
