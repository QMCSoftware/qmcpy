#!/usr/bin/env python3
"""
Harden a demo notebook for Colab and classify it in the Colab manifest.

This script is intentionally conservative:
- it updates one notebook at a time
- it inserts a standard badge/bootstrap before the first substantive code cell
- it only auto-classifies notebooks that are not already in enabled or disabled
- it can force-regenerate Colab bootstrap cells for notebooks already in enabled
- it validates the result with the existing strict Colab checks
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from check_colab_notebooks import (
    DEFAULT_MANIFEST,
    EXTRA_PIP_DEPENDENCIES,
    REPO_ROOT,
    as_source_list,
    badge_markup,
    cell_source_text,
    discovered_notebooks,
    early_non_install_code_cells,
    imported_modules,
    is_any_badge_cell,
    is_any_install_cell,
    local_module_matches,
    load_json,
    pip_install_lines,
    validate_disabled_notebook,
    validate_enabled_notebook,
    validate_manifest,
    manifest_sets,
    validate_strict_enabled_notebook,
)


EXTRA_IMPORT_DEPENDENCIES = {
    **EXTRA_PIP_DEPENDENCIES,
    "botorch": ("botorch",),
    "gpytorch": ("gpytorch",),
    "ipywidgets": ("ipywidgets",),
    "seaborn": ("seaborn",),
    "sklearn": ("scikit-learn", "sklearn"),
    "sympy": ("sympy",),
    "torch": ("torch",),
    "umbridge": ("umbridge",),
    "yfinance": ("yfinance",),
}
LATEX_MARKERS = (
    "text.usetex",
    "\\usepackage",
    "dvipng",
    "latexmk",
    "computer modern",
)
COLAB_BADGE_IMAGE_FRAGMENT = "colab.research.google.com/assets/colab-badge.svg"


def dump_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)
        handle.write("\n")


def discovered_imports(cells: list[dict]) -> set[str]:
    modules: set[str] = set()
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        modules.update(
            imported_modules(
                cell_source_text(cell),
                location=f"cell {idx + 1}",
            )
        )
    return modules


def local_repo_import_matches(
    notebook_path: Path, cells: list[dict]
) -> dict[str, list[Path]]:
    notebook_dir = notebook_path.parent.resolve()
    matches_by_module: dict[str, list[Path]] = {}
    for module in sorted(discovered_imports(cells)):
        if module == "qmcpy":
            continue
        matches = local_module_matches(notebook_dir, module)
        if matches:
            matches_by_module[module] = matches
    return matches_by_module


def needs_latex_setup(cells: list[dict]) -> bool:
    source = "\n".join(
        cell_source_text(cell)
        for cell in cells
        if cell.get("cell_type") in {"code", "markdown"}
    )
    return any(marker in source for marker in LATEX_MARKERS)


def extra_pip_packages(cells: list[dict]) -> list[str]:
    early_cells = [cell for _, cell in early_non_install_code_cells(cells)]
    modules = discovered_imports(early_cells)
    packages: list[str] = []
    for module, names in sorted(EXTRA_IMPORT_DEPENDENCIES.items()):
        if module in modules:
            package = names[0]
            if package not in packages:
                packages.append(package)
    all_install_lines = [
        line
        for cell in cells
        if cell.get("cell_type") == "code"
        for line in pip_install_lines(cell_source_text(cell))
    ]
    for _, names in sorted(EXTRA_IMPORT_DEPENDENCIES.items()):
        package = names[0]
        if package in packages:
            continue
        if any(name.lower() in line for name in names for line in all_install_lines):
            packages.append(package)
    return packages


def extra_repo_paths(notebook_path: Path, cells: list[dict]) -> list[str]:
    notebook_dir = notebook_path.parent.resolve()
    repo_matches = local_repo_import_matches(notebook_path, cells)
    rel_paths: list[str] = []

    for module in sorted(repo_matches):
        for match in repo_matches[module]:
            if match.name == "__init__.py" and match.parent.name == module:
                parent = match.parent.parent.resolve()
            else:
                parent = match.parent.resolve()
            if parent == notebook_dir:
                continue
            if parent == REPO_ROOT:
                continue
            rel_parent = parent.relative_to(REPO_ROOT).as_posix()
            if rel_parent not in rel_paths:
                rel_paths.append(rel_parent)

    return rel_paths


def bootstrap_cell_source(notebook_path: Path, manifest: dict, cells: list[dict]) -> list[str]:
    notebook_dir_rel = notebook_path.parent.relative_to(REPO_ROOT).as_posix()
    packages = extra_pip_packages(cells)
    rel_paths = extra_repo_paths(notebook_path, cells)
    latex_setup = needs_latex_setup(cells)
    needs_repo_clone = bool(local_repo_import_matches(notebook_path, cells))

    lines = [
        "# @title Execute this cell to install dependencies\n",
        "try:\n",
        "  import google.colab\n",
    ]

    if needs_repo_clone:
        lines.extend(
            [
                "  import sys\n",
                "  import os\n",
                '  repo_root = "/content/QMCSoftware"\n',
                f'  notebook_dir = f"{{repo_root}}/{notebook_dir_rel}"\n',
                "  if not os.path.isdir(repo_root):\n",
                f"    !git clone -q --depth 1 https://github.com/{manifest['repo']} {{repo_root}}\n",
            ]
        )

    lines.append("  !pip install -q qmcpy\n")

    if packages:
        lines.append(f"  !pip install -q {' '.join(packages)}\n")

    if latex_setup:
        lines.append(
            '  !tmp=$(mktemp) && if { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends texlive-latex-base texlive-fonts-recommended texlive-latex-extra cm-super dvipng; } >"$tmp" 2>&1; then rm -f "$tmp"; else status=$?; cat "$tmp"; rm -f "$tmp"; exit $status; fi\n'
        )

    if needs_repo_clone:
        lines.extend(
            [
                "  os.chdir(notebook_dir)\n",
                "  if notebook_dir not in sys.path:\n",
                "    sys.path.insert(0, notebook_dir)\n",
            ]
        )
        for rel_path in rel_paths:
            lines.extend(
                [
                    f'  extra_path = f"{{repo_root}}/{rel_path}"\n',
                    "  if extra_path not in sys.path:\n",
                    "    sys.path.insert(0, extra_path)\n",
                ]
            )

    lines.extend(["except:\n", "  pass"])
    return lines

def badge_stripped_cell(cell: dict) -> dict | None:
    if not is_any_badge_cell(cell):
        return cell

    kept_lines = [
        line
        for line in as_source_list(cell.get("source", []))
        if COLAB_BADGE_IMAGE_FRAGMENT not in line
    ]
    if not "".join(kept_lines).strip():
        return None

    cleaned_cell = copy.deepcopy(cell)
    cleaned_cell["source"] = kept_lines
    return cleaned_cell


def remove_any_badge_cells(cells: list[dict]) -> list[dict]:
    cleaned_cells: list[dict] = []
    for cell in cells:
        cleaned_cell = badge_stripped_cell(cell)
        if cleaned_cell is not None:
            cleaned_cells.append(cleaned_cell)
    return cleaned_cells


def badge_bootstrap_insert_index(cells: list[dict]) -> int:
    insert_at = 0
    while insert_at < len(cells) and cells[insert_at].get("cell_type") == "markdown":
        insert_at += 1
    first_code_cell = next(
        (idx for idx, cell in enumerate(cells) if cell.get("cell_type") == "code"),
        len(cells),
    )
    return min(insert_at, first_code_cell)


def remove_existing_bootstrap_cells(cells: list[dict]) -> list[dict]:
    return [cell for cell in cells if not is_any_install_cell(cell)]


def pending_unclassified(
    manifest: dict, current_notebook: str | None = None
) -> set[str]:
    enabled, disabled = manifest_sets(manifest)
    missing = discovered_notebooks() - enabled - set(disabled)
    if current_notebook is not None:
        missing.discard(current_notebook)
    return missing


def harden_notebook(notebook_path: Path, manifest_path: Path) -> None:
    manifest = load_json(manifest_path)
    notebook_payload = load_json(notebook_path)

    original_manifest = copy.deepcopy(manifest)
    original_notebook = copy.deepcopy(notebook_payload)

    # Cells that survive unchanged: remove only the old badge/bootstrap cells.
    # These dict objects are never modified; they are passed through as-is.
    kept_cells = remove_any_badge_cells(
        remove_existing_bootstrap_cells(list(notebook_payload.get("cells", [])))
    )

    insert_at = badge_bootstrap_insert_index(kept_cells)

    badge_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            badge_markup(
                manifest["repo"],
                manifest["git_ref"],
                notebook_path.relative_to(REPO_ROOT).as_posix(),
            )
        ],
    }
    bootstrap_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        # Pass kept_cells (not the final list) so the source scanner only sees
        # the original notebook code cells, not the badge cell we just built.
        "source": bootstrap_cell_source(notebook_path, manifest, kept_cells),
    }

    # Build the final cell list purely by concatenation — kept_cells are untouched.
    cells = kept_cells[:insert_at] + [badge_cell, bootstrap_cell] + kept_cells[insert_at:]

    notebook_payload["cells"] = cells

    notebook_rel = notebook_path.relative_to(REPO_ROOT).as_posix()
    enabled = list(manifest.get("enabled", []))
    disabled = dict(manifest.get("disabled", {}))
    if notebook_rel in disabled:
        raise ValueError(
            f"{notebook_rel} is already classified as disabled; harden_colab_notebook does not reclassify disabled notebooks."
        )
    if notebook_rel not in enabled:
        enabled.append(notebook_rel)
    enabled = sorted(enabled)
    manifest["enabled"] = enabled
    manifest["disabled"] = disabled

    try:
        dump_json(notebook_path, notebook_payload)
        dump_json(manifest_path, manifest)

        reloaded_manifest = load_json(manifest_path)
        errors = validate_manifest(
            reloaded_manifest,
            allowed_missing=pending_unclassified(reloaded_manifest, notebook_rel),
        )
        errors.extend(
            validate_enabled_notebook(
                notebook_path,
                reloaded_manifest["repo"],
                reloaded_manifest["git_ref"],
            )
        )
        errors.extend(validate_strict_enabled_notebook(notebook_path))
        if errors:
            raise RuntimeError("\n".join(errors))
    except Exception:
        dump_json(notebook_path, original_notebook)
        dump_json(manifest_path, original_manifest)
        raise


def error_summary(exc: Exception) -> str:
    return str(exc).splitlines()[0] if str(exc) else "unknown error"


def validate_target_notebook(notebook_path: Path, notebook_rel: str) -> None:
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_rel}")
    if notebook_path.suffix != ".ipynb":
        raise ValueError(f"Expected a notebook path ending in .ipynb: {notebook_rel}")


def manifest_notebook_paths(manifest_path: Path, mode: str) -> list[Path]:
    manifest = load_json(manifest_path)
    enabled, disabled = manifest_sets(manifest)
    if mode == "enabled":
        notebook_rels = enabled
    elif mode == "unclassified":
        notebook_rels = discovered_notebooks() - enabled - set(disabled)
    else:  # pragma: no cover - internal guard
        raise ValueError(f"Unknown notebook selection mode: {mode}")
    return sorted((REPO_ROOT / notebook_rel).resolve() for notebook_rel in notebook_rels)


def disable_notebook(notebook_path: Path, manifest_path: Path, reason: str) -> None:
    manifest = load_json(manifest_path)
    original_manifest = copy.deepcopy(manifest)
    notebook_payload = load_json(notebook_path)
    original_notebook = copy.deepcopy(notebook_payload)
    notebook_rel = notebook_path.relative_to(REPO_ROOT).as_posix()
    enabled = sorted(
        path for path in manifest.get("enabled", []) if path != notebook_rel
    )
    manifest["enabled"] = enabled
    disabled = dict(manifest.get("disabled", {}))
    disabled[notebook_rel] = reason
    manifest["disabled"] = disabled
    try:
        notebook_payload["cells"] = remove_existing_bootstrap_cells(
            remove_any_badge_cells(list(notebook_payload.get("cells", [])))
        )
        dump_json(notebook_path, notebook_payload)
        dump_json(manifest_path, manifest)
        reloaded_manifest = load_json(manifest_path)
        errors = validate_manifest(
            reloaded_manifest,
            allowed_missing=pending_unclassified(reloaded_manifest, notebook_rel),
        )
        errors.extend(validate_disabled_notebook(notebook_path))
        if errors:
            raise RuntimeError("\n".join(errors))
    except Exception:
        dump_json(notebook_path, original_notebook)
        dump_json(manifest_path, original_manifest)
        raise


def harden_batch(
    notebook_paths: list[Path],
    manifest_path: Path,
    *,
    disable_failures: bool,
) -> tuple[list[str], list[tuple[str, str]]]:
    successes: list[str] = []
    failures: list[tuple[str, str]] = []

    for notebook_path in notebook_paths:
        notebook_rel = notebook_path.relative_to(REPO_ROOT).as_posix()
        try:
            validate_target_notebook(notebook_path, notebook_rel)
            harden_notebook(notebook_path, manifest_path)
            successes.append(notebook_rel)
        except Exception as exc:
            summary = error_summary(exc)
            if disable_failures:
                disable_notebook(
                    notebook_path,
                    manifest_path,
                    f"Automatic Colab hardening failed: {summary}",
                )
            failures.append((notebook_rel, summary))

    return successes, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Harden a demo notebook for Colab and enable it in the Colab manifest."
    )
    parser.add_argument(
        "--notebook",
        help="Repo-relative path to the notebook to harden, e.g. demos/plot_proj_function.ipynb",
    )
    parser.add_argument(
        "--all-unclassified",
        action="store_true",
        help="Attempt to harden every notebook under demos/ that is not yet listed in enabled or disabled.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate the Colab bootstrap for enabled notebook(s). Without --notebook, processes every enabled notebook.",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help=f"Path to the Colab manifest (default: {DEFAULT_MANIFEST})",
    )
    args = parser.parse_args()
    if args.notebook and args.all_unclassified:
        parser.error("specify either --notebook or --all-unclassified, not both")
    if args.all_unclassified and args.force:
        parser.error("specify either --all-unclassified or --force, not both")
    if not args.notebook and not args.all_unclassified and not args.force:
        parser.error("specify --notebook, --all-unclassified, or --force")
    return args


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    if args.notebook:
        notebook_path = (REPO_ROOT / args.notebook).resolve()
        try:
            validate_target_notebook(notebook_path, args.notebook)
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        manifest = load_json(manifest_path)
        enabled, disabled = manifest_sets(manifest)
        notebook_rel = notebook_path.relative_to(REPO_ROOT).as_posix()
        if notebook_rel in disabled:
            raise SystemExit(
                f"{notebook_rel} is already classified as disabled; update the manifest manually before reclassifying it."
            )
        if notebook_rel in enabled:
            harden_notebook(notebook_path, manifest_path)
            print(f"Hardened {notebook_rel} for Colab.")
            return 0
        try:
            harden_notebook(notebook_path, manifest_path)
            print(f"Hardened {notebook_rel} for Colab and added it to enabled.")
            return 0
        except Exception as exc:
            summary = error_summary(exc)
            disable_notebook(
                notebook_path,
                manifest_path,
                f"Automatic Colab hardening failed: {summary}",
            )
            print(
                f"Could not harden {notebook_rel}; restored the notebook and added it to disabled."
            )
            print(f"Reason: {summary}")
            return 1

    mode = "enabled" if args.force else "unclassified"
    successes, failures = harden_batch(
        manifest_notebook_paths(manifest_path, mode),
        manifest_path,
        disable_failures=not args.force,
    )
    for notebook_rel in successes:
        print(f"Hardened {notebook_rel} for Colab.")
    if failures:
        print("")
        print("Not yet hardened:")
        for notebook_rel, error in failures:
            print(f"- {notebook_rel}: {error}")
    print("")
    print(
        f"Hardened {len(successes)} notebook(s); {len(failures)} notebook(s) still need manual follow-up."
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
