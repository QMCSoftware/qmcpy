#!/usr/bin/env python3
"""
Validate Colab support metadata for demo notebooks.

Every notebook under ``demos/`` must be explicitly classified in the manifest:
- enabled: notebook should expose the expected Colab badge and bootstrap cell
- disabled: notebook is intentionally excluded, with a reason
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import warnings
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMOS_DIR = REPO_ROOT / "demos"
DEFAULT_MANIFEST = Path(__file__).with_name("colab_notebooks_manifest.json")

CORE_BOOTSTRAP_FRAGMENT = "import google.colab"
BOOTSTRAP_CELL_MARKER = "# @title Execute this cell to install dependencies"
REPO_QMCPY_INSTALL_FRAGMENTS = (
    "git+https://github.com/QMCSoftware/QMCSoftware",
    "pip install -q -e",
    "pip install -e",
)
EXTRA_PIP_DEPENDENCIES = {
    "QuantLib": ("QuantLib", "quantlib"),
    "parsl": ("parsl",),
    "seaborn": ("seaborn",),
    "skopt": ("scikit-optimize", "skopt"),
    "tueplots": ("tueplots",),
}
UMBRIDGE_MARKERS = ("import umbridge", "from umbridge", "UMBridgeWrapper", "HTTPModel(")
EARLY_EXTRA_DEPENDENCY_CODE_CELLS = 3
REPO_FETCH_FRAGMENTS = ("git clone", "raw.githubusercontent.com", "wget ", "curl ")
PATH_SETUP_FRAGMENTS = ("sys.path.insert", "os.chdir(", "%cd ", "cd ")
IGNORED_NOTEBOOK_NAME_PREFIXES = (".tmp", "._tmp")
EXTRA_DEPS_MARKER = "# colab-deps:"


def load_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path}:{exc.lineno}:{exc.colno}: invalid JSON: {exc.msg}"
        ) from exc


def as_source_list(source: str | list[str]) -> list[str]:
    if isinstance(source, str):
        return source.splitlines(keepends=True)
    return list(source)


def cell_source_text(cell: dict) -> str:
    return "".join(as_source_list(cell.get("source", [])))


def python_source_for_ast(source: str) -> str:
    filtered_lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("!", "%")):
            continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines)


def imported_modules(source: str, location: str = "<unknown>") -> set[str]:
    cleaned = python_source_for_ast(source)
    if not cleaned.strip():
        return set()
    try:
        # Notebook code often contains valid runtime strings such as LaTeX
        # preambles that trigger irrelevant SyntaxWarnings during static parsing.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(cleaned, filename=location)
    except SyntaxError:
        return set()

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def badge_markup(repo: str, git_ref: str, notebook_path: str) -> str:
    return (
        "[![Open In Colab]"
        "(https://colab.research.google.com/assets/colab-badge.svg)]"
        "(https://colab.research.google.com/github/"
        f"{repo}/blob/{git_ref}/{notebook_path})"
    )


def is_any_badge_cell(cell: dict) -> bool:
    if cell.get("cell_type") != "markdown":
        return False
    source = cell_source_text(cell)
    return "Open In Colab" in source or "colab.research.google.com" in source


def has_expected_badge(cell: dict, repo: str, git_ref: str, notebook_path: str) -> bool:
    return (
        cell.get("cell_type") == "markdown"
        and badge_markup(repo, git_ref, notebook_path) in cell_source_text(cell)
    )


def is_any_install_cell(cell: dict) -> bool:
    if cell.get("cell_type") != "code":
        return False
    source_lines = cell_source_text(cell).splitlines()
    return bool(source_lines) and source_lines[0].strip() == BOOTSTRAP_CELL_MARKER


def pip_install_lines(source: str) -> list[str]:
    return [
        line.lower()
        for line in source.splitlines()
        if line.lstrip().lower().startswith(("!pip install", "%pip install"))
    ]


def installs_qmcpy(source: str) -> bool:
    install_lines = pip_install_lines(source)
    if any("qmcpy" in line for line in install_lines):
        return True
    return any(fragment in source for fragment in REPO_QMCPY_INSTALL_FRAGMENTS)


def installs_packages(source: str, package_names: tuple[str, ...]) -> bool:
    install_lines = pip_install_lines(source)
    return any(
        package.lower() in line
        for line in install_lines
        for package in package_names
    )


def declared_extra_pip_packages(cells: list[dict]) -> list[str]:
    """Escape hatch for deps outside EXTRA_PIP_DEPENDENCIES/EXTRA_IMPORT_DEPENDENCIES:
    a `# colab-deps: pkg-a, pkg-b` comment anywhere in a code cell."""
    packages: list[str] = []
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        for line in cell_source_text(cell).splitlines():
            stripped = line.strip()
            if not stripped.startswith(EXTRA_DEPS_MARKER):
                continue
            for name in stripped[len(EXTRA_DEPS_MARKER):].split(","):
                name = name.strip()
                if name and name not in packages:
                    packages.append(name)
    return packages


def is_bootstrap_cell(cell: dict) -> bool:
    source = cell_source_text(cell)
    return is_any_install_cell(cell) and (
        CORE_BOOTSTRAP_FRAGMENT in source
        and "except ImportError:" in source
        and "if IN_COLAB:" in source
        and installs_qmcpy(source)
    )


def notebook_code_cells(cells: list[dict]) -> list[tuple[int, dict]]:
    return [
        (idx, cell) for idx, cell in enumerate(cells) if cell.get("cell_type") == "code"
    ]


def early_non_install_code_cells(
    cells: list[dict], limit: int = EARLY_EXTRA_DEPENDENCY_CODE_CELLS
) -> list[tuple[int, dict]]:
    early_cells: list[tuple[int, dict]] = []
    for idx, cell in notebook_code_cells(cells):
        if is_any_install_cell(cell):
            continue
        early_cells.append((idx, cell))
        if len(early_cells) >= limit:
            break
    return early_cells


def imported_modules_in_cells(indexed_cells: list[tuple[int, dict]]) -> set[str]:
    modules: set[str] = set()
    for idx, cell in indexed_cells:
        modules.update(
            imported_modules(cell_source_text(cell), location=f"cell {idx + 1}")
        )
    return modules


def code_source_upto(cells: list[dict], end_index: int) -> str:
    return "\n".join(
        cell_source_text(cell)
        for idx, cell in enumerate(cells)
        if cell.get("cell_type") == "code" and idx <= end_index
    )


def find_first_module_import(cells: list[dict], module: str) -> int | None:
    for idx, cell in notebook_code_cells(cells):
        location = f"cell {idx + 1}"
        if module in imported_modules(cell_source_text(cell), location=location):
            return idx
    return None


def local_module_matches(notebook_dir: Path, module: str) -> list[Path]:
    matches: list[Path] = []
    direct_file = notebook_dir / f"{module}.py"
    package_init = notebook_dir / module / "__init__.py"
    if direct_file.exists():
        matches.append(direct_file)
    if package_init.exists():
        matches.append(package_init)
    for candidate in notebook_dir.rglob(f"{module}.py"):
        if candidate not in matches and "__pycache__" not in candidate.parts:
            matches.append(candidate)
    return matches


def validate_strict_enabled_notebook(path: Path) -> list[str]:
    notebook_path = path.relative_to(REPO_ROOT).as_posix()
    notebook_dir = path.parent
    payload = load_json(path)
    cells = payload.get("cells", [])
    errors: list[str] = []
    full_source = "\n".join(
        cell_source_text(cell) for _, cell in notebook_code_cells(cells)
    )
    early_imports = imported_modules_in_cells(early_non_install_code_cells(cells))

    for module, package_names in sorted(EXTRA_PIP_DEPENDENCIES.items()):
        if module not in early_imports:
            continue
        first_import_index = find_first_module_import(cells, module)
        if first_import_index is None:
            continue
        source_before_import = code_source_upto(cells, first_import_index)
        if not installs_packages(source_before_import, package_names):
            errors.append(
                f"{notebook_path}: imports '{module}' without installing {package_names[0]!r} before that import."
            )

    imported_roots = sorted(
        {
            module
            for idx, cell in notebook_code_cells(cells)
            for module in imported_modules(
                cell_source_text(cell), location=f"cell {idx + 1}"
            )
        }
    )
    if "umbridge" in imported_roots or any(marker in full_source for marker in UMBRIDGE_MARKERS):
        errors.append(
            f"{notebook_path}: depends on UM-Bridge and should be classified as Colab-disabled."
        )

    for module in imported_roots:
        local_matches = local_module_matches(notebook_dir, module)
        if not local_matches:
            continue
        first_import_index = find_first_module_import(cells, module)
        if first_import_index is None:
            continue
        source_before_import = code_source_upto(cells, first_import_index)
        if not any(fragment in source_before_import for fragment in REPO_FETCH_FRAGMENTS):
            errors.append(
                f"{notebook_path}: imports local module '{module}' without fetching repo files first."
            )
        if not any(fragment in source_before_import for fragment in PATH_SETUP_FRAGMENTS):
            errors.append(
                f"{notebook_path}: imports local module '{module}' without updating the working directory or sys.path first."
            )
        nested_match = next(
            (match for match in local_matches if match.parent != notebook_dir), None
        )
        if nested_match is not None:
            rel_parent = nested_match.parent.relative_to(notebook_dir).as_posix()
            if rel_parent and rel_parent not in source_before_import:
                errors.append(
                    f"{notebook_path}: imports local module '{module}' from '{rel_parent}' without referencing that path in Colab setup."
                )

    return errors


def manifest_sets(manifest: dict) -> tuple[set[str], dict[str, str]]:
    enabled = set(manifest.get("enabled", []))
    disabled = dict(manifest.get("disabled", {}))
    return enabled, disabled


def is_discoverable_notebook(path: Path) -> bool:
    return ".ipynb_checkpoints" not in path.parts and not path.name.startswith(
        IGNORED_NOTEBOOK_NAME_PREFIXES
    )


def discovered_notebooks() -> set[str]:
    return {
        path.relative_to(REPO_ROOT).as_posix()
        for path in DEMOS_DIR.rglob("*.ipynb")
        if is_discoverable_notebook(path)
    }


def validate_manifest(manifest: dict, allowed_missing: set[str] | None = None) -> list[str]:
    errors: list[str] = []
    enabled, disabled = manifest_sets(manifest)
    discovered = discovered_notebooks()
    declared = enabled | set(disabled)
    allowed_missing = allowed_missing or set()

    overlap = enabled & set(disabled)
    if overlap:
        errors.append(
            "Manifest paths cannot be both enabled and disabled: "
            + ", ".join(sorted(overlap))
        )

    missing = (discovered - declared) - allowed_missing
    if missing:
        errors.append(
            "Manifest is missing notebook classifications for: "
            + ", ".join(sorted(missing))
        )

    extra = declared - discovered
    if extra:
        errors.append(
            "Manifest references notebooks that do not exist: "
            + ", ".join(sorted(extra))
        )

    for notebook_path, reason in sorted(disabled.items()):
        if not isinstance(reason, str) or not reason.strip():
            errors.append(f"Disabled notebook is missing a reason: {notebook_path}")

    if not manifest.get("repo"):
        errors.append("Manifest must define a non-empty 'repo' value.")
    if not manifest.get("git_ref"):
        errors.append("Manifest must define a non-empty 'git_ref' value.")

    return errors


def validate_enabled_notebook(path: Path, repo: str, git_ref: str) -> list[str]:
    notebook_path = path.relative_to(REPO_ROOT).as_posix()
    payload = load_json(path)
    cells = payload.get("cells", [])
    errors: list[str] = []

    badge_positions = [
        idx
        for idx, cell in enumerate(cells)
        if has_expected_badge(cell, repo, git_ref, notebook_path)
    ]
    any_badge_positions = [
        idx for idx, cell in enumerate(cells) if is_any_badge_cell(cell)
    ]
    bootstrap_positions = [
        idx for idx, cell in enumerate(cells) if is_bootstrap_cell(cell)
    ]
    any_install_positions = [
        idx for idx, cell in enumerate(cells) if is_any_install_cell(cell)
    ]
    first_substantive_code = next(
        (
            idx
            for idx, cell in enumerate(cells)
            if cell.get("cell_type") == "code" and not is_bootstrap_cell(cell)
        ),
        None,
    )

    if not badge_positions:
        errors.append(f"{notebook_path}: missing the expected Colab badge markup.")
    if not bootstrap_positions:
        errors.append(
            f"{notebook_path}: missing a Colab bootstrap cell with the core qmcpy install command."
        )

    if len(badge_positions) > 1:
        errors.append(
            f"{notebook_path}: expected one matching Colab badge, found {len(badge_positions)}."
        )
    if len(any_badge_positions) != len(badge_positions):
        errors.append(
            f"{notebook_path}: found Colab badge markup that does not match the expected notebook URL."
        )
    if len(bootstrap_positions) > 1:
        errors.append(
            f"{notebook_path}: expected one Colab bootstrap cell, found {len(bootstrap_positions)}."
        )
    if len(any_install_positions) != len(bootstrap_positions):
        errors.append(
            f"{notebook_path}: found google.colab setup code that does not include the core qmcpy bootstrap."
        )
    if badge_positions and bootstrap_positions and badge_positions[0] > bootstrap_positions[0]:
        errors.append(
            f"{notebook_path}: Colab badge must appear before the Colab bootstrap cell."
        )

    if (
        badge_positions
        and first_substantive_code is not None
        and badge_positions[0] > first_substantive_code
    ):
        errors.append(
            f"{notebook_path}: Colab badge must appear before the first substantive code cell."
        )

    if (
        bootstrap_positions
        and first_substantive_code is not None
        and bootstrap_positions[0] > first_substantive_code
    ):
        errors.append(
            f"{notebook_path}: Colab bootstrap cell must appear before the first substantive code cell."
        )

    return errors


def validate_disabled_notebook(path: Path) -> list[str]:
    notebook_path = path.relative_to(REPO_ROOT).as_posix()
    payload = load_json(path)
    cells = payload.get("cells", [])
    errors: list[str] = []

    if any(is_any_badge_cell(cell) for cell in cells):
        errors.append(
            f"{notebook_path}: manifest marks this notebook as Colab-disabled, but a badge is present."
        )
    if any(is_any_install_cell(cell) for cell in cells):
        errors.append(
            f"{notebook_path}: manifest marks this notebook as Colab-disabled, but a Colab install cell is present."
        )

    return errors


def run_check(manifest_path: Path, strict: bool = False) -> int:
    manifest = load_json(manifest_path)
    errors = validate_manifest(manifest)
    enabled, disabled = manifest_sets(manifest)
    repo = manifest["repo"]
    git_ref = manifest["git_ref"]

    for notebook_path in sorted(enabled):
        errors.extend(
            validate_enabled_notebook(REPO_ROOT / notebook_path, repo, git_ref)
        )

    for notebook_path in sorted(disabled):
        errors.extend(validate_disabled_notebook(REPO_ROOT / notebook_path))

    if strict:
        for notebook_path in sorted(enabled):
            errors.extend(validate_strict_enabled_notebook(REPO_ROOT / notebook_path))

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(
        "Colab notebook check passed: "
        f"{len(enabled)} enabled, {len(disabled)} disabled, "
        f"{len(enabled) + len(disabled)} total."
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Colab badge/bootstrap cells for demo notebooks."
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help=f"Path to the Colab manifest (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Run additional static Colab-readiness checks for enabled notebooks.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    return run_check(manifest_path, strict=args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
