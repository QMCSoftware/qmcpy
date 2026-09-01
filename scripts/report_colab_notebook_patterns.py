#!/usr/bin/env python3
"""
Group demo notebooks by Colab badge/bootstrap cell pattern.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from check_colab_notebooks import (
    DEFAULT_MANIFEST,
    REPO_ROOT,
    cell_source_text,
    is_any_badge_cell,
    is_bootstrap_cell,
    load_json,
    manifest_sets,
    validate_manifest,
)

def first_matching_cell(
    cells: list[dict],
    predicate,
) -> tuple[int | None, str]:
    for idx, cell in enumerate(cells, start=1):
        if predicate(cell):
            return idx, cell_source_text(cell)
    return None, ""


def extra_install_commands(source: str) -> list[str]:
    commands: list[str] = []
    for raw_line in source.splitlines():
        line = raw_line.strip()
        if not line.startswith("!pip install"):
            continue
        if "qmcpy" in line.lower():
            continue
        if " -q " in line:
            commands.append(line.split(" -q ", 1)[1].strip())
        else:
            commands.append(line.split("!pip install", 1)[1].strip())
    return commands


def pattern_family(source: str) -> str:
    if not source:
        return "Enabled but missing bootstrap cell"

    has_repo_clone = "git clone" in source
    has_path_setup = "os.chdir(" in source or "sys.path.insert" in source
    has_apt = "apt-get" in source
    extra_installs = extra_install_commands(source)

    if has_repo_clone and extra_installs:
        return "Repo-local bootstrap + extra pip installs"
    if has_repo_clone or has_path_setup:
        return "Repo-local bootstrap"
    if has_apt:
        return "LaTeX bootstrap"
    if extra_installs:
        return "Extra pip bootstrap"
    return "Basic qmcpy bootstrap"


def exact_variant(source: str) -> str:
    if not source:
        return "missing bootstrap cell"

    labels = ["qmcpy"]
    if "git clone" in source:
        labels.append("repo-clone")
    if "os.chdir(" in source or "sys.path.insert" in source:
        labels.append("path-setup")
    if "apt-get" in source:
        labels.append("apt")

    extra_installs = extra_install_commands(source)
    if extra_installs:
        labels.append(f"extra-pip: {', '.join(extra_installs)}")
    else:
        labels.append("extra-pip: none")

    return " | ".join(labels)


def placement_label(badge_position: int | None, bootstrap_position: int | None) -> str:
    badge = "missing" if badge_position is None else str(badge_position)
    bootstrap = "missing" if bootstrap_position is None else str(bootstrap_position)
    return f"badge cell {badge}, bootstrap cell {bootstrap}"


def print_grouped_notebooks(
    title: str,
    groups: dict[str, list[str]],
    details: dict[str, dict[str, str]] | None = None,
) -> None:
    print(title)
    for group_name in sorted(groups):
        notebooks = sorted(groups[group_name])
        print(f"- {group_name} ({len(notebooks)} notebooks)")
        for notebook in notebooks:
            suffix = ""
            if details is not None:
                detail = details.get(group_name, {}).get(notebook)
                if detail:
                    suffix = f" [{detail}]"
            print(f"  - {notebook}{suffix}")
    print()


def run_report(manifest_path: Path) -> int:
    manifest = load_json(manifest_path)
    errors = validate_manifest(manifest)
    if errors:
        for error in errors:
            print(error)
        return 1

    enabled, disabled = manifest_sets(manifest)

    family_groups: dict[str, list[str]] = defaultdict(list)
    placement_groups: dict[str, list[str]] = defaultdict(list)
    family_details: dict[str, dict[str, str]] = defaultdict(dict)

    for notebook_path in sorted(enabled):
        payload = load_json(REPO_ROOT / notebook_path)
        cells = payload.get("cells", [])
        badge_position, _ = first_matching_cell(cells, is_any_badge_cell)
        bootstrap_position, bootstrap_source = first_matching_cell(
            cells, is_bootstrap_cell
        )
        family = pattern_family(bootstrap_source)
        exact = exact_variant(bootstrap_source)
        placement = placement_label(badge_position, bootstrap_position)
        extra_installs = extra_install_commands(bootstrap_source)

        family_groups[family].append(notebook_path)
        placement_groups[placement].append(notebook_path)

        if extra_installs:
            family_details[family][notebook_path] = ", ".join(extra_installs)

    disabled_groups: dict[str, list[str]] = defaultdict(list)
    for notebook_path, reason in sorted(disabled.items()):
        disabled_groups[reason].append(notebook_path)

    total = len(enabled) + len(disabled)
    print("Colab notebook pattern report")
    print(f"Manifest: {manifest_path.relative_to(REPO_ROOT).as_posix()}")
    print(f"Enabled: {len(enabled)}")
    print(f"Disabled: {len(disabled)}")
    print(f"Total: {total}")
    print()

    print_grouped_notebooks(
        "Enabled pattern families",
        family_groups,
        details=family_details,
    )
    print_grouped_notebooks("Badge/bootstrap placement", placement_groups)
    print_grouped_notebooks("Disabled notebooks by reason", disabled_groups)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group demo notebooks by Colab badge/bootstrap cell pattern."
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help=f"Path to the Colab manifest (default: {DEFAULT_MANIFEST})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_report(Path(args.manifest).resolve())


if __name__ == "__main__":
    raise SystemExit(main())
