#!/usr/bin/env python3
"""
Execute notebooks from a GitHub branch checkout, with Colab-friendly defaults.

Typical Colab usage:

    !python scripts/execute_notebooks_from_branch.py \
        --branch colab_easy_fix_redo \
        --notebook demos/quickstart.ipynb

The script writes executed notebooks to an output directory and does not modify
the source notebooks in the checkout.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_REPO_URL = "https://github.com/QMCSoftware/QMCSoftware.git"
DEFAULT_BRANCH = "develop"
DEFAULT_CHECKOUT_DIR = Path("/content/QMCSoftware")
DEFAULT_OUTPUT_DIR = Path("/content/qmcpy_executed_notebooks")
DEFAULT_MANIFEST = "scripts/colab_notebooks_manifest.json"
DISABLED_BOOTSTRAP_SOURCE = [
    "# Colab bootstrap cell skipped by scripts/execute_notebooks_from_branch.py.\n",
    "# The script is already running inside the selected checkout and executes\n",
    "# with the current runtime environment.\n",
    "pass\n",
]


def run(command: list[str], cwd: Path | None = None) -> None:
    label = " ".join(command)
    if cwd is not None:
        print(f"$ (cd {cwd} && {label})", flush=True)
    else:
        print(f"$ {label}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def ensure_checkout(
    repo_url: str,
    branch: str,
    checkout_dir: Path,
    fresh: bool,
    update_existing: bool,
) -> Path:
    checkout_dir = checkout_dir.expanduser().resolve()
    if fresh and checkout_dir.exists():
        shutil.rmtree(checkout_dir)

    if (checkout_dir / ".git").exists():
        if update_existing:
            run(["git", "fetch", "origin", branch], cwd=checkout_dir)
            run(["git", "checkout", branch], cwd=checkout_dir)
            run(["git", "pull", "--ff-only", "origin", branch], cwd=checkout_dir)
        else:
            print(f"Using existing checkout without updating: {checkout_dir}", flush=True)
    elif checkout_dir.exists() and any(checkout_dir.iterdir()):
        raise SystemExit(
            f"{checkout_dir} exists and is not an empty git checkout. "
            "Pass --fresh or choose another --checkout-dir."
        )
    else:
        checkout_dir.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                branch,
                repo_url,
                str(checkout_dir),
            ]
        )

    return checkout_dir


def load_manifest_notebooks(repo_root: Path, manifest_rel: str) -> list[str]:
    manifest_path = repo_root / manifest_rel
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    return list(manifest.get("enabled", []))


def selected_notebooks(args: argparse.Namespace, repo_root: Path) -> list[str]:
    notebooks = list(args.notebook)
    if args.all_enabled:
        notebooks.extend(load_manifest_notebooks(repo_root, args.manifest))
    if not notebooks:
        raise SystemExit("No notebooks selected. Pass --notebook or --all-enabled.")

    seen: set[str] = set()
    deduped: list[str] = []
    for notebook in notebooks:
        normalized = notebook.strip().lstrip("./")
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def install_runner_dependencies(python: str) -> None:
    run([python, "-m", "pip", "install", "-q", "nbconvert", "nbformat", "ipykernel"])


def cell_source_text(cell: dict) -> str:
    source = cell.get("source", [])
    if isinstance(source, str):
        return source
    return "".join(source)


def is_colab_bootstrap_cell(cell: dict) -> bool:
    source = cell_source_text(cell)
    return (
        cell.get("cell_type") == "code"
        and "import google.colab" in source
        and "pip install" in source
    )


def execution_input_path(notebook_path: Path, skip_colab_bootstrap: bool) -> Path:
    if not skip_colab_bootstrap:
        return notebook_path

    with notebook_path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)

    changed = False
    for cell in notebook.get("cells", []):
        if not is_colab_bootstrap_cell(cell):
            continue
        cell["source"] = DISABLED_BOOTSTRAP_SOURCE
        cell["execution_count"] = None
        cell["outputs"] = []
        changed = True

    if not changed:
        return notebook_path

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".ipynb",
        prefix=".tmp_branch_execute_",
        dir=notebook_path.parent,
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(notebook, handle)

    return temp_path


def execute_notebook(
    repo_root: Path,
    notebook_rel: str,
    output_dir: Path,
    timeout: int,
    kernel: str,
    allow_errors: bool,
    skip_colab_bootstrap: bool,
) -> None:
    notebook_path = repo_root / notebook_rel
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_rel}")

    output_path = output_dir / notebook_rel
    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_path = execution_input_path(notebook_path, skip_colab_bootstrap)

    try:
        command = [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            str(input_path),
            "--output",
            str(output_path),
            "--ExecutePreprocessor.timeout",
            str(timeout),
            "--ExecutePreprocessor.kernel_name",
            kernel,
        ]
        if allow_errors:
            command.append("--allow-errors")

        run(command, cwd=repo_root)
    finally:
        if input_path != notebook_path and input_path.exists():
            input_path.unlink()



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone/fetch a branch and execute selected notebooks from it."
    )
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument(
        "--checkout-dir",
        type=Path,
        default=DEFAULT_CHECKOUT_DIR,
        help="Directory for the branch checkout. Defaults to /content/QMCSoftware.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for executed notebook outputs.",
    )
    parser.add_argument(
        "--notebook",
        action="append",
        default=[],
        help="Notebook path relative to the repo root. May be passed multiple times.",
    )
    parser.add_argument(
        "--all-enabled",
        action="store_true",
        help="Execute every enabled notebook listed in the Colab manifest.",
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--kernel", default="python3")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete and reclone --checkout-dir before running.",
    )
    parser.add_argument(
        "--no-update-existing",
        action="store_true",
        help="If --checkout-dir is already a git checkout, use it as-is without fetching, checking out, or pulling.",
    )
    parser.add_argument(
        "--install-runner-deps",
        action="store_true",
        help="Install nbconvert/nbformat/ipykernel before executing notebooks.",
    )
    parser.add_argument(
        "--execute-colab-bootstrap",
        action="store_true",
        help=(
            "Execute Colab bootstrap cells unchanged. By default they are skipped "
            "in temporary execution copies because google.colab frontend messages "
            "can break nbconvert kernels."
        ),
    )
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="Write executed notebooks even if cells raise errors.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.install_runner_deps:
        install_runner_dependencies(sys.executable)

    repo_root = ensure_checkout(
        args.repo_url,
        args.branch,
        args.checkout_dir,
        fresh=args.fresh,
        update_existing=not args.no_update_existing,
    )
    output_dir = args.output_dir.expanduser().resolve()
    notebooks = selected_notebooks(args, repo_root)

    failures: list[tuple[str, str]] = []
    for index, notebook in enumerate(notebooks, start=1):
        print(f"\n[{index}/{len(notebooks)}] Executing {notebook}", flush=True)
        try:
            execute_notebook(
                repo_root,
                notebook,
                output_dir,
                timeout=args.timeout,
                kernel=args.kernel,
                allow_errors=args.allow_errors,
                skip_colab_bootstrap=not args.execute_colab_bootstrap,
            )
        except Exception as exc:  # noqa: BLE001 - print all failures at the end
            failures.append((notebook, str(exc)))
            print(f"FAILED: {notebook}: {exc}", file=sys.stderr, flush=True)

    if failures:
        print("\nNotebook execution failed:", file=sys.stderr)
        for notebook, error in failures:
            print(f"- {notebook}: {error}", file=sys.stderr)
        return 1

    print(f"\nExecuted {len(notebooks)} notebook(s). Outputs are in {output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
