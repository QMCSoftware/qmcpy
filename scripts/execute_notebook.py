#!/usr/bin/env python3
"""
Execute a notebook from this checkout without writing run artifacts into the repo.

Example:

    python3 scripts/execute_notebook.py demos/DAKOTA_Genz/dakota_genz.ipynb

The script creates/reuses a temporary virtual environment, installs the notebook
runner dependencies, copies the notebook to a temporary working directory,
executes that copy, writes the executed notebook to /private/tmp by default, and
removes large run artifacts such as x_full_dakota.txt from the temp directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PREFERRED_TMP = Path("/private/tmp")
TMP_ROOT = PREFERRED_TMP if PREFERRED_TMP.exists() else Path(tempfile.gettempdir())
DEFAULT_VENV = TMP_ROOT / "qmcpy-notebook-exec-venv"
DEFAULT_OUTPUT_DIR = TMP_ROOT / "qmcpy_executed_notebooks"
DEFAULT_PACKAGES = ["nbconvert", "ipykernel", "pandas", "matplotlib", "gdown", "."]
DEFAULT_CLEANUP_GLOBS = ["x_full_dakota.txt"]


def run(
    command: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    label = " ".join(command)
    if cwd is None:
        print(f"$ {label}", flush=True)
    else:
        print(f"$ (cd {cwd} && {label})", flush=True)
    return subprocess.run(command, cwd=cwd, env=env, check=False)


def create_venv(venv_dir: Path) -> None:
    if (venv_dir / "bin" / "python").exists():
        return
    print(f"Creating virtual environment: {venv_dir}", flush=True)
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    venv.EnvBuilder(with_pip=True).create(venv_dir)


def install_packages(python: Path, packages: list[str]) -> None:
    if not packages:
        return
    result = run([str(python), "-m", "pip", "install", "-q", *packages], cwd=REPO_ROOT)
    if result.returncode:
        raise SystemExit(result.returncode)


def runner_python(args: argparse.Namespace) -> Path:
    if args.python:
        python = args.python.expanduser()
        if not args.skip_install:
            packages = [] if args.no_default_packages else list(DEFAULT_PACKAGES)
            packages.extend(args.package)
            install_packages(python, packages)
        return python

    venv_dir = args.venv.expanduser().resolve()
    create_venv(venv_dir)
    python = venv_dir / "bin" / "python"
    if not args.skip_install:
        packages = [] if args.no_default_packages else list(DEFAULT_PACKAGES)
        packages.extend(args.package)
        install_packages(python, packages)
    return python


def run_repo_checks(args: argparse.Namespace) -> None:
    if args.skip_repo_checks:
        return

    checks = [
        [sys.executable, "scripts/check_colab_notebooks.py", "--strict"],
        ["git", "diff", "--check"],
    ]
    for command in checks:
        result = run(command, cwd=REPO_ROOT)
        if result.returncode:
            raise SystemExit(result.returncode)


def notebook_errors(executed_path: Path) -> list[tuple[int, str, str]]:
    with executed_path.open(encoding="utf-8") as handle:
        notebook = json.load(handle)

    errors: list[tuple[int, str, str]] = []
    for index, cell in enumerate(notebook.get("cells", [])):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors.append(
                    (
                        index,
                        output.get("ename", "<unknown>"),
                        output.get("evalue", ""),
                    )
                )
    return errors


def execution_env() -> dict[str, str]:
    env = os.environ.copy()
    cache_paths = {
        "MPLCONFIGDIR": TMP_ROOT / "qmcpy-mpl-cache",
        "XDG_CACHE_HOME": TMP_ROOT / "qmcpy-xdg-cache",
        "IPYTHONDIR": TMP_ROOT / "qmcpy-ipython",
    }
    for name, path in cache_paths.items():
        path.mkdir(parents=True, exist_ok=True)
        env.setdefault(name, str(path))
    env.setdefault("JUPYTER_PLATFORM_DIRS", "1")
    return env


def cleanup_run_dir(run_dir: Path, cleanup_globs: list[str]) -> None:
    for pattern in cleanup_globs:
        for path in run_dir.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)


def execute_notebook(args: argparse.Namespace, python: Path) -> Path:
    notebook_path = args.notebook.expanduser().resolve()
    if not notebook_path.exists():
        raise SystemExit(f"Notebook not found: {notebook_path}")
    if not notebook_path.is_relative_to(REPO_ROOT):
        raise SystemExit(f"Notebook must be inside repo root: {REPO_ROOT}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else output_dir / notebook_path.name
    )

    if args.work_dir:
        run_dir = args.work_dir.expanduser().resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        return execute_notebook_in_dir(args, python, notebook_path, run_dir, output_path)

    if args.keep_work_dir:
        run_dir = Path(tempfile.mkdtemp(prefix="qmcpy-notebook-run-", dir=TMP_ROOT))
        print(f"Keeping work directory: {run_dir}", flush=True)
        return execute_notebook_in_dir(args, python, notebook_path, run_dir, output_path)

    with tempfile.TemporaryDirectory(
        prefix="qmcpy-notebook-run-",
        dir=TMP_ROOT,
    ) as run_dir_text:
        return execute_notebook_in_dir(
            args,
            python,
            notebook_path,
            Path(run_dir_text),
            output_path,
        )


def execute_notebook_in_dir(
    args: argparse.Namespace,
    python: Path,
    notebook_path: Path,
    run_dir: Path,
    output_path: Path,
) -> Path:
    run_notebook = run_dir / notebook_path.name
    shutil.copy2(notebook_path, run_notebook)

    command = [
        str(python),
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        run_notebook.name,
        "--output",
        str(output_path),
        f"--ExecutePreprocessor.timeout={args.timeout}",
        f"--ExecutePreprocessor.kernel_name={args.kernel}",
    ]
    if args.allow_nbconvert_errors:
        command.append("--allow-errors")

    try:
        result = run(command, cwd=run_dir, env=execution_env())
        if result.returncode:
            raise SystemExit(result.returncode)
        return output_path
    finally:
        if not args.keep_artifacts:
            cleanup_run_dir(run_dir, args.cleanup_glob)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute a notebook in a temporary working directory."
    )
    parser.add_argument("notebook", type=Path, help="Notebook path inside this repo.")
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    parser.add_argument(
        "--python",
        type=Path,
        help="Use this Python interpreter instead of creating/reusing --venv.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Do not install packages into the runner Python.",
    )
    parser.add_argument(
        "--no-default-packages",
        action="store_true",
        help="Do not install the default runner package set.",
    )
    parser.add_argument(
        "--package",
        action="append",
        default=[],
        help="Extra pip package to install in the runner Python. May be repeated.",
    )
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--kernel", default="python3")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Use this working directory instead of a temporary directory.",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep an automatically-created working directory after execution.",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Do not remove cleanup-glob matches from the working directory.",
    )
    parser.add_argument(
        "--cleanup-glob",
        action="append",
        default=list(DEFAULT_CLEANUP_GLOBS),
        help="Artifact glob to remove from the working directory after execution.",
    )
    parser.add_argument(
        "--skip-repo-checks",
        action="store_true",
        help="Skip check_colab_notebooks.py --strict and git diff --check.",
    )
    parser.add_argument(
        "--allow-nbconvert-errors",
        action="store_true",
        help="Pass --allow-errors to nbconvert so it writes output after cell errors.",
    )
    parser.add_argument(
        "--allow-cell-errors",
        action="store_true",
        help="Return success even if the executed notebook contains error outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_repo_checks(args)
    python = runner_python(args)
    output_path = execute_notebook(args, python)

    print(f"Executed notebook written to: {output_path}", flush=True)
    errors = notebook_errors(output_path)
    if errors:
        print("\nNotebook error outputs:", file=sys.stderr)
        for cell_index, name, value in errors:
            print(f"- cell {cell_index}: {name}: {value}", file=sys.stderr)
        if not args.allow_cell_errors:
            return 1

    print("Notebook execution completed without error outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
