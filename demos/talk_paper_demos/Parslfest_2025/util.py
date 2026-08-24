"""
Utility functions for ParslFest 2025 notebooks.
Contains common helper functions used across multiple notebooks.
"""

import os
import sys
import subprocess
import re


def find_repo_root(start=None):
    """
    Find the repository root by looking for pyproject.toml.

    Args:
        start: Starting directory (default: current working directory)

    Returns:
        str: Path to the repository root

    Raises:
        FileNotFoundError: If repository root not found
    """
    if start is None:
        start = os.getcwd()

    cur = start
    while True:
        if os.path.exists(os.path.join(cur, "pyproject.toml")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise FileNotFoundError("Repository root not found (no pyproject.toml)")
        cur = parent


def setup_environment(output_dir="output"):
    """
    Common setup: add booktests to path and create output directory.

    Args:
        output_dir: Directory for output files (default: "output")

    Returns:
        str: The output directory path
    """
    sys.path.append(os.path.join(find_repo_root(), "test", "booktests"))
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def run_make_command(cmd, output_file, is_debug=False, tests=None, env=None):
    """
    Run a make command and capture output to a file.

    Args:
        cmd: Base make command (e.g., "booktests_no_docker")
        output_file: Path to write output
        is_debug: If True, use debug test set
        tests: Custom test string (overrides is_debug default)
        env: Environment variables dict (optional)

    Returns:
        bool: True if command succeeded
    """
    is_linux = sys.platform.startswith("Linux")

    if tests is None and is_debug:
        tests = "tb_quickstart tb_qmcpy_intro tb_lattice_random_generator"

    if tests:
        make_cmd = ["make", cmd, f"TESTS={tests}"]
    else:
        make_cmd = ["make", cmd]

    if is_linux:
        make_cmd = ["taskset", "-c", "0"] + make_cmd

    repo_root = find_repo_root()

    with open(output_file, "wb") as out_f:
        try:
            subprocess.run(
                make_cmd,
                cwd=repo_root,
                stdout=out_f,
                stderr=subprocess.STDOUT,
                check=True,
                env=env,
            )
            return True
        except subprocess.CalledProcessError:
            return False


def parse_total_time(output_file, pattern=r"Ran \d+ tests? in ([\d\.]+)s"):
    """
    Parse total time from test output file.

    Args:
        output_file: Path to output file
        pattern: Regex pattern to match time

    Returns:
        float: Parsed time or 0.0 if not found
    """
    with open(output_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
        match = re.search(pattern, text)
        return float(match.group(1)) if match else 0.0
