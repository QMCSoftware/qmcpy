#!/usr/bin/env python3
"""
Script to generate tb_*.py test files for notebooks in the demos directory.
Can be called from the Makefile or used standalone.
"""

import os
import argparse
from pathlib import Path


def generate_test_file(notebook_path, output_dir=None):
    """
    Generate a tb_*.py test file for a given notebook.

    Args:
        notebook_path (str): Path to the notebook file
        output_dir (str, optional): Directory to save the test file.
                                   Defaults to current directory.

    Returns:
        str: Path to the generated test file
    """
    # Get notebook name and convert to test file name
    notebook_name = Path(notebook_path).stem
    test_name = notebook_name.replace("-", "_")  # .replace('.', '_')

    # Determine output directory
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

    # Create the test file path
    test_file_path = output_dir / f"tb_{test_name}.py"

    # Calculate relative path from test file to notebook
    try:
        notebook_path = Path(notebook_path)
        if notebook_path.is_absolute():
            # Try to make it relative to the test directory
            try:
                rel_path = os.path.relpath(notebook_path, output_dir)
            except ValueError:
                # Can't make relative path, use absolute
                rel_path = str(notebook_path)
        else:
            rel_path = str(notebook_path)
    except Exception:
        rel_path = str(notebook_path)
    rel_path = Path(rel_path).as_posix()

    # Generate the test file content
    test_content = f"""import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    @testbook('{rel_path}', execute=True, timeout=TB_TIMEOUT)
    def test_{test_name}_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
"""

    # Write the test file
    with open(test_file_path, "w") as f:
        f.write(test_content)

    print(f"Generated test file: {test_file_path}")
    return str(test_file_path)


def generate_missing_tests(demos_dir="../../demos", output_dir=None):
    """
    Generate test files for all notebooks in demos directory that don't have tests.

    Args:
        demos_dir (str): Path to the demos directory
        output_dir (str, optional): Directory to save test files

    Returns:
        list: List of generated test file paths
    """
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

    demos_path = Path(demos_dir)
    if not demos_path.exists():
        print(f"Demos directory not found: {demos_path}")
        return []

    generated_files = []

    # Find all notebooks in demos directory
    for notebook_path in demos_path.glob("**/*.ipynb"):
        notebook_name = notebook_path.stem

        # Skip all notebooks in demos/talk_paper_demos/Parslfest_2025/
        # These are presentation/demo notebooks and should not have tests generated
        if "Parslfest_2025" in str(notebook_path) or notebook_path.name in ["archive.ipynb"]:
            continue

        # Skip scratch notebooks (gitignored via "*tmp*" or "sc_*"): they can
        # exist transiently and would otherwise leave a generated tb_*.py
        # permanently pointing at a now-missing notebook.
        if "tmp" in notebook_path.name or notebook_path.name.startswith("sc_"):
            continue

        # Convert notebook name to test file name
        test_name = notebook_name.replace("-", "_")  # .replace('.', '_')
        test_file_path = output_dir / f"tb_{test_name}.py"

        # Check if test file already exists
        if not test_file_path.exists():
            print(f"Missing test for: {notebook_path}")

            # Calculate relative path from test directory to notebook
            try:
                rel_notebook_path = os.path.relpath(notebook_path, output_dir)
            except ValueError:
                rel_notebook_path = str(notebook_path)

            generated_file = generate_test_file(rel_notebook_path, output_dir)
            generated_files.append(generated_file)

    return generated_files


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description="Generate tb_*.py test files for notebook demos"
    )
    parser.add_argument(
        "--notebook", "-n", help="Path to a specific notebook to generate test for"
    )
    parser.add_argument(
        "--demos-dir",
        "-d",
        default="../../demos",
        help="Path to demos directory (default: ../../demos)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Output directory for test files (default: current directory)",
    )
    parser.add_argument(
        "--check-missing",
        "-c",
        action="store_true",
        help="Generate tests for all missing notebooks in demos directory",
    )

    args = parser.parse_args()

    if args.notebook:
        # Generate test for specific notebook
        generate_test_file(args.notebook, args.output_dir)
    elif args.check_missing:
        # Generate tests for all missing notebooks
        generated = generate_missing_tests(args.demos_dir, args.output_dir)
        if generated:
            print(f"\nGenerated {len(generated)} test files:")
            for f in generated:
                print(f"  {f}")
        else:
            print("No missing test files found.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
