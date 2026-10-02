import json
from pathlib import Path
import tempfile
import unittest

from scripts.check_ref_style import check_markdown_file, check_notebook_file


class TestCheckRefStyle(unittest.TestCase):

    def test_markdown_code_is_not_treated_as_a_citation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.md"
            path.write_text(
                "Use `values[1]` in this example.\n\n"
                "```python\n"
                "values[2]\n"
                "```\n",
                encoding="utf-8",
            )

            self.assertEqual(check_markdown_file(path), [])

    def test_bibliography_heading_defines_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.md"
            path.write_text(
                "This method is described in [1].\n\n"
                "## Bibliography\n\n"
                "[1] A. Author, Example, 2024.\n",
                encoding="utf-8",
            )

            self.assertEqual(check_markdown_file(path), [])

    def test_notebook_code_spans_and_fences_are_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.ipynb"
            path.write_text(
                json.dumps({
                    "cells": [{
                        "cell_type": "markdown",
                        "source": [
                            "Use `values[1]`.\n",
                            "```python\n",
                            "values[2]\n",
                            "```\n",
                        ],
                    }],
                }),
                encoding="utf-8",
            )

            self.assertEqual(check_notebook_file(path), [])


if __name__ == "__main__":
    unittest.main()
