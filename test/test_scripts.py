from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import harden_colab_notebook as harden


class TestHardenColabNotebook(unittest.TestCase):
    """Tests for Colab hardening helper functions."""

    @staticmethod
    def _markdown_cell(lines: list[str]) -> dict:
        return {"cell_type": "markdown", "metadata": {}, "source": lines}

    @staticmethod
    def _code_cell(source: str) -> dict:
        return {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [source],
        }

    def test_badge_stripped_cell_preserves_intro_text(self):
        cell = self._markdown_cell(
            [
                "# ML Sensitivity Indices\n",
                "\n",
                "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
                "(https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/iris.ipynb)\n",
                "\n",
                "This notebook demonstrates QMCPy's support for vectorized sensitivity index computation.\n",
            ]
        )

        cleaned = harden.badge_stripped_cell(cell)

        self.assertIsNotNone(cleaned)
        text = "".join(cleaned["source"])
        self.assertIn("# ML Sensitivity Indices", text)
        self.assertIn("vectorized sensitivity index computation", text)
        self.assertNotIn("Open In Colab", text)
        self.assertNotIn("colab-badge.svg", text)

    def test_remove_any_badge_cells_drops_badge_only_cells(self):
        cells = [
            self._markdown_cell(
                [
                    "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
                    "(https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/iris.ipynb)\n"
                ]
            ),
            self._code_cell("print('kept')\n"),
        ]

        cleaned = harden.remove_any_badge_cells(cells)

        self.assertEqual(cleaned, [cells[1]])

    def test_badge_bootstrap_insert_index_keeps_leading_markdown_block(self):
        cells = [
            self._markdown_cell(["# Title\n"]),
            self._markdown_cell(["Description paragraph.\n"]),
            self._code_cell("print('first code')\n"),
        ]

        self.assertEqual(harden.badge_bootstrap_insert_index(cells), 2)

    def test_extra_pip_packages_preserves_later_explicit_installs(self):
        cells = [
            self._code_cell("import qmcpy as qp\n"),
            self._code_cell("import ipywidgets as widgets\n"),
            self._code_cell(
                "try:\n"
                "    import QuantLib as ql\n"
                "except ModuleNotFoundError:\n"
                "    !pip install -q QuantLib\n"
            ),
        ]

        self.assertEqual(
            harden.extra_pip_packages(cells),
            ["ipywidgets", "QuantLib"],
        )
