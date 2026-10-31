import tempfile
import unittest
from pathlib import Path

import nbformat

from __init__ import BaseNotebookTest


class NotebookTests(BaseNotebookTest):

    def test_saved_results(self):
        notebook_path, _ = self.locate_notebook(
            "../../demos/mixed_distributions_blog_analysis.ipynb"
        )
        notebook = nbformat.read(notebook_path, as_version=4)
        nbformat.validate(notebook)
        code_cells = [
            cell for cell in notebook.cells
            if cell.cell_type == "code"
            and not cell.source.startswith("# @title Execute this cell")
        ]
        self.assertEqual(
            [cell.execution_count for cell in code_cells],
            list(range(1, len(code_cells) + 1)),
        )
        outputs = [output for cell in code_cells for output in cell.outputs]
        self.assertFalse(any(output.output_type == "error" for output in outputs))
        self.assertEqual(
            sum("image/png" in output.get("data", {}) for output in outputs), 4
        )

    def test_mixed_distributions_blog_analysis_notebook(self):
        """Exercise every cell without rerunning the full publication study."""
        notebook_path, _ = self.locate_notebook(
            "../../demos/mixed_distributions_blog_analysis.ipynb"
        )
        original = Path(notebook_path).read_bytes()
        notebook = nbformat.read(notebook_path, as_version=4)
        source = "\n".join(
            cell.source for cell in notebook.cells if cell.cell_type == "code"
        )
        with tempfile.TemporaryDirectory() as output_dir:
            replacements = {
                "sample_sizes = 2 ** np.arange(6, 16)":
                    "sample_sizes = 2 ** np.arange(6, 11)",
                "n_cases100 = 100": "n_cases100 = 2",
                "mc_n = 1_000_000": "mc_n = 20_000",
                'output_dir = repo_root / "demos" / "mixed_distributions_blog_outputs"':
                    f"output_dir = Path({output_dir!r})",
            }
            # Fail if a notebook edit makes a workload override stop matching.
            for old in replacements:
                self.assertEqual(source.count(old), 1, old)
            self.run_notebook(notebook_path, replacements=replacements)
            self.assertEqual(
                {path.name for path in Path(output_dir).iterdir()},
                {
                    "mixture_density_1d.png",
                    "equal_cost_1d.png",
                    "equal_cost_fixed_10d.png",
                    "equal_cost_random_10d.png",
                },
            )
        self.assertEqual(Path(notebook_path).read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
