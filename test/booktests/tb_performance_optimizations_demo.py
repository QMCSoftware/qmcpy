import unittest
from __init__ import BaseNotebookTest


class NotebookTests(BaseNotebookTest):

    def test_performance_optimizations_demo_notebook(self):
        notebook_path, _ = self.locate_notebook("../../demos/performance_optimizations_demo.ipynb")
        replacements = {
            "%timeit -r 7 -o": "%timeit -n 1 -r 1 -o",
            "if os.environ.get('QMCPY_RERUN_DOWNSTREAM_DEMOS'):": "if False:",
        }
        self.run_notebook(notebook_path, replacements)


if __name__ == "__main__":
    unittest.main()
