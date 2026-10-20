import unittest
from __init__ import BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    def test_pymc_model_notebook(self):
        notebook_path, _ = self.locate_notebook("../../demos/pymc_model.ipynb")
        replacements = {}
        self.run_notebook(notebook_path, replacements)

if __name__ == '__main__':
    unittest.main()
