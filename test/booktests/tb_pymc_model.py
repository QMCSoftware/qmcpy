import unittest
from __init__ import BaseNotebookTest
# from .__init__ import BaseNotebookTest # uncomment when running just this file

class NotebookTests(BaseNotebookTest):

    def test_pymc_model_notebook(self):
        notebook_path, _ = self.locate_notebook("../../demos/pymc_model.ipynb")
        replacements = {
            "pm.sample()": "pm.sample(draws=10, tune=10)",
            "pm.sample(initvals=": "pm.sample(draws=10, tune=10, initvals=",
            "n_replications=2**7": "n_replications=2",
        }
        self.run_notebook(notebook_path, replacements)

if __name__ == '__main__':
    unittest.main()
