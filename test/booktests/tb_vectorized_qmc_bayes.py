import unittest, pytest
from __init__ import BaseNotebookTest


@pytest.mark.slow
class NotebookTests(BaseNotebookTest):

    def test_vectorized_qmc_bayes_notebook(self):
        notebook_path, _ = self.locate_notebook(
            "../../demos/vectorized_qmc_bayes.ipynb"
        )
        # Reduce sample/grid sizes and loosen tolerances to speed execution
        replacements = {
            "n = 2**6": "n = 2**3",
            "tpax = 32": "tpax = 4",
            "abs_tol = 1e-1": "abs_tol = 5e-1",
            "abs_tol=.025": "abs_tol=.2",
            "abs_tol=.05": "abs_tol=.2",
            "abs_tol=.005": "abs_tol=.1",
            "n_restarts_optimizer = 16": "n_restarts_optimizer = 1",
            "n_limit=2**18": "n_limit=2**10",
            "xplt = np.linspace(0,1,100)": "xplt = np.linspace(0,1,20)",
        }
        self.run_notebook(notebook_path, replacements)


if __name__ == "__main__":
    unittest.main()
