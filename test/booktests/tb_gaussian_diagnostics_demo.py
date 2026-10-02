import unittest, pytest
import os
from __init__ import BaseNotebookTest, pip_install


@pytest.mark.slow
class NotebookTests(BaseNotebookTest):

    def setUp(self):
        super().setUp()
        # Install required packages
        pip_install("matplotlib", "scipy")
        # Create outputs directory if needed
        os.makedirs("outputs_nb", exist_ok=True)

    def test_gaussian_diagnostics_demo_notebook(self):
        notebook_path, _ = self.locate_notebook(
            "../../demos/gaussian_diagnostics/gaussian_diagnostics_demo.ipynb"
        )
        # Speed up heavy computations and plotting in the demo
        replacements = {
            "nRep = 20": "nRep = 2",
            "nRep = 5": "nRep = 2",
            "npts = 2 ** 6": "npts = 2 ** 4",
            "npts = 2 ** 10": "npts = 2 ** 6",
            "lnthetarange = np.arange(-2, 2.2, 0.2)": "lnthetarange = np.arange(-2, 2.2, 1.6)",
            "lnorderrange = np.arange(-1, 1.1, 0.1)": "lnorderrange = np.arange(-1, 1.1, 0.8)",
            "xtol=1e-3": "xtol=1e-1",
            "N1 = int(2 ** np.floor(16 / dim))": "N1 = int(2 ** np.floor(8 / dim))",
        }
        self.run_notebook(notebook_path, replacements=replacements)


if __name__ == "__main__":
    unittest.main()
