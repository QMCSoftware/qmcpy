import unittest
from __init__ import BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    def test_gbm_demo_notebook(self):
        notebook_path, notebook_dir = self.locate_notebook(
            "../../demos/GBM/gbm_demo.ipynb"
        )
        symlinks_to_fix = [
            "config.py",
            "data_util.py",
            "latex_util.py",
            "plot_util.py",
            "qmcpy_util.py",
            "quantlib_util.py",
        ]
        self.fix_symlinks(notebook_dir, symlinks_to_fix)
        replacements = {
            "cf.is_debug = IN_COLAB": "cf.is_debug = True",
            "n_samples = 2**12": "n_samples = 4",
            "sampler = qp.Lattice(2**7, seed=42)": "sampler = qp.Lattice(4, seed=42)",
            "qp.IIDStdUniform(2**8, seed=42)": "qp.IIDStdUniform(4, seed=42)",
            "qp.Lattice(2**8, seed=42)": "qp.Lattice(4, seed=42)",
            "'n_paths': 2**14": "'n_paths': 4",
            "'n_steps': 252": "'n_steps': 4",
            "replications = 8": "replications = 1",
            "%timeit -n 10 -r 3 -o": "%timeit -n 2 -r 1 -o",
            "plt.savefig": "#plt.savefig",
            "n_plot=50": "n_plot=8",
            "options=['IIDStdUniform', 'Sobol', 'Halton', 'Lattice'], value='IIDStdUniform'": "options=['Sobol'], value='Sobol'",
            "'IIDStdUniform', 'Sobol', 'Halton', 'Lattice'": "'Sobol'",
            "S0, mu, sigma, T, n_samples = 100.0, 0.05, 0.20, 1.0, 2**12": "S0, mu, sigma, T, n_samples = 100.0, 0.05, 0.20, 1.0, 4",
            "n=32": "n=4",
            # Additional replacements to reduce computation
            "2**n": "4",
            "max=8": "max=2",
        }

        stop_at_pattern = "Output results to LaTeX"
        self.run_notebook(notebook_path, replacements, stop_at_pattern=stop_at_pattern)


if __name__ == "__main__":
    unittest.main()
