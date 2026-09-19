import shutil
import tempfile
import unittest
from pathlib import Path

import nbformat

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
        replacements = {
            "'n_paths': 2**14": "'n_paths': 8",
            "'n_steps': 252": "'n_steps': 4",
            "'replications': 8": "'replications': 2",
            "replications = 8": "replications = 2",
            "n_paths=2**12": "n_paths=8",
            "n_steps=252": "n_steps=4",
            "%timeit -n 10 -r 3 -o": "%timeit -n 1 -r 1 -o",
            "2**12": "8",
            "2**7": "4",
            "2**8": "4",
            "n=32": "n=4",
            "2**n": "8",
            "max=8, step=1, value=7": "max=2, step=1, value=2",
            "qp.Sobol(252)": "qp.Sobol(4, seed=cf.QMCPY_SEED)",
            "qp.Lattice(252)": "qp.Lattice(4, seed=cf.QMCPY_SEED)",
            "qp.Halton(252)": "qp.Halton(4, seed=cf.QMCPY_SEED)",
        }

        # Run every cell, including file exports, without changing demo artifacts.
        with tempfile.TemporaryDirectory() as tmp:
            test_dir = Path(tmp) / "GBM"
            shutil.copytree(
                notebook_dir,
                test_dir,
                ignore=shutil.ignore_patterns("__pycache__", "images", "outputs"),
            )
            self.fix_symlinks(test_dir, symlinks_to_fix)
            test_path = test_dir / Path(notebook_path).name
            notebook = nbformat.read(test_path, as_version=4)
            setup = next(
                cell for cell in notebook.cells
                if cell.cell_type == "code" and "cf.is_debug =" in cell.source
            )
            setup.source += """
# Keep both sweep axes and every sampler, using tiny CI workloads.
cf.get_experiment_configurations = lambda: {
    'time_steps': {'fixed_paths': 8, 'range': [4, 8], 'series_name': 'Time Steps'},
    'paths': {'fixed_steps': 4, 'range': [4, 8], 'series_name': 'Paths'},
}
"""
            notebook.cells.append(nbformat.v4.new_code_cell("""
assert params_qp['replications'] == replications == 2
stats = ['Mean', 'Std Dev', 'Mean Absolute Error', 'Std Dev Error']
errors = ['Mean SE', 'Std Dev SE', 'MAE SE', 'Std Dev Error SE']
for table in (results_df, sweep_results_df):
    empirical = table[table['Method'] != 'Theoretical']
    assert np.isfinite(empirical[stats + errors].to_numpy()).all()
    assert (empirical[errors].to_numpy() >= 0).all()
    qmcpy_rows = empirical[empirical['Method'] == 'QMCPy']
    assert set(qmcpy_rows['Sampler']) == set(cf.get_sampler_configurations()['all_samplers'])
    assert (qmcpy_rows['Mean SE'] > 0).all()  # Randomizations must differ.
assert set(sweep_results_df['Series']) == {'Time Steps', 'Paths'}
assert set(ablation_df['Construction']) == {'PCA', 'Cholesky', 'BrownianBridge'}
assert len(ablation_df) == 3 * len(cf.get_sampler_configurations()['all_samplers'])
assert np.isfinite(ablation_df[['Mean Absolute Error', 'Std Dev Error', 'Runtime (s)']].to_numpy()).all()
assert os.path.isfile('outputs/gbm_comparison_table.tex')
assert os.path.isfile('outputs/parameter_sweep_results.csv')
assert all(os.path.isfile(f'images/figure_{i}.png') for i in range(5, 9))
plt.close('all')
"""))
            notebook.cells[-1].pop("id", None)  # Match the notebook's v4.4 schema.
            nbformat.write(notebook, test_path)
            self.run_notebook(test_path, replacements)


if __name__ == "__main__":
    unittest.main()
