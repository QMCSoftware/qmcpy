import re
import unittest, pytest
from __init__ import BaseNotebookTest
import tb_data_portfolio_allocation

@pytest.mark.slow
class NotebookTests(BaseNotebookTest):

    def test_data_portfolio_allocation(self):
        """Run the data portfolio allocation notebook test"""
        # Force run even if skip decorator is present
        test_instance = tb_data_portfolio_allocation.NotebookTests()
        test_instance.setUp()
        # Call the actual test method directly, bypassing skip
        notebook_path, _ = test_instance.locate_notebook('../../demos/portfolio/data_portfolio_allocation.ipynb')
        test_instance.run_notebook(notebook_path)

    def test_portfolio_allocation_demo_notebook(self):
        notebook_path, _ = self.locate_notebook(
            "../../demos/portfolio/portfolio_allocation_demo.ipynb"
        )

        replacements = {
            "n_ports = [2**13, 2**14, 2**15]": "n_ports = [2**7, 2**8]",
            # measure_runtime's sweep sizes and run_backtest_case's replication counts moved
            # from notebook-cell locals into sampler_util.py function bodies (now parameters
            # with matching defaults), so they're no longer in the notebook's own cell text to
            # patch directly; override them at the (still-in-notebook) call sites instead.
            "df_runtime = su.measure_runtime(sampler_types, transform=TRANSFORM_METH)":
                "df_runtime = su.measure_runtime(sampler_types, transform=TRANSFORM_METH, "
                "dimensions=[5, 10], num_samples=[2**m for m in range(8, 11)], fixed_dimension=10)",
            # Shrink Section 4's loops to the smallest ticker universe.
            "for n_tickers in (4, 10, 20, 40):": "for n_tickers in (4,):",
            re.compile(r"num_ports_section4 = 2\*\*\d+(  # or 2\*\*\d+)?"): "num_ports_section4 = 2**7",
            "su.run_backtest_case(n_tickers, 'OOS', section4_data, sampler_types, num_ports=num_ports_section4, principal=principal, transform=TRANSFORM_METH)":
                "su.run_backtest_case(n_tickers, 'OOS', section4_data, sampler_types, num_ports=num_ports_section4, principal=principal, transform=TRANSFORM_METH, oos_replications=5)",
        }

        self.run_notebook(notebook_path, replacements)

if __name__ == "__main__":
    unittest.main()
