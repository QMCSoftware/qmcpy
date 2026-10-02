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
            # Regex (not literal): these sweep bounds keep getting retuned, which dead-ended literal keys twice; anchoring on the stable prefix survives future retuning.
            re.compile(r"dimensions = \[5, 10, 20[^\]]*\]"): "dimensions = [5, 10]",
            re.compile(r"range\(8,\s*\d+\)"): "range(8, 11)",
            "fixed_dimension = 50": "fixed_dimension = 10",
            'tickers1, description1 = load_assets(f"data/df10_{start_date}_to_{end_date}.csv.gz")': 'tickers1, description1 = ["AAPL", "ABNB"], ["Apple", "Airbnb"]',
            # Shrinks Section 4's loops to just 4 tickers (independent of the tickers1 shrink above).
            "for n_tickers in (4, 10, 20, 40):": "for n_tickers in (4,):",
            re.compile(r"num_ports_section4 = 2\*\*\d+(  # or 2\*\*\d+)?"): "num_ports_section4 = 2**7",
            "replications = 50": "replications = 5",
        }

        self.run_notebook(notebook_path, replacements)

if __name__ == "__main__":
    unittest.main()
