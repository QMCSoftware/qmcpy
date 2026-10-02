import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

@unittest.skip("Skipped in workflow - called from tb_portfolio_allocation_demo.py instead")
class NotebookTests(BaseNotebookTest):

    @testbook('../../demos/portfolio/data_portfolio_allocation.ipynb', execute=True, timeout=TB_TIMEOUT)
    def test_data_portfolio_allocation_notebook(self, tb):
        pass

if __name__ == '__main__':
    unittest.main()
