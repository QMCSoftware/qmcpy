import unittest
from pathlib import Path

from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

NOTEBOOK = Path(__file__).resolve().parents[2] / "demos" / "brownian_bridge.ipynb"


class NotebookTests(BaseNotebookTest):

    @testbook(NOTEBOOK.as_posix(), execute=True, timeout=TB_TIMEOUT)
    def test_brownian_bridge_notebook(self, tb):
        pass


if __name__ == "__main__":
    unittest.main()
