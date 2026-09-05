import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest


class NotebookTests(BaseNotebookTest):

    @testbook(
        "../../demos/true_measure_domain_inclusion.ipynb",
        execute=True,
        timeout=TB_TIMEOUT,
    )
    def test_true_measure_domain_inclusion_notebook(self, tb):
        pass


if __name__ == "__main__":
    unittest.main()
