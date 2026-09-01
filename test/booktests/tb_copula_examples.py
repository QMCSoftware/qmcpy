import unittest
from __init__ import TB_TIMEOUT, BaseNotebookTest


class NotebookTests(BaseNotebookTest):

    def test_copula_examples_notebook(self):
        notebook_path, _ = self.locate_notebook(
            "../../demos/copula_examples.ipynb"
        )
        replacements = {
            "fig.savefig": "# fig.savefig",
        }
        self.run_notebook(notebook_path, replacements=replacements, timeout=TB_TIMEOUT)


if __name__ == "__main__":
    unittest.main()
