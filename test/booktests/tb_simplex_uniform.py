import unittest

from __init__ import BaseNotebookTest


class NotebookTests(BaseNotebookTest):

    def test_simplex_uniform_notebook(self):
        notebook_path, _ = self.locate_notebook("../../demos/simplex_uniform.ipynb")
        self.run_notebook(notebook_path)


if __name__ == "__main__":
    unittest.main()
