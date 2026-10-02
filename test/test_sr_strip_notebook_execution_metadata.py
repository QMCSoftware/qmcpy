from pathlib import Path
import tempfile
import unittest

import pytest

# nbformat belongs to the notebook stack omitted from the slim `test_core`
# extra (see pyproject.toml / CONTRIBUTING.md), so skip rather than fail
# collection where it's absent, as the other optional-stack tests do.
nbformat = pytest.importorskip("nbformat")

from scripts.strip_notebook_execution_metadata import strip_execution_metadata


class TestStripNotebookExecutionMetadata(unittest.TestCase):

    def test_removes_null_execution_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.ipynb"
            notebook = nbformat.v4.new_notebook()
            cell = nbformat.v4.new_code_cell("1 + 1")
            cell.metadata["execution"] = None
            notebook.cells = [cell]
            nbformat.write(notebook, path)

            self.assertTrue(strip_execution_metadata(path))
            updated = nbformat.read(path, as_version=nbformat.NO_CONVERT)

        self.assertNotIn("execution", updated.cells[0].metadata)


if __name__ == "__main__":
    unittest.main()
