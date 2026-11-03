from pathlib import Path
import tempfile
import unittest

import nbformat

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

    def test_check_mode_does_not_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.ipynb"
            notebook = nbformat.v4.new_notebook()
            cell = nbformat.v4.new_code_cell("1 + 1")
            cell.metadata["execution"] = {"iopub.status.busy": "timestamp"}
            notebook.cells = [cell]
            nbformat.write(notebook, path)
            original = path.read_bytes()

            self.assertTrue(strip_execution_metadata(path, check=True))
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
