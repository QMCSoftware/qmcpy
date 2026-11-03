from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.check_notebook_execution import check_notebook_file, main


class TestCheckNotebookExecution(unittest.TestCase):

    @staticmethod
    def _write_notebook(path, cells):
        path.write_text(json.dumps({"cells": cells}), encoding="utf-8")

    def _assert_invalid_shape(self, notebook):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.ipynb"
            path.write_text(json.dumps(notebook), encoding="utf-8")
            with self.assertRaises(ValueError):
                check_notebook_file(path)
            with redirect_stdout(StringIO()) as report:
                status = main([str(path), "--strict"])
        self.assertEqual(status, 1)
        self.assertIn("invalid-notebook", report.getvalue())

    def test_notebook_must_be_object(self):
        self._assert_invalid_shape([])

    def test_cells_must_be_list(self):
        self._assert_invalid_shape({"cells": None})

    def test_cell_must_be_object(self):
        self._assert_invalid_shape({"cells": [None]})

    def test_invalid_code_fields(self):
        valid = {"cell_type": "code", "source": ["1"], "outputs": [], "execution_count": 1}
        invalid_fields = (
            {"outputs": None},
            {"outputs": [None]},
            {"source": None},
            {"source": [None]},
            {"outputs": [{"output_type": "error", "evalue": None}]},
        )
        for fields in invalid_fields:
            with self.subTest(fields=fields):
                self._assert_invalid_shape({"cells": [{**valid, **fields}]})

    def test_sequential_and_nonsequential_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.ipynb"
            self._write_notebook(
                path,
                [
                    {"cell_type": "code", "execution_count": 1, "outputs": []},
                    {"cell_type": "code", "execution_count": 2, "outputs": []},
                ],
            )
            self.assertEqual(check_notebook_file(path), [])

            self._write_notebook(
                path,
                [
                    {"cell_type": "code", "execution_count": 1, "outputs": []},
                    {"cell_type": "code", "execution_count": 3, "outputs": []},
                ],
            )
            findings = check_notebook_file(path)

        self.assertEqual([finding[1] for finding in findings], ["execution-count-not-sequential"])

    def test_generated_colab_bootstrap_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.ipynb"
            for install in ("!pip install -q qmcpy\n", "!pip install -q -e {repo_root}\n"):
                with self.subTest(install=install):
                    self._write_notebook(
                        path,
                        [
                            {
                                "cell_type": "code",
                                "execution_count": None,
                                "outputs": [],
                                "source": [
                                    "# @title Execute this cell to install dependencies\n",
                                    "import google.colab\n",
                                    install,
                                ],
                            },
                            {"cell_type": "code", "execution_count": 1, "outputs": []},
                        ],
                    )
                    self.assertEqual(check_notebook_file(path), [])

    def test_invalid_diff_ref(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.ipynb"
            self._write_notebook(path, [{"cell_type": "code", "execution_count": 1}])
            with patch("scripts.check_notebook_execution._changed_files", side_effect=RuntimeError("bad ref")):
                with redirect_stdout(StringIO()) as report, redirect_stderr(StringIO()) as errors:
                    status = main([str(path), "--diff", "missing", "--strict"])

        self.assertEqual(status, 1)
        self.assertEqual(report.getvalue(), "")
        self.assertIn("ERROR: --diff missing: bad ref", errors.getvalue())

    def test_empty_code_cell_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.ipynb"
            self._write_notebook(
                path,
                [
                    {"cell_type": "code", "execution_count": 1, "outputs": [], "source": ["x = 1\n"]},
                    {"cell_type": "code", "execution_count": None, "outputs": [], "source": ["   \n", "\n"]},
                    {"cell_type": "code", "execution_count": 2, "outputs": [], "source": ["y = 2\n"]},
                ],
            )

            self.assertEqual(check_notebook_file(path), [])

    def test_empty_code_cell_error_output_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.ipynb"
            self._write_notebook(
                path,
                [{
                    "cell_type": "code",
                    "execution_count": None,
                    "outputs": [{
                        "output_type": "error",
                        "ename": "ValueError",
                        "evalue": "retained error",
                    }],
                    "source": [],
                }],
            )
            findings = check_notebook_file(path)

        self.assertEqual([finding[1] for finding in findings], ["cell-has-error-output"])

    def test_error_output_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.ipynb"
            self._write_notebook(
                path,
                [{
                    "cell_type": "code",
                    "execution_count": 1,
                    "outputs": [{
                        "output_type": "error",
                        "ename": "ValueError",
                        "evalue": "bad value",
                    }],
                }],
            )
            findings = check_notebook_file(path)

        self.assertEqual([finding[1] for finding in findings], ["cell-has-error-output"])

    def test_strict_mode_fails_for_malformed_notebook(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.ipynb"
            path.write_text("not json", encoding="utf-8")

            with redirect_stdout(StringIO()):
                status = main([str(path), "--strict", "--quiet"])

        self.assertEqual(status, 1)


if __name__ == "__main__":
    unittest.main()
