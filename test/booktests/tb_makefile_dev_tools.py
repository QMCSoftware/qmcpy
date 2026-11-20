import unittest
from testbook import testbook
from __init__ import TB_TIMEOUT, BaseNotebookTest

class NotebookTests(BaseNotebookTest):

    def test_makefile_dev_tools_notebook(self):
        expected = [
            "check_format", "check_notebook_execution_changed",
            "check_ref_style_changed", "check", "tests_fast",
        ]
        notebook = self._notebook_test_path("../../demos/makefile_dev_tools.ipynb")
        with testbook(notebook, execute=True, timeout=TB_TIMEOUT) as tb:
            self.assertEqual(tb.ref("steps")("prepush"), expected)
            pipelines = {name: tb.ref("steps")(name) for name in ("format", "check", "prepush")}
            tb.inject("REPO = None")
            for name, targets in pipelines.items():
                self.assertEqual(tb.ref("steps")(name), targets)

if __name__ == '__main__':
    unittest.main()
