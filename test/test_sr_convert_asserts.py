import shutil
import tempfile
import textwrap
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from scripts import convert_asserts


class TestConvertAsserts(unittest.TestCase):

    def setUp(self):
        self.tmp_path = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp_path, ignore_errors=True)

    def _write(self, source):
        path = self.tmp_path / "sample.py"
        path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
        return path

    def test_converts_assert_message_and_preserves_inline_comment(self):
        source = textwrap.dedent(
            '''
            def positive(x):
                assert x > 0, f"expected positive x, got {x}"  # public input
                return x
            '''
        ).lstrip()

        result = convert_asserts.transform_source(source)

        self.assertEqual(result.converted_lines, (2,))
        self.assertEqual(result.skipped_lines, ())
        self.assertIn("if not (x > 0):  # public input", result.source)
        self.assertIn(
            'raise AssertionError(f"expected positive x, got {x}")',
            result.source,
        )

        namespace = {}
        exec(result.source, namespace)
        self.assertEqual(namespace["positive"](2), 2)
        with self.assertRaisesRegex(AssertionError, "expected positive x, got -1"):
            namespace["positive"](-1)

    def test_preserves_multiline_condition_and_message(self):
        source = textwrap.dedent(
            '''
            def bounded(x):
                assert (
                    0 <= x <= 1
                ), (
                    f"x outside [0, 1]: {x}"
                )
            '''
        ).lstrip()

        result = convert_asserts.transform_source(source)

        self.assertIn("if not (\n        0 <= x <= 1\n    ):", result.source)
        self.assertIn(
            'raise AssertionError(\n            f"x outside [0, 1]: {x}"\n        )',
            result.source,
        )
        compile(result.source, "sample.py", "exec")

    def test_supports_an_explicit_exception_already_in_scope(self):
        source = "def positive(x):\n    assert x > 0, 'positive required'\n"

        result = convert_asserts.transform_source(source, exception="ValueError")

        namespace = {}
        exec(result.source, namespace)
        with self.assertRaisesRegex(ValueError, "positive required"):
            namespace["positive"](0)

    def test_preserves_a_tuple_as_one_exception_argument(self):
        source = "def f():\n    assert False, ('left', 'right')\n"

        result = convert_asserts.transform_source(source)

        namespace = {}
        exec(result.source, namespace)
        with self.assertRaises(AssertionError) as context:
            namespace["f"]()
        self.assertEqual(context.exception.args, (("left", "right"),))

    def test_check_mode_reports_without_writing(self):
        path = self._write(
            '''
            def positive(x):
                assert x > 0
            '''
        )
        original = path.read_text(encoding="utf-8")
        output = StringIO()

        with redirect_stdout(output):
            status = convert_asserts.main(["--check", str(path)])

        self.assertEqual(status, 1)
        self.assertIn("1 file(s) would change", output.getvalue())
        self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_skips_assert_mixed_with_other_one_line_statements(self):
        source = "def f(x):\n    assert x; return x\n"

        result = convert_asserts.transform_source(source)

        self.assertEqual(result.source, source)
        self.assertEqual(result.converted_lines, ())
        self.assertEqual(result.skipped_lines, (2,))

    def test_skips_assert_in_a_one_line_compound_suite(self):
        source = "def f(x):\n    if x: assert x > 0\n"

        result = convert_asserts.transform_source(source)

        self.assertEqual(result.source, source)
        self.assertEqual(result.converted_lines, ())
        self.assertEqual(result.skipped_lines, (2,))

    def test_rejects_an_exception_expression(self):
        error = StringIO()

        with redirect_stderr(error):
            status = convert_asserts.main(
                ["--exception", "ValueError()", "unused.py"]
            )

        self.assertEqual(status, 2)
        self.assertIn("exception must be a name already in scope", error.getvalue())


if __name__ == "__main__":
    unittest.main()
