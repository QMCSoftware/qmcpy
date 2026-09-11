import shutil
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from scripts import annotate_public_api_types


class TestAnnotatePublicAPITypes(unittest.TestCase):

    def setUp(self):
        self.tmp_path = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp_path, ignore_errors=True)

    def _write(self, source):
        path = self.tmp_path / "sample.py"
        path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
        return path

    def test_annotates_public_method_inputs_and_output(self):
        path = self._write(
            '''
            import numpy as np

            class Model:

                def evaluate(self, x, scale=1.0):
                    """Evaluate the model.

                    Args:
                        x (np.ndarray): Evaluation points.
                        scale (float): Output scale.

                    Returns:
                        np.ndarray: Scaled values.
                    """
                    return scale * x
            '''
        )

        result = annotate_public_api_types.update_file(path)
        source = path.read_text(encoding="utf-8")

        self.assertTrue(result.changed)
        self.assertEqual(len(result.updates), 3)
        self.assertIn(
            "def evaluate(self, x: np.ndarray, scale: float = 1.0) -> np.ndarray:",
            source,
        )

    def test_annotates_constructor_and_adds_none_return(self):
        path = self._write(
            '''
            class Body:

                def __init__(self, mass):
                    """Initialize a body.

                    Args:
                        mass (float): Body mass.
                    """
                    self.mass = mass
            '''
        )

        annotate_public_api_types.update_file(path)

        self.assertIn(
            "def __init__(self, mass: float) -> None:",
            path.read_text(encoding="utf-8"),
        )

    def test_annotates_decorated_public_method(self):
        path = self._write(
            '''
            class Model:

                @staticmethod
                def normalize(x):
                    """Normalize a value.

                    Args:
                        x (float): Value to normalize.

                    Returns:
                        float: Normalized value.
                    """
                    return x
            '''
        )

        result = annotate_public_api_types.update_file(path)

        self.assertTrue(result.changed)
        self.assertIn(
            "def normalize(x: float) -> float:",
            path.read_text(encoding="utf-8"),
        )

    def test_preserves_existing_annotation_and_reports_conflict(self):
        path = self._write(
            '''
            def scale(x: int) -> float:
                """Scale a value.

                Args:
                    x (float): Value to scale.

                Returns:
                    float: Scaled value.
                """
                return float(x)
            '''
        )
        original = path.read_text(encoding="utf-8")

        result = annotate_public_api_types.update_file(path)

        self.assertFalse(result.changed)
        self.assertEqual(len(result.conflicts), 1)
        self.assertEqual(result.conflicts[0].slot, "x")
        self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_skips_type_whose_name_is_not_available(self):
        path = self._write(
            '''
            def evaluate(x):
                """Evaluate points.

                Args:
                    x (ArrayLike): Evaluation points.
                """
                return x
            '''
        )
        original = path.read_text(encoding="utf-8")

        result = annotate_public_api_types.update_file(path)

        self.assertFalse(result.changed)
        self.assertEqual(len(result.skips), 1)
        self.assertIn("not available", result.skips[0].reason)
        self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_skips_types_that_contradict_literal_defaults(self):
        path = self._write(
            '''
            import numpy as np

            def evaluate(x=None, tolerance=0.5):
                """Evaluate points.

                Args:
                    x (np.ndarray): Evaluation points.
                    tolerance (np.ndarray): Error tolerance.
                """
                return x
            '''
        )
        original = path.read_text(encoding="utf-8")

        result = annotate_public_api_types.update_file(path)

        self.assertFalse(result.changed)
        self.assertEqual(len(result.skips), 2)
        self.assertTrue(any("not optional" in skip.reason for skip in result.skips))
        self.assertTrue(any("conflicts" in skip.reason for skip in result.skips))
        self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_ignores_private_and_nested_functions(self):
        path = self._write(
            '''
            def _private(x):
                """Private helper.

                Args:
                    x (int): Value.
                """
                return x

            def public():
                """Return a nested callable."""

                def nested(x):
                    """Nested helper.

                    Args:
                        x (int): Value.
                    """
                    return x

                return nested
            '''
        )

        result = annotate_public_api_types.update_file(path)

        self.assertFalse(result.changed)
        self.assertEqual(result.updates, ())

    def test_check_mode_reports_without_writing(self):
        path = self._write(
            '''
            def scale(x):
                """Scale a value.

                Args:
                    x (float): Value to scale.
                """
                return 2 * x
            '''
        )
        original = path.read_text(encoding="utf-8")
        output = StringIO()

        with redirect_stdout(output):
            status = annotate_public_api_types.main(
                ["--check", "--root", str(self.tmp_path), str(path)]
            )

        self.assertEqual(status, 1)
        self.assertIn("1 file(s) would change", output.getvalue())
        self.assertEqual(path.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
