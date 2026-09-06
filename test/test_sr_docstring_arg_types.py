import shutil
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from scripts import add_docstring_arg_types


class TestAddDocstringArgTypes(unittest.TestCase):

    def setUp(self):
        self.tmp_path = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp_path, ignore_errors=True)

    def _write(self, source):
        path = self.tmp_path / "sample.py"
        path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
        return path

    def test_adds_annotation_types_to_existing_google_args(self):
        path = self._write(
            '''
            def calculate_velocity(
                distance: float,
                time: float,
                acceleration: float = 0.0,
            ) -> float:
                """Calculate velocity.

                Args:
                    distance: Distance traveled.
                    time: Elapsed time.
                    acceleration: Constant acceleration.

                Returns:
                    float: Velocity.
                """
                return distance / time + acceleration * time
            '''
        )

        result = add_docstring_arg_types.update_file(path)
        source = path.read_text(encoding="utf-8")

        self.assertTrue(result.changed)
        self.assertEqual(
            [(update.argument, update.annotation) for update in result.updates],
            [
                ("distance", "float"),
                ("time", "float"),
                ("acceleration", "float"),
            ],
        )
        self.assertIn("distance (float): Distance traveled.", source)
        self.assertIn("time (float): Elapsed time.", source)
        self.assertIn("acceleration (float): Constant acceleration.", source)

    def test_check_mode_reports_without_writing(self):
        path = self._write(
            '''
            def scale(x: int) -> int:
                """Scale x.

                Args:
                    x: Value to scale.

                Returns:
                    Scaled value.
                """
                return 2 * x
            '''
        )
        original = path.read_text(encoding="utf-8")

        output = StringIO()
        with redirect_stdout(output):
            status = add_docstring_arg_types.main(
                ["--check", "--include-outputs", str(path)]
            )

        self.assertEqual(status, 1)
        self.assertIn("1 file(s) would change", output.getvalue())
        self.assertIn("1 output type update(s)", output.getvalue())
        self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_preserves_existing_type_unless_overwrite_is_requested(self):
        path = self._write(
            '''
            def scale(x: float):
                """Scale x.

                Args:
                    x (int): Value to scale.
                """
                return 2 * x
            '''
        )

        result = add_docstring_arg_types.update_file(path)
        self.assertFalse(result.changed)
        self.assertIn("x (int):", path.read_text(encoding="utf-8"))

        result = add_docstring_arg_types.update_file(path, overwrite_existing=True)
        self.assertTrue(result.changed)
        self.assertIn("x (float):", path.read_text(encoding="utf-8"))

    def test_updates_public_constructor_without_documenting_self(self):
        path = self._write(
            '''
            class Body:

                def __init__(self, mass: float):
                    """Initialize a body.

                    Args:
                        mass: Body mass.
                    """
                    self.mass = mass
            '''
        )

        result = add_docstring_arg_types.update_file(path)

        self.assertEqual(len(result.updates), 1)
        self.assertEqual(result.updates[0].argument, "mass")
        self.assertIn("mass (float): Body mass.", path.read_text(encoding="utf-8"))

    def test_normalizes_multiline_annotations_and_colon_spacing(self):
        path = self._write(
            '''
            def first(
                values: list[
                    float
                ],
            ):
                """Return the first value.

                Args:
                    values :  Values to inspect.
                """
                return values[0]
            '''
        )

        result = add_docstring_arg_types.update_file(path)
        source = path.read_text(encoding="utf-8")

        self.assertTrue(result.changed)
        self.assertIn("values (list[float]): Values to inspect.", source)

    def test_does_not_infer_a_type_for_an_unannotated_argument(self):
        path = self._write(
            '''
            def scale(x):
                """Scale a value.

                Args:
                    x: Value to scale.
                """
                return 2 * x
            '''
        )
        original = path.read_text(encoding="utf-8")

        result = add_docstring_arg_types.update_file(path)

        self.assertFalse(result.changed)
        self.assertEqual(result.updates, [])
        self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_adds_return_type_when_output_sync_is_requested(self):
        path = self._write(
            '''
            def norm(x: float) -> float:
                """Compute a norm.

                Args:
                    x: Input value.

                Returns:
                    Computed norm.
                """
                return abs(x)
            '''
        )

        result = add_docstring_arg_types.update_file(path, include_outputs=True)
        source = path.read_text(encoding="utf-8")

        self.assertTrue(result.changed)
        self.assertIn("x (float): Input value.", source)
        self.assertIn("float: Computed norm.", source)
        self.assertEqual(
            [update.section for update in result.updates],
            ["Args", "Returns"],
        )

    def test_replaces_existing_output_type_only_when_requested(self):
        path = self._write(
            '''
            def norm(x) -> float:
                """Compute a norm.

                Returns:
                    int: Computed norm.
                """
                return abs(x)
            '''
        )

        result = add_docstring_arg_types.update_file(path, include_outputs=True)
        self.assertFalse(result.changed)
        self.assertIn("int: Computed norm.", path.read_text(encoding="utf-8"))

        result = add_docstring_arg_types.update_file(
            path,
            include_outputs=True,
            overwrite_existing=True,
        )
        self.assertTrue(result.changed)
        self.assertIn("float: Computed norm.", path.read_text(encoding="utf-8"))

    def test_extracts_item_type_for_yields_section(self):
        path = self._write(
            '''
            from typing import Iterator

            def indices(n: int) -> Iterator[int]:
                """Yield indices.

                Args:
                    n: Number of indices.

                Yields:
                    Next index.
                """
                yield from range(n)
            '''
        )

        result = add_docstring_arg_types.update_file(path, include_outputs=True)
        source = path.read_text(encoding="utf-8")

        self.assertTrue(result.changed)
        self.assertIn("n (int): Number of indices.", source)
        self.assertIn("int: Next index.", source)


if __name__ == "__main__":
    unittest.main()
