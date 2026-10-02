import importlib.util
import tempfile
import unittest
from pathlib import Path

# Load by file path: a package import would run test/booktests/__init__.py
# first, which needs psutil/testbook/nbformat/matplotlib -- not installed in
# the minimal "Core Unit Tests" env, and not needed by generate_test.py itself.
_GENERATE_TEST_PATH = Path(__file__).resolve().parent / "booktests" / "generate_test.py"
_spec = importlib.util.spec_from_file_location("_generate_test_under_test", _GENERATE_TEST_PATH)
generate_test = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generate_test)


class TestGenerateBooktests(unittest.TestCase):

    def test_skips_scratch_notebooks(self):
        with tempfile.TemporaryDirectory() as tmp:
            demos = Path(tmp) / "demos" / "portfolio"
            demos.mkdir(parents=True)
            (demos / "_RUN_tmp.ipynb").write_text("{}")
            (demos / "real_demo.ipynb").write_text("{}")
            out_dir = Path(tmp) / "booktests"
            out_dir.mkdir()

            generated = generate_test.generate_missing_tests(
                demos_dir=str(demos.parent), output_dir=out_dir
            )

            self.assertEqual([Path(p).name for p in generated], ["tb_real_demo.py"])
            self.assertFalse((out_dir / "tb__RUN_tmp.py").exists())


if __name__ == "__main__":
    unittest.main()
