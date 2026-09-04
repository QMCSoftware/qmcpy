from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import check_colab_notebooks as check
from scripts import harden_colab_notebook as harden
from scripts import smoke_test_colab_notebooks as smoke


def markdown_cell(source: str, cell_id: str = "markdown") -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code_cell(source: str, cell_id: str = "code") -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


class TestColabNotebooks(unittest.TestCase):

    def _tmp_path(self) -> Path:
        """Fresh temp directory, removed after the test (pytest ``tmp_path``)."""
        path = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, path, ignore_errors=True)
        return path

    def _setattr(self, target, name, value):
        """Set ``target.name = value`` for the test only (pytest ``monkeypatch``)."""
        patcher = mock.patch.object(target, name, value)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _colab_repo(self):
        """Build a throwaway repo layout and point the scripts at it.

        Returns ``(notebook_path, manifest_path)`` (pytest ``colab_repo``).
        """
        tmp_path = self._tmp_path()
        demos_dir = tmp_path / "demos"
        demos_dir.mkdir()
        notebook_path = demos_dir / "example.ipynb"
        notebook = {
            "cells": [
                markdown_cell("# Example\n", "title"),
                code_cell("import math\n", "imports"),
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        notebook_path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")

        manifest_path = tmp_path / "manifest.json"
        manifest = {
            "repo": "QMCSoftware/QMCSoftware",
            "git_ref": "develop",
            "enabled": [],
            "disabled": {},
        }
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")

        self._setattr(check, "REPO_ROOT", tmp_path)
        self._setattr(check, "DEMOS_DIR", demos_dir)
        self._setattr(harden, "REPO_ROOT", tmp_path)
        self._setattr(smoke, "REPO_ROOT", tmp_path)
        return notebook_path, manifest_path

    def test_badge_stripping_preserves_intro_and_drops_badge_only_cells(self):
        intro = markdown_cell(
            "# ML Sensitivity Indices\n\n"
            "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            "(https://colab.research.google.com/github/QMCSoftware/QMCSoftware/"
            "blob/develop/demos/iris.ipynb)\n\n"
            "This notebook demonstrates sensitivity indices.\n"
        )
        badge_only = markdown_cell(
            "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            "(https://colab.research.google.com/github/QMCSoftware/QMCSoftware/"
            "blob/develop/demos/iris.ipynb)\n"
        )

        cleaned_intro = harden.badge_stripped_cell(intro)
        self.assertIsNotNone(cleaned_intro)
        self.assertIn("# ML Sensitivity Indices", check.cell_source_text(cleaned_intro))
        self.assertIn("sensitivity indices", check.cell_source_text(cleaned_intro))
        self.assertNotIn("Open In Colab", check.cell_source_text(cleaned_intro))
        self.assertEqual(
            harden.remove_any_badge_cells([badge_only, code_cell("pass\n")]),
            [code_cell("pass\n")],
        )

    def test_is_any_badge_cell_rejects_spoofed_hostname(self):
        spoofed = markdown_cell(
            "[click](https://evil.example/colab.research.google.com/assets/colab-badge.svg)\n"
        )
        genuine = markdown_cell(
            "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            "(https://colab.research.google.com/github/QMCSoftware/QMCSoftware/"
            "blob/develop/demos/iris.ipynb)\n"
        )

        self.assertFalse(check.is_any_badge_cell(spoofed))
        self.assertTrue(check.is_any_badge_cell(genuine))

    def test_bootstrap_detection_uses_marker_and_real_install_command(self):
        misleading = code_cell(
            '"""import google.colab\n# @title Execute this cell to install dependencies\n'
            '!pip install qmcpy\n"""\n'
        )
        comment_only = code_cell(
            "# @title Execute this cell to install dependencies\n"
            "# import google.colab\n"
            "# !pip install qmcpy\n"
        )
        self.assertFalse(check.is_any_install_cell(misleading))
        self.assertFalse(check.is_bootstrap_cell(misleading))
        self.assertTrue(check.is_any_install_cell(comment_only))
        self.assertFalse(check.is_bootstrap_cell(comment_only))

        tmp_path = self._tmp_path()
        self._setattr(harden, "REPO_ROOT", tmp_path)
        notebook_path = tmp_path / "demos" / "example.ipynb"
        notebook_path.parent.mkdir()
        source = "".join(
            harden.bootstrap_cell_source(
                notebook_path,
                {"repo": "QMCSoftware/QMCSoftware"},
                [],
            )
        )
        generated = code_cell(source)
        self.assertTrue(check.is_bootstrap_cell(generated))
        self.assertIn("except ImportError:", source)
        self.assertIn("if IN_COLAB:", source)
        self.assertNotIn("except:\n", source)
        compile(smoke.rewrite_shell_magics(source), "<bootstrap>", "exec")

    def test_extra_pip_packages_preserves_later_explicit_installs(self):
        cells = [
            code_cell("import qmcpy as qp\n"),
            code_cell("import ipywidgets as widgets\n"),
            code_cell(
                "try:\n"
                "    import QuantLib as ql\n"
                "except ModuleNotFoundError:\n"
                "    !pip install -q QuantLib\n"
            ),
            code_cell("!pip install -q seaborn\n"),
        ]

        self.assertEqual(
            harden.extra_pip_packages(cells), ["QuantLib", "ipywidgets", "seaborn"]
        )

    def test_needs_latex_setup_detects_tueplots(self):
        cells = [
            code_cell("import qmcpy as qp\n"),
            code_cell(
                "from tueplots import bundles\n"
                "pyplot.rcParams.update(bundles.probnum2025())\n"
            ),
        ]

        self.assertTrue(harden.needs_latex_setup(cells))

    def test_imported_modules_survives_magic_only_block_body(self):
        # A shell-magic line as the *only* statement in a block used to leave an
        # empty `if:`/`try:` body, making ast.parse raise and silently hiding
        # every import in the cell (not just the magic line itself).
        source = (
            "import os\n"
            "from util import helper\n"
            "if True:\n"
            "    !echo hi\n"
        )
        self.assertEqual(check.imported_modules(source), {"os", "util"})

    def test_local_module_matches_finds_ancestor_directory(self):
        tmp_path = self._tmp_path()
        self._setattr(check, "DEMOS_DIR", tmp_path)
        (tmp_path / "util.py").write_text("", encoding="utf-8")
        notebook_dir = tmp_path / "output"
        notebook_dir.mkdir()

        matches = check.local_module_matches(notebook_dir, "util")

        self.assertEqual(matches, [tmp_path / "util.py"])

    def test_extra_pip_packages_honors_colab_deps_marker(self):
        cells = [
            code_cell("import qmcpy as qp\n"),
            code_cell(
                "# colab-deps: plotly, some-package\n"
                "import plotly\n"
            ),
        ]

        self.assertEqual(
            harden.extra_pip_packages(cells), ["plotly", "some-package"]
        )

    def test_dump_notebook_preserves_existing_json_indent(self):
        tmp_path = self._tmp_path()
        notebook_path = tmp_path / "example.ipynb"
        notebook = {
            "cells": [code_cell("pass\n")],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        original_source = json.dumps(notebook, indent=2) + "\n"

        harden.dump_notebook(notebook_path, notebook, original_source)

        self.assertEqual(
            notebook_path.read_text(encoding="utf-8"), original_source
        )

    def test_harden_check_smoke_round_trip_is_idempotent(self):
        notebook_path, manifest_path = self._colab_repo()
        harden.harden_notebook(notebook_path, manifest_path)

        self.assertEqual(check.run_check(manifest_path, strict=True), 0)
        smoke_notebook, source_indices = smoke.build_smoke_notebook(notebook_path, 1)
        self.assertEqual(len(smoke_notebook["cells"]), len(source_indices))

        sentinel = object()
        old_modules = {
            name: sys.modules.get(name, sentinel) for name in ("google", "google.colab")
        }
        old_environment = {
            name: os.environ.get(name, sentinel)
            for name in ("QMC_COLAB_SMOKE", "QMC_COLAB_SMOKE_REPO_ROOT", "QMC_COLAB_SMOKE_NOTEBOOK_DIR")
        }
        namespace: dict = {}
        try:
            for cell in smoke_notebook["cells"]:
                if cell["cell_type"] == "code":
                    exec(check.cell_source_text(cell), namespace)
        finally:
            for name, value in old_modules.items():
                if value is sentinel:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value
            for name, value in old_environment.items():
                if value is sentinel:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

        self._setattr(
            harden,
            "dump_notebook",
            lambda *_args, **_kwargs: self.fail("unchanged notebook was rewritten"),
        )
        self._setattr(
            harden,
            "dump_json",
            lambda *_args, **_kwargs: self.fail("unchanged manifest was rewritten"),
        )
        harden.harden_notebook(notebook_path, manifest_path)

    def test_checker_rejects_wrong_badge(self):
        notebook_path, manifest_path = self._colab_repo()
        harden.harden_notebook(notebook_path, manifest_path)
        notebook = check.load_json(notebook_path)
        badge = next(cell for cell in notebook["cells"] if check.is_any_badge_cell(cell))
        badge["source"] = [check.cell_source_text(badge).replace("develop", "wrong-ref")]
        notebook_path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")

        self.assertEqual(check.run_check(manifest_path, strict=True), 1)

    def test_harden_failure_does_not_disable_notebook(self):
        notebook_path, manifest_path = self._colab_repo()
        original_manifest = manifest_path.read_text(encoding="utf-8")
        self._setattr(
            harden,
            "harden_notebook",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("failure")),
        )

        successes, failures = harden.harden_batch([notebook_path], manifest_path)

        self.assertEqual(successes, [])
        self.assertEqual(failures, [("demos/example.ipynb", "failure")])
        self.assertEqual(manifest_path.read_text(encoding="utf-8"), original_manifest)

    def test_smoke_batch_continues_after_a_notebook_failure(self):
        def fake_build(notebook_path: Path, cells_after_bootstrap: int):
            return {"cells": []}, []

        def fake_execute(notebook_path: Path, smoke_nb, source_indices, timeout):
            if "broken" in notebook_path.as_posix():
                raise RuntimeError("boom")

        self._setattr(smoke, "build_smoke_notebook", fake_build)
        self._setattr(smoke, "execute_smoke_notebook", fake_execute)

        passed, failed = smoke.smoke_test_batch(
            ["demos/broken.ipynb", "demos/ok.ipynb"], cells_after_bootstrap=1, timeout=60
        )

        self.assertEqual(passed, ["demos/ok.ipynb"])
        self.assertEqual(failed, [("demos/broken.ipynb", "boom")])


if __name__ == "__main__":
    unittest.main()
