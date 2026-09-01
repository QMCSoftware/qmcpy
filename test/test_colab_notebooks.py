from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

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


@pytest.fixture
def colab_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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

    monkeypatch.setattr(check, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(check, "DEMOS_DIR", demos_dir)
    monkeypatch.setattr(harden, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(smoke, "REPO_ROOT", tmp_path)
    return notebook_path, manifest_path


def test_badge_stripping_preserves_intro_and_drops_badge_only_cells():
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
    assert cleaned_intro is not None
    assert "# ML Sensitivity Indices" in check.cell_source_text(cleaned_intro)
    assert "sensitivity indices" in check.cell_source_text(cleaned_intro)
    assert "Open In Colab" not in check.cell_source_text(cleaned_intro)
    assert harden.remove_any_badge_cells([badge_only, code_cell("pass\n")]) == [
        code_cell("pass\n")
    ]


def test_bootstrap_detection_uses_marker_and_real_install_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    misleading = code_cell(
        '"""import google.colab\n# @title Execute this cell to install dependencies\n'
        '!pip install qmcpy\n"""\n'
    )
    comment_only = code_cell(
        "# @title Execute this cell to install dependencies\n"
        "# import google.colab\n"
        "# !pip install qmcpy\n"
    )
    assert not check.is_any_install_cell(misleading)
    assert not check.is_bootstrap_cell(misleading)
    assert check.is_any_install_cell(comment_only)
    assert not check.is_bootstrap_cell(comment_only)

    monkeypatch.setattr(harden, "REPO_ROOT", tmp_path)
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
    assert check.is_bootstrap_cell(generated)
    assert "except ImportError:" in source
    assert "if IN_COLAB:" in source
    assert "except:\n" not in source
    compile(smoke.rewrite_shell_magics(source), "<bootstrap>", "exec")


def test_extra_pip_packages_preserves_later_explicit_installs():
    cells = [
        code_cell("import qmcpy as qp\n"),
        code_cell("import ipywidgets as widgets\n"),
        code_cell(
            "try:\n"
            "    import QuantLib as ql\n"
            "except ModuleNotFoundError:\n"
            "    !pip install -q QuantLib\n"
        ),
    ]

    assert harden.extra_pip_packages(cells) == ["ipywidgets", "QuantLib"]


def test_dump_notebook_preserves_existing_json_indent(tmp_path: Path):
    notebook_path = tmp_path / "example.ipynb"
    notebook = {
        "cells": [code_cell("pass\n")],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    original_source = json.dumps(notebook, indent=2) + "\n"

    harden.dump_notebook(notebook_path, notebook, original_source)

    assert notebook_path.read_text(encoding="utf-8") == original_source


def test_harden_check_smoke_round_trip_is_idempotent(
    colab_repo, monkeypatch: pytest.MonkeyPatch
):
    notebook_path, manifest_path = colab_repo
    harden.harden_notebook(notebook_path, manifest_path)

    assert check.run_check(manifest_path, strict=True) == 0
    smoke_notebook, source_indices = smoke.build_smoke_notebook(notebook_path, 1)
    assert len(smoke_notebook["cells"]) == len(source_indices)

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

    monkeypatch.setattr(
        harden,
        "dump_notebook",
        lambda *_args, **_kwargs: pytest.fail("unchanged notebook was rewritten"),
    )
    monkeypatch.setattr(
        harden,
        "dump_json",
        lambda *_args, **_kwargs: pytest.fail("unchanged manifest was rewritten"),
    )
    harden.harden_notebook(notebook_path, manifest_path)


def test_checker_rejects_wrong_badge(colab_repo):
    notebook_path, manifest_path = colab_repo
    harden.harden_notebook(notebook_path, manifest_path)
    notebook = check.load_json(notebook_path)
    badge = next(cell for cell in notebook["cells"] if check.is_any_badge_cell(cell))
    badge["source"] = [check.cell_source_text(badge).replace("develop", "wrong-ref")]
    notebook_path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")

    assert check.run_check(manifest_path, strict=True) == 1


def test_harden_failure_does_not_disable_notebook(
    colab_repo, monkeypatch: pytest.MonkeyPatch
):
    notebook_path, manifest_path = colab_repo
    original_manifest = manifest_path.read_text(encoding="utf-8")
    monkeypatch.setattr(
        harden,
        "harden_notebook",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("failure")),
    )

    successes, failures = harden.harden_batch([notebook_path], manifest_path)

    assert successes == []
    assert failures == [("demos/example.ipynb", "failure")]
    assert manifest_path.read_text(encoding="utf-8") == original_manifest
