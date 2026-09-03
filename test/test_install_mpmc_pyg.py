"""Tests for the platform-specific MPMC dependency installer."""

import subprocess
from types import SimpleNamespace

import pytest

from qmcpy.util import install_mpmc_pyg


def _torch(version="2.12.1+cpu", cuda=None, hip=None):
    return SimpleNamespace(
        __version__=version,
        version=SimpleNamespace(cuda=cuda, hip=hip),
    )


def test_torch_versions_include_baseline_fallback():
    """Wheel lookup tries an exact patch release, then its minor baseline."""
    assert install_mpmc_pyg.torch_versions("2.12.1+cpu") == ["2.12.1", "2.12.0"]
    assert install_mpmc_pyg.torch_versions("2.12.0") == ["2.12.0"]

    with pytest.raises(RuntimeError, match="Unable to parse torch version"):
        install_mpmc_pyg.torch_versions("development")


@pytest.mark.parametrize(
    ("torch_module", "expected"),
    [
        (_torch(), "cpu"),
        (_torch(cuda="12.6"), "cu126"),
        (_torch(cuda="13.0.1"), "cu130"),
    ],
)
def test_accelerator_tag(torch_module, expected):
    """PyTorch build metadata maps to the expected PyG wheel tag."""
    assert install_mpmc_pyg.accelerator_tag(torch_module) == expected


def test_accelerator_tag_rejects_rocm():
    """The installer directs unsupported ROCm users to upstream guidance."""
    with pytest.raises(RuntimeError, match="does not currently support ROCm"):
        install_mpmc_pyg.accelerator_tag(_torch(hip="6.3"))


def test_main_retries_with_torch_minor_baseline(monkeypatch):
    """A missing exact wheel page falls back to the minor baseline page."""
    calls = []

    def fake_run(*args):
        calls.append(args)
        if args[-1].endswith("torch-2.12.1+cpu.html"):
            raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(install_mpmc_pyg, "run", fake_run)

    install_mpmc_pyg.main(_torch())

    assert calls[0][-1] == "torch-geometric>=2.6.1"
    assert calls[1][-1] == "https://data.pyg.org/whl/torch-2.12.1+cpu.html"
    assert calls[2][-1] == "https://data.pyg.org/whl/torch-2.12.0+cpu.html"
    assert "--only-binary" in calls[1]


def test_main_falls_back_to_official_source_release(monkeypatch):
    """A wheel-index outage falls back to PyG's pinned source release."""
    calls = []

    def fake_run(*args):
        calls.append(args)
        if install_mpmc_pyg.PYG_LIB_REQUIREMENT in args:
            raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(install_mpmc_pyg, "run", fake_run)

    install_mpmc_pyg.main(_torch())

    assert len(calls) == 4
    assert "--no-build-isolation" in calls[-1]
    assert calls[-1][-1] == install_mpmc_pyg.PYG_LIB_SOURCE


def test_main_explains_that_torch_must_be_installed(monkeypatch):
    """Running the helper before installing the extra gives a useful error."""
    def missing_torch(_name):
        raise ModuleNotFoundError("No module named 'torch'", name="torch")

    monkeypatch.setattr(install_mpmc_pyg.importlib, "import_module", missing_torch)

    with pytest.raises(RuntimeError, match=r"install 'qmcpy\[mpmc\]'"):
        install_mpmc_pyg.main()


def test_main_warns_when_pyg_lib_unavailable(monkeypatch, capsys):
    """pyg_lib is optional: when every install path fails, main() warns and returns."""
    def fail_pyg_lib(*args):
        if (
            install_mpmc_pyg.PYG_LIB_REQUIREMENT in args
            or install_mpmc_pyg.PYG_LIB_SOURCE in args
        ):
            raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(install_mpmc_pyg, "run", fail_pyg_lib)

    install_mpmc_pyg.main(_torch())  # must not raise

    assert "could not install the optional pyg_lib" in capsys.readouterr().out
