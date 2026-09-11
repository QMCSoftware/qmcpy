"""Tests for the platform-specific MPMC dependency installer."""

import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from qmcpy.util import install_mpmc_pyg


def _torch(version="2.12.1+cpu", cuda=None, hip=None):
    return SimpleNamespace(
        __version__=version,
        version=SimpleNamespace(cuda=cuda, hip=hip),
    )


class TestInstallMPMCPyG(unittest.TestCase):

    def test_torch_versions_include_baseline_fallback(self):
        """Wheel lookup tries an exact patch release, then its minor baseline."""
        self.assertEqual(
            install_mpmc_pyg.torch_versions("2.12.1+cpu"), ["2.12.1", "2.12.0"]
        )
        self.assertEqual(install_mpmc_pyg.torch_versions("2.12.0"), ["2.12.0"])

        with self.assertRaisesRegex(RuntimeError, "Unable to parse torch version"):
            install_mpmc_pyg.torch_versions("development")

    def test_accelerator_tag(self):
        """PyTorch build metadata maps to the expected PyG wheel tag."""
        cases = [
            (_torch(), "cpu"),
            (_torch(cuda="12.6"), "cu126"),
            (_torch(cuda="13.0.1"), "cu130"),
        ]
        for torch_module, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(
                    install_mpmc_pyg.accelerator_tag(torch_module), expected
                )

    def test_accelerator_tag_rejects_rocm(self):
        """The installer directs unsupported ROCm users to upstream guidance."""
        with self.assertRaisesRegex(RuntimeError, "does not currently support ROCm"):
            install_mpmc_pyg.accelerator_tag(_torch(hip="6.3"))

    def test_main_retries_with_torch_minor_baseline(self):
        """A missing exact wheel page falls back to the minor baseline page."""
        calls = []

        def fake_run(*args):
            calls.append(args)
            if args[-1].endswith("torch-2.12.1+cpu.html"):
                raise subprocess.CalledProcessError(1, args)

        with patch.object(install_mpmc_pyg, "run", fake_run):
            install_mpmc_pyg.main(_torch())

        self.assertEqual(calls[0][-1], "torch-geometric>=2.6.1")
        self.assertEqual(
            calls[1][-1], "https://data.pyg.org/whl/torch-2.12.1+cpu.html"
        )
        self.assertEqual(
            calls[2][-1], "https://data.pyg.org/whl/torch-2.12.0+cpu.html"
        )
        self.assertIn("--only-binary", calls[1])

    def test_main_explains_that_torch_must_be_installed(self):
        """Running the helper before installing the extra gives a useful error."""
        def missing_torch(_name):
            raise ModuleNotFoundError("No module named 'torch'", name="torch")

        with patch.object(
            install_mpmc_pyg.importlib, "import_module", missing_torch
        ):
            with self.assertRaisesRegex(RuntimeError, r"install 'qmcpy\[mpmc\]'"):
                install_mpmc_pyg.main()

    def test_main_reports_missing_wheel(self):
        """Exhausting candidate wheel pages reports the build that failed."""
        def fail_pyg_lib(*args):
            if "pyg_lib>=0.6.0" in args:
                raise subprocess.CalledProcessError(1, args)

        with patch.object(install_mpmc_pyg, "run", fail_pyg_lib):
            with self.assertRaisesRegex(RuntimeError, r"torch 2\.12\.1\+cpu \(cpu\)"):
                install_mpmc_pyg.main(_torch())


if __name__ == "__main__":
    unittest.main()
