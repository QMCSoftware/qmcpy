"""Install the platform-specific ``pyg_lib`` runtime used by MPMC."""

import importlib
import re
import subprocess
import sys


PYG_LIB_REQUIREMENT = "pyg_lib>=0.6.0"
PYG_LIB_SOURCE = "git+https://github.com/pyg-team/pyg-lib.git@0.8.0"
TORCH_GEOMETRIC_REQUIREMENT = "torch-geometric>=2.6.1"


def run(*args):
    """Run an installation command and show it to the user."""
    print("+", " ".join(args), flush=True)
    subprocess.check_call(list(args))


def torch_versions(version):
    """Return exact and major/minor baseline versions for PyG wheel lookup."""
    match = re.match(r"(\d+\.\d+\.\d+)", version)
    if match is None:
        raise RuntimeError(f"Unable to parse torch version: {version}")

    full = match.group(1)
    major, minor, _ = full.split(".")
    baseline = f"{major}.{minor}.0"
    return [full] if baseline == full else [full, baseline]


def accelerator_tag(torch_module):
    """Return the PyG wheel accelerator tag for an installed PyTorch build."""
    torch_build = torch_module.version
    if getattr(torch_build, "hip", None):
        raise RuntimeError(
            "QMCPy's MPMC installer does not currently support ROCm wheels. "
            "See the PyG installation guide: "
            "https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html."
        )

    cuda_version = getattr(torch_build, "cuda", None)
    if cuda_version is None:
        return "cpu"

    match = re.match(r"(\d+)\.(\d+)", cuda_version)
    if match is None:
        raise RuntimeError(
            f"Unable to parse PyTorch CUDA version: {cuda_version}"
        )
    return f"cu{match.group(1)}{match.group(2)}"


def wheel_urls(torch_version, accelerator):
    """Return candidate PyG wheel pages for the installed PyTorch build."""
    return [
        f"https://data.pyg.org/whl/torch-{version}+{accelerator}.html"
        for version in torch_versions(torch_version)
    ]


def main(torch_module=None):
    """Install the PyG dependencies MPMC needs beyond what PyPI resolves.

    ``torch-geometric`` is required and comes from PyPI.  ``pyg_lib`` is only an
    optional accelerator -- MPMC and ``torch-geometric`` run without it -- so
    when no wheel exists for this torch build and the source build fails, we
    warn and carry on instead of aborting the install.
    """
    if torch_module is None:
        try:
            torch_module = importlib.import_module("torch")
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "PyTorch is not installed. Install QMCPy's MPMC extra first: "
                "python -m pip install 'qmcpy[mpmc]'"
            ) from error

    print(f"Detected torch {torch_module.__version__}", flush=True)
    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "--prefer-binary",
        TORCH_GEOMETRIC_REQUIREMENT,
    )

    if not _install_pyg_lib(torch_module):
        print(
            "WARNING: could not install the optional pyg_lib accelerator for "
            f"torch {torch_module.__version__} "
            f"({accelerator_tag(torch_module)}); MPMC falls back to "
            "torch-geometric's native (slower) scatter path.",
            flush=True,
        )


def _install_pyg_lib(torch_module):
    """Best-effort pyg_lib install; return True on success, False otherwise."""
    accelerator = accelerator_tag(torch_module)
    for wheel_url in wheel_urls(torch_module.__version__, accelerator):
        print(f"Trying pyg_lib wheels from {wheel_url}", flush=True)
        try:
            run(
                sys.executable,
                "-m",
                "pip",
                "install",
                "--prefer-binary",
                "--only-binary",
                "pyg_lib",
                PYG_LIB_REQUIREMENT,
                "--find-links",
                wheel_url,
            )
            return True
        except subprocess.CalledProcessError:
            pass

    print(
        "Pre-built pyg_lib wheels were unavailable; trying the official "
        "source release",
        flush=True,
    )
    try:
        run(
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            PYG_LIB_SOURCE,
        )
        return True
    except subprocess.CalledProcessError:
        return False


if __name__ == "__main__":
    main()
