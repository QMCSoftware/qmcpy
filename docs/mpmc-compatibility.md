# QMCPy MPMC Compatibility Matrix

`qmcpy.discrete_distribution.mpmc` depends on the PyTorch Geometric stack, so its support window is narrower than the core QMCPy package. This page records the compatibility policy we should optimize for when pinning dependencies, adding tests, and reviewing MPMC pull requests.

## Recommended Baseline

- Treat MPMC as an optional feature, not part of the minimum QMCPy dependency set.
- Require PyTorch and `torch-geometric`; `pyg_lib` is an optional accelerator. When the compiled radius-graph backend is unavailable or fails to load, MPMC uses a native `torch.cdist` fallback. Neither `pyg_lib` nor `torch-cluster` is required to run MPMC.
- For reproducible local work and future CI pinning, prefer a modern PyTorch line with matching `data.pyg.org` wheels installed by `qmcpy-install-mpmc`.
- `unittests.yml` runs the full suite on `3.10`-`3.14` plus a slim `core-tests` tier on `3.9` (see [Minimum Python Version by Role](CONTRIBUTING.md#minimum-python-version-by-role)); neither installs MPMC.

## Support Policy

| Python | Linux / macOS / Windows | MPMC status | Dependency guidance | CI expectation |
|---|---|---|---|---|
| `3.14` | Target | Supported | `torch >= 2.10`, `torch-geometric >= 2.6.1`; optional `pyg_lib >= 0.6.0` from the matching `data.pyg.org` wheel index | Run MPMC doctests and unit tests |
| `3.13` | Target | Supported | `torch >= 2.10`, `torch-geometric >= 2.6.1`; optional `pyg_lib >= 0.6.0` | Run MPMC doctests and unit tests |
| `3.12` | Target | Supported | `torch >= 2.10`, `torch-geometric >= 2.6.1`; optional `pyg_lib >= 0.6.0` | Run MPMC doctests and unit tests |
| `3.10` to `3.11` | Best effort | Not a release blocker for MPMC | May work with matching PyTorch / PyG wheels, but not required by current CI policy | Optional manual testing only |

Python `3.9` is covered only by the slim `core-tests` tier, which never installs MPMC's PyTorch Geometric stack (see [Minimum Python Version by Role](CONTRIBUTING.md#minimum-python-version-by-role)).

The distinction is intentional:

- Core QMCPy still has a wider Python support window.
- MPMC should track the support window of current PyTorch and PyG releases, which is substantially newer.

## CI Policy

The current CI split is:

- `alltests.yml`: the only workflow that installs the MPMC stack (`qmcpy-install-mpmc`) and runs `make doctests_mpmc` plus the MPMC unit tests, on Python `3.13`. The steps are not OS-gated: Ubuntu alone on feature-branch pushes, all three OSes on full sweeps. Dependency validation requires PyTorch and `torch-geometric`; a missing `pyg_lib` accelerator does not block the tests.
- `unittests.yml`: `3.10`-`3.14`, each version on one operating system, plus a `core-tests` tier on all three OSes for `3.9`. Neither installs `torch-geometric` or calls `qmcpy-install-mpmc`, so MPMC tests normally skip. `test/test_dd_mpmc.py` checks only for PyTorch and `torch-geometric`; when both are available, the tests run with or without `pyg_lib`.

See [MPMC Coverage by OS](ci-testing.md#mpmc-coverage-by-os) for the per-operating-system breakdown.

This keeps required MPMC coverage in `alltests.yml`. Widening the `unittests.yml` matrix alone does not add MPMC coverage; a job must install the required PyTorch and `torch-geometric` dependencies.

## Local Developer Commands

Install the usual test and MPMC extras first, then optionally add the platform-specific accelerator with QMCPy's installed helper command:

```bash
python -m pip install -e ".[test,test_torch,test_gpytorch,test_botorch,mpmc]"
qmcpy-install-mpmc
```

The `mpmc` extra contains dependencies available from PyPI. The helper handles `pyg_lib` separately because its wheel page depends on the installed PyTorch version and accelerator build, which standard project metadata cannot select. If no matching wheel or source build is available, the helper warns and continues; MPMC can use its native PyTorch fallback.

Then run the MPMC-specific checks:

```bash
make doctests_mpmc
WITH_MPMC=1 make tests_no_docker
```

## Why `pyg_lib` Instead of `torch-cluster`?

The current PyG installation guide says:

- PyG is available for Python `3.10` through `3.14`.
- From PyG `2.3` onward, a basic install no longer needs external packages beyond PyTorch.
- `torch-cluster` is no longer required as a separate package because that functionality moved into `pyg-lib >= 0.6.0`.

For QMCPy MPMC, that makes `pyg_lib` the default path we should maintain first.

## References

[1] "PyTorch Geometric installation guide." [Online]. Available: [https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html). [Accessed: Sep. 17, 2026].

[2] "PyTorch 2.10.0." [Online]. Available: [https://pypi.org/project/torch/2.10.0/](https://pypi.org/project/torch/2.10.0/). [Accessed: Sep. 17, 2026].

[3] "torch-geometric." [Online]. Available: [https://pypi.org/project/torch-geometric/](https://pypi.org/project/torch-geometric/). [Accessed: Sep. 17, 2026].

[4] "PyG wheel index for `torch-2.10.0+cpu`." [Online]. Available: [https://data.pyg.org/whl/torch-2.10.0+cpu.html](https://data.pyg.org/whl/torch-2.10.0+cpu.html). [Accessed: Sep. 17, 2026].
