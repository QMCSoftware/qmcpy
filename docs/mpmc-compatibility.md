# QMCPy MPMC Compatibility Matrix

`qmcpy.discrete_distribution.mpmc` depends on the PyTorch Geometric stack, so its support window is narrower than the core QMCPy package. This page records the compatibility policy we should optimize for when pinning dependencies, adding tests, and reviewing MPMC pull requests.

## Recommended Baseline

- Treat MPMC as an optional feature, not part of the minimum QMCPy dependency set.
- Prefer `pyg_lib` plus `torch-geometric`; do not require `torch-cluster` as a separate dependency.
- For reproducible local work and future CI pinning, prefer a modern PyTorch line with matching `data.pyg.org` wheels installed by `qmcpy-install-mpmc`.
- Keep older Python jobs in `unittests.yml` for core QMCPy coverage, but do not require them to run MPMC.

## Support Policy

| Python | Linux / macOS / Windows | MPMC status | Dependency guidance | CI expectation |
|---|---|---|---|---|
| `3.14` | Target | Supported | `torch >= 2.10`, `torch-geometric >= 2.6.1`, `pyg_lib >= 0.6.0` from the matching `data.pyg.org` wheel index | Run MPMC doctests and unit tests |
| `3.13` | Target | Supported | `torch >= 2.10`, `torch-geometric >= 2.6.1`, `pyg_lib >= 0.6.0` | Run MPMC doctests and unit tests |
| `3.12` | Target | Supported | `torch >= 2.10`, `torch-geometric >= 2.6.1`, `pyg_lib >= 0.6.0` | Run MPMC doctests and unit tests |
| `3.10` to `3.11` | Best effort | Not a release blocker for MPMC | May work with matching PyTorch / PyG wheels, but not required by current CI policy | Optional manual testing only |
| `3.5` to `3.9` | Legacy core-package coverage only | Not supported for MPMC | Do not spend CI budget trying to keep MPMC running here | No MPMC doctests or unit tests |

The distinction is intentional:

- Core QMCPy still has a wider Python support window.
- MPMC should track the support window of current PyTorch and PyG releases, which is substantially newer.

## CI Policy

The current CI split should be:

- `alltests.yml`: full-sweep validation on Linux, macOS, and Windows for Python `3.13`, including `make doctests_mpmc` and the standard unit-test suite.
- `unittests.yml`: a broader version sampler for the repository, with explicit MPMC jobs on Python `3.12`, `3.13`, and `3.14`.
- Older `unittests.yml` jobs: keep them for core QMCPy regressions, but do not require MPMC there.

This gives one place to enforce modern MPMC compatibility without forcing the entire repository to abandon older Python jobs immediately.

## Local Developer Commands

Install the usual test and MPMC extras first, then add the platform-specific
PyG runtime with QMCPy's installed helper command:

```bash
python -m pip install -e ".[test,test_torch,test_gpytorch,test_botorch,mpmc]"
qmcpy-install-mpmc
```

The `mpmc` extra contains dependencies available from PyPI. The helper handles
`pyg_lib` separately because its wheel page depends on the installed PyTorch
version and accelerator build, which standard project metadata cannot select.

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

- [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
- [PyTorch 2.10.0 on PyPI](https://pypi.org/project/torch/2.10.0/)
- [torch-geometric on PyPI](https://pypi.org/project/torch-geometric/)
- [PyG wheel index for `torch-2.10.0+cpu`](https://data.pyg.org/whl/torch-2.10.0+cpu.html)
