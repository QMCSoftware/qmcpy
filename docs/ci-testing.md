# CI/CD Testing

This page summarizes QMCPy's current GitHub Actions CI layout.

## Workflows

| Workflow | Trigger | Runner / Python | Main work |
|---|---|---|---|
| `alltests.yml` | Feature-branch `push` | `ubuntu`, Python `3.13` | <ul><li>Non-Docker doctests</li><li>MPMC doctests and unit tests (Ubuntu only)</li><li>`unittests`</li><li>Coverage upload</li></ul> |
| `alltests.yml` | `push` to `develop` or `master`; PR into `develop` or `master`; branch name ending in `choi`; `workflow_dispatch` | `ubuntu`, `macos`, `windows`; Python `3.13` | <ul><li>Doctests</li><li>MPMC doctests and unit tests on all three OSes</li><li>`unittests`</li><li>Coverage upload</li><li>Booktests</li><li>Linux-only UMBridge doctests when Docker is available</li></ul> |
| `unittests.yml` (`tests` job) | `push` to `develop` or `master`; PR into `develop` or `master`; `workflow_dispatch` | `ubuntu`, `macos`, `windows`; Python `3.10` to `3.14` | <ul><li>Install test and optional extras</li><li>Run `unittests`</li><li>No MPMC stack installed, so MPMC unit tests skip</li></ul> |
| `unittests.yml` (`core-tests` job) | same as above | `ubuntu`, `macos`, `windows`; Python `3.9` | <ul><li>Build and install the no-extra user wheel, check dependencies, and import outside the source tree</li><li>Install `test_core`, then run `unittests_core` (no booktests)</li><li>Blocking test of the declared support-policy floor, on every supported OS</li></ul> |
| `unittests.yml` (`prerelease-tests` job) | same as above | `ubuntu`; Python `3.15.0-rc.1` | <ul><li>Uses `actions/setup-python` with `allow-prereleases` (conda-forge has no 3.15)</li><li>Install `test_core`, run `unittests_core`</li><li>Non-blocking: expected to fail until `scipy` and `scikit-learn` ship cp315 wheels</li></ul> |
| `docs.yml` | `push` to `master` | `ubuntu`, Python `3.13` | <ul><li>`uml`</li><li>`copydocs`</li><li>`mkdocs gh-deploy --force`</li></ul> |
| `pep8.yml` | `push` to `develop` or `master`; `workflow_dispatch` | `ubuntu`, Python `3.13` | <ul><li>`check_pep8`</li><li>Open a badge-update pull request if badge assets change</li></ul> |
| `pypi-stats.yml` | Weekly schedule; `workflow_dispatch` | `ubuntu`, Python `3.13` | <ul><li>Regenerate PyPI download statistics</li><li>Publish updated files</li></ul> |

There is no nightly CI schedule.

## Policy

- Linux is the default feedback path and runs on every push.
- macOS and Windows in `alltests.yml` are reserved for `develop`/`master` pushes, pull requests into those branches, branches whose name ends in `choi`, and manual runs.
- `concurrency` cancels superseded runs in both workflows; in `alltests.yml`, `push` and `pull_request` use separate groups so a PR does not inherit cancelled sibling checks from a same-SHA push.
- Both `unittests.yml` and `alltests.yml` pass `matrix.python-version` to `setup-miniconda` and assert the running interpreter before any test runs, so their version labels are real. Steps that touch Python use a profile-loading shell (`bash -el {0}` on Unix, `pwsh` on Windows); the default non-login shell silently falls back to the conda base interpreter, which is how these matrices previously went green without testing the versions they named.
- Booktests are skipped on feature-branch pushes and run only in the full sweep.
- `unittests.yml` is tiered: `tests` installs the full `test` extra (needing Python `3.10`+ via `pytest >= 9.0.3` and `parsl >= 2026.01.05`), `core-tests` verifies the built no-extra wheel before installing the slim `test_core` extra, and `prerelease-tests` looks ahead to the next interpreter. Test modules self-skip through `pytest.importorskip` when an optional stack is missing.
- The pre-release tier is informational and never gates a merge. Promote a version out of it into the `tests` matrix once the job passes; `Programming Language :: Python :: 3.15` is deliberately **not** in `pyproject.toml` classifiers until then.
- UMBridge doctests run only on Linux full sweeps with Docker available.
- MPMC steps in `alltests.yml` are **not** OS-gated: they run on every OS the matrix selects. See [MPMC Coverage by OS](#mpmc-coverage-by-os).
- `workflow_dispatch` means manually triggered workflow.

## MPMC Coverage by OS

MPMC needs a platform-specific `pyg_lib` wheel that PyPI does not carry, installed separately by `qmcpy-install-mpmc`. Only `alltests.yml` does that, and its MPMC steps carry no `if: runner.os` condition, so they run on every OS the matrix selects.

| Workflow / trigger | Python | Ubuntu | macOS | Windows |
|---|---|---|---|---|
| `alltests.yml`, full sweep | `3.13` | Run | Run | Run |
| `alltests.yml`, feature-branch `push` | `3.13` | Run | Not in matrix | Not in matrix |
| `unittests.yml` (`tests`) | `3.10`-`3.14` | Skipped | Skipped | Skipped |
| `unittests.yml` (`core-tests`) | `3.9` | Skipped | Skipped | Skipped |

"Run" covers both the MPMC doctests (`make doctests_mpmc`) and the MPMC unit tests in `test/test_dd_mpmc.py`. `unittests.yml` never calls `qmcpy-install-mpmc`, so those tests skip there via `pytest.importorskip("pyg_lib")` and its jobs pass without exercising MPMC — treat `alltests.yml` as the only source of MPMC signal. See [mpmc-compatibility.md](mpmc-compatibility.md) for the version-support policy behind this split.

## Related Docs

- [tests.md](tests.md): local Makefile targets and coverage commands.
- [booktests.md](booktests.md): notebook-test mechanics and developer commands.

When workflow files `.github/workflows/*.yml` change, update this page together with `mkdocs.yml`, `README.md`, and [tests.md](tests.md) if applicable.
