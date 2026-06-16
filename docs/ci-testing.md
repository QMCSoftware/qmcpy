# CI/CD Testing

This page summarizes QMCPy's current GitHub Actions CI layout.

## Workflows

| Workflow | Trigger | Runner / Python | Main work |
|---|---|---|---|
| `alltests.yml` | Feature-branch `push` | `ubuntu`, Python `3.13` | <ul><li>Non-Docker doctests</li><li>`unittests`</li><li>Coverage upload</li></ul> |
| `alltests.yml` | `push` to `develop` or `master`; PR into `develop` or `master`; `workflow_dispatch` | `ubuntu`, `macos`, `windows`; Python `3.13` | <ul><li>Doctests</li><li>`unittests`</li><li>Coverage upload</li><li>Booktests</li><li>Linux-only UMBridge doctests when Docker is available</li></ul> |
| `unittests.yml` | `push` to `develop` or `master`; PR into `develop` or `master`; `workflow_dispatch` | `ubuntu`, `macos`,  `windows`; Python `3.5` to `3.14` except for `3.13` | <ul><li>Install test and optional extras</li><li>Run `unittests`</li></ul> |
| `docs.yml` | `push` to `master` | `ubuntu`, Python `3.13` | <ul><li>`uml`</li><li>`copydocs`</li><li>`mkdocs gh-deploy --force`</li></ul> |
| `pep8.yml` | `push` to `develop` or `master`; `workflow_dispatch` | `ubuntu`, Python `3.13` | <ul><li>`check_pep8`</li><li>Open a badge-update pull request if badge assets change</li></ul> |
| `pypi-stats.yml` | Weekly schedule; `workflow_dispatch` | `ubuntu`, Python `3.13` | <ul><li>Regenerate PyPI download statistics</li><li>Publish updated files</li></ul> |

There is no nightly CI schedule.

## Policy

- Linux is the default feedback path and runs on every push.
- macOS and Windows in `alltests.yml` are reserved for `develop`/`master` pushes, pull requests into those branches, and manual runs.
- `concurrency` cancels superseded runs in both workflows; in `alltests.yml`, `push` and `pull_request` use separate groups so a PR does not inherit cancelled sibling checks from a same-SHA push.
- `alltests.yml` pins Miniconda base Python to `3.13`; `unittests.yml` still uses the base environment without explicitly passing `matrix.python-version` into `setup-miniconda`.
- Booktests are skipped on feature-branch pushes and run only in the full sweep.
- UMBridge doctests run only on Linux full sweeps with Docker available.
- `workflow_dispatch` means manually triggered workflow.

## Related Docs

- [tests.md](tests.md): local Makefile targets and coverage commands.
- [booktests.md](booktests.md): notebook-test mechanics and developer commands.

When workflow files `.github/workflows/*.yml` change, update this page together with `mkdocs.yml`, `README.md`, and [tests.md](tests.md) if applicable.
