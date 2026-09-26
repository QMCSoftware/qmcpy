# QMCSoftware Agent Guidance

## Purpose and authority

This file is a concise entry point for coding agents. It does not replace the repository's contributor policies. Before substantive work, read and follow:

- [`CONTRIBUTING.md`](CONTRIBUTING.md) for issues, branches, pull requests, setup, and validation;
- [`docs/good_practices.md`](docs/good_practices.md) for implementation, testing, documentation, and review expectations; and
- [`docs/ai-assisted-contributions.md`](docs/ai-assisted-contributions.md) for required verification and pull-request disclosure.

## Required workflow

- Inspect `git status` and preserve unrelated work before editing.
- Connect each change to a GitHub issue.
- Synchronize with `origin/develop`, create a focused feature branch from `develop`, and open a pull request back into `develop`.
- Do not commit directly to `develop` or `master`, and never force-push shared branches.
- Coding agents must not merge pull requests; a human maintainer performs merges after the required human reviews and CI checks are complete.
- Keep the change scoped to its issue. Do not modify other repositories unless the user separately authorizes work there.
- Use the GitHub issue and pull request as the operational handoff for parallel work. Do not introduce repository-wide task or status files without maintainer agreement.
- Disclose substantive AI assistance in the pull-request template and state what was independently verified.

## Multi-machine and parallel work

- Before resuming work, fetch the remote state and confirm the current branch, its upstream, and the associated issue and pull-request status. Fast-forward an existing feature branch when possible; if local and remote history diverge, stop and reconcile the histories without force-pushing.
- Before changing machines or handing work to another collaborator, push a coherent feature-branch state and update the issue or pull request with the remaining work, validation completed, and any unresolved questions. Do not leave the only copy of active work on one machine.
- Do not edit the same feature branch concurrently on multiple machines or with multiple collaborators. Use separate issue-linked branches for genuinely parallel work and combine them through reviewable pull requests.

## Validation and generated content

- Run the smallest relevant checks described in the contributor guides, plus broader tests when the change affects shared behavior.
- Run `git diff --check` and review the complete diff before committing.
- When documentation is affected, build or render the relevant documentation and inspect the result.
- Do not edit generated files as if they were authoritative source. Keep source data, generators, committed outputs, and build instructions synchronized as required by the existing workflow.
- Never bypass failing tests or review safeguards merely to complete a task.

## Release safeguards

- `develop` is the integration branch and `master` is updated only through the periodic release process in [`docs/RELEASE.md`](docs/RELEASE.md).
- Coding agents must not merge `develop` into `master`, change a release version, publish to TestPyPI or PyPI, create or push release tags, create GitHub releases, or announce a release. A release maintainer must explicitly authorize and perform those operations.
- Never inspect, expose, commit, or reproduce package-index credentials, API tokens, or other release secrets.

## Repository boundaries

QMCSoftware contains QMCPy source, tests, demos, papers, and technical documentation. Organization-wide website content belongs in the separate `qmcsoftware-website` repository. Coordinate cross-repository changes as separate, explicitly scoped tasks and publish dependencies before updating their consumers.
