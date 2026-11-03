# Kernel and discrepancy collaboration

**Status:** Draft for discussion. This branch establishes guidance; it does
not implement or approve a new public API.

**Collaborators:** Fred Hickernell and [Laasya (`Laasya-73`)](https://github.com/Laasya-73).

**Coordination:** [Issue #646](https://github.com/QMCSoftware/qmcpy/issues/646).
Planning branch: `codex/kernel-discrepancy-plan`; integration target: `develop`.

## Read this first

Both collaborators and their AI assistants should use this document together
with the repository's [agent guidance](https://github.com/QMCSoftware/qmcpy/blob/develop/AGENTS.md),
[contribution workflow](https://github.com/QMCSoftware/qmcpy/blob/develop/CONTRIBUTING.md),
[good practices](good_practices.md), and [AI policy](ai-assisted-contributions.md).

The design document holds scope, agreed decisions, and open design questions.
The issue and its linked PRs hold current ownership, work in progress, next
actions, and validation evidence. Read their latest updates before resuming;
this initial document is not evidence that a later task is still unclaimed.

The first joint action is to compare the starting code below, discuss the open
decisions, and agree on a small first implementation. Record each agreement
here through a reviewed change before treating it as an implementation contract.

## Goal and scope

Extend QMCPy's existing kernels with a coherent discrepancy interface for
point-set-versus-measure and point-set-versus-point-set comparisons. Keep a
path toward two-sample tests and GP/Bayesian-cubature use without replacing the
existing kernel model prematurely.

Retain the [original MATH 565 convergence plan](https://github.com/fjhickernell/HickernellAcademicLib/blob/83a47281d7763eb8b4b6b2bc1f44412fa531b7f7/classlib/math_565_%E2%86%92_qmcpy_kernels_discrepancy_convergence_plan.md)
as provenance. It uses former repository names, includes placeholder sections,
and predates current QMCPy capabilities; its proposed dataclasses and module
layout are not an approved migration specification.

The initial implementation should reuse existing kernel evaluation and
analytical-integral hooks, establish a small dense reference calculation, and
validate it independently. An existing squared-exponential kernel with the
uniform unit-cube target is a candidate; the collaborators must confirm the
first supported pair and deliverable. Broader kernel/measure families, prefix
calculations, and structure-specific acceleration can follow in separate PRs.

## Verified starting point

The QMCPy inventory below was checked at
[`3c89e7fc`](https://github.com/QMCSoftware/qmcpy/tree/3c89e7fc);
recheck affected interfaces when `develop` advances.

| Source | Existing capability and integration constraint |
| --- | --- |
| [QMCPy kernel API](api/kernels.md) and [AbstractKernel](https://github.com/QMCSoftware/qmcpy/blob/3c89e7fc/qmcpy/kernel/abstract_kernel.py) | Existing NumPy/PyTorch evaluation, parameter batches, derivatives, and trainable parameters. Calls broadcast arrays with final coordinate dimension; they do not automatically turn two `(n, d)` and `(m, d)` arrays into a Gram matrix. Preserve these capabilities. |
| Existing `single_integral_01d` and `double_integral_01d` kernel methods | These are unit-cube integrals, not expectations for an arbitrary `TrueMeasure`. Gaussian/squared-exponential and SI/DSI families implement them; not every kernel does. |
| [Classlib discrepancy prototype](https://github.com/fjhickernell/HickernellAcademicLib/tree/83a47281d7763eb8b4b6b2bc1f44412fa531b7f7/classlib/discrepancy) | Contains sample/sample and sample/analytical-measure comparisons, prefix calculations, and several kernels. The uniform-measure adapter has analytical formulas for the centered kernel. It is reference code to evaluate, not a package to copy wholesale. |
| [Kronecker helpers](https://github.com/QMCSoftware/qmcpy/blob/3c89e7fc/qmcpy/discrete_distribution/kronecker.py) and [MPMC utilities](https://github.com/QMCSoftware/qmcpy/blob/3c89e7fc/qmcpy/discrete_distribution/mpmc/utils.py) | Existing specialized discrepancy calculations and losses provide comparison cases. Their weight and output conventions need explicit reconciliation with a general interface. |

Related work includes [historical discrepancy PR #378](https://github.com/QMCSoftware/qmcpy/pull/378),
[Lattice kronecker PR #556](https://github.com/QMCSoftware/qmcpy/pull/556),
[shift/NaN issue #612](https://github.com/QMCSoftware/qmcpy/issues/612), and
[shape issue #629](https://github.com/QMCSoftware/qmcpy/issues/629).
Check their latest state and avoid duplicating pending fixes. Do not base this
feature on an unmerged branch without an explicit dependency decision.

## Decisions to resolve together

1. **Public API and compatibility.** Use discrepancy terminology and explain
   its relationship to maximum mean discrepancy (MMD). Agree on public names,
   result shapes, and extension points before adding classes. Prefer extending
   `AbstractKernel` and existing measure concepts over a parallel hierarchy.
2. **Squared values and estimators.** Distinguish empirical discrepancy,
   squared discrepancy, and an unbiased estimator of squared discrepancy.
   The prototype's `mmd` and prefix helpers have different default output
   conventions; unsquared outputs currently clip negative squared estimates.
   Specify negative-estimate handling and samples too small for the requested
   estimator rather than inheriting silent clipping or estimator switching.
3. **Kernel–measure compatibility.** List supported pairs and the exact
   single/double-integral formulas needed by each operation. Retain the old
   plan's requirement for analytical integrals; unsupported pairs must fail
   clearly rather than silently use Monte Carlo. Validate that the supplied
   measure adapter and kernel parameters actually match.
4. **Three kinds of weights.** Name coordinate/kernel weights,
   sample/quadrature weights, and weights on a sequence of prefix objectives
   separately. The prototype centered kernel uses `gamma**2`, while current
   SI/DSI and MPMC coordinate weights enter linearly. Agree on normalization
   and conversion instead of carrying parameter values across unchanged.
5. **Array, backend, and resource contracts.** Specify accepted dimensions,
   batching, dtype/device behavior, and memory limits. Begin with a dense
   reference; define blocked evaluation of the same finite sum before large
   examples. Decide
   whether the first implementation needs PyTorch/autograd support.
6. **Fast paths.** Add lattice/digital-net acceleration only for documented
   kernel/design combinations. Verify agreement with the dense reference and
   measure scaling; the old plan's general fast-path ambition is not a
   complexity guarantee for every kernel or point set.

## First implementation acceptance criteria

After the design decisions above are agreed, create focused implementation
issues with named owners and concrete completion conditions. The initial PR
should include:

- A documented contract for the selected comparison and supported pairs,
  including shapes, normalization, outputs, and invalid-input behavior.
- Small deterministic checks against independently derived formulas or
  trusted existing implementations, including diagonal terms and estimator
  differences; compare dense and blocked calculations when blocking is added.
- Edge cases for insufficient samples, mismatched dimensions, unsupported
  kernel–measure pairs, weight conventions, and finite numerical results.
- A reproducible demo and doctestable examples explaining the interpretation
  and limits of the reported discrepancy.
- Focused tests, affected documentation checks, AI-use disclosure, CI, and
  the required two human reviews before a maintainer merges.

The planning PR does not close the feature issue or establish that any of
these implementation criteria have already been met.

## How Fred, Laasya, and their AIs work together

1. **Start with current evidence.** Inspect `git status`, fetch `origin`, and
   read this document plus the issue and relevant PR. Preserve existing work;
   fast-forward when safe and reconcile divergence without a force-push.
2. **Claim a concrete task.** Record the owner, intended files, branch, and
   acceptance criteria in the issue/PR before editing. One active editor uses
   a given branch at a time, including across machines and AI sessions.
3. **Use review comments for planning.** Discuss proposed changes in the draft
   PR; take turns editing the planning branch. For parallel implementation,
   create separate issue-linked branches from current `develop` and focused
   PRs back into `develop`. State dependencies between PRs explicitly.
4. **Make a durable handoff.** Push coherent work, update this document when
   a design decision changes, and leave an issue/PR update with what changed,
   exact validation commands/results, remaining work, and unresolved decisions.
   Do not leave the only copy in a local checkout or private chat.
5. **Keep the scope here.** Classlib and course repositories are read-only
   reference sources for this task. A migration there requires its own scope
   and publication sequence. Do not merge PRs or publish a release through an
   AI agent; maintainers retain those responsibilities.

Laasya has repository write access, verified when this branch was created.
Either collaborator can fetch and check out `codex/kernel-discrepancy-plan`.
Write access does not make concurrent edits to the same branch safe; agree on
the current editor or use separate branches.

## Separate machines and Codex projects

Each collaborator works in their own local clone of `QMCSoftware/qmcpy`.
A local Codex project should point to that clone as its primary folder; its
name can differ on each machine. The GitHub branch, this guidance, and the
issue/PR are the shared record; local chats are supporting context.

To begin, fetch `origin` and check out the planning branch. If it is not
already local, use:

```bash
git fetch origin
git switch --track origin/codex/kernel-discrepancy-plan
```

If the branch already exists locally, switch to it, inspect the worktree, and
fast-forward with `git pull --ff-only` only when the local work is safe to
update. Do not discard uncommitted work or reset a diverged branch.

A useful first prompt to either AI is:

> Read AGENTS.md and docs/kernel-discrepancy-design.md, then read issue #646
> and its linked PRs. Summarize the current decisions, task owner, validation,
> and next action. Check the branch and worktree before editing; do not begin
> a task already being edited by the other collaborator.
