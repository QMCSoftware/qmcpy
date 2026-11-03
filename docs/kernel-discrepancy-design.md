# Kernel and discrepancy collaboration

**Status:** Draft for discussion, updated after auditing `develop` at
`3c89e7fc`. This branch establishes guidance; it does not implement or approve
a new public API. See the [audit and reproductions](kernel-discrepancy-audit.md)
and the [paper copies and reading guide](references/README.md).

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
point-set-versus-measure and point-set-versus-point-set comparisons. Fred's
requirements are:

- Cheap discrepancy evaluation for periodic shift-invariant (SI) kernels with
  matching lattice sequences and digitally shift-invariant (DSI) kernels with
  matching digital sequences.
- Reusable structured Gram-matrix operations for Bayesian credible intervals
  and related kernel methods, preserving existing backend/batch capabilities.
- A general dense or blocked path for kernels, measures, or designs without
  a validated fast pairing. Not every kernel supports cheap evaluation.
- Cancellation-resistant evaluation of small discrepancies, spectra, and
  credible-interval factors, following the joint Jagadeeswaran–Hickernell
  papers [1], [2].

Keep a path toward two-sample tests and GP/Bayesian-cubature use without
replacing the existing kernel model prematurely.

Retain the [original MATH 565 convergence plan](https://github.com/fjhickernell/HickernellAcademicLib/blob/83a47281d7763eb8b4b6b2bc1f44412fa531b7f7/classlib/math_565_%E2%86%92_qmcpy_kernels_discrepancy_convergence_plan.md)
as provenance. It uses former repository names, includes placeholder sections,
and predates current QMCPy capabilities; its proposed dataclasses and module
layout are not an approved migration specification.

The first supported fast examples should include one existing SI/lattice
pair and one existing DSI/digital-net pair, with the uniform unit-cube target.
Agree on the exact classes and admissible sampler settings. Reuse existing
kernel evaluation and analytical-integral hooks, with small dense reference
calculations for independent validation. A squared-exponential or another
nonmatching kernel can demonstrate the general path; it does not inherit the
torus/digital-group shortcut merely by being stationary.

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
| [SI/DSI examples](https://github.com/QMCSoftware/qmcpy/blob/3c89e7fc/qmcpy/kernel/si_dsi_kernels.py) and [Bayesian cubature](https://github.com/QMCSoftware/qmcpy/blob/3c89e7fc/qmcpy/stopping_criterion/abstract_cub_bayes_ld_g.py) | FFT/FWHT products and solves already appear in public-kernel doctests; private Bayesian paths already use first-column spectra. The gap is a shared, validated operator/discrepancy interface. The audit also found baseline correctness defects that must not become the new contract. |

Related work includes [historical discrepancy PR #378](https://github.com/QMCSoftware/qmcpy/pull/378),
[Lattice kronecker PR #556](https://github.com/QMCSoftware/qmcpy/pull/556),
[shift/NaN issue #612](https://github.com/QMCSoftware/qmcpy/issues/612), and
[shape issue #629](https://github.com/QMCSoftware/qmcpy/issues/629).
Check their latest state and avoid duplicating pending fixes. Do not base this
feature on an unmerged branch without an explicit dependency decision.

## Structured and general computation paths

Treat speed as a property of the kernel, design, target measure, and requested
operation together. The operator should expose its supported capabilities and
requirements instead of assuming every invariant kernel is fast on every design.

For a complete, compatible design of `n = 2**m` points in `d` dimensions:

| Pairing | Equal-weight uniform-target squared discrepancy | Structured Gram operations |
| --- | --- | --- |
| Periodic SI kernel + rank-1 lattice | Evaluate one correctly indexed kernel/remainder column and reduce it: `O(nd)` for fixed-cost coordinate kernels. No FFT is needed for this scalar. | `O(nd + n log n)` spectral setup; `O(n log n)` matvec/solve; `O(n)` log determinant after the spectrum, subject to a valid positive spectrum. |
| DSI kernel + compatible digital net | The analogous one-column reduction: `O(nd)`. | FWHT spectral setup/products/solves with analogous costs and explicit binary precision/order conventions. |
| General/nonmatching pair | Dense or blocked pairwise reference calculation, typically `O(n^2 d)` kernel work; retain analytical target integrals only for supported pairs. | Dense assembly/storage are quadratic; a general dense factorization is typically cubic. Blocking controls evaluation memory and does not create a fast factorization. |

Here, cheap Gram matrices mean a representation supporting cheap operations,
not a promise to return all `n*n` entries cheaply. Materializing the full array
still has quadratic output/storage cost. For fixed kernel parameters, retain
the generator column and spectrum using linear additional storage. Repeated
hyperparameter fitting incurs setup/likelihood work for each parameter trial;
arbitrary-query cross-covariances and posterior variances have separate costs.

Compatibility checks must cover complete dyadic point sets, ordering and
transform normalization, common shifts, digital linear scrambling, binary
precision, interlacing, independent sampler replications, and kernel batches.
`fftbr` expects bit-reversed ordering; other lattice orders need a validated
permutation/transform. Do not enable the simple FWHT route for nested uniform
(Owen/NUS) scrambling, arbitrary prefixes, or arbitrary permutations without
an additional valid construction. Pooling independent replications does not
automatically yield one structured Gram matrix. Target transformations,
derivatives, and observation-noise models need their own compatibility contract.

The proposed Gram interface should provide first-column/remainder data,
spectrum, matvec, solve, log determinant, kernel integrals, and explicit dense
materialization. Final method names remain a design decision. Preserve scalar
scale and parameter conventions when connecting it to Bayesian consumers.

## Cancellation resistance is part of the interface

For the normalized product form in [1, sec. 4.4], compute the remainder
directly. With coordinate terms `a_j`, use `r_1 = a_1` and
`r_j = r_(j-1) * (1 + a_j) + a_j`; do not form the product and subtract one.

For ordinary SI/DSI kernel values with constant cube integral `S`, retain
`K = S * 11^T + R` as separate components. Transform the directly evaluated
remainder column to obtain its constant-mode eigenvalue `rho_0`. For a valid
matching equal-weight design, squared discrepancy is `rho_0 / n`. For
normalized sample weights `w`, it is `w^T R w`; nonnormalized weights require
the additional term `S * (sum(w) - 1)^2`.

In the normalized credible-factor calculation of [1], evaluate
`rho_0 / (n*S + rho_0)` rather than `1 - n*S/lambda_0`. This is a
dimensionless factor, not by itself every model's posterior variance or full
Bayes half-width. Preserve the paper's mean/scale estimation, normalization,
and the distinct MLE/FULL/GCV formulas. A general measure or derivative
kernel cannot inherit these formulas without verifying their assumptions.

Current public kernels have an internal constant/remainder recurrence, but
ordinary evaluation adds the constant back. Thus `kernel(stable=True) - S`
can still destroy the small answer. The new interface must expose and carry
the remainder and its spectrum through discrepancy and Bayesian operations.
Reconstructing them as `K - S*11^T` or `lambda_0 - n*S` defeats that purpose.

Provide a convenient public remainder-evaluation function; a candidate name
is `kernel_minus_constant(...)`. Its contract is direct evaluation of the
small remainder, with the same array/backend/batch conventions as the full
kernel and access to the associated constant `S`. Naming and return type
remain to be reviewed. Reuse the internal recurrence without requiring users
to prepare its low-level component/derivative arguments. Initially define
this for ordinary SI/DSI values; derivative combinations and general
measure-centered kernels need their own reviewed contracts. Unsupported
kernels must not silently implement it as full-kernel subtraction while
claiming cancellation resistance.

Require high-precision or independently known tiny-discrepancy cases, small
coordinate weights, increasing dimension/sample size, scale changes, and
NumPy/PyTorch dtype checks. Include a case where subtraction returns zero but
the reference is positive; an absolute tolerance that accepts zero is
insufficient. Define zero/singular cases and distinguish harmless roundoff,
negative unbiased estimates, and materially invalid covariance spectra.
Unconditional absolute values or undocumented jitter must not hide defects.

## Implementation sequence

1. Triage the [audited baseline defects](kernel-discrepancy-audit.md) into
   focused bug work and independent regression cases. Correctness fixes are
   separate from introducing a new abstraction.
2. Agree on kernel/measure, estimator/output, weight, and stable-remainder
   contracts. Establish a dense/blocked reference without replacing the
   existing object model.
3. Expose validated SI/lattice and DSI/digital-net discrepancy and structured
   Gram operations, preserving remainder information and explicit fast-path
   eligibility. Validate both against dense and high-precision references.
4. Connect Bayesian consumers to the shared operations. Check signed
   integrands, all supported MLE/FULL/GCV modes, hyperparameter conversions,
   and stable credible factors; do not require parity with known defects.
5. Expand kernel/measure families, prefix operations, and other uses through
   focused follow-up issues, with explicit cost and capability documentation.

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
6. **Fast paths and stable representations.** Agree on capability dispatch,
   transform/order metadata, remainder/spectrum access, singular behavior,
   and dense fallback. Validate matrix symmetry/positive semidefiniteness,
   operations, and discrepancy separately; matching a dense multiplication
   alone does not establish a valid covariance.

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
- SI/lattice and DSI/digital-net structured-versus-dense products/solves,
  symmetry and spectra, common-shift invariance, ordering/scrambling rejection,
  independent replications, and supported Torch parameter gradients.
- Cancellation-sensitive discrepancies and credible factors checked against
  high precision, including explicit preservation of a tiny positive result.
- Bayesian signed-integrand and MLE/FULL/GCV regression checks, and clearly
  measured setup, repeated-operation, and dense-output costs.
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

## References

[1] R. Jagadeeswaran and F. J. Hickernell, "Fast automatic Bayesian cubature
using lattice sampling," *Statistics and Computing*, vol. 29, pp. 1215–1229,
2019, doi: [10.1007/s11222-019-09895-9](https://doi.org/10.1007/s11222-019-09895-9).
See sec. 4.4, printed p. 1224, in the [local paper copy](references/jagadeeswaran-hickernell-2019-lattice.pdf).

[2] R. Jagadeeswaran and F. J. Hickernell, "Fast Automatic Bayesian Cubature
Using Sobol’ Sampling," in *Advances in Modeling and Simulation: Festschrift
for Pierre L’Ecuyer*, Z. Botev, A. Keller, C. Lemieux, and B. Tuffin, Eds.,
pp. 301–318, 2022, doi: [10.1007/978-3-031-10193-9_15](https://doi.org/10.1007/978-3-031-10193-9_15).
See sec. 3.3, printed pp. 307–310, in the [local paper copy](references/jagadeeswaran-hickernell-2022-sobol.pdf).
