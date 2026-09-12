# Good Practices for Contributors

This page collects the shared contribution expectations that help QMCPy stay scientifically correct, reproducible, and reviewable. Use it together with the [contributing guide](https://qmcsoftware.github.io/QMCSoftware/CONTRIBUTING/), which covers repository workflow and local setup, plus the [AI-assisted contributions policy](ai-assisted-contributions.md), the [test targets guide](tests.md), and the [notebook test guide](booktests.md).

## Start from an Issue and Keep the Scope Clear

- Connect every bug fix, feature, refactor, or documentation update to an issue.
- Keep each pull request focused on one topic so reviewers can reason about the mathematical and API impact.
- For architectural or mathematically subtle changes, open a draft PR early and schedule a review meeting before merge.

## Tests Are Required

We expect tests for every change that affects behavior, documentation, or user workflows.

### Cover the Changed Behavior

- Add or update **unit tests** in `test/` for new logic, bug fixes, edge cases, invalid inputs, shapes, finite outputs, and meaningful invariants.
- Add or update **doctests** when public docstrings, examples, or usage patterns change.
- Add or update **notebook tests** when a demo or blog notebook changes.

### Keep Tests Stable and Meaningful

- Use deterministic seeds or deterministic generators in tests and examples.
- Keep tests small enough to run locally and in CI.
- When speeding up or stabilizing tests, keep tolerances, sample sizes, and expected outputs strong enough to catch real regressions. If you relax a check, explain why the weaker threshold is still meaningful.
- Match the existing test style in the file and use the repository's assertion helpers or test framework methods consistently.

Run the smallest relevant checks before requesting review; see the contributing guide and test guides for the exact commands.

When notebook-backed content changes:

- Use the notebook-focused checks for `demos/` and blog content.
- Keep executable Python snippets under `docs/` runnable as well.
- If one notebook cell is unusually slow, prefer skipping that cell or reducing the workload rather than skipping the entire notebook test.

## Write Google-Style Docstrings

QMCPy documentation is built from docstrings, so public APIs should document their behavior clearly and consistently.

- Use **Google-style docstrings** for public classes, methods, and functions.
- Document parameters, return values, shapes, assumptions, and any stochastic behavior.
- Include short doctestable examples when they clarify expected use.
- Update docstrings at the same time as the implementation so the rendered API docs do not drift from the code.

## Extend the Existing Object Model

New functionality should fit the existing QMCPy class hierarchy instead of introducing parallel designs without discussion.

- Inherit from the closest existing QMCPy abstract or base class rather than directly from `object`.
- Reuse established interfaces and field names where possible.
- Typical extension points include `DiscreteDistribution`, `TrueMeasure`, `Integrand`, `StoppingCriterion`, and `AccumulateData`.
- If a change does not fit the current hierarchy, raise that design question in an issue or draft PR before committing to a new abstraction.

The [components overview](components.md) and the QMCSoftware blog post on [object classes in QMCPy](https://qmcsoftware.org/blogs/visualizing-the-internals-of-object-classes-in-qmcpy/) provide useful background on the current architecture.

## Validate Links, Metadata, and CI Scope

Several reviews focused on avoidable cleanup that is easy to catch before requesting review.

### Links, Names, and Metadata

- Verify external URLs, raw data links, and referenced file paths before opening a PR.
- When a PR deletes or moves an already-published page, keep its URL alive with a `redirect_maps` entry under the `redirects` plugin in `mkdocs.yml`. Deleting a page deletes its links too, so only `make check_removed_urls` (deployed `sitemap.xml` vs current `docs/` sources) catches this, not `make check_links`.
- Keep public names exact across code, docs, nav labels, notebooks, PR titles, and data files, especially for package names, publication years, and schema keys.
- Use concrete metadata values when possible. For example, prefer specific supported languages over vague labels such as "Multiple".

### Build Pipeline and Generated Artifacts

- Avoid committing generated outputs, copied raw data, or other bulky artifacts when a source URL or regeneration step is sufficient.
- Keep CI and dependency changes as narrow as possible, and explain in the PR description why each new extra, workflow step, or version pin is needed.
- If you add generated documentation, data-driven tables, or helper scripts, keep the source files, generator, committed outputs, and docs build pipeline in sync. If regeneration is manual, document the exact command and commit the refreshed output together with the source change.

### Docs and Assets

- If you add custom HTML or CSS to the docs, verify that it renders correctly in both Material light and dark themes and on narrow screens without depending on missing third-party assets.
- Prefer shell-friendly filenames without spaces for assets that may be referenced from scripts, CI, or command lines.
- For large binary artifacts such as slides, prefer reproducible source materials plus a short README, and use Git LFS or external hosting when normal git history would become heavy.
- In docs, prefer unambiguous wording and stable statuses over tentative or ambiguous phrases.

### Code Hygiene

- Remove unused imports, trailing whitespace, and other style-only churn before requesting review.
- Use explicit runtime exceptions such as `ParameterError` for invalid user inputs instead of relying on `assert` statements in production code.

## Add Demos or Blogs as Notebooks

User-facing methods, new workflows, and mathematically important additions should usually come with an executable notebook.
- Put demos and tutorials in `.ipynb` files under `demos/`.
- If a contribution is best explained as a blog post, keep the blog content backed by a notebook when practical.
- Keep notebooks lightweight, deterministic, and suitable for docs rendering and CI.
- Include the mathematical rationale, key assumptions, validation evidence, and a minimal example.

## Requesting Review

Request review when the contribution is ready for technical evaluation, not while core pieces are still missing.

- Open a **draft PR** if you want early feedback on design, mathematics, or scope.
- Request formal review only after the relevant tests pass locally and the required docstrings, docs, and notebooks are in place.
- If AI tools substantively affected the change, disclose that use in the PR description and summarize what you independently verified.
- Summarize the numerical goal, API impact, issue link, and commands you ran in the PR description.
- If you changed CI, dependency pins, notebook runtime, or external data references, explain that scope explicitly in the PR description.
- Call out any remaining risks, approximations, or open questions explicitly.
- For complex mathematical changes, ask for a review meeting in addition to GitHub review comments.

## Requesting Re-Review

Re-request review when you have addressed prior comments and the branch is ready for another pass.

- After substantial updates, post a short summary of what changed since the last review.
- Re-run the relevant tests after addressing review feedback, especially if behavior or interfaces changed.
- Re-request review from the same reviewers when their previous concerns have been addressed.
- If new commits materially change the design or numerical behavior, mention that directly so reviewers know to re-check the affected areas.

## Before Merge

- Ensure required reviews are complete.
- Ensure CI is green for the relevant jobs.
- Make sure docs, tests, and notebooks changed together when the contribution changed public behavior.
- Delete the feature branch after a successful merge.
