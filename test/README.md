# Test Targets Guide

This document describes the available test targets in the Makefile for QMCSoftware. All targets use pytest with parallel execution (via `pytest-xdist`) when available.

## Quick Reference

| Target | Purpose | Speed | Use Case |
|--------|---------|-------|----------|
| `make tests_fast` ⚡ | **Recommended**: All tests in parallel | Fast | Default choice: doctests + unittests + booktests concurrently |
| `make tests_no_docker` | All tests sequentially (no Docker) | Slow | Conservative validation; isolates flaky tests |
| `make tests` | All tests with Docker support | Very Slow | Complete validation with GPU/Docker dependencies |
| `make unittests` | Unit tests only | Fast | Quick feedback on code changes |
| `make doctests_no_docker` | Doctests (excludes GPU/Docker) | Moderate | Validate docstring examples |
| `make doctests` | All doctests with Docker | Slow | Full docstring validation |
| `make booktests_no_docker` | Jupyter notebook tests | Slow | Validate demo notebooks |
| `make booktests_parallel_no_docker` | Notebook tests with Parsl parallelization | Variable | Distributed notebook execution |
| `make check_colab_notebooks` | Audit enabled notebooks for Colab-readiness | Fast | Catch missing pip installs, repo-local imports, and source-install drift |
| `make check_colab_notebooks_smoke` | Execute Colab notebook setup smoke tests | Fast | Run bootstrap plus early import/setup cells for enabled notebooks |
| `make harden_colab_notebook [NOTEBOOK=...]` | Insert Colab bootstrap and classify notebook(s) | Fast | Harden one notebook, or attempt to harden unclassified demo notebooks |
| `make report_colab_notebook_patterns` | Group notebooks by Colab bootstrap family | Fast | Audit which notebooks use basic, extra-pip, LaTeX, or repo-local setup cells |
| `make coverage` | Display coverage report | Instant | View test coverage summary |
| `make delcoverage` | Reset coverage tracking | Instant | Start fresh coverage analysis |


## Detailed Descriptions

## Scope

This short guide highlights four practical areas:
- Local tests in the `test/` directory and the Makefile targets that run them (for example `make unittests`, `make booktests_no_docker`, and `make doctests_no_docker`).
- Local coverage targets in the Makefile (for example `make coverage` and `make delcoverage`) and how to collect coverage locally using `--cov-append` or `coverage run --append`.
- Remote CI: the GitHub Actions workflow at `.github/workflows/alltests.yml` (referred to here as `alltests.yml`) which runs matrix jobs and invokes the Makefile targets for full validation.
- Remote coverage publishing to Codecov and how to add a Codecov badge to the project `README.md`.


### Core Test Targets (Recommended)

#### `make tests_fast` ⚡ **RECOMMENDED**
**Fastest option**: Runs doctests, unittests, and booktests **concurrently** in background processes.
- **Parallelization**: All three test families run simultaneously, not sequentially
- **Cleanup**: Removes invalid distribution artifacts before running
- **Time**: ~30–60 seconds (depending on CPU cores and notebook complexity)
- **Coverage**: Full summary report at the end
- **Use when**: You want comprehensive testing with maximum speed
- **log files**: in `test/booktests/logs/`

#### `make tests_no_docker`
Runs all tests sequentially: doctests, unittests, and generates coverage reports (excludes Docker-dependent tests).
- **Sequence**: doctests_no_docker → unittests → coverage report
- **Time**: ~60–120 seconds
- **Coverage**: Full summary report
- **Use when**: Pre-commit or CI/CD validation (without Docker)

#### `make tests`
Runs all tests **sequentially** with full Docker support (for GPU-heavy and UMBridge tests).
- **Includes**: Full doctests suite (with umbridge and markdown validation)
- **Time**: ~120–180+ seconds
- **Coverage**: Full summary report
- **Use when**: Complete validation with Docker dependencies available

---

### Doctest Targets

#### `make doctests_no_docker` (Composite)
Runs doctests excluding GPU and Docker dependencies.
- **Composition**: `doctests_minimal` + `doctests_torch` + `doctests_gpytorch` + `doctests_botorch`
- **Time**: ~15–30 seconds
- **Use when**: Validating docstring examples

#### `make doctests` (Composite)
Runs all doctests including Docker-dependent UMBridge and markdown validation.
- **Composition**: `doctests_markdown` + `doctests_minimal` + `doctests_torch` + `doctests_gpytorch` + `doctests_botorch` + `doctests_umbridge`
- **Time**: Variable (Docker startup overhead)
- **Use when**: Full docstring validation with Docker available

#### `make doctests_minimal` (Building Block)
Core doctests excluding all optional dependencies (PyTorch, GPyTorch, BoTorch, UMBridge).
- **Modules tested**: Main qmcpy modules
- **Time**: ~5–10 seconds
- **Note**: Usually called via `doctests_no_docker` or `doctests`; rarely used standalone

#### `make doctests_torch` (Building Block)
Doctests for PyTorch-dependent modules.
- **Modules tested**: `qmcpy/fast_transform/ft_pytorch.py`, `qmcpy/kernel/*.py`, `qmcpy/util/*shift*.py`
- **Time**: ~3–5 seconds
- **Note**: Usually called via `doctests_no_docker`; rarely used standalone

#### `make doctests_gpytorch` (Building Block)
Doctests for GPyTorch integration.
- **Modules tested**: `qmcpy/stopping_criterion/pf_gp_ci.py`
- **Time**: ~3–5 seconds
- **Note**: Usually called via `doctests_no_docker`; rarely used standalone

#### `make doctests_botorch` (Building Block)
Doctests for BoTorch integration.
- **Modules tested**: `qmcpy/integrand/hartmann6d.py`
- **Time**: ~2–3 seconds
- **Note**: Usually called via `doctests_no_docker`; rarely used standalone

#### `make doctests_umbridge`
Doctests for UMBridge wrapper (requires Docker).
- **Modules tested**: `qmcpy/integrand/umbridge_wrapper.py`
- **Time**: ~10–15 seconds (includes Docker startup)
- **Dependencies**: Docker must be running
- **Note**: Usually called via `doctests`; rarely used standalone

#### `make doctests_markdown`
Validates embedded Python code in markdown files under `docs/`.
- **Tools**: Uses `phmutest` (markdown test utility)
- **Time**: ~3–5 seconds
- **Note**: Usually called via `doctests`; rarely used standalone

#### Running doctests for a single file
Call pytest's `--doctest-modules` flag directly on the file, for example,
  ```bash
  python -m pytest --doctest-modules qmcpy/discrete_distribution/lattice/lattice.py
  ```

#### Suppressing an expected doctest warning (`conftest.py`)

For warnings intentionally triggered by a doctest, add a file-specific filter to `DOCTEST_WARNING_FILTERS` in the root-level `conftest.py`. This avoids hiding the warning globally or importing the test-only `pytest` dependency in library code.

Only filter expected, documented warnings. Fix unexpected warnings at their source.

---

### Unit & Notebook Test Targets

#### `make unittests`
Runs unit tests from the `test/` directory using pytest with parallel workers (when `pytest-xdist` is installed).
- **Time**: ~13–30 seconds (depending on CPU cores)
- **Coverage**: Incremental coverage report appended to `.coverage`
- **Use when**: Quick feedback on code changes

#### `make booktests_no_docker`
Generates and runs tests from Jupyter notebooks in the `demos/` folder using unittest discovery.
- **Automatically generates** missing test files in `test/booktests/`
- **Cleans** cache and temporary files before running
- **Time**: Highly variable (5 seconds to 5+ minutes depending on notebook complexity)
- **Coverage**: Incremental
- **Use when**: Validating demo notebooks or documentation

#### `make booktests_parallel_no_docker`
Runs notebook tests with **Parsl distributed parallelization** for compute-heavy demos.
- **Parallelization**: Uses Parsl framework for task scheduling
- **Cleanup**: Removes temporary outputs (EPS, JPG, PDF, PNG files, logs, etc.)
- **Time**: Highly variable (depends on Parsl workers and notebook complexity)
- **Optional parameters**: `TESTS="tb_notebook1 tb_notebook2"` to run specific tests
- **Dependencies**: Parsl must be installed and configured
- **Use when**: Running large notebook suites with distributed compute resources


---

### Helper / Internal Targets

#### `make check_booktests`
Validates that all Jupyter notebooks in `demos/` have corresponding test files in `test/booktests/`.
- **Output**: Lists missing test files and notebook/test file counts
- **Note**: Called automatically by `booktests_no_docker`; rarely used standalone

#### `make generate_booktests`
Auto-generates missing test stub files for notebooks.
- **Output**: Reports any generated files
- **Note**: Called automatically by `booktests_no_docker`; rarely used standalone

#### `make check_colab_notebooks`
Runs the strict static Colab-readiness checks.
- **Behavior**: Validates the manifest, badge and bootstrap placement, early dependencies, and repo-local imports
- **Use when**: You change a demo notebook or its Colab setup

#### `make check_colab_notebooks_smoke`
Runs a lightweight execution smoke test for each Colab-enabled notebook.
- **Execution scope**: Simulates a Colab runtime, rewrites shell install commands to no-ops, then executes the bootstrap cell plus up to `$(SMOKE_CODE_CELLS)` smoke-safe import/setup code cells
- **Purpose**: Catch runtime regressions in early import/setup logic that static checks miss
- **CI usage**: Invoked in Linux CI after test dependencies are installed
- **Default depth**: `SMOKE_CODE_CELLS=2`

#### `make harden_colab_notebook [NOTEBOOK=...]`
Hardens one notebook, or if `NOTEBOOK` is omitted, scans `demos/` for notebooks that are not yet listed in either `enabled` or `disabled`.
- **What it does**: Inserts the badge, adds a generated Colab bootstrap cell, infers common extra pip dependencies, and adds repo-local `sys.path` setup when needed
- **Classification rule**: Existing `disabled` entries are left untouched; unclassified notebooks are added to `enabled` only after hardening validates. Failures remain unclassified for manual review
- **Force mode**: `make harden_colab_notebook FORCE=1` regenerates the Open in Colab badge and the `# @title Execute this cell to install dependencies` cell for every notebook already listed in `enabled`; `make harden_colab_notebook NOTEBOOK=... FORCE=1` does the same for one notebook
- **Cell order**: The generated `import google.colab` bootstrap cell is always inserted after the Open in Colab badge
- **Validation**: Runs the existing Colab checks after rewriting; if validation fails, the notebook and manifest are restored and the failure is reported
- **Examples**: `make harden_colab_notebook NOTEBOOK=demos/plot_proj_function.ipynb`, `make harden_colab_notebook`, and `make harden_colab_notebook FORCE=1`

#### `make report_colab_notebook_patterns`
Groups notebooks already classified in `scripts/colab_notebooks_manifest.json` by the current Colab badge/bootstrap cell pattern.
- **Pattern families**: Reports basic `qmcpy`-only bootstrap cells, extra-pip variants, LaTeX setup cells, repo-clone/path-setup cells, and disabled notebooks grouped by reason
- **Dependency details**: Lists extra install commands for notebooks that need more than `qmcpy`
- **Placement summary**: Reports where the badge and bootstrap cells appear, for example `badge cell 1, bootstrap cell 2`
- **Use when**: You want to batch-normalize notebook Colab setup or review which notebooks will be affected by bootstrap changes

Every enabled notebook, grouped by its Colab bootstrap pattern family (regenerate with `make report_colab_notebook_patterns`):

- **Basic qmcpy bootstrap** (28): [acceptance_rejection.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/acceptance_rejection.ipynb), [asian-option-mlqmc.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/asian-option-mlqmc.ipynb), [brownian_bridge.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/brownian_bridge.ipynb), [control_variates.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/control_variates.ipynb), [copula_examples.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/copula_examples.ipynb), [Iteration_Log_Tolerance_Demo.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/demo_resume_data/Iteration_Log_Tolerance_Demo.ipynb), [digital_net_b2.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/digital_net_b2.ipynb), [gaussian_diagnostics_demo.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/gaussian_diagnostics/gaussian_diagnostics_demo.ipynb), [korobov_hammersley_latinhypercube_demos.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/korobov_hammersley_latinhypercube_demos.ipynb), [lattice_random_generator.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/lattice_random_generator.ipynb), [lebesgue_integration.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/lebesgue_integration.ipynb), [linear-scrambled-halton.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/linear-scrambled-halton.ipynb), [nei_demo.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/nei_demo.ipynb), [plot_proj_function.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/plot_proj_function.ipynb), [pricing_options.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/pricing_options.ipynb), [product_measure.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/product_measure.ipynb), [qei-demo-for-blog.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/qei-demo-for-blog.ipynb), [qmcpy-logo.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/qmcpy-logo.ipynb), [qmcpy_intro.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/qmcpy_intro.ipynb), [quickstart.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/quickstart.ipynb), [ray_tracing.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/ray_tracing.ipynb), [sample_scatter_plots.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/sample_scatter_plots.ipynb), [scipywrapper_demo.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/scipywrapper_dependence_custom/scipywrapper_demo.ipynb), [some_true_measures.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/some_true_measures.ipynb), [statistics_for_TrueMeasure.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/statistics_for_TrueMeasure.ipynb), [sorokin_thesis_2025.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/talk_paper_demos/SorokinThesis2025/sorokin_thesis_2025.ipynb), [pydata_chi_2023.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/talk_paper_demos/pydata_chi_2023.ipynb), [why_add_q_to_mc_blog.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/talk_paper_demos/why_add_q_to_mc_blog/why_add_q_to_mc_blog.ipynb)
- **Extra pip bootstrap** (4): [iris.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/iris.ipynb), [joss2026.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/talk_paper_demos/JOSS2026/joss2026.ipynb), [MCQMC_2020_QMC_Software_Tutorial.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/talk_paper_demos/MCQMC_Tutorial_2020/MCQMC_2020_QMC_Software_Tutorial.ipynb), [Sorokin_random_LD_seq_QMC_fast_kernel_methods_2026.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/talk_paper_demos/Sorokin_random_LD_seq_QMC_fast_kernel_methods_2026/Sorokin_random_LD_seq_QMC_fast_kernel_methods_2026.ipynb)
- **LaTeX bootstrap** (4): [dakota_genz.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/DAKOTA_Genz/dakota_genz.ipynb), [elliptic-pde.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/elliptic-pde.ipynb), [vectorized_qmc.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/vectorized_qmc.ipynb), [vectorized_qmc_bayes.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/vectorized_qmc_bayes.ipynb)
- **Repo-local bootstrap** (7): [gbm_examples.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/GBM/gbm_examples.ipynb), [accuracy_and_resume.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/demo_resume_data/accuracy_and_resume.ipynb), [resume_examples.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/demo_resume_data/resume_examples.ipynb), [01_sequential.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/talk_paper_demos/Parslfest_2025/01_sequential.ipynb), [02_parallel.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/talk_paper_demos/Parslfest_2025/02_parallel.ipynb), [03_visualize_speedup.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/talk_paper_demos/Parslfest_2025/03_visualize_speedup.ipynb), [01_sequential_output.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/talk_paper_demos/Parslfest_2025/output/01_sequential_output.ipynb)
- **Repo-local bootstrap + extra pip installs** (1): [gbm_demo.ipynb](https://colab.research.google.com/github/QMCSoftware/QMCSoftware/blob/develop/demos/GBM/gbm_demo.ipynb)

#### `make coverage`
Displays the current coverage report (must run other targets first to accumulate coverage data).
- **Output**: Terminal summary of coverage percentages per file/module
- **Note**: Coverage data is appended from previous runs; use `make delcoverage` to reset

#### `make delcoverage`
Deletes `.coverage` and `coverage.json` files to reset coverage tracking.
- **Use before**: Running a fresh coverage report without accumulated data

---

## Currently Active Targets: Justification

---

### Target Dependency Graph

```
tests_fast ⚡ (RECOMMENDED)
├── doctests_no_docker
│   ├── doctests_minimal
│   ├── doctests_torch
│   ├── doctests_gpytorch
│   └── doctests_botorch
├── unittests
└── booktests_no_docker
    ├── check_booktests
    ├── generate_booktests
    └── clean_local_only_files

tests_no_docker
├── doctests_no_docker
├── unittests
└── coverage

tests (full with Docker)
├── doctests
│   ├── doctests_markdown
│   ├── doctests_minimal
│   ├── doctests_torch
│   ├── doctests_gpytorch
│   ├── doctests_botorch
│   └── doctests_umbridge
├── unittests
└── coverage

booktests_parallel_no_docker
├── check_booktests
├── generate_booktests
├── clean_local_only_files
└── [Parsl distributed execution]
```

### Automatic Parallel Execution
- If `pytest-xdist` is installed, tests run with `-n auto` (detected by `scripts/pytest_xdist.py`)
- Override: `make PYTEST_XDIST="-n 4" unittests` to force 4 workers

### Selective Notebook Tests
Run specific notebook tests:
```bash
make booktests_no_docker TESTS="tb_quickstart tb_pricing_options"
```

### Environment Cleanup
The test targets automatically call `make clean_local_only_files` to remove corrupted distribution artifacts (e.g., invalid seaborn entries).

## Reproducibility

All tests use deterministic seeds (e.g., `seed=42`, `seed=7`) where applicable to ensure reproducible results across runs.

## Workflow Recommendations

### For Development
```bash
make unittests              # Quick feedback (13–30s)
make doctests_no_docker     # Validate docstrings (15–30s)
```

### Before Committing
```bash
make tests_fast             # Comprehensive, fast (30–60s)
```

### For CI/CD or Full Validation
```bash
make tests_no_docker        # Sequential, safe (60–120s)
```

## Troubleshooting

**Issue**: Tests fail with "command not found" or pytest not recognized
- **Solution**: Ensure the conda environment is activated: `conda activate qmcpy`

**Issue**: Tests are slow or not parallelized
- **Solution**: Install `pytest-xdist`: `pip install pytest-xdist`

**Issue**: Warnings about invalid distributions
- **Solution**: Cleanup runs automatically; if warnings persist, recreate the environment or reinstall the affected package

**Issue**: Coverage numbers seem low or cumulative
- **Solution**: Reset coverage with `make delcoverage`, then run tests

---

## Coverage Report Strategy

### Overview
QMCSoftware uses a **multi-platform unified coverage report** approach in GitHub Actions CI. Coverage data from all test types (doctests, unittests, booktests) running on all platforms (Ubuntu, macOS, Windows) is combined into a single coverage percentage.

### Official Coverage Metric (Unit Tests Only)

Although QMCSoftware executes **doctests**, **unit tests**, and **booktests**, only **unit tests** are used as the **official coverage metric** for evaluating code coverage.

This distinction is intentional and follows standard software engineering best practices.

#### Rationale

- **Unit tests (`test/`)** are:
  - Deterministic and isolated
  - Designed to explicitly exercise control flow, edge cases, and error paths
  - Stable across platforms and environments
  - The industry-standard basis for coverage metrics

- **Doctests** validate correctness of documented examples, but:
  - Often execute high-level workflows
  - May implicitly cover large portions of code without asserting behavior
  - Can inflate coverage without increasing test rigor

- **Booktests (notebook tests)**:
  - Execute demonstration notebooks end-to-end
  - Are environment- and runtime-dependent
  - Primarily validate documentation and reproducibility, not fine-grained logic

For these reasons, **doctests and booktests are excluded from the official coverage percentage** to avoid overstating test completeness.

#### Local vs CI Coverage

- **Local development coverage**:
  - Developers may accumulate coverage across test types for diagnostic purposes
  - HTML reports may include additional lines exercised by doctests or booktests

- **Official coverage evaluation**:
  - Coverage reported for review, grading, or quality metrics is derived **only from unit tests**
  - This ensures coverage reflects intentional, maintainable, and reviewable tests

This policy ensures that coverage metrics remain **meaningful, reproducible, and aligned with software quality goals**, rather than reflecting incidental execution paths.


### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Matrix Job: tests (windows-latest, macos-latest, ubuntu-latest)  │
├─────────────────────────────────────────────────────────┤
│  1. Clean old coverage files (.coverage*, coverage.json)│
│  2. Run doctests (with --cov-append)                    │
│  3. Run unittests (with --cov-append)                   │
│  4. Run booktests (with --cov-append)                   │
│  5. Upload .coverage & coverage.json as artifacts       │
└─────────────────────────────────────────────────────────┘
```

### Key Syntax & Configuration

#### 1. Coverage Configuration (`.coveragerc`)
Cross-platform coverage combining requires relative paths to handle different OS path formats (Windows `C:\`, macOS `/Users/`, Ubuntu `/home/`):

**`.coveragerc` settings:**
- `relative_files = True` – Store paths relative to project root
- `[paths]` section maps all OS path variants to common source location

#### 2. Makefile Test Targets (Coverage Append Mode)
All test targets use `--cov-append` (pytest) or `coverage run --append` to accumulate coverage within each OS runner:

**Doctest targets:** `doctests_minimal`, `doctests_torch`, `doctests_gpytorch`, `doctests_botorch`, `doctests_umbridge` – all use `--cov-append`

**Unit test target:** `unittests` – uses `--cov-append`

**Notebook test targets:**
- `booktests_no_docker` – uses `coverage run --append`
- `booktests_parallel_no_docker` – Parsl runner internally uses `coverage run --append`
- `booktests_parallel_pytest` – uses `--cov-append`

**Key flags:**
- `--cov qmcpy/` – Target package for coverage measurement
- `--cov-append` – Append to existing `.coverage` data (pytest-cov)
- `coverage run --append` – Append mode for unittest-based tests
- `--cov-report term` – Terminal output after each test run
- `--cov-report json` – Generate `coverage.json` for tracking

#### 3. GitHub Actions Workflow

- **Clean coverage at start of each matrix job:**

## CI & Coverage (summary)

- **GitHub Actions:** The main CI workflow is `.github/workflows/alltests.yml` (referred to in this document as `alltests.yml`). It runs a matrix across OSes, and calls Makefile targets.

    _Note_: The project CI is configured to upload coverage to Codecov.

A second workflow, `.github/workflows/unittests.yml`, runs a matrix across Python versions and is triggered by updates to `develop` and `master` branches only.

### Local Coverage Workflow

**Run tests and view coverage locally:**

- Clean old coverage
- Run tests (accumulates coverage with --cov-append)


### Benefits

1. **Comprehensive test coverage** – Includes doctests, unittests, and notebook tests
2. **Artifact persistence** – HTML and XML reports available 
3. **Incremental local testing** – `--cov-append` allows building coverage across multiple test runs
4. **CI/CD integration ready** – XML output compatible with Codecov, Coveralls, etc.

### Troubleshooting

**Coverage numbers seem wrong or incomplete:**
- Run `make delcoverage` to clean old data before starting fresh
- Ensure all test commands use `--cov-append` or `coverage run --append`

**Coverage combining fails locally:**
- Ensure `coverage` package is installed: `pip install coverage`

---

## See Also

- `Makefile` – Full test target definitions (in project root)
- `.github/workflows/alltests.yml` – CI all test workflow
- `.github/workflows/unittests.yml` - CI unit test workflow
- `make clean_local_only_files` – Artifact cleanup utility
- `scripts/pytest_xdist.py` – Parallel execution detection helper
