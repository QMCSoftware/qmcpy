# Notebook Tests in QMCPy

## Contents

- **`.gitignore`**: ignores temporary files and generated test outputs in this directory.
- **`__init__.py`**: package marker for the `test.booktests` module.
- **`generate_test.py`**: script that generates `tb_*.py` test files from notebooks in `demos/`.
- **`parsl_test_runner.py`**: helper harness used to run Parsl-based notebook tests and coordinate workers.
- **`README.md`**: this documentation file describing how to run and manage the notebook tests.

## Overview

To execute an individual testbook file, e.g., `tb_Sorokin_random_LD_seq_QMC_fast_kernel_methods_2026.py`, run the following command in a terminal:

```{bash}
    cd test/booktests && python -m pytest tb_Sorokin_random_LD_seq_QMC_fast_kernel_methods_2026.py -v
```

To execute all testbook files sequentially, run the following command in a terminal:

```{bash}
    make booktests_no_docker
```

To execute all testbook files in parallel using Parsl, run the following command in a terminal:

```{bash}
    make booktests_parallel_no_docker
```

To execute say two testbook files sequentially, run the following command in a terminal:

```{bash}
    make booktests_no_docker TESTS="tb_Argonne_2023_Talk_Figures tb_Purdue_Talk_Figures"
```

To execute say two testbook files in parallel, run the following command in a terminal:

```{bash}
    make booktests_parallel_no_docker TESTS="tb_Argonne_2023_Talk_Figures tb_Purdue_Talk_Figures"
```

For a demo, see the Jupyter notebook, `demos/talk_paper_demos/Parslfest_2025/`.


## Design Patterns: Using `setUp/helpers` vs. `@testbook` Decorator**

*   Generated files such as `tb_iris.py` uses `@testbook` for a standalone notebook, requiring no special setup.
*   GBM notebooks such as `gbm_examples.py` rely on local modules and sometimes have broken symlinks, needing setup for correct imports.
*   Running notebooks from their directory (via `setUp`) ensures consistent relative paths and imports, which the decorator doesn't reliably handle.
*   `BaseNotebookTest`'s `setUp/tearDown` methods handle resource management, beneficial for long-running demos.
*   Shared helper functions in `__init__.py` centralize complex logic (location, symlink fixing, running) for reusability and clarity.
