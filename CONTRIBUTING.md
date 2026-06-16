# Contributing

Thank you for your interest in contributing to the QMCPy package!

Please submit **pull requests (PRs)** to the `develop` branch and **issues** using a template from `.github/ISSUE_TEMPLATE/`

After a feature branch has been successfully merged, it is best practice to delete it. This action keeps the repository tidy and prevents the accumulation of stale branches.

For planned releases, open related PRs with enough lead time so review can begin at least one week before the release date.

For complex or mathematical contributions, schedule at least one PR review meeting before merging. 

If you develop a new component, please consider writing a blog for the[qmcpy documentation](https://qmcsoftware.github.io/QMCSoftware/) including a brief summary of the mathematical rationale, key assumptions,  validation evidence (tests, benchmarks, or references), and examples.

Join team communications by reaching out to us at [qmc-software@googlegroups.com](mailto:qmc-software@googlegroups.com)

## Installation

In a git enabled terminal (e.g. [bash](https://gitforwindows.org/) for Windows) with [miniconda](https://docs.conda.io/en/latest/miniconda.html) installed and C compilers enabled (Windows users may need to install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)), run

~~~bash
git clone https://github.com/QMCSoftware/QMCSoftware.git
cd QMCSoftware
git checkout develop
conda create --name qmcpy python=3.13
conda activate qmcpy
pip install -e .[dev]
~~~

While `dev` contains the most complete set of install dependencies, a number of other install dependency groups can be found in our `pyproject.toml` file. If running in the `zsh` terminal you may need to use

~~~bash
pip install -e ".[dev]"
~~~

### 📚 Using `qmcpy` in courses (`class` extra)

`qmcpy` provides a `class` optional dependency group that installs a
complete teaching environment (JupyterLab, plotting, statistics, and utilities)
in addition to `qmcpy` itself.

For a typical course setup, you can do:
```bash
git clone https://github.com/QMCSoftware/QMCSoftware.git
cd QMCSoftware
pip install -e ".[class]"
```

or for a heavy-duty version
```bash
pip install -e ".[class,dev]"
```

## Tests

Doctests and unittests take a few minute to run with

~~~bash
pip install -e ".[dev,docs,test]"
make tests_no_docker
~~~

Optionally, you may install [Docker](https://www.docker.com/products/docker-desktop/) and then run all tests with

~~~bash
make tests
~~~

Please see the targets in the makefile for more granular control over tests.

## Documentation

### Ensure `pyreverse` is on your PATH

`pyreverse` must be available as a command-line tool. If it is not, verify your PATH as below.

* MacOS / Linux 

~~~bash
conda activate qmcpy
# check that pyreverse is found
which pyreverse || echo "pyreverse not found"
pyreverse --help
~~~

Alternative:
	
~~~bash
	# add user scripts dir to PATH (zsh example; use ~/.bashrc for bash)
	echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
	# preferred: open a new terminal so the new PATH is picked up
	source ~/.zshrc
~~~

* Windows (cmd or PowerShell)

~~~powershell
conda activate qmcpy
# check that pyreverse is found
where pyreverse
pyreverse --help
~~~

If `where pyreverse` cannot find the command, ensure your Python Scripts directory is on your `PATH`. A common way to locate it is:

~~~powershell
python -m site --user-base
# then add "<that-path>\Scripts" to your PATH
~~~

You can update PATH via System settings or in your PowerShell profile (`$PROFILE`).

### Build the documentation

On MacOS / Linux (and on Windows via Git Bash, WSL, or any environment with `make`):

~~~bash
make doc
~~~

### Download PDF documentation

In the built HTML documentation:

1. Navigate to the **“Printable Docs”** section.
2. Use your browser’s print dialog:
   - **Windows / Linux:** `Ctrl+P`  
   - **MacOS:** `Cmd+P`
3. Choose **“Save as PDF”** and save to your preferred location.


## Demos

Demos are Jupyter notebooks which may be launched using the command

~~~bash
jupyter-lab
~~~

## Other Developer Tools

The [Developers Tools](https://qmcpy.org/references-for-python-and-mathematical-software-development/) page on [qmcpy.org](https://qmcpy.org) documents additional tools we have found helpful for mathematical software development and presentation.

## VSCode Tips

[VSCode](https://code.visualstudio.com) (Visual Studio Code) is the IDE of choice for many of our developers. Here we compile some helpful notes regarding additional setup for VSCode.

- Run `CMD`+`p` then `> Python: Select Interpreter` then select the `('qmcpy')` choice from the dropdown to link the qmcpy environment into your workspace. Now when you open a terminal, your command line should read `(qmcpy) username@...` which indicates the qmcpy environment has been automatically activated. Also, when debugging the qmcpy environment will be automatically used.
- Go to `File` and click `Save Workspace as...` to save a `qmcpy` workspace for future development.

Some VSCode extensions we found useful include

- Python
- Jupyter
- Markdown Preview Enhanced
- eps-preview, which requires
    - Postscript Language
    - pdf2svg
- Git Graph
- Code Spell Checker
