# Contributing

Thank you for your interest in contributing to the QMCPy library! This library is the product of many hours of labor from many contributors. Join team communications by reaching out to us at [qmc-software@googlegroups.com](mailto:qmc-software@googlegroups.com).

## Good Practices

To preserve the integrity of this library, we have instituted some good practices for developing features, improving performance, and fixing bugs. Please read [this document](docs/good_practices.md) to acquaint yourself with them.

### AI-Assisted Contributions

QMCPy welcomes AI assistance, but contributors and reviewers remain responsible for correctness, reproducibility, licensing, and citations. If AI affects your code, tests, demos, documentation, or pull request text, follow the [AI-assisted contributions policy](docs/ai-assisted-contributions.md) and disclose that use in your pull request.

### Issues

All improvements to QMCPy should be connected to an **issue** using a template from `.github/ISSUE_TEMPLATE/`.

- If you are looking for a way to contribute, search the issues and contact the person who started the issue, if you would like to help.

- If you identify an improvement that is not in an issue, you may submit an issue yourself.

### Feature Branches

If you have not yet installed the QMCPy library, see [Installation](#installation) below.

You should do all your work on a feature branch that is created from the `develop` branch; see [Branches](#branches) below. Once you have something ready, submit a **pull request (PR)** to the `develop` branch and request reviews from at least two team members. Tools such as GitHub Copilot may provide supplemental feedback, but they do not replace human review or approval. It may help to have a brief PR review Zoom meeting with the code reviewers to walk them through the changes.

After a feature branch has been approved by two code reviewers, you may merge it into `develop`. After a successful merge, it is best practice to delete your feature branch on GitHub. This action keeps the repository tidy and prevents the accumulation of stale branches.

We periodically release the contents of `develop` to `master`. Contact the team for the next release date. Plan to submit your pull request to `develop` at least one week before the release date. If your contribution does not make it into the next release, we hope that it will make it into the one after that.

### Blogs

Blog prose is maintained in [`QMCSoftware/QMCSoftware.github.io`](https://github.com/QMCSoftware/QMCSoftware.github.io), not in this repository's MkDocs documentation site. Propose and publish blog posts there; published articles appear on the [QMCSoftware Blog](https://qmcsoftware.org/blogs/).

Runnable examples remain in this repository. For every article backed by a notebook:

1. Keep the executable notebook under `demos/` as the reproducible source and link the article to that exact repository path. Moving the article prose to the website is not a reason to delete its notebook.
2. Edit and execute the notebook in the QMCPy development environment, from its containing directory when it uses relative imports or helper files.
3. Add or update the matching `test/booktests/tb_*.py` test and run that focused test before updating the website article. See [`test/booktests/README.md`](https://github.com/QMCSoftware/QMCSoftware/blob/develop/test/booktests/README.md) for commands.
4. Update the website article separately, then verify that its source-notebook link still resolves.
5. If the article had a page on this repository's MkDocs site, do not just delete it: add a `redirect_maps` entry for its old path under the `redirects` plugin in `mkdocs.yml`, then confirm with `make check_removed_urls`.


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

The `dev` extra includes QMCPy's PyPI-hosted MPMC dependencies. MPMC additionally
requires a platform-specific `pyg_lib` wheel that is not available from PyPI.
After installing `dev`, let the QMCPy installer select the wheel page matching
the installed PyTorch build:

~~~bash
qmcpy-install-mpmc
~~~

For an MPMC installation without the complete development environment, use:

~~~bash
pip install -e ".[mpmc]"
qmcpy-install-mpmc
~~~

## 📚 Using `qmcpy` In Courses (`class` Extra)

`qmcpy` provides a `class` optional dependency group that installs a complete teaching environment (JupyterLab, plotting, statistics, and utilities) in addition to `qmcpy` itself.

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

## Branches

### For Main Repository Collaborators

Branch directly from `develop` inside the `QMCSoftware/` repository. This allows other team members to easily review your work by checking out your branch with

```bash
git fetch origin
git checkout <branch-name>
```

### For External Contributors (Forks)

Fork the repository to your personal account and create your branch there. Main repository collaborators can review or test your forked branch without having to clone your repo. For example, say a main repository collaborator wants to check out the `develop` branch on the `git@github.com:MyGitHubUsername/QMCSoftware.git` fork. The main repository contributor may call this remote fork the `MyGitHubUsername-fork` and call the branch name `MyGitHubUsername-develop` within our repo to avoid conflict with the origin `develop` branch. The following commands accomplish this.

```bash
# Add the fork as a remote source
git remote add MyGitHubUsername-fork git@github.com:MyGitHubUsername/QMCSoftware.git

# Download the fork's branch data
git fetch MyGitHubUsername-fork

# Create your local branch tracking the fork's branch
git checkout -b MyGitHubUsername-develop MyGitHubUsername-fork/develop
```

When new changes are pushed to the `develop` branch on the fork `git@github.com:MyGitHubUsername/QMCSoftware.git`, the main repo collaborator may then run

```bash
# 1. Switch to the local branch tracking your fork
git checkout MyGitHubUsername-develop

# 2. Pull the new changes directly from your fork's branch
git pull MyGitHubUsername-fork develop
```

## Tests

Doctests and unittests take a few minutes to run with

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

### Ensure `pyreverse` Is On Your PATH

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

### Build the Documentation

On MacOS / Linux (and on Windows via Git Bash, WSL, or any environment with `make`):

~~~bash
make doc
~~~

### Download PDF Documentation

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
