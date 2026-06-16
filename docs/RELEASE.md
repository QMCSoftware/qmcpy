# QMCPy Release Documentation

The following is a guide for those in charge of making new releases of QMCPy.

## Setup

In order to publish a release, you will need to have an account on the two websites:

- https://pypi.org
- https://test.pypi.org

Notes:

- These accounts will be different from each other. PyPI is the standard Python package index where users can `pip install qmcpy` from. TestPyPI is a separate package index meant for developers to test their release before publishing on the actual PyPI. In order to release `qmcpy`, you will need to contact Aleksei to request access to the actual `qmcpy` project on both websites.
- When setting up your PyPI account, please have another team member invite you as a collaborator. However, you will need to have two factor authentication (2FA) setup for your account before you can be added. We will not go over the steps on setting up 2FA as that depends on the vendor and method (mobile app or security key) you choose. Please research on the setup procedure for the 2FA vendor and method you pick. Some common 2FA authenticators you can install on your smartphone include Google Authenticator and Microsoft Authenticator.

Please setup and save your PyPI and TestPyPI API keys (do not commit to a public git repo). They should look something like the following 

```
[pypi]
  username = __token__
  password = ...

[testpypi]
  username = __token__
  password = ...
```


## Sanity Checks


```bash
git checkout develop
git pull 
pip install -e .
make doctests_no_docker
make doc
```

Ensure all tests are passing and the docs are compiling locally and look good. Again, if the GitHub actions are passing then of course everything should work locally, but it is always good to check. 

## TestPyPI release 

To set the release version, edit the `__version__` attribute in `qmcpy/__init__.py`. Note that versions ending in a letter, e.g. `2.1.1a`, are alpha releases that are not automatically downloaded. Note also that the version number must be greater than the current version number. For example, if the latest version was `2.1` then you could do `2.1.1` or `2.2` but cannot go back to `2.0.1`. See [this guide of Python versioning](https://packaging.python.org/en/latest/discussions/versioning/) for more details. 

Please install the [PDM](https://pdm-project.org/latest/) Python package and dependency manager, the build system we use as specified in the `pyproject.toml` file.

With PDM installed, you can follow the [PDM publish guide](https://pdm-project.org/latest/usage/publish/). Specifically, we first publish to TestPyPI with the command 

```bash 
pdm publish --repository testpypi
```

When prompted for a username put `__token__`, and when prompted for a password put the password for your API token for TestPyPI. If successful, you will get a link to the release on TestPyPI. Note that locally, your build may be found at `QMCSoftware/dist`. 

To actually test our TestPyPI release, it is a good idea to pretend you are a new user and create a fresh environment in which you try some basic QMCPy commands. Here are some commands to run to create a fresh environment and run some basic tests 

```bash 
conda create --name tmp python=3.13
conda activate tmp 
pip install -i https://test.pypi.org/simple/ qmcpy==???
python
```

Of course you will need to replace ??? with the version name. If the `pip install -i ...` command fails, try 

```bash 
pip install --index-url https://pypi.org/simple --extra-index-url https://test.pypi.org/simple "qmcpy==2.1"
```

Once in the Python console in the `tmp` environment, try running some basic commands such as 

```python
import qmcpy as qp 
import numpy as np
qp.DigitalNetB2(3)(8) 
qp.DigitalNetB2(3,replications=2)(8) 
qp.KernelShiftInvar(2)(np.random.rand(3,1,2),np.random.rand(1,4,2))
exit()
```

It is also good to check things work with torch. Still in the `tmp` env, run 

```bash 
pip install torch 
python
```

Back in the Python console, run 

```python
import qmcpy as qp 
import torch 
qp.fwht_torch(torch.rand(8))
qp.KernelShiftInvar(2,torchify=True)(torch.rand(3,1,2),torch.rand(1,4,2))
```

If you find errors, please fix them and try another release on TestPyPI until everything looks good. Note that you will need to increase the version number if making a new release, so it is good practice to make only small increments, e.g., if you want to release version `2.2` but that one failed, make the next try `2.2.0.1` and the one after that `2.2.0.2` and so on. Otherwise, if you do say `2.3`, then the next time someone goes to make a release of `2.3` it will give errors because that version has already been released on TestPyPI. 

## PyPI Release 

Back in the `qmcpy` environment, to make the actual PyPI release that users `pip install qmcpy` with, it is as simple as running 

```bash 
pdm publish
```

As with TestPyPI, the username is `__token__` and you will need your PyPI API password. 

Back in your `tmp` environment, you may try a 

```bash 
pip install qmcpy
python
```

just to make sure everything gets installed ok. In the Python console, you can run the same sanity checks as above, but it is probably not necessary. It is probably sufficient just to make sure the correct version is installed with 

```python 
import qmcpy as qp
qp.__version__
```


## Pull `develop` branch into `master` 

Make a pull request from `develop` into `master` and merge it in. 

## GitHub Release 

The PyPI release contains barebones tools from `QMCSoftware/qmcpy/`. The GitHub release contains additional project components such as tests and demos. To create a GitHub release, navigate to [QMCPy's GitHub release page](https://github.com/QMCSoftware/QMCSoftware/releases) and click `Draft a new release`. For `Tag:Select tag`, click `Create new tag` and enter, for example, `v2.1`. The target should be the master branch. Give the release a title like `QMCPy v2.1`. Click `Generate release notes` to automatically collect PR and other release notes. 

## Cleanup 

Delete the `tmp` environment with 

```bash 
conda env remove --name tmp 
```

## Advertising QMCPy

Please share the QMCPy release with team members and the greater community if possible. In the past we have posted both the PyPI and GitHub release links to the group on Slack.  


## Additional Post-Release Tasks

- Update this document if necessary.
- Draft an announcement in `docs/release` for NA Digest, `qmcpy.org`, etc.  Add it to `mkdocs.yml` for team to review.
- Coordinate branch cleanup and communicate to contributors.
- Review Release Issue; close or reassign sub‑issues to a future release as appropriate.
- If critical issues appear, prepare and publish a patch release (e.g., vX.Y.1).

