# Emit pytest-xdist argument if available; can be overridden on the make command line
PYTEST_XDIST ?= $(shell python scripts/pytest_xdist.py 2>/dev/null)
PYTEST ?=
PYTHON ?= python3
SMOKE_CODE_CELLS ?= 2
WITH_MPMC ?= 0
HAS_MPMC ?= $(shell python -c "import importlib.util; mods=('torch','pyg_lib','torch_geometric'); print(int(all(importlib.util.find_spec(m) is not None for m in mods)))" 2>/dev/null || echo 0)

# set environment variable for documentation
export JUPYTER_PLATFORM_DIRS=1

##########################################################
# Coverage artifacts (local-only; should be gitignored)
##########################################################
ARTIFACTS_DIR ?= artifacts
COV_DIR ?= $(ARTIFACTS_DIR)/coverage
LOG_DIR ?= $(ARTIFACTS_DIR)/logs

UNIT_COV_DIR ?= $(COV_DIR)/unit
DOCTEST_COV_DIR ?= $(COV_DIR)/doctests
BOOKTEST_COV_DIR ?= $(COV_DIR)/booktests

ensure_artifacts:
	@mkdir -p $(UNIT_COV_DIR) $(DOCTEST_COV_DIR) $(BOOKTEST_COV_DIR) $(LOG_DIR)

##########################################################
# utilities
##########################################################
# This helps locate generated or local-only folders like build, .pytest_cache, etc.
find_local_only_files:
	chmod +x scripts/find_local_only_folders.sh
	./scripts/find_local_only_folders.sh

clean_local_only_files:
	rm -fr test/booktests/.ipynb_checkpoints/ .pytest_cache/ .ruff_cache/ __pycache__/ */__pycache__/ */*/__pycache__/ raw.githubusercontent.com/ */raw.githubusercontent.com/ */*/raw.githubusercontent.com/ site/ build/ .pdm-build/ artifacts/logs/ artifacts/booktests/ */*/logs/ */*/runinfo/
	chmod +x scripts/find_local_only_folders.sh > /dev/null 2>&1
	for f in $(shell ./scripts/find_local_only_folders.sh > /dev/null 2>&1); do \
		rm -f "$$f"; > /dev/null 2>&1; \
	done

clean_coverage:
	rm -fr artifacts/coverage/ .coverage* test/booktests/.coverage*

TEST_STYLE_PATH ?= test
# Check test/test_*.py against two suite conventions: (1) written as a
# unittest.TestCase subclass ("object class"), not bare pytest functions;
# (2) named test_<area>_*.py where <area> is the qmcpy subpackage under test
# (dd ft ig kn sc tm ut) or a cross-cutting bucket (ee sr).
# Informational by default; pass --strict to make it fail
# (e.g. STRICT=--strict make check_test_style).
check_test_style:
	@$(PYTHON) scripts/check_test_style.py $(TEST_STYLE_PATH) $(STRICT)

##########################################################
# Doctests
##########################################################
doctests_minimal: ensure_artifacts
	@mkdir -p $(DOCTEST_COV_DIR)/minimal
	COVERAGE_FILE=$(DOCTEST_COV_DIR)/minimal/.coverage \
	python -m pytest $(PYTEST_XDIST) -x --cov qmcpy/ --cov-report term --cov-report json:$(DOCTEST_COV_DIR)/minimal/coverage.json --no-header --cov-append \
		--doctest-modules qmcpy/ \
		--ignore qmcpy/fast_transform/ft_pytorch.py \
		--ignore qmcpy/stopping_criterion/pf_gp_ci.py \
		--ignore qmcpy/kernel/ \
		--ignore qmcpy/util/dig_shift_invar_ops.py \
		--ignore qmcpy/util/shift_invar_ops.py \
		--ignore qmcpy/util/exact_gpytorch_gression_model.py \
		--ignore qmcpy/integrand/umbridge_wrapper.py \
		--ignore qmcpy/integrand/hartmann6d.py \
		--ignore qmcpy/discrete_distribution/mpmc/ \

doctests_torch: ensure_artifacts
	@mkdir -p $(DOCTEST_COV_DIR)/torch
	COVERAGE_FILE=$(DOCTEST_COV_DIR)/torch/.coverage \
	python -m pytest $(PYTEST_XDIST) -x --cov qmcpy/ --cov-report term --cov-report json:$(DOCTEST_COV_DIR)/torch/coverage.json --no-header --cov-append \
		--doctest-modules qmcpy/fast_transform/ft_pytorch.py \
		--doctest-modules qmcpy/kernel/*.py \
		--doctest-modules qmcpy/util/dig_shift_invar_ops.py \
		--doctest-modules qmcpy/util/shift_invar_ops.py \

doctests_gpytorch: ensure_artifacts
	@mkdir -p $(DOCTEST_COV_DIR)/gpytorch
	COVERAGE_FILE=$(DOCTEST_COV_DIR)/gpytorch/.coverage \
	python -m pytest $(PYTEST_XDIST) -x --cov qmcpy/ --cov-report term --cov-report json:$(DOCTEST_COV_DIR)/gpytorch/coverage.json --no-header --cov-append \
		--doctest-modules qmcpy/stopping_criterion/pf_gp_ci.py \

doctests_botorch: ensure_artifacts
	@mkdir -p $(DOCTEST_COV_DIR)/botorch
	COVERAGE_FILE=$(DOCTEST_COV_DIR)/botorch/.coverage \
	python -m pytest $(PYTEST_XDIST) -x --cov qmcpy/ --cov-report term --cov-report json:$(DOCTEST_COV_DIR)/botorch/coverage.json --no-header --cov-append \
		--doctest-modules qmcpy/integrand/hartmann6d.py \

doctests_mpmc:
	@mkdir -p $(DOCTEST_COV_DIR)/mpmc
	COVERAGE_FILE=$(DOCTEST_COV_DIR)/mpmc/.coverage \
	python -m pytest $(PYTEST_XDIST) -x --cov qmcpy/ --cov-report term --cov-report json:$(DOCTEST_COV_DIR)/mpmc/coverage.json --no-header --cov-append \
		--doctest-modules qmcpy/discrete_distribution/mpmc/*.py \

doctests_umbridge: ensure_artifacts # https://github.com/UM-Bridge/umbridge/issues/96
	@mkdir -p $(DOCTEST_COV_DIR)/umbridge
	@docker --version
	COVERAGE_FILE=$(DOCTEST_COV_DIR)/umbridge/.coverage \
	python -m pytest $(PYTEST_XDIST) -x --cov qmcpy/ --cov-report term --cov-report json:$(DOCTEST_COV_DIR)/umbridge/coverage.json --no-header --cov-append \
		--doctest-modules qmcpy/integrand/umbridge_wrapper.py \

doctests_markdown:
	@phmutest docs/*.md --replmode --log -c


doctests_no_docker_no_mpmc: doctests_minimal doctests_torch doctests_gpytorch doctests_botorch

doctests_no_docker: doctests_minimal doctests_torch doctests_gpytorch doctests_botorch doctests_mpmc

doctests_no_mpmc: doctests_minimal doctests_torch doctests_gpytorch doctests_botorch doctests_umbridge

doctests: doctests_markdown doctests_minimal doctests_torch doctests_gpytorch doctests_botorch doctests_umbridge doctests_mpmc

##########################################################
# Unit Tests in `test/` folder (OFFICIAL coverage)
##########################################################
unittests: ensure_artifacts
	@mkdir -p $(UNIT_COV_DIR)
	@PYTHON_BIN=$$(command -v python 2>/dev/null || { [ -n "$$CONDA_PREFIX" ] && command -v "$$CONDA_PREFIX/bin/python" 2>/dev/null; } || { command -v conda >/dev/null 2>&1 && conda run -n qmcpy python -c 'import sys; print(sys.executable)' 2>/dev/null; } || command -v python3 2>/dev/null); \
	if [ -z "$$PYTHON_BIN" ]; then \
		echo "No Python interpreter found (tried: python, $$CONDA_PREFIX/bin/python, python3)."; \
		exit 127; \
	fi; \
	COVERAGE_FILE=$(UNIT_COV_DIR)/.coverage \
	"$$PYTHON_BIN" -m pytest $(PYTEST_XDIST) -x $(PYTEST_EXTRA_ARGS) \
		--cov=qmcpy \
		--cov-report term \
		--cov-report json:$(UNIT_COV_DIR)/coverage.json \
		--no-header \
		test/ -W ignore::DeprecationWarning

# Core unit tests only: skips test/booktests/ (needs the notebook stack); other
# modules self-skip via pytest.importorskip. Pairs with the `test_core` extra so
# interpreters at the `requires-python` floor can run this. Unlike `unittests`
# this omits -x: on a compatibility run the full list of failures is the point.
unittests_core: ensure_artifacts
	@mkdir -p $(UNIT_COV_DIR)
	COVERAGE_FILE=$(UNIT_COV_DIR)/.coverage \
	python -m pytest $(PYTEST_XDIST) $(PYTEST_EXTRA_ARGS) \
		--cov=qmcpy \
		--cov-report term \
		--cov-report json:$(UNIT_COV_DIR)/coverage.json \
		--no-header -rs \
		--ignore=test/booktests \
		test/ -W ignore::DeprecationWarning

tests_no_docker_no_mpmc: doctests_no_docker_no_mpmc unittests coverage

##########################################################
# Unit Tests for `*.ipynb` in `demos/` folder
##########################################################
generate_booktests:
	@echo "\nGenerating missing booktest files..."
	cd test/booktests/ && python generate_test.py --check-missing

check_colab_notebooks:  # faster
	$(PYTHON) -m scripts.check_colab_notebooks --strict

check_colab_notebooks_smoke:  # slower; executes bootstrap + a few cells of every enabled notebook
	$(PYTHON) -m scripts.smoke_test_colab_notebooks --cells-after-bootstrap $(SMOKE_CODE_CELLS)

harden_colab_notebook:  # Add Colab button if necessary
	@if [ -n "$(NOTEBOOK)" ]; then \
		if [ -n "$(FORCE)" ]; then \
			$(PYTHON) -m scripts.harden_colab_notebook --notebook "$(NOTEBOOK)" --force; \
		else \
			$(PYTHON) -m scripts.harden_colab_notebook --notebook "$(NOTEBOOK)"; \
		fi; \
	elif [ -n "$(FORCE)" ]; then \
		$(PYTHON) -m scripts.harden_colab_notebook --force; \
	else \
		$(PYTHON) -m scripts.harden_colab_notebook --all-unclassified; \
	fi

report_colab_notebook_patterns:
	$(PYTHON) -m scripts.report_colab_notebook_patterns

open_colab_notebook:  # Open NOTEBOOK in Colab from the current branch, but only when it differs from COLAB_BASE (default develop); usage: make open_colab_notebook NOTEBOOK=demos/foo.ipynb [COLAB_BASE=develop]
	@nb="$(NOTEBOOK)"; nb="$${nb#./}"; base="$${COLAB_BASE:-develop}"; \
	if [ -z "$$nb" ]; then echo "Usage: make open_colab_notebook NOTEBOOK=demos/path/to.ipynb [COLAB_BASE=develop]"; exit 2; fi; \
	case "$$nb" in *.ipynb) ;; *) echo "Not a .ipynb file: $$nb"; exit 2;; esac; \
	branch=$$(git rev-parse --abbrev-ref HEAD); \
	slug=$$($(PYTHON) -c "import json; wprint(json.load(open('scripts/colab_notebooks_manifest.json'))['repo'])" 2>/dev/null); \
	[ -n "$$slug" ] || slug=$$(git remote get-url origin 2>/dev/null | sed -E 's#(git@github\.com:|https://github\.com/)##; s#\.git$$##'); \
	if [ -z "$$slug" ]; then echo "Cannot determine the GitHub owner/repo (manifest 'repo' or 'origin' remote)."; exit 1; fi; \
	git fetch -q origin "$$base" "$$branch" 2>/dev/null || true; \
	if ! git rev-parse -q --verify "origin/$$branch" >/dev/null; then \
		echo "Branch '$$branch' is not on origin -- push it first (Colab loads notebooks from GitHub)."; exit 1; \
	fi; \
	if ! git rev-parse -q --verify "origin/$$base" >/dev/null; then \
		echo "Base '$$base' is not a branch on origin -- set COLAB_BASE to a pushed branch (e.g. develop)."; exit 1; \
	fi; \
	if ! git ls-tree -r --name-only "origin/$$branch" | grep -qxF "$$nb"; then \
		echo "'$$nb' is not committed on origin/$$branch -- commit and push it first."; exit 1; \
	fi; \
	git diff --quiet "origin/$$branch" -- "$$nb" || \
		echo "note: local '$$nb' differs from origin/$$branch; Colab shows the pushed version."; \
	if [ "$$branch" = "$$base" ]; then \
		ref="$$base"; echo "On '$$base' -- opening the $$base version."; \
	elif ! git ls-tree -r --name-only "origin/$$base" | grep -qxF "$$nb"; then \
		ref="$$branch"; echo "'$$nb' is new (not on origin/$$base) -- opening the '$$branch' version."; \
	elif git diff --quiet "origin/$$base" "origin/$$branch" -- "$$nb"; then \
		ref="$$base"; echo "'$$nb' is unchanged vs origin/$$base -- the standard badge covers it; opening the $$base version."; \
	else \
		ref="$$branch"; echo "'$$nb' differs from origin/$$base -- opening the '$$branch' version."; \
	fi; \
	url="https://colab.research.google.com/github/$$slug/blob/$$ref/$$nb"; \
	echo "$$url"; \
	$(PYTHON) -m webbrowser "$$url" >/dev/null 2>&1 || echo "(could not auto-open a browser; copy the URL above)"

open_colab_notebook_gist:  # Upload NOTEBOOK from the working tree to a throwaway secret gist and open it in Colab (needs the gh CLI); usage: make open_colab_notebook_gist NOTEBOOK=demos/foo.ipynb
	@nb="$(NOTEBOOK)"; nb="$${nb#./}"; \
	if [ -z "$$nb" ]; then echo "Usage: make open_colab_notebook_gist NOTEBOOK=demos/path/to.ipynb"; exit 2; fi; \
	if [ ! -f "$$nb" ]; then echo "No such file: $$nb"; exit 2; fi; \
	case "$$nb" in *.ipynb) ;; *) echo "Not a .ipynb file: $$nb"; exit 2;; esac; \
	if ! command -v gh >/dev/null 2>&1; then \
		echo "The 'gh' CLI is required (https://cli.github.com), then run 'gh auth login'."; exit 1; \
	fi; \
	base=$$(basename "$$nb"); \
	url=$$(gh gist create --desc "qmcpy Colab preview of $$nb (safe to delete)" "$$nb") || exit 1; \
	id=$${url##*/}; \
	login=$$(gh api user -q .login 2>/dev/null); \
	colab="https://colab.research.google.com/gist/$${login:+$$login/}$$id/$$base"; \
	echo "gist (secret):    $$url"; \
	echo "colab:            $$colab"; \
	echo "delete when done: gh gist delete $$id"; \
	echo "note: sibling .py helpers won't resolve from a gist; the bootstrap cell falls back to develop."; \
	$(PYTHON) -m webbrowser "$$colab" >/dev/null 2>&1 || echo "(could not auto-open a browser; copy the colab URL above)"

open_notebook:  # Open NOTEBOOK from the working tree in local JupyterLab; usage: make open_notebook NOTEBOOK=demos/foo.ipynb
	@nb="$(NOTEBOOK)"; nb="$${nb#./}"; \
	if [ -z "$$nb" ]; then echo "Usage: make open_notebook NOTEBOOK=demos/path/to.ipynb"; exit 2; fi; \
	if [ ! -f "$$nb" ]; then echo "No such file: $$nb"; exit 2; fi; \
	case "$$nb" in *.ipynb) ;; *) echo "Not a .ipynb file: $$nb"; exit 2;; esac; \
	if command -v jupyter >/dev/null 2>&1; then exec jupyter lab "$$nb"; \
	else exec $(PYTHON) -m jupyterlab "$$nb"; fi

check_booktests:
	rm -fr demos/.ipynb_checkpoints/*checkpoint.ipynb && \
	find demos -name '*.ipynb' | while read nb; do \
		base=$$(basename "$$nb" .ipynb); \
		test_base=$$(echo "$$base" | sed 's/[-.]/_/g'); \
		if echo "$$nb" | grep -q "Parslfest_2025"; then \
			continue; \
		fi; \
		if ! ls test/booktests/tb_"$$test_base".py > /dev/null 2>&1; then \
			echo "    Missing test for: $$nb -> Expected: test/booktests/tb_$$test_base.py"; \
		fi; \
	done
	@echo "Total notebooks:  $$(find demos -name '*.ipynb' | wc -l)"
	@echo "Total test files: $$(find test/booktests -name 'tb_*.py' | wc -l)"

tests_no_mpmc: doctests_no_mpmc unittests coverage

booktests_no_docker: check_booktests generate_booktests clean_local_only_files ensure_artifacts
	@echo "\nNotebook tests"
	@mkdir -p $(BOOKTEST_COV_DIR)
	set -e && \
	cd test/booktests/ && \
	if [ -z "$(TESTS)" ]; then \
		PYTHONWARNINGS="ignore::UserWarning,ignore::DeprecationWarning,ignore::FutureWarning,ignore::ImportWarning" \
		COVERAGE_FILE=../../$(BOOKTEST_COV_DIR)/.coverage \
		python -W ignore -m coverage run --append --source=../../qmcpy/ -m unittest discover -s . -p "*.py" -v --failfast; \
	else \
		PYTHONWARNINGS="ignore::UserWarning,ignore::DeprecationWarning,ignore::FutureWarning,ignore::ImportWarning" \
		COVERAGE_FILE=../../$(BOOKTEST_COV_DIR)/.coverage \
		python -W ignore -m coverage run --append --source=../../qmcpy/ -m unittest $(TESTS) -v --failfast; \
	fi && \
	cd ../..

# coverage is done in function run_single_test() in test/booktests/parsl_test_runner.py
booktests_parallel_no_docker: check_booktests generate_booktests clean_local_only_files
	@echo "\nNotebook tests with Parsl"
	cd test/booktests/ && \
	rm -fr *.eps *.jpg *.pdf *.png *.part *.txt *.log && rm -fr logs && rm -fr runinfo prob_failure_gp_ci_plots && \
	PYTHONWARNINGS="ignore::UserWarning,ignore::DeprecationWarning,ignore::FutureWarning,ignore::ImportWarning" \
	python parsl_test_runner.py $(TESTS) -v --failfast && \
	cd ../..

# Windows-compatible parallel booktests using pytest-xdist instead of Parsl
booktests_parallel_pytest: check_booktests generate_booktests clean_local_only_files ensure_artifacts
	@mkdir -p $(BOOKTEST_COV_DIR)
	cd test/booktests/ && \
	PYTHONWARNINGS="ignore::UserWarning,ignore::DeprecationWarning,ignore::FutureWarning,ignore::ImportWarning" \
	COVERAGE_FILE=../../$(BOOKTEST_COV_DIR)/.coverage \
	python -W ignore -m pytest $(PYTEST_XDIST) $(PYTEST) -v tb_*.py \
		--cov=qmcpy \
		--cov-append \
		--cov-report=term \
		--cov-report=json:../../$(BOOKTEST_COV_DIR)/coverage.json && \
	cd ../..

##########################################################
# Combinations of Above Tests
##########################################################
tests:
	set -e && $(MAKE) doctests && $(MAKE) unittests && $(MAKE) coverage

tests_no_docker:
	@echo "Running environment cleanup for invalid distributions (dry-run will be skipped, applying changes)..."
	@if [ "$(WITH_MPMC)" = "1" ] || [ "$(HAS_MPMC)" = "1" ]; then \
		DOCTESTS_TARGET=doctests_no_docker; \
		UNITTESTS_ARGS=""; \
	else \
		DOCTESTS_TARGET=doctests_no_docker_no_mpmc; \
		UNITTESTS_ARGS="--ignore=test/test_dd_mpmc.py"; \
	fi && \
	set -e && $(MAKE) $$DOCTESTS_TARGET && $(MAKE) unittests PYTEST_EXTRA_ARGS="$$UNITTESTS_ARGS"

# Fast test target: run doctests, unittests, booktests concurrently
tests_fast:
	@echo "Running fast tests: doctests and unittests concurrently (splitting CPU cores)."
	@make clean_local_only_files clean_coverage && \
	if [ "$(WITH_MPMC)" = "1" ] || [ "$(HAS_MPMC)" = "1" ]; then \
		DOCTESTS_TARGET=doctests_no_docker; \
		UNITTESTS_ARGS=""; \
	else \
		DOCTESTS_TARGET=doctests_no_docker_no_mpmc; \
		UNITTESTS_ARGS="--ignore=test/test_dd_mpmc.py"; \
	fi && \
	set -e && \
	$(MAKE) $$DOCTESTS_TARGET & \
	$(MAKE) unittests PYTEST_EXTRA_ARGS="$$UNITTESTS_ARGS" & \
	$(MAKE) booktests_parallel_no_docker  & \
	wait
	$(MAKE) coverage

##########################################################
# Local Coverage Reports and Tools
##########################################################
coverage: ensure_artifacts # https://github.com/marketplace/actions/coverage-badge
	@echo ""
	@echo "============================================================"
	@echo "OFFICIAL COVERAGE REPORT (UNIT TESTS ONLY)"
	@echo "Source: $(UNIT_COV_DIR)/.coverage"
	@echo "Doctests and booktests are NOT included."
	@echo "============================================================"
	@echo ""
	COVERAGE_FILE=$(UNIT_COV_DIR)/.coverage \
	python -m coverage report -m

combine-coverage-local: ensure_artifacts  # Combine coverage files and build reports locally (NOT official)
	@echo "Combining coverage files from $(COV_DIR)/ into coverage-data/ and generating reports"
	@mkdir -p coverage-data && \
	rm -rf coverage-data/* && \
	# Prefer new artifact layout first
	find $(COV_DIR) -type f -name '.coverage*' -exec cp {} coverage-data/ \; 2>/dev/null || true; \
	find $(COV_DIR) -type f -name 'coverage.json' -exec cp {} coverage-data/ \; 2>/dev/null || true; \
	# Backwards-compat: if artifacts had nothing, try repo root
	if [ -z "$$(find coverage-data -type f \( -name '.coverage*' -o -name 'coverage.json' \) 2>/dev/null)" ]; then \
		echo "No coverage found under $(COV_DIR)/. Trying repo root (.coverage*, coverage.json) ..."; \
		cp -r .coverage* coverage.json coverage-data/ 2>/dev/null || true; \
	fi; \
	# Final check: must have something to combine
	if [ -z "$$(find coverage-data -type f -name '.coverage*' 2>/dev/null)" ]; then \
		echo "No coverage data found. Run tests first (e.g., make unittests / make doctests / make booktests_*)"; \
		exit 1; \
	fi; \
	python scripts/combine_coverage.py --dir coverage-data --outdir coverage_html --keep

coverage_html: ensure_artifacts
	@mkdir -p $(UNIT_COV_DIR)/html
	@echo ""
	@echo "============================================================"
	@echo "GENERATING UNIT TEST COVERAGE HTML (OFFICIAL)"
	@echo "Output: $(UNIT_COV_DIR)/html/index.html"
	@echo "============================================================"
	@echo ""
	COVERAGE_FILE=$(UNIT_COV_DIR)/.coverage \
	python -m coverage html -d $(UNIT_COV_DIR)/html

delcoverage:
	@rm -f .coverage coverage.json test/booktests/.coverage
	@rm -rf $(COV_DIR)
	@rm -rf .pytest_cache

##########################################################
# Make UML class diagrams
##########################################################
uml:
	# UML Diagrams
	#	Discrete Distributions
	@pyreverse -k qmcpy/discrete_distribution/ -o svg 1>/dev/null && mv classes.svg docs/api/umls/discrete_distribution_overview.svg
	#	Discrete Distribution Specific
	@pyreverse qmcpy/discrete_distribution/ -o svg 1>/dev/null && mv classes.svg docs/api/umls/discrete_distribution_specific.svg
	#	True Measure Overview
	@pyreverse -k qmcpy/true_measure/ -o svg 1>/dev/null && mv classes.svg docs/api/umls/true_measure_overview.svg
	#	True Measure Specific
	@pyreverse qmcpy/true_measure/ -o svg 1>/dev/null && mv classes.svg docs/api/umls/true_measure_specific.svg
	#	Integrand Overview
	@pyreverse -k qmcpy/integrand/ -o svg 1>/dev/null && mv classes.svg docs/api/umls/integrand_overview.svg
	#	Integrand Specific
	@pyreverse qmcpy/integrand/ -o svg 1>/dev/null && mv classes.svg docs/api/umls/integrand_specific.svg
	#	Stopping Criterion Overview
	@pyreverse -k qmcpy/stopping_criterion/ -o svg 1>/dev/null && mv classes.svg docs/api/umls/stopping_criterion_overview.svg
	#	Stopping Criterion Specific
	@pyreverse qmcpy/stopping_criterion/ -o svg 1>/dev/null && mv classes.svg docs/api/umls/stopping_criterion_specific.svg
	#	Kernel Overview
	@pyreverse -k qmcpy/kernel/ -o svg 1>/dev/null && mv classes.svg docs/api/umls/kernel_overview.svg
	#	Kernel Specific
	@pyreverse qmcpy/kernel/ -o svg 1>/dev/null && mv classes.svg docs/api/umls/kernel_specific.svg

##########################################################
# Documentation with `mkdocs`
#
# Run `mkdocs build -v` to debug. It generates HTML in the site/ folder.
# You can enter `open site/index.html` to open the local pages in a browser.
# (However, the search function may be slow.)
#
# Use `mkdocs serve` to run a local server. The webpages are stored in a temporary folder and will be deleted when the server is stopped.
##########################################################
copydocs:  # mkdocs only looks for content in the docs/ folder, so we have to copy it there
	@rm -rf docs/paper docs/demos
	@cp README.md docs/README.md
	@cp AGENTS.md docs/AGENTS.md
	@perl -0pi -e 's!\(docs/good_practices\.md\)!\(good_practices.md\)!g' docs/AGENTS.md
	@perl -0pi -e 's!\(docs/ai-assisted-contributions\.md\)!\(ai-assisted-contributions.md\)!g' docs/AGENTS.md
	@perl -0pi -e 's!\(docs/RELEASE\.md\)!\(RELEASE.md\)!g' docs/AGENTS.md
	@perl -0pi -e 's!\(docs/assets/pep8-badge\.svg\)!\(assets/pep8-badge.svg\)!g' docs/README.md
	@perl -0pi -e 's!\(docs/qmc-software\.md\)!\(qmc-software.md\)!g' docs/README.md
	@cp CONTRIBUTING.md docs/CONTRIBUTING.md
	@# Rewrite repo-root-relative link for the copied MkDocs page.
	@perl -0pi -e 's!\(docs/good_practices\.md\)!\(good_practices.md\)!g' docs/CONTRIBUTING.md
	@perl -0pi -e 's!\(docs/ai-assisted-contributions\.md\)!\(ai-assisted-contributions.md\)!g' docs/CONTRIBUTING.md
	@cp community.md docs/community.md
	@cp -r demos docs
	@find docs/demos -mindepth 2 -name README.md -delete
	@cp -r paper docs
	@rm -f docs/paper/README.md
	@./scripts/render_paper_for_mkdocs.sh
	@cp test/booktests/README.md docs/booktests.md
	@cp test/README.md docs/tests.md
	@python scripts/make_qmc_software_page.py
	@mkdir -p docs/stats
	@cp stats/pypi_downloads.md docs/stats/pypi_downloads.md
	@cp docs/assets/logos/qmcpy_logo.png docs/apple-touch-icon.png
	@cp docs/assets/logos/qmcpy_logo.png docs/apple-touch-icon-precomposed.png
	@cp docs/assets/logos/qmcpy_logo.png docs/favicon.ico
	@cp QMCPy_Shared_Leadership.md docs/

runmkdocserve:
	@PORT=$${MKDOCS_PORT:-8000}; \
	while lsof -iTCP:$$PORT -sTCP:LISTEN >/dev/null 2>&1; do \
		PORT=$$((PORT+1)); \
	done; \
	echo "Starting mkdocs on http://127.0.0.1:$$PORT"; \
	NO_MKDOCS_2_WARNING=1 JUPYTER_PLATFORM_DIRS=1 mkdocs serve -a 127.0.0.1:$$PORT
	NO_MKDOCS_2_WARNING=1 JUPYTER_PLATFORM_DIRS=1 mkdocs serve -a 127.0.0.1:$$PORT

doc: uml copydocs runmkdocserve

docnouml: copydocs runmkdocserve

check_links: copydocs  # internal links + anchors only; fast, no network, safe for CI
	@NO_MKDOCS_2_WARNING=1 mkdocs build -q -d site
	@python scripts/check_links.py site

check_links_external: copydocs  # also checks http/https links; slow and network-flaky, run locally
	@NO_MKDOCS_2_WARNING=1 mkdocs build -q -d site
	@python scripts/check_links.py site --external

# The targets above check links inside the new site; these check the other
# direction -- already-published URLs that would 404 after the next deploy.
check_removed_urls: copydocs  # fetches the deployed sitemap.xml; needs network
	@python scripts/check_removed_urls.py

check_removed_urls_verify: copydocs  # also HTTP-checks every redirect target
	@python scripts/check_removed_urls.py --verify-redirects

##########################################################
# PEP8
##########################################################
PYLINT ?= pylint
PYLINT_BASE ?= develop

check_pep8:
	@$(PYLINT) qmcpy --exit-zero --disable=R,C,E0401 --ignored-modules=qmctoolscl

check_pep8_changed:
	@set -e; \
	changed_files="$$( \
		{ \
			git diff --name-only --diff-filter=ACMR "$(PYLINT_BASE)...HEAD" -- '*.py'; \
			git diff --name-only --diff-filter=ACMR HEAD -- '*.py'; \
			git ls-files --others --exclude-standard -- '*.py'; \
		} | sort -u \
	)"; \
	if [ -z "$$changed_files" ]; then \
		echo "No changed Python files relative to $(PYLINT_BASE)."; \
	else \
		echo "Running pylint on changed Python files relative to $(PYLINT_BASE):"; \
		printf '%s\n' "$$changed_files"; \
		$(PYLINT) --disable=R,C,E0401 --ignored-modules=qmctoolscl $$changed_files; \
	fi

pep8: update_pep8_badge

update_pep8_badge:
	@mkdir -p $(LOG_DIR) docs/assets
	@make check_pep8 > $(LOG_DIR)/pylint.out
	@python3 scripts/update_pep8_badge.py $(LOG_DIR)/pylint.out docs/assets/pep8-badge.json docs/assets/pep8-badge.svg


##########################################################
# Formatting
##########################################################

FORMAT_PATH ?= .
MARKDOWN_UNWRAP_PATH ?= $(FORMAT_PATH)

format:
	$(MAKE) flatten_qmcpy_imports
	@echo ""
	$(MAKE) markdown-unwrap MARKDOWN_UNWRAP_PATH="$(MARKDOWN_UNWRAP_PATH)"
	@echo ""
	$(MAKE) rm_trailing_whitespace FORMAT_PATH="$(FORMAT_PATH)"
	$(MAKE) harden_colab_notebook

flatten_qmcpy_imports:
	$(PYTHON) scripts/flatten_qmcpy_imports.py

markdown-unwrap:
	$(PYTHON) scripts/unwrap_markdown.py "$(MARKDOWN_UNWRAP_PATH)"

rm_trailing_whitespace:
	$(PYTHON) scripts/remove_trailing_whitespace.py "$(FORMAT_PATH)"
