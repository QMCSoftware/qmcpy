# Emit pytest-xdist argument if available; can be overridden on the make command line
PYTEST_XDIST ?= $(shell python scripts/pytest_xdist.py 2>/dev/null)
PYTEST ?=

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
	rm -fr artifacts/coverage/ .coverage*

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

doctests_umbridge: ensure_artifacts # https://github.com/UM-Bridge/umbridge/issues/96
	@mkdir -p $(DOCTEST_COV_DIR)/umbridge
	@docker --version
	COVERAGE_FILE=$(DOCTEST_COV_DIR)/umbridge/.coverage \
	python -m pytest $(PYTEST_XDIST) -x --cov qmcpy/ --cov-report term --cov-report json:$(DOCTEST_COV_DIR)/umbridge/coverage.json --no-header --cov-append \
		--doctest-modules qmcpy/integrand/umbridge_wrapper.py \

doctests_markdown:
	@phmutest docs/*.md --replmode --log -c

doctests: doctests_markdown doctests_minimal doctests_torch doctests_gpytorch doctests_botorch doctests_umbridge

doctests_no_docker: doctests_minimal doctests_torch doctests_gpytorch doctests_botorch

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
	"$$PYTHON_BIN" -m pytest $(PYTEST_XDIST) -x \
		--cov=qmcpy \
		--cov-report term \
		--cov-report json:$(UNIT_COV_DIR)/coverage.json \
		--no-header \
		test/ -W ignore::DeprecationWarning

##########################################################
# Unit Tests for `*.ipynb` in `demos/` folder
##########################################################
generate_booktests:
	@echo "\nGenerating missing booktest files..."
	cd test/booktests/ && python generate_test.py --check-missing

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
	set -e && $(MAKE) doctests_no_docker && $(MAKE) unittests  

# Fast test target: run doctests, unittests, booktests concurrently
tests_fast:
	@echo "Running fast tests: doctests and unittests concurrently (splitting CPU cores)."
	@make clean_local_only_files && \
	set -e && \
	$(MAKE) doctests_no_docker & \
	$(MAKE) unittests & \
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
# run ` mkdocs build -v` to debug
##########################################################
copydocs:  # mkdocs only looks for content in the docs/ folder, so we have to copy it there
	@rm -rf docs/paper docs/demos
	@cp README.md docs/README.md 
	@perl -0pi -e 's!\(docs/assets/pep8-badge\.svg\)!\(assets/pep8-badge.svg\)!g' docs/README.md
	@perl -0pi -e 's!\(docs/qmc-software\.md\)!\(qmc-software.md\)!g' docs/README.md
	@cp CONTRIBUTING.md docs/CONTRIBUTING.md 
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

runmkdocserve:
	@PORT=$${MKDOCS_PORT:-8000}; \
	while lsof -iTCP:$$PORT -sTCP:LISTEN >/dev/null 2>&1; do \
		PORT=$$((PORT+1)); \
	done; \
	echo "Starting mkdocs on http://127.0.0.1:$$PORT"; \
	NO_MKDOCS_2_WARNING=1 JUPYTER_PLATFORM_DIRS=1 mkdocs serve -a 127.0.0.1:$$PORT
	
doc: uml copydocs runmkdocserve

docnouml: copydocs runmkdocserve

##########################################################
# PEP8
##########################################################
check_pep8:
	@pylint qmcpy --exit-zero --disable=R,C,E0401

pep8: update_pep8_badge

update_pep8_badge:
	@mkdir -p $(LOG_DIR) docs/assets
	@make check_pep8 > $(LOG_DIR)/pylint.out
	@python3 scripts/update_pep8_badge.py $(LOG_DIR)/pylint.out docs/assets/pep8-badge.json docs/assets/pep8-badge.svg
