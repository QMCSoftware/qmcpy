PYTHON ?= python3
NOTEBOOKS := 01_sequential.ipynb 02_parallel.ipynb 03_visualize_speedup.ipynb
OUTDIR := output
KERNEL := qmcpy
TIMEOUT := 3600

.PHONY: all sequential parallel visualize clean

all: sequential parallel visualize

sequential:
	@echo "Running 01_sequential.ipynb (sequential)"
	$(PYTHON) -m nbconvert --to notebook --execute 01_sequential.ipynb \
		--ExecutePreprocessor.kernel_name=$(KERNEL) --ExecutePreprocessor.timeout=$(TIMEOUT) \
		--output-dir=$(OUTDIR) --output=01_sequential_output.ipynb

parallel:
	@echo "Running 02_parallel.ipynb (parallel) for workers  2"
	@for w in  2 ; do \
		 echo " -> running with $$w workers"; \
		 PARSL_MAX_WORKERS=$$w $(PYTHON) -m nbconvert --to notebook --execute 02_parallel.ipynb \
		 --ExecutePreprocessor.kernel_name=$(KERNEL) --ExecutePreprocessor.timeout=$(TIMEOUT) \
		 --output-dir=$(OUTDIR) --output=02_parallel_workers_$$w.ipynb; \
		done

visualize:
	@echo "Running 03_visualize_speedup.ipynb (visualize)"
	$(PYTHON) -m nbconvert --to notebook --execute 03_visualize_speedup.ipynb \
		--ExecutePreprocessor.kernel_name=$(KERNEL) --ExecutePreprocessor.timeout=$(TIMEOUT) \
		--output-dir=$(OUTDIR) --output=03_visualize_output.ipynb

clean:
	@echo "Cleaning output files"
	-rm -f $(OUTDIR)/*.ipynb $(OUTDIR)/*.txt $(OUTDIR)/*.csv
