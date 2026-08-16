PYTHON ?= .venv/bin/python3

DATASETS := electricity pamap2 mhealth har pemsbay metrla gsod

# ---------------------------------------------------------------------------
# Download targets (one per dataset)
# ---------------------------------------------------------------------------

data/raw/electricity/.done:
	$(PYTHON) scripts/download_electricity.py
	touch $@

data/raw/pamap2/.done:
	$(PYTHON) scripts/download_pamap2.py
	touch $@

data/raw/mhealth/.done:
	$(PYTHON) scripts/download_mhealth.py
	touch $@

data/raw/har/.done:
	$(PYTHON) scripts/download_har.py
	touch $@

data/raw/pemsbay/.done:
	$(PYTHON) scripts/download_pemsbay.py
	touch $@

data/raw/metrla/.done:
	$(PYTHON) scripts/download_metrla.py
	touch $@

data/raw/gsod/.done:
	$(PYTHON) scripts/download_gsod.py
	touch $@

download: $(foreach d,$(DATASETS),data/raw/$(d)/.done)

# ---------------------------------------------------------------------------
# Preprocessing targets (one per dataset)
# ---------------------------------------------------------------------------

data/processed/electricity.parquet: data/raw/electricity/.done scripts/preprocess_electricity.py
	$(PYTHON) scripts/preprocess_electricity.py

data/processed/pamap2.parquet: data/raw/pamap2/.done scripts/preprocess_pamap2.py
	$(PYTHON) scripts/preprocess_pamap2.py

data/processed/mhealth.parquet: data/raw/mhealth/.done scripts/preprocess_mhealth.py
	$(PYTHON) scripts/preprocess_mhealth.py

data/processed/har.parquet: data/raw/har/.done scripts/preprocess_har.py
	$(PYTHON) scripts/preprocess_har.py

data/processed/pemsbay.parquet: data/raw/pemsbay/.done scripts/preprocess_pemsbay.py
	$(PYTHON) scripts/preprocess_pemsbay.py

data/processed/metrla.parquet: data/raw/metrla/.done scripts/preprocess_metrla.py
	$(PYTHON) scripts/preprocess_metrla.py

data/processed/gsod.parquet: data/raw/gsod/.done scripts/preprocess_gsod.py
	$(PYTHON) scripts/preprocess_gsod.py

preprocess: $(foreach d,$(DATASETS),data/processed/$(d).parquet)

# ---------------------------------------------------------------------------
# Experiment targets
# Each dataset is a single invocation that produces all result CSVs.
# The script handles all window sizes and algorithms internally, with
# resume support (skips existing CSVs). Use make -j6 to run datasets
# in parallel.
# ---------------------------------------------------------------------------

define EXPERIMENT_RULE
results/$(1).done: data/processed/$(1).parquet scripts/run_experiment.py
	$$(PYTHON) scripts/run_experiment.py $(1)
	touch $$@
endef

$(foreach d,$(DATASETS),$(eval $(call EXPERIMENT_RULE,$(d))))

experiments: $(foreach d,$(DATASETS),results/$(d).done)

# ---------------------------------------------------------------------------
# Tables and figures
# ---------------------------------------------------------------------------

results/autoregressive.csv: $(foreach d,$(DATASETS),results/$(d).done) scripts/run_autoregressive.py scripts/run_experiment.py
	$(PYTHON) scripts/run_autoregressive.py

autoregressive: results/autoregressive.csv

results/tables.done: $(foreach d,$(DATASETS),results/$(d).done) results/autoregressive.csv scripts/generate_tables.py
	$(PYTHON) scripts/generate_tables.py
	touch $@

tables: results/tables.done

paper/fig_group.pdf paper/fig_temporal.pdf: $(foreach d,$(DATASETS),results/$(d).done) scripts/plot_results.R scripts/theme_paper.R
	Rscript scripts/plot_results.R

paper/fig_sweep.pdf: $(foreach d,$(DATASETS),results/$(d).done) scripts/plot_window_sweep.R scripts/theme_paper.R
	Rscript scripts/plot_window_sweep.R

paper/fig_autoregressive.pdf: results/autoregressive.csv $(foreach d,$(DATASETS),results/$(d).done) scripts/plot_autoregressive.R scripts/theme_paper.R
	Rscript scripts/plot_autoregressive.R

figures: paper/fig_group.pdf paper/fig_temporal.pdf paper/fig_autoregressive.pdf

# ---------------------------------------------------------------------------
# Top-level targets
# ---------------------------------------------------------------------------

all: tables figures

clean:
	rm -rf data/processed/*.parquet results/*.csv results/*.done results/tables.tex

distclean: clean
	rm -rf $(foreach d,$(DATASETS),data/raw/$(d))

.PHONY: download preprocess experiments autoregressive tables figures all clean distclean
