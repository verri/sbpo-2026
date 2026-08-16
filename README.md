# Cross-Validation Strategies for Multi-Series Time Series Forecasting

Source code for reproducing the experiments in:

> Filipe Alves Neto Verri, Jorge Luiz Franco, and Marcos Gonçalves Quiles.
> **On Cross-Validation Strategies for Autoregressive Forecasting.**
> Simpósio Brasileiro de Pesquisa Operacional (SBPO), 2026.

The paper compares four cross-validation strategies (KFold, GroupKFold,
GroupShuffleSplit, TimeSeriesSplit) for multi-series forecasting with
global models, under two deployment scenarios: forecasting the future of
series seen in training, and forecasting series never seen in training.

## Requirements

- Python 3.10+
- R 4.x with `ggplot2`, `dplyr`, `readr`, `tidyr`, `patchwork`

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Reproducing the Experiments

```bash
make download          # fetch all seven datasets
make preprocess        # convert to tidy parquet files
make -j1 experiments   # run all CV strategies x algorithms x window sizes
make autoregressive    # multi-step evaluation (h=1,5,10)
make tables figures    # generate LaTeX tables and PDF figures
```

Each step is incremental: completed stages are skipped on re-run.

## Datasets

| Dataset     | Domain   | Groups                 | Sampling | Target       |
|-------------|----------|------------------------|----------|--------------|
| Electricity | Energy   | 370 clients            | 1 week   | power (MW)   |
| PEMS-BAY    | Traffic  | 100 sensors            | 2 hours  | speed (mph)  |
| METR-LA     | Traffic  | 173 sensors            | 2 hours  | speed (mph)  |
| GSOD        | Weather  | 300 stations           | 1 week   | temp (C)     |
| PAMAP2      | Activity | 88 subject-action      | ~1 Hz    | heart rate   |
| MHEALTH     | Activity | 120 subject-action     | 5 Hz     | ECG lead 1   |
| HAR         | Activity | 180 subject-action     | 5 Hz     | accel x      |

The sampling column gives the resolution *after* preprocessing: every
dataset is block-averaged over non-overlapping, group-local windows
(PAMAP2 from 100 Hz, MHEALTH and HAR from 50 Hz).  Groups left with
fewer than `2 * w_max + h = 40` samples are dropped.

## Repository Structure

```
Makefile                     # Top-level orchestration
datasets.json                # Dataset metadata (URLs, DOIs, citations)
requirements.txt             # Python dependencies
scripts/
  download_*.py              # One download script per dataset
  preprocess_*.py            # One preprocess script per dataset
  run_experiment.py          # All CV strategies for one dataset
  run_autoregressive.py      # Multi-step autoregressive evaluation
  generate_tables.py         # LaTeX table generation
  summarize_results.py       # Console summary of CV vs test RMSE
  describe_data.py           # Summary stats of preprocessed data
  ts_sensitivity.py          # TimeSeriesSplit fold-aggregation sensitivity
  plot_results.R             # CV calibration figures
  plot_autoregressive.R      # Autoregressive evaluation figure
  plot_window_sweep.R        # RMSE vs window size figure
  theme_paper.R              # Shared ggplot2 theme
data/
  raw/                       # Downloaded datasets (not committed)
  processed/                 # Tidy parquet files (not committed)
results/                     # CSV outputs from experiments
```

Raw and preprocessed data are not distributed here: `make download`
fetches every dataset from its original source, listed with URLs and
DOIs in `datasets.json`.

## Reproducibility

Three fixed seeds (42, 43, 44) control the held-out group selection and
every stochastic cross-validation and model operation, so a full re-run
reproduces the published numbers.  Package versions are pinned in
`requirements.txt`.  The full grid is 7 datasets x 11 window sizes x 5
algorithms x 3 seeds x 4 CV strategies, or 47,355 model fits.

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{verri2026crossvalidation,
  author    = {Verri, Filipe Alves Neto and
               Franco, Jorge Luiz and
               Quiles, Marcos Gon\c{c}alves},
  title     = {On Cross-Validation Strategies for Autoregressive Forecasting},
  booktitle = {Simp\'osio Brasileiro de Pesquisa Operacional (SBPO)},
  year      = {2026}
}
```
