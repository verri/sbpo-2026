#!/usr/bin/env Rscript
# Generate calibration plots comparing CV bias vs actual test RMSE.
#
# For each holdout scenario, plots the signed percentage bias
# (Delta% = 100 * (CV mean - test) / test) for every (dataset,
# algorithm, CV strategy) triple, with whiskers showing the 95%
# Nadeau-Bengio confidence interval around the CV mean (translated
# into Delta% space).  The dashed zero line marks perfect calibration:
# when the whisker crosses it, the test RMSE lies inside the CV's NB
# confidence interval for that cell.
#
# Produces two PDFs and matching CSVs with the plotted values:
#   paper/fig_temporal.pdf / paper/fig_temporal.csv — temporal holdout
#   paper/fig_group.pdf    / paper/fig_group.csv    — group holdout
#
# Usage: Rscript scripts/plot_results.R

source("scripts/theme_paper.R")
library(tidyverse)

# ---------------------------------------------------------------------------
# Nadeau-Bengio CI configuration
# ---------------------------------------------------------------------------

N_FOLDS <- 10
N_SEEDS <- 3
# t_{0.025, df} with df = N_SEEDS * (N_FOLDS - 1) = 27.
NB_T_CRIT <- 2.052

# Variance inflation factor 1/k + n_test/n_train per strategy.  For the
# expanding-window TimeSeriesSplit (sklearn default: n_test = N/(k+1),
# n_train_i = i * N/(k+1)) we average n_test/n_train across folds.
STRATEGY_NB_FACTOR <- c(
  kfold         = 1 / N_FOLDS + 1 / (N_FOLDS - 1),
  group_kfold   = 1 / N_FOLDS + 1 / (N_FOLDS - 1),
  group_shuffle = 1 / N_FOLDS + 0.20 / 0.80,
  timeseries    = 1 / N_FOLDS + mean(1 / seq_len(N_FOLDS))
)

# ---------------------------------------------------------------------------
# Load all result CSVs
# ---------------------------------------------------------------------------

results_dir <- "results"
# Only consume files produced by the seeded run_experiment.py (e.g. *_s42.csv).
csv_files <- list.files(results_dir, pattern = "_s\\d+\\.csv$",
                        full.names = TRUE)
if (length(csv_files) == 0) stop("No seeded CSV files found in results/")

MAX_WINDOW <- 15

df <- csv_files |>
  map_dfr(read_csv, show_col_types = FALSE) |>
  filter(window_size <= MAX_WINDOW,
         dataset %in% names(dataset_labels))

# Aggregate fold scores into mean + NB-corrected SE per
# (dataset, algorithm, cv_strategy, window_size).  For TimeSeriesSplit
# we estimate the fold-score variance from the second half of folds
# only (fold >= N_FOLDS/2), where the expanding training set has grown
# large enough to stabilise; this mitigates the heteroscedasticity that
# plain NB pooling would inflate into implausibly wide intervals on
# small datasets (see Bengio & Grandvalet, 2004).
TS_SD_MIN_FOLD <- N_FOLDS %/% 2  # keep folds >= 5 for TimeSeries SD

cv_stats <- df |>
  filter(metric == "val_rmse") |>
  mutate(keep_for_sd = (cv_strategy != "timeseries") |
                      (fold >= TS_SD_MIN_FOLD)) |>
  group_by(dataset, window_size, algorithm, cv_strategy) |>
  summarise(
    cv_mean = mean(value),
    cv_sd   = sd(value[keep_for_sd]),
    .groups = "drop"
  ) |>
  mutate(
    nb_factor = STRATEGY_NB_FACTOR[cv_strategy],
    cv_nb_se  = cv_sd * sqrt(nb_factor / N_SEEDS)
  )

# Test RMSE is collapsed across seeds (median) so each (dataset, window,
# algorithm, cv_strategy) has a single value before pivoting.
test <- df |>
  filter(metric == "test_rmse") |>
  group_by(dataset, window_size, algorithm, cv_strategy) |>
  summarise(value = median(value), .groups = "drop")

test_wide <- test |>
  select(dataset, window_size, algorithm, cv_strategy, value) |>
  pivot_wider(names_from = cv_strategy, values_from = value) |>
  rename(test_temporal = temporal_osa, test_group = group_osa)

# ---------------------------------------------------------------------------
# Build plot data for one scenario: Delta% and NB CI per (ds, algo, strat)
# ---------------------------------------------------------------------------

build_plot_data <- function(cv_stats, test_wide, test_col) {
  best_windows <- test_wide |>
    group_by(dataset, algorithm) |>
    slice_min(!!sym(test_col), n = 1, with_ties = FALSE) |>
    select(dataset, algorithm, window_size)

  stats_at_best <- cv_stats |>
    inner_join(best_windows,
               by = c("dataset", "algorithm", "window_size"))

  test_at_best <- test_wide |>
    inner_join(best_windows,
               by = c("dataset", "algorithm", "window_size")) |>
    transmute(dataset, algorithm, test_rmse = !!sym(test_col))

  stats_at_best |>
    inner_join(test_at_best, by = c("dataset", "algorithm")) |>
    mutate(
      delta_pct     = 100 * (cv_mean - test_rmse) / test_rmse,
      delta_se_pct  = 100 * cv_nb_se / test_rmse,
      delta_ci_low  = delta_pct - NB_T_CRIT * delta_se_pct,
      delta_ci_high = delta_pct + NB_T_CRIT * delta_se_pct
    )
}

# ---------------------------------------------------------------------------
# Plot function: dodged Delta% points with NB CI whiskers per algorithm
# ---------------------------------------------------------------------------

make_delta_plot <- function(plot_data) {
  n_datasets <- n_distinct(plot_data$dataset)
  n_cols <- min(n_datasets, 4)
  dodge <- position_dodge(width = 0.7)

  ggplot(plot_data,
         aes(x = cv_strategy, y = delta_pct, group = algorithm)) +
    geom_hline(yintercept = 0, linetype = "dashed",
               linewidth = 0.3, colour = "grey40") +
    geom_linerange(
      aes(ymin = delta_ci_low, ymax = delta_ci_high),
      position = dodge, linewidth = 0.4, colour = "grey15"
    ) +
    facet_wrap(~ dataset, ncol = n_cols, scales = "free_y") +
    labs(x = NULL, y = expression(Delta * "%")) +
    theme_paper() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))
}

# ---------------------------------------------------------------------------
# Generate plots
# ---------------------------------------------------------------------------

out_dir <- "paper"
dir.create(out_dir, showWarnings = FALSE)

write_calibration_csv <- function(plot_data, path) {
  plot_data |>
    transmute(
      cv            = cv_strategy,
      dataset       = dataset,
      algorithm     = algorithm,
      cv_mean       = round(cv_mean, 3),
      cv_nb_se      = round(cv_nb_se, 3),
      test_rmse     = round(test_rmse, 3),
      delta_pct     = round(delta_pct, 2),
      delta_ci_low  = round(delta_ci_low, 2),
      delta_ci_high = round(delta_ci_high, 2)
    ) |>
    arrange(dataset, cv, algorithm) |>
    write_csv(path)
  cat("Saved", path, "\n")
}

# Temporal holdout.
temporal_data <- build_plot_data(cv_stats, test_wide, "test_temporal") |>
  prettify()
if (nrow(temporal_data) > 0) {
  n_ds <- n_distinct(temporal_data$dataset)
  n_cols <- min(n_ds, 4)
  fig_h <- 1.2 + 1.1 * ceiling(n_ds / n_cols)
  p <- make_delta_plot(temporal_data)
  ggsave(file.path(out_dir, "fig_temporal.pdf"), p,
         width = PAPER_WIDTH, height = fig_h, device = cairo_pdf)
  cat("Saved", file.path(out_dir, "fig_temporal.pdf"), "\n")
  write_calibration_csv(temporal_data,
                        file.path(out_dir, "fig_temporal.csv"))
}

# Group holdout.
group_data <- build_plot_data(cv_stats, test_wide, "test_group") |>
  prettify()
if (nrow(group_data) > 0) {
  n_ds <- n_distinct(group_data$dataset)
  n_cols <- min(n_ds, 4)
  fig_h <- 1.2 + 1.1 * ceiling(n_ds / n_cols)
  p <- make_delta_plot(group_data)
  ggsave(file.path(out_dir, "fig_group.pdf"), p,
         width = PAPER_WIDTH, height = fig_h, device = cairo_pdf)
  cat("Saved", file.path(out_dir, "fig_group.pdf"), "\n")
  write_calibration_csv(group_data,
                        file.path(out_dir, "fig_group.csv"))
}

cat("Done.\n")
