#!/usr/bin/env Rscript
# Generate autoregressive evaluation figure.
#
# Shows median CV bias (Delta%) vs autoregressive horizon (h=1, 5, 10)
# for three representative datasets (one homogeneous, one heterogeneous,
# one with dramatic strategy ranking differences).  Faceted by dataset
# (columns) and scenario (rows).  Full per-dataset numbers are reported
# in results/autoregressive.csv.
#
# Produces: paper/fig_autoregressive.pdf
#
# Usage: Rscript scripts/plot_autoregressive.R

source("scripts/theme_paper.R")
library(tidyverse)

# ---------------------------------------------------------------------------
# Load autoregressive results
# ---------------------------------------------------------------------------

ar_path <- "results/autoregressive.csv"
if (!file.exists(ar_path)) stop("No autoregressive results found: ", ar_path)

ar <- read_csv(ar_path, show_col_types = FALSE)

# Restrict to three representative datasets for the figure.  The full
# grid is still saved in results/autoregressive.csv and reported in
# tab_autoregressive.
FIG_DATASETS <- c("electricity", "metrla", "mhealth")
ar <- ar |> filter(dataset %in% FIG_DATASETS)

# Collapse across seeds (median) so downstream joins stay unique.
if ("seed" %in% names(ar)) {
  ar <- ar |>
    group_by(dataset, window_size, algorithm, horizon, scenario) |>
    summarise(rmse = median(rmse), .groups = "drop")
}

# ---------------------------------------------------------------------------
# Load CV fold scores to compute CV mean at the same windows
# ---------------------------------------------------------------------------

results_dir <- "results"
csv_files <- list.files(results_dir, pattern = "_s\\d+\\.csv$",
                        full.names = TRUE)

MAX_WINDOW <- 15

df <- csv_files |>
  map_dfr(read_csv, show_col_types = FALSE) |>
  filter(window_size <= MAX_WINDOW,
         dataset %in% names(dataset_labels))

cv <- df |> filter(metric == "val_rmse")

# CV mean at the windows used in AR experiments.
ar_keys <- ar |> distinct(dataset, algorithm, window_size)

cv_mean <- cv |>
  inner_join(ar_keys, by = c("dataset", "algorithm", "window_size")) |>
  group_by(dataset, algorithm, cv_strategy) |>
  summarise(cv_mean = mean(value), .groups = "drop")

# ---------------------------------------------------------------------------
# Build plot data: median bias across algorithms
# ---------------------------------------------------------------------------

scenario_labels <- c("temporal" = "Temporal holdout",
                     "group"    = "Group holdout")

# Expand: for each AR row, join all 4 CV strategies.
plot_data <- ar |>
  inner_join(cv_mean, by = c("dataset", "algorithm"),
             relationship = "many-to-many") |>
  filter(!is.na(rmse), rmse > 0) |>
  mutate(bias_pct = (cv_mean - rmse) / rmse * 100)

pd <- plot_data |>
  group_by(dataset, scenario, cv_strategy, horizon) |>
  summarise(
    med  = median(bias_pct),
    q25  = quantile(bias_pct, 0.25),
    q75  = quantile(bias_pct, 0.75),
    .groups = "drop"
  ) |>
  mutate(
    dataset     = factor(dataset_labels[dataset],     levels = dataset_labels),
    scenario    = factor(scenario_labels[scenario],    levels = scenario_labels),
    cv_strategy = factor(strategy_labels[cv_strategy], levels = strategy_labels)
  ) |>
  filter(!is.na(dataset), !is.na(cv_strategy))

# ---------------------------------------------------------------------------
# Plot — monochrome with linetypes (matching existing figure style)
# ---------------------------------------------------------------------------

p <- ggplot(pd, aes(x = horizon, y = med,
                     linetype = cv_strategy, shape = cv_strategy)) +
  geom_line(colour = "grey20", linewidth = 0.45) +
  geom_point(colour = "grey10", size = 1.5) +
  facet_wrap(~ scenario + dataset, scales = "free_y", ncol = 3) +
  scale_x_continuous(breaks = c(1, 5, 10)) +
  scale_linetype_manual(values = c("solid", "dashed", "dotdash", "dotted")) +
  scale_shape_manual(values = c(16, 17, 15, 4)) +
  labs(
    x = "Horizon (steps ahead)",
    y = expression(Delta * "% (CV bias)"),
    linetype = NULL, shape = NULL
  ) +
  theme_paper() +
  theme(
    legend.position  = "bottom",
    axis.text.x      = element_text(angle = 0, hjust = 0.5)
  )

out_dir <- "paper"
dir.create(out_dir, showWarnings = FALSE)
ggsave(file.path(out_dir, "fig_autoregressive.pdf"), p,
       width = PAPER_WIDTH, height = 3.0, device = cairo_pdf)
cat("Saved", file.path(out_dir, "fig_autoregressive.pdf"), "\n")
