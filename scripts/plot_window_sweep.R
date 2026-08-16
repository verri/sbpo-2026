#!/usr/bin/env Rscript
# Plot RMSE vs window size for each dataset, averaged across algorithms.
#
# For each (dataset, cv_strategy):
#   - Solid line: group holdout test RMSE (mean across algorithms)
#   - Dashed line: temporal holdout test RMSE (mean across algorithms)
#
# Faceted by dataset. One line colour per CV strategy.
#
# Produces paper/fig_sweep.pdf
#
# Usage: Rscript scripts/plot_window_sweep.R

source("scripts/theme_paper.R")
library(tidyverse)

MAX_WINDOW <- 15

results_dir <- "results"
csv_files <- list.files(results_dir, pattern = "_s\\d+\\.csv$",
                        full.names = TRUE)
if (length(csv_files) == 0) stop("No seeded CSV files found in results/")

df <- csv_files |>
  map_dfr(read_csv, show_col_types = FALSE) |>
  filter(window_size <= MAX_WINDOW,
         dataset %in% names(dataset_labels))

test <- df |> filter(metric == "test_rmse")

# Average test RMSE across algorithms per (dataset, window, scenario).
test_avg <- test |>
  mutate(scenario = recode(cv_strategy,
    "group_osa"    = "Group holdout",
    "temporal_osa" = "Temporal holdout"
  )) |>
  group_by(dataset, window_size, scenario) |>
  summarise(rmse = mean(value), .groups = "drop")

test_avg <- test_avg |>
  mutate(dataset = factor(dataset_labels[dataset], levels = dataset_labels)) |>
  filter(!is.na(dataset))

n_datasets <- n_distinct(test_avg$dataset)
n_cols <- min(n_datasets, 3)
n_rows <- ceiling(n_datasets / n_cols)

p <- ggplot(test_avg, aes(x = window_size, y = rmse,
                           linetype = scenario, shape = scenario)) +
  geom_line(linewidth = 0.5, colour = "grey30") +
  geom_point(size = 1.5, colour = "grey30") +
  scale_linetype_manual(
    name = NULL,
    values = c("Group holdout" = "solid", "Temporal holdout" = "dashed")
  ) +
  scale_shape_manual(
    name = NULL,
    values = c("Group holdout" = 16, "Temporal holdout" = 1)
  ) +
  scale_x_continuous(breaks = seq(5, 15, by = 2)) +
  facet_wrap(~ dataset, scales = "free_y", ncol = n_cols) +
  labs(
    x = "Window size",
    y = "RMSE (mean across algorithms)"
  ) +
  theme_paper()

fig_h <- 1.8 + 2.2 * n_rows
out_path <- file.path("paper", "fig_sweep.pdf")
ggsave(out_path, p, width = PAPER_WIDTH, height = fig_h, device = cairo_pdf)
cat("Saved", out_path, "\n")
