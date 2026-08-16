# scripts/theme_paper.R
# Shared ggplot2 theme for SBPO 2026 paper figures.
# Tufte-inspired: minimal ink, serif font, no box borders.
#
# Source this file from any plot script:
#   source("scripts/theme_paper.R")

library(ggplot2)

# ---------------------------------------------------------------------------
# Paper dimensions (measured from compiled PDF)
# ---------------------------------------------------------------------------

PAPER_WIDTH <- 5.993  # inches (433.12 pt)

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

theme_paper <- function(base_size = 9, base_family = "serif") {
  theme_minimal(base_size = base_size, base_family = base_family) %+replace%
    theme(
      # Uniform text size across all elements
      plot.title        = element_text(size = base_size, face = "plain",
                                       hjust = 0, margin = margin(b = 4)),
      strip.text        = element_text(size = base_size, face = "plain",
                                       hjust = 0, margin = margin(b = 2)),
      axis.title        = element_text(size = base_size),
      axis.text         = element_text(size = base_size,
                                       colour = "grey30"),
      legend.text       = element_text(size = base_size),
      legend.title      = element_text(size = base_size),

      # Grid: only light horizontal lines
      panel.grid.major.x = element_blank(),
      panel.grid.minor   = element_blank(),
      panel.grid.major.y = element_line(linewidth = 0.3, colour = "grey85"),

      # No panel border or strip background
      panel.border       = element_blank(),
      strip.background   = element_blank(),

      # Axes
      axis.line.x        = element_line(linewidth = 0.3, colour = "grey50"),
      axis.ticks.x       = element_line(linewidth = 0.3, colour = "grey50"),
      axis.ticks.y       = element_blank(),
      axis.ticks.length  = unit(2, "pt"),

      # Legend
      legend.position    = "bottom",
      legend.key.size    = unit(10, "pt"),
      legend.margin      = margin(t = -2),
      legend.box.margin  = margin(t = -4),

      # Margins and spacing
      plot.margin        = margin(4, 6, 4, 4),
      panel.spacing      = unit(10, "pt")
    )
}

# ---------------------------------------------------------------------------
# Shared labels
# ---------------------------------------------------------------------------

dataset_labels <- c(
  "electricity" = "Electricity",
  "pemsbay"     = "PEMS-BAY",
  "metrla"      = "METR-LA",
  "gsod"        = "GSOD",
  "pamap2"      = "PAMAP2",
  "mhealth"     = "MHEALTH",
  "har"         = "HAR"
)

strategy_labels <- c(
  "kfold"         = "KFold",
  "group_kfold"   = "GroupKFold",
  "group_shuffle" = "GroupShuffle",
  "timeseries"    = "TimeSeries"
)

algo_labels <- c(
  "ridge" = "Ridge",
  "gbr"   = "HistGBR",
  "mlp"   = "MLP",
  "knn"   = "KNN",
  "et"    = "ExtraTrees"
)

# ---------------------------------------------------------------------------
# Prettify helper: recode dataset, algorithm, strategy to display labels
# ---------------------------------------------------------------------------

prettify <- function(data) {
  data |>
    dplyr::mutate(
      cv_strategy = factor(strategy_labels[cv_strategy],
                           levels = strategy_labels),
      algorithm   = factor(algo_labels[algorithm],
                           levels = algo_labels),
      dataset     = factor(dataset_labels[dataset],
                           levels = dataset_labels)
    ) |>
    dplyr::filter(!is.na(cv_strategy), !is.na(algorithm), !is.na(dataset))
}
