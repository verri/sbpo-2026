#!/usr/bin/env Rscript
# Statistical tests for CV bias and coverage across CV strategies.
#
#   Bias      |Delta%| ~ strategy * scenario + (1 | dataset/algorithm)
#             - Linear mixed-effects model (raw and rank-transformed).
#             - Critical-difference diagram (Demsar 2006) per scenario.
#
#   Coverage  covered ~ strategy * scenario + (1 | dataset/algorithm)
#             - Mixed-effects logistic regression (binomial).
#
# Interaction p-values support the paper's thesis that the best CV
# strategy depends on the deployment scenario.
#
# Requires: tidyverse, lmerTest, lme4
# Usage: Rscript scripts/test_interaction.R

source("scripts/theme_paper.R")
suppressPackageStartupMessages({
  library(tidyverse)
  library(lmerTest)
})

MAX_WINDOW <- 15
N_FOLDS <- 10
N_SEEDS <- 3
# t_{0.025, df} with df = N_SEEDS * (N_FOLDS - 1) = 27.
NB_T_CRIT <- 2.052
SPLIT_STRATEGIES <- c("kfold", "group_kfold", "group_shuffle", "timeseries")

# Nadeau-Bengio variance inflation factor 1/k + n_test/n_train per strategy.
STRATEGY_NB_FACTOR <- c(
  kfold         = 1 / N_FOLDS + 1 / (N_FOLDS - 1),
  group_kfold   = 1 / N_FOLDS + 1 / (N_FOLDS - 1),
  group_shuffle = 1 / N_FOLDS + 0.20 / 0.80,
  timeseries    = 1 / N_FOLDS + mean(1 / seq_len(N_FOLDS))
)

# ---------------------------------------------------------------------------
# Load seeded result CSVs
# ---------------------------------------------------------------------------

results_dir <- "results"
csv_files <- list.files(results_dir, pattern = "_s\\d+\\.csv$",
                        full.names = TRUE)
if (length(csv_files) == 0) stop("No seeded CSV files found in results/")

df <- csv_files |>
  map_dfr(read_csv, show_col_types = FALSE) |>
  filter(window_size <= MAX_WINDOW,
         dataset %in% names(dataset_labels))

# ---------------------------------------------------------------------------
# Aggregate across seeds (Section 2.5 conventions)
# ---------------------------------------------------------------------------

# For TimeSeriesSplit we estimate fold-score SD from the second half of
# folds only (fold >= N_FOLDS/2), where the expanding training set has
# stabilised; see Bengio & Grandvalet (2004) on fold-level heteroscedasticity.
TS_SD_MIN_FOLD <- N_FOLDS %/% 2

cv <- df |>
  filter(metric == "val_rmse",
         cv_strategy %in% SPLIT_STRATEGIES) |>
  mutate(keep_for_sd = (cv_strategy != "timeseries") |
                      (fold >= TS_SD_MIN_FOLD)) |>
  group_by(dataset, window_size, algorithm, cv_strategy) |>
  summarise(cv_mean = mean(value),
            cv_min  = min(value),
            cv_max  = max(value),
            cv_sd   = sd(value[keep_for_sd]),
            .groups = "drop") |>
  mutate(
    nb_factor  = STRATEGY_NB_FACTOR[cv_strategy],
    cv_nb_se   = cv_sd * sqrt(nb_factor / N_SEEDS),
    cv_ci_low  = cv_mean - NB_T_CRIT * cv_nb_se,
    cv_ci_high = cv_mean + NB_T_CRIT * cv_nb_se
  )

test_wide <- df |>
  filter(metric == "test_rmse") |>
  group_by(dataset, window_size, algorithm, cv_strategy) |>
  summarise(value = median(value), .groups = "drop") |>
  pivot_wider(names_from = cv_strategy, values_from = value) |>
  rename(test_temporal = temporal_osa, test_group = group_osa)

pick_best <- function(tbl, col) {
  tbl |>
    group_by(dataset, algorithm) |>
    slice_min(!!sym(col), n = 1, with_ties = FALSE) |>
    transmute(dataset, algorithm, window_size,
              test_rmse = !!sym(col))
}

best <- bind_rows(
  pick_best(test_wide, "test_temporal") |> mutate(scenario = "temporal"),
  pick_best(test_wide, "test_group")    |> mutate(scenario = "group")
)

long <- best |>
  inner_join(cv, by = c("dataset", "algorithm", "window_size"),
             relationship = "many-to-many") |>
  mutate(
    delta_abs = abs(100 * (cv_mean - test_rmse) / test_rmse),
    covered   = as.integer(test_rmse >= cv_ci_low & test_rmse <= cv_ci_high),
    strategy  = factor(cv_strategy, levels = SPLIT_STRATEGIES),
    scenario  = factor(scenario, levels = c("temporal", "group")),
    dataset   = factor(dataset),
    algorithm = factor(algorithm)
  )

n_cells <- n_distinct(long |> select(dataset, algorithm, scenario))
cat(sprintf("Rows fed to models: %d  (cells: %d, expected 70)\n",
            nrow(long), n_cells))

# ---------------------------------------------------------------------------
# 1) Bias: linear mixed-effects model on |Delta%|
# ---------------------------------------------------------------------------

cat("\n================  BIAS (raw |Delta%|)  ================\n")
m <- lmer(delta_abs ~ strategy * scenario + (1 | dataset/algorithm),
          data = long)
aov_tbl <- anova(m, type = 2)
print(aov_tbl)
inter_p <- aov_tbl["strategy:scenario", "Pr(>F)"]
cat(sprintf("Interaction p-value: %.4g\n", inter_p))

# ---------------------------------------------------------------------------
# 2) Bias: rank-transformed (robust to heavy tails)
# ---------------------------------------------------------------------------

cat("\n================  BIAS (ranks within cell)  ================\n")
long_rank <- long |>
  group_by(dataset, algorithm, scenario) |>
  mutate(rank_delta = rank(delta_abs, ties.method = "average")) |>
  ungroup()

m_rank <- lmer(rank_delta ~ strategy * scenario + (1 | dataset/algorithm),
               data = long_rank)
aov_rank <- anova(m_rank, type = 2)
print(aov_rank)
inter_p_rank <- aov_rank["strategy:scenario", "Pr(>F)"]
cat(sprintf("Interaction p-value (ranks): %.4g\n", inter_p_rank))

mean_ranks <- long_rank |>
  group_by(strategy, scenario) |>
  summarise(mean_rank = mean(rank_delta), .groups = "drop")
cat("\nMean rank per strategy x scenario:\n")
print(mean_ranks |>
        pivot_wider(names_from = scenario, values_from = mean_rank))

# ---------------------------------------------------------------------------
# 3) Critical-difference diagram per scenario (Nemenyi, Demsar 2006)
# ---------------------------------------------------------------------------

# Studentized range / sqrt(2) critical values at alpha = 0.05 (Demsar Table 5).
Q_ALPHA_05 <- c(`2` = 1.960, `3` = 2.343, `4` = 2.569,
                `5` = 2.728, `6` = 2.850, `7` = 2.949,
                `8` = 3.031, `9` = 3.102, `10` = 3.164)

cd_stats <- function(sub) {
  k <- n_distinct(sub$strategy)
  N <- n_distinct(sub |> select(dataset, algorithm))
  q <- Q_ALPHA_05[as.character(k)]
  cd <- q * sqrt(k * (k + 1) / (6 * N))
  ranks <- sub |>
    group_by(strategy) |>
    summarise(mean_rank = mean(rank_delta), .groups = "drop") |>
    arrange(mean_rank)
  list(ranks = ranks, cd = cd, k = k, N = N)
}

find_cliques <- function(r, cd) {
  # Maximal intervals of sorted ranks with spread < cd
  k <- length(r)
  out <- list()
  for (i in seq_len(k)) {
    j <- i
    while (j < k && r[j + 1] - r[i] < cd) j <- j + 1
    if (j > i) {
      dominated <- length(out) > 0 &&
        any(vapply(out, \(cl) cl[1] <= i && cl[2] >= j, logical(1)))
      if (!dominated) out[[length(out) + 1]] <- c(i, j)
    }
  }
  out
}

cd_diagram <- function(cd, ranks, title) {
  k <- nrow(ranks)
  r <- ranks$mean_rank
  cliques <- find_cliques(r, cd)
  mid <- (1 + k) / 2

  labs_df <- ranks |>
    mutate(
      side      = if_else(mean_rank <= mid, "left", "right"),
      lbl       = strategy_labels[as.character(strategy)]
    ) |>
    group_by(side) |>
    mutate(
      rail_order = if (first(side) == "left")
                     rank(mean_rank, ties.method = "first")
                   else rank(-mean_rank, ties.method = "first"),
      rail_y     = -0.12 - 0.13 * (rail_order - 1),
      label_x    = if_else(side == "left", 1 - 0.15, k + 0.15),
      label_h    = if_else(side == "left", 1, 0)
    ) |>
    ungroup()

  cliques_df <- if (length(cliques) > 0) {
    tibble(
      id      = seq_along(cliques),
      x_start = vapply(cliques, \(cl) r[cl[1]], numeric(1)),
      x_end   = vapply(cliques, \(cl) r[cl[2]], numeric(1)),
      y       = -0.04 - 0.04 * (seq_along(cliques) - 1)
    )
  } else tibble()

  y_min <- min(labs_df$rail_y) - 0.05
  y_max <- 0.42

  p <- ggplot() +
    annotate("segment", x = 1, xend = k, y = 0, yend = 0,
             linewidth = 0.4, colour = "grey20") +
    annotate("segment", x = seq_len(k), xend = seq_len(k),
             y = 0, yend = 0.06, linewidth = 0.3, colour = "grey20") +
    annotate("text", x = seq_len(k), y = 0.13,
             label = as.character(seq_len(k)),
             family = "serif", size = 2.8, colour = "grey20") +
    geom_segment(data = labs_df,
                 aes(x = mean_rank, xend = mean_rank,
                     y = 0, yend = rail_y),
                 linewidth = 0.3, colour = "grey40") +
    geom_segment(data = labs_df,
                 aes(x = mean_rank, xend = label_x,
                     y = rail_y, yend = rail_y),
                 linewidth = 0.3, colour = "grey40") +
    geom_text(data = labs_df,
              aes(x = label_x, y = rail_y,
                  label = lbl, hjust = label_h),
              family = "serif", size = 2.8, colour = "grey20") +
    # CD bracket
    annotate("segment", x = 1, xend = 1 + cd, y = 0.24, yend = 0.24,
             linewidth = 0.5, colour = "grey20") +
    annotate("segment", x = c(1, 1 + cd), xend = c(1, 1 + cd),
             y = 0.21, yend = 0.27, linewidth = 0.3, colour = "grey20") +
    annotate("text", x = 1 + cd / 2, y = 0.35,
             label = sprintf("CD = %.2f", cd),
             family = "serif", size = 2.8, colour = "grey20") +
    scale_x_continuous(limits = c(-0.5, k + 2.6), expand = c(0, 0)) +
    scale_y_continuous(limits = c(y_min, y_max), expand = c(0, 0)) +
    labs(title = title) +
    theme_void(base_size = 9, base_family = "serif") +
    theme(plot.title = element_text(hjust = 0.5, size = 9, family = "serif",
                                    margin = margin(t = 4, b = 2)))

  if (nrow(cliques_df) > 0) {
    p <- p + geom_segment(data = cliques_df,
                          aes(x = x_start - 0.02, xend = x_end + 0.02,
                              y = y, yend = y),
                          linewidth = 0.6, colour = "grey10",
                          lineend = "round")
  }
  p
}

cd_temp  <- cd_stats(long_rank |> filter(scenario == "temporal"))
cd_group <- cd_stats(long_rank |> filter(scenario == "group"))

cat(sprintf("\nNemenyi CD (alpha = 0.05): temporal = %.3f  group = %.3f\n",
            cd_temp$cd, cd_group$cd))

p_temp  <- cd_diagram(cd_temp$cd,  cd_temp$ranks,
                      "Temporal holdout")
p_group <- cd_diagram(cd_group$cd, cd_group$ranks,
                      "Group holdout")

out_pdf <- "paper/fig_cd.pdf"
cairo_pdf(out_pdf, width = PAPER_WIDTH, height = 1.25)
grid::grid.newpage()
grid::pushViewport(grid::viewport(layout = grid::grid.layout(1, 2)))
print(p_temp,  vp = grid::viewport(layout.pos.row = 1, layout.pos.col = 1))
print(p_group, vp = grid::viewport(layout.pos.row = 1, layout.pos.col = 2))
dev.off()
cat("Wrote", out_pdf, "\n")

# ---------------------------------------------------------------------------
# 4) Coverage: mixed-effects logistic regression
# ---------------------------------------------------------------------------

cat("\n================  COVERAGE (GLMM)  ================\n")
m_cov <- lme4::glmer(covered ~ strategy * scenario + (1 | dataset/algorithm),
                     data = long, family = binomial,
                     control = lme4::glmerControl(optimizer = "bobyqa"))
m_cov_add <- update(m_cov, . ~ strategy + scenario + (1 | dataset/algorithm))

lrt <- anova(m_cov_add, m_cov)
print(lrt)
inter_p_cov <- lrt$`Pr(>Chisq)`[2]
cat(sprintf("Interaction LRT p-value (coverage): %.4g\n", inter_p_cov))

cov_rates <- long |>
  group_by(strategy, scenario) |>
  summarise(coverage = mean(covered), n = n(), .groups = "drop") |>
  pivot_wider(names_from = scenario, values_from = c(coverage, n))
cat("\nMarginal coverage rate per strategy x scenario:\n")
print(cov_rates)

# ---------------------------------------------------------------------------
# 5) Model-selection regret: Friedman test per scenario
# ---------------------------------------------------------------------------

cat("\n================  REGRET (Friedman)  ================\n")

cv_selected <- cv |>
  group_by(dataset, algorithm, cv_strategy) |>
  slice_min(cv_mean, n = 1, with_ties = FALSE) |>
  transmute(dataset, algorithm, cv_strategy, cv_selected_w = window_size)

test_long <- test_wide |>
  pivot_longer(c(test_temporal, test_group),
               names_to = "scenario", values_to = "test_rmse") |>
  mutate(scenario = recode(scenario,
                           "test_temporal" = "temporal",
                           "test_group"    = "group"))

test_at_cv <- cv_selected |>
  ungroup() |>
  inner_join(test_long, by = c("dataset", "algorithm",
                               "cv_selected_w" = "window_size"),
             relationship = "many-to-many") |>
  rename(test_cv = test_rmse)

oracle_df <- best |> rename(test_oracle = test_rmse, oracle_w = window_size)

regret <- test_at_cv |>
  inner_join(oracle_df, by = c("dataset", "algorithm", "scenario"),
             relationship = "many-to-many") |>
  mutate(regret = 100 * (test_cv - test_oracle) / test_oracle,
         cell   = paste(dataset, algorithm, sep = ":"))

regret_friedman <- function(sub) {
  m <- sub |>
    ungroup() |>
    select(cell, cv_strategy, regret) |>
    pivot_wider(names_from = cv_strategy, values_from = regret) |>
    as.data.frame()
  rn <- m$cell
  m <- as.matrix(m[, -1])
  rownames(m) <- rn
  friedman.test(m)
}

ft_temp  <- regret_friedman(regret |> filter(scenario == "temporal"))
ft_group <- regret_friedman(regret |> filter(scenario == "group"))

cat("\nFriedman on regret (temporal):\n"); print(ft_temp)
cat("\nFriedman on regret (group):\n");    print(ft_group)

regret_summary <- regret |>
  group_by(scenario, cv_strategy) |>
  summarise(median = median(regret), mean = mean(regret), .groups = "drop")
cat("\nMedian / mean regret per strategy x scenario:\n")
print(regret_summary)

# ---------------------------------------------------------------------------
# Emit compact LaTeX macros for the paper
# ---------------------------------------------------------------------------

out_tex <- "paper/stat_interaction.tex"
lines <- c(
  "% Auto-generated by scripts/test_interaction.R",
  sprintf("\\newcommand{\\InteractionP}{%.3f}",      inter_p),
  sprintf("\\newcommand{\\InteractionF}{%.2f}",
          aov_tbl["strategy:scenario", "F value"]),
  sprintf("\\newcommand{\\InteractionDf}{%g, %.1f}",
          aov_tbl["strategy:scenario", "NumDF"],
          aov_tbl["strategy:scenario", "DenDF"]),
  sprintf("\\newcommand{\\InteractionPRank}{%.3f}",  inter_p_rank),
  sprintf("\\newcommand{\\InteractionFRank}{%.2f}",
          aov_rank["strategy:scenario", "F value"]),
  sprintf("\\newcommand{\\CoverageInterP}{%.3f}",    inter_p_cov),
  sprintf("\\newcommand{\\NemenyiCDTemp}{%.2f}",     cd_temp$cd),
  sprintf("\\newcommand{\\NemenyiCDGroup}{%.2f}",    cd_group$cd),
  sprintf("\\newcommand{\\RegretFPTemp}{%.3f}",      ft_temp$p.value),
  sprintf("\\newcommand{\\RegretFPGroup}{%.3f}",     ft_group$p.value)
)
writeLines(lines, out_tex)
cat("\nWrote", out_tex, "\n")
