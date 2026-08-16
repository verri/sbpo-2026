#!/usr/bin/env python3
"""Summarize experiment results: compare CV estimates with actual test RMSE.

Two analyses, one per holdout scenario:
1. Group holdout: selects window with lowest group test RMSE per (dataset, algo).
2. Temporal holdout: selects window with lowest temporal test RMSE per (dataset, algo).

For each, reports per CV strategy:
- Mean and range (min, max) of CV fold RMSE scores
- Actual test RMSE for both scenarios (at the selected window)
- Signed % difference: (mean_cv - test) / test * 100
- Whether the test RMSE falls within the fold range [min, max]

The fold range is used instead of a confidence interval because CV folds
share training data and are not independent — a percentile-based CI from
5 correlated observations is not statistically meaningful.
"""

import glob
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")


MAX_WINDOW = 15


def load_all_results():
    files = sorted(glob.glob(str(RESULTS_DIR / "*_s*.csv")))
    if not files:
        raise FileNotFoundError("No seeded result CSVs found in results/")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df[df["window_size"] <= MAX_WINDOW]


def print_analysis(cv, test_wide, scenario, test_col, label):
    """Run analysis for one holdout scenario."""

    # For each (dataset, algorithm), find the window with lowest test RMSE.
    best_windows = (
        test_wide.loc[test_wide.groupby(["dataset", "algorithm"])[test_col].idxmin()]
        [["dataset", "algorithm", "window_size"]]
    )

    print("=" * 80)
    print(f"  Best window per (dataset, algorithm) — lowest {label} test RMSE")
    print("=" * 80)
    for _, row in best_windows.iterrows():
        print(f"  {row['dataset']:<15} {row['algorithm']:<10} w={int(row['window_size'])}")

    # Filter CV and test to best windows.
    cv_best = cv.merge(best_windows, on=["dataset", "algorithm", "window_size"])
    test_best = test_wide.merge(best_windows, on=["dataset", "algorithm", "window_size"])

    # CV stats at best window.
    cv_stats = (
        cv_best.groupby(["dataset", "algorithm", "cv_strategy"])["value"]
        .agg(cv_mean="mean", cv_min="min", cv_max="max")
        .reset_index()
    )

    merged = cv_stats.merge(test_best, on=["dataset", "algorithm"])

    # Compute metrics for both test scenarios.
    for sc, tc in [("temporal", "test_temporal"), ("group", "test_group")]:
        merged[f"pct_diff_{sc}"] = (
            (merged["cv_mean"] - merged[tc]) / merged[tc] * 100
        )
        merged[f"in_range_{sc}"] = (
            (merged[tc] >= merged["cv_min"]) &
            (merged[tc] <= merged["cv_max"])
        )

    print(f"\n  CV vs Test RMSE (window chosen by best {label} RMSE)\n")

    for ds, ds_grp in merged.groupby("dataset"):
        print(f"  {ds}")
        print(f"  {'algo':<8} {'strategy':<16} "
              f"{'CV mean':>8} {'[min':>8} {'max]':>8} "
              f"{'T.test':>8} {'%diff':>8} {'inR':>4} "
              f"{'G.test':>8} {'%diff':>8} {'inR':>4}")
        print("  " + "-" * 100)
        for _, row in ds_grp.sort_values(["algorithm", "cv_strategy"]).iterrows():
            print(f"  {row['algorithm']:<8} {row['cv_strategy']:<16} "
                  f"{row['cv_mean']:>8.2f} {row['cv_min']:>8.2f} {row['cv_max']:>8.2f} "
                  f"{row['test_temporal']:>8.2f} {row['pct_diff_temporal']:>+7.1f}% "
                  f"{'Y' if row['in_range_temporal'] else 'N':>3} "
                  f"{row['test_group']:>8.2f} {row['pct_diff_group']:>+7.1f}% "
                  f"{'Y' if row['in_range_group'] else 'N':>3}")
        print()


def summarize():
    df = load_all_results()

    cv = df[df["metric"] == "val_rmse"]
    # Collapse test RMSE across seeds (median) before pivoting so that each
    # (dataset, window, algorithm, cv_strategy) maps to a single value.
    test = (
        df[df["metric"] == "test_rmse"]
        .groupby(["dataset", "window_size", "algorithm", "cv_strategy"])["value"]
        .median()
        .reset_index()
    )

    # Pivot test results: one row per (dataset, window, algorithm).
    test_wide = test.pivot_table(
        index=["dataset", "window_size", "algorithm"],
        columns="cv_strategy",
        values="value",
    ).reset_index()
    test_wide.columns.name = None
    test_wide = test_wide.rename(columns={
        "temporal_osa": "test_temporal",
        "group_osa": "test_group",
    })

    # Analysis 1: best window for group holdout.
    print_analysis(cv, test_wide, "group", "test_group", "group")

    print()

    # Analysis 2: best window for temporal holdout.
    print_analysis(cv, test_wide, "temporal", "test_temporal", "temporal")

    print("  %diff = (CV_mean - test) / test * 100")
    print("    positive = CV overestimates error (pessimistic)")
    print("    negative = CV underestimates error (optimistic, dangerous)")
    print("  inR = test RMSE falls within [min, max] of CV fold scores")
    print("  Note: CV folds share training data and are NOT independent;")
    print("        the fold range is descriptive, not a confidence interval.")


if __name__ == "__main__":
    summarize()
