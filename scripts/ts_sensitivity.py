"""Sensitivity of the TimeSeriesSplit results to the fold-aggregation rule.

Reproduces the numbers quoted in Section 3.4 of the paper.

The main analysis (``generate_tables.py``) estimates the TimeSeriesSplit
fold-score SD from the second half of folds, where the expanding training
set has grown large enough to stabilise, while the CV mean averages all
ten folds.  The two therefore rest on different fold subsets.  This script
recomputes the TimeSeriesSplit calibration under three aggregation rules:

    paper      mean over all folds,     SD over folds >= 5   (as published)
    all        mean over all folds,     SD over all folds
    late-both  mean over folds >= 5,    SD over folds >= 5

Under ``late-both`` the Nadeau--Bengio variance inflation factor and the
t quantile are recomputed for the five retained folds.

Only TimeSeriesSplit is affected; the other three strategies are identical
under all three rules, which the script asserts.

Usage (from the project root):

    .venv/bin/python3 scripts/ts_sensitivity.py
"""

import numpy as np
import pandas as pd

import generate_tables as G

MODES = ["paper", "all", "late-both"]

# Folds are 0-based; ``late`` keeps folds 5..9, i.e. expanding steps i = 6..10
# where fold i trains on i * N / (k + 1) samples.
LATE_FOLD = G.N_FOLDS // 2
LATE_STEPS = range(LATE_FOLD + 1, G.N_FOLDS + 1)

# Nadeau--Bengio factor 1/k + mean_i(n_test / n_train_i) over the retained
# folds, with k = 5 rather than 10.
NB_FACTOR_LATE = 1.0 / len(LATE_STEPS) + np.mean([1.0 / i for i in LATE_STEPS])
# t_{0.025, df} with df = N_SEEDS * (5 - 1) = 12.
NB_T_CRIT_LATE = 2.145

SCENARIOS = [("temporal", "test_temporal"), ("group", "test_group")]


def load_cv_and_test():
    """Return (fold-level CV scores, per-window test RMSE) as in the paper."""
    df = G.load_all_results()
    cv = df[df["metric"] == "val_rmse"]
    test = (
        df[df["metric"] == "test_rmse"]
        .groupby(["dataset", "window_size", "algorithm", "cv_strategy"])["value"]
        .median()
        .reset_index()
    )
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
    return cv, test_wide


def build(cv, test_wide, test_col, mode):
    """Per-cell CV mean, NB interval, and test RMSE under one aggregation rule."""
    best_windows = (
        test_wide
        .groupby(["dataset", "algorithm"])[test_col]
        .idxmin()
        .map(lambda i: test_wide.loc[i, ["dataset", "algorithm", "window_size"]])
    )
    best_windows = pd.DataFrame(best_windows.tolist()).drop_duplicates()
    merged = cv.merge(best_windows, on=["dataset", "algorithm", "window_size"])

    is_ts = merged["cv_strategy"] == "timeseries"
    late = merged["fold"].astype(int) >= LATE_FOLD
    keep_all = pd.Series(True, index=merged.index)

    if mode == "paper":
        keep_mean, keep_sd = keep_all, (~is_ts) | late
    elif mode == "all":
        keep_mean, keep_sd = keep_all, keep_all
    elif mode == "late-both":
        keep_mean = keep_sd = (~is_ts) | late
    else:
        raise ValueError(f"unknown mode: {mode}")

    key = ["dataset", "algorithm", "cv_strategy"]
    stats = (
        merged.loc[keep_mean].groupby(key)["value"].mean().rename("cv_mean").reset_index()
        .merge(
            merged.loc[keep_sd].groupby(key)["value"].std(ddof=1)
            .rename("cv_sd").reset_index(),
            on=key,
        )
    )

    # Under late-both the TimeSeriesSplit estimate rests on 5 folds, so its
    # inflation factor and t quantile change; other strategies keep theirs.
    ts_cell = stats["cv_strategy"] == "timeseries"
    use_late = ts_cell & (mode == "late-both")
    factor = stats["cv_strategy"].map(G.STRATEGY_NB_FACTOR).where(~use_late, NB_FACTOR_LATE)
    t_crit = np.where(use_late, NB_T_CRIT_LATE, G.NB_T_CRIT)

    half_width = t_crit * stats["cv_sd"] * np.sqrt(factor / G.N_SEEDS)
    stats["ci_low"] = stats["cv_mean"] - half_width
    stats["ci_high"] = stats["cv_mean"] + half_width
    stats["half_width"] = half_width

    test_best = test_wide.merge(best_windows, on=["dataset", "algorithm", "window_size"])
    out = stats.merge(test_best, on=["dataset", "algorithm"])
    truth = out[test_col].values
    out["covered"] = (truth >= out["ci_low"].values) & (truth <= out["ci_high"].values)
    out["bias_pct"] = (out["cv_mean"].values - truth) / truth * 100
    out["half_width_pct"] = out["half_width"].values / truth * 100
    return out


def mean_abs_bias(cell_data, strategy):
    """Paper convention: mean over datasets of |mean signed bias over algorithms|."""
    d = cell_data[cell_data["cv_strategy"] == strategy]
    return d.groupby("dataset")["bias_pct"].mean().abs().mean()


def signed_bias(cell_data, strategy):
    """Same aggregation, keeping the sign: positive means pessimistic."""
    d = cell_data[cell_data["cv_strategy"] == strategy]
    return d.groupby("dataset")["bias_pct"].mean().mean()


def main():
    cv, test_wide = load_cv_and_test()
    print(f"NB factor:  published (k=10) = {G.STRATEGY_NB_FACTOR['timeseries']:.4f}   "
          f"late-both (k=5) = {NB_FACTOR_LATE:.4f}")
    print(f"t quantile: published (df=27) = {G.NB_T_CRIT}   "
          f"late-both (df=12) = {NB_T_CRIT_LATE}\n")

    for scenario, test_col in SCENARIOS:
        cells = {m: build(cv, test_wide, test_col, m) for m in MODES}

        # The aggregation rule must not touch the other three strategies.
        for strategy in ["kfold", "group_kfold", "group_shuffle"]:
            counts = {m: int(cells[m].query("cv_strategy == @strategy")["covered"].sum())
                      for m in MODES}
            assert len(set(counts.values())) == 1, (strategy, counts)

        print(f"{scenario.upper()} HOLDOUT")
        print(f"  {'Strategy':<14}" + "".join(f"{m:>30}" for m in MODES))
        for strategy in G.STRATEGY_ORDER:
            row = f"  {G.STRATEGY_LABELS[strategy]:<14}"
            for m in MODES:
                d = cells[m][cells[m]["cv_strategy"] == strategy]
                row += (f"{int(d['covered'].sum())}/{len(d)}  "
                        f"bias={mean_abs_bias(cells[m], strategy):5.1f}%"
                        f"  w={d['half_width_pct'].median():5.1f}%").rjust(30)
            print(row)

        # Signed bias and interval width under the published rule; these back
        # the claim in Section 2.3 that no strategy is optimistic on average
        # and that KFold's intervals are the narrowest.
        print("  published rule, signed bias (+ = pessimistic) and cells pessimistic:")
        for strategy in G.STRATEGY_ORDER:
            d = cells["paper"][cells["paper"]["cv_strategy"] == strategy]
            print(f"    {G.STRATEGY_LABELS[strategy]:<14}"
                  f"{signed_bias(cells['paper'], strategy):+7.1f}%"
                  f"   {int((d['bias_pct'] > 0).sum())}/{len(d)} cells"
                  f"   median half-width {d['half_width_pct'].median():5.1f}%")

        ts = {m: cells[m][cells[m]["cv_strategy"] == "timeseries"] for m in MODES}
        print(f"  TimeSeries widest interval: "
              + ", ".join(f"{m}={ts[m]['half_width_pct'].max():.0f}% of test RMSE"
                          for m in MODES))
        pamap = {m: ts[m][ts[m]["dataset"] == "pamap2"]["bias_pct"].mean() for m in MODES}
        print("  TimeSeries PAMAP2 bias:     "
              + ", ".join(f"{m}={pamap[m]:+.1f}%" for m in MODES))
        print()


if __name__ == "__main__":
    main()
