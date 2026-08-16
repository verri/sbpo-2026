#!/usr/bin/env python3
"""Generate LaTeX tables comparing CV fold ranges vs actual test RMSE.

Produces:
    paper/tab_group.tex             — detailed, best window for group holdout
    paper/tab_temporal.tex          — detailed, best window for temporal holdout
    paper/tab_group_summary.tex     — compact summary for group holdout
    paper/tab_temporal_summary.tex  — compact summary for temporal holdout
    paper/tab_scale.tex             — dataset scale and experiment count
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path("results")
DATA_DIR = Path("data/processed")
OUT_DIR = Path("paper")
MAX_WINDOW = 15
MIN_WINDOW = 5

STRATEGY_LABELS = {
    "kfold": "KFold",
    "group_kfold": "GroupKFold",
    "group_shuffle": "GroupShuffle",
    "timeseries": "TimeSeries",
}

ALGO_LABELS = {
    "ridge": "Ridge",
    "gbr": "HistGBR",
    "mlp": "MLP",
    "knn": "KNN",
    "et": "ExtraTrees",
}

DATASET_SHORT = {
    "electricity": "Electricity",
    "pemsbay": "PEMS-BAY",
    "metrla": "METR-LA",
    "gsod": "GSOD",
    "pamap2": "PAMAP2",
    "mhealth": "MHEALTH",
    "har": "HAR",
}

DATASET_DETAIL = {
    "electricity": "Electricity (daily MW, 370 clients)",
    "pemsbay": "PEMS-BAY (hourly mph, 100 sensors)",
    "metrla": "METR-LA (hourly mph, 173 sensors)",
    "gsod": r"GSOD (daily {\textdegree}C, 300 stations)",
    "pamap2": r"PAMAP2 ({\raise.17ex\hbox{$\scriptstyle\sim$}}9\,Hz, 89 groups)",
    "mhealth": r"MHEALTH (50\,Hz, 120 groups)",
    "har": r"HAR (50\,Hz, 180 groups)",
}

# Display order.
DATASET_ORDER = ["electricity", "pemsbay", "metrla", "gsod", "pamap2", "mhealth", "har"]
ALGO_ORDER = ["ridge", "gbr", "mlp", "knn", "et"]
STRATEGY_ORDER = ["kfold", "group_kfold", "group_shuffle", "timeseries"]

N_WINDOWS = MAX_WINDOW - MIN_WINDOW + 1
N_CV_STRATEGIES = len(STRATEGY_ORDER)
N_FOLDS = 10
N_SEEDS = 3

# Nadeau--Bengio variance inflation factor 1/k + n_test/n_train per strategy.
# For the expanding-window TimeSeriesSplit (sklearn default: n_test = N/(k+1),
# n_train_i = i * N/(k+1)) we average n_test/n_train across folds.
STRATEGY_NB_FACTOR = {
    "kfold":         1.0 / N_FOLDS + 1.0 / (N_FOLDS - 1),
    "group_kfold":   1.0 / N_FOLDS + 1.0 / (N_FOLDS - 1),
    "group_shuffle": 1.0 / N_FOLDS + 0.20 / 0.80,
    "timeseries":    1.0 / N_FOLDS + sum(1.0 / i for i in range(1, N_FOLDS + 1)) / N_FOLDS,
}
# t_{0.025, df} with df = N_SEEDS * (N_FOLDS - 1) = 27.
NB_T_CRIT = 2.052


def load_all_results():
    files = sorted(glob.glob(str(RESULTS_DIR / "*_s*.csv")))
    if not files:
        raise FileNotFoundError("No seeded CSV files found in results/")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["window_size"] <= MAX_WINDOW]
    # Exclude datasets not in our display order.
    df = df[df["dataset"].isin(DATASET_ORDER)]
    return df


def build_table_data(cv, test_wide, test_col):
    """Build merged data at best window for a given test scenario."""
    best_windows = (
        test_wide
        .groupby(["dataset", "algorithm"])[test_col]
        .idxmin()
        .map(lambda i: test_wide.loc[i, ["dataset", "algorithm", "window_size"]])
    )
    best_windows = pd.DataFrame(best_windows.tolist()).drop_duplicates()

    merged = cv.merge(best_windows, on=["dataset", "algorithm", "window_size"])
    # For TimeSeriesSplit, estimate fold-score SD from the second half of
    # folds only (fold >= N_FOLDS/2), where the expanding training set has
    # grown large enough to stabilise; this mitigates the heteroscedasticity
    # of early TimeSeriesSplit folds that plain pooling inflates.
    ts_mask = merged["cv_strategy"] == "timeseries"
    fold_int = merged["fold"].astype(int)
    keep_for_sd = (~ts_mask) | (fold_int >= N_FOLDS // 2)
    cv_stats = (
        merged.groupby(["dataset", "algorithm", "cv_strategy"])
        .agg(
            cv_mean=("value", "mean"),
            cv_min=("value", "min"),
            cv_max=("value", "max"),
        )
        .reset_index()
    )
    sd_rows = (
        merged.loc[keep_for_sd]
        .groupby(["dataset", "algorithm", "cv_strategy"])["value"]
        .std(ddof=1)
        .rename("cv_sd")
        .reset_index()
    )
    cv_stats = cv_stats.merge(
        sd_rows, on=["dataset", "algorithm", "cv_strategy"]
    )
    nb_se = cv_stats["cv_sd"] * np.sqrt(
        cv_stats["cv_strategy"].map(STRATEGY_NB_FACTOR) / N_SEEDS
    )
    cv_stats["cv_nb_se"] = nb_se
    cv_stats["cv_ci_low"] = cv_stats["cv_mean"] - NB_T_CRIT * nb_se
    cv_stats["cv_ci_high"] = cv_stats["cv_mean"] + NB_T_CRIT * nb_se

    test_best = test_wide.merge(best_windows, on=["dataset", "algorithm", "window_size"])
    return cv_stats.merge(test_best, on=["dataset", "algorithm"])


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt(val, precision=2):
    return f"{val:.{precision}f}"


def fmt_pct(val):
    """Format a signed percentage with LaTeX minus."""
    if val >= 0:
        return f"+{val:.1f}"
    return f"$-${abs(val):.1f}"


def fmt_count(n):
    """Format an integer with comma thousand separators."""
    return f"{n:,}"


# ---------------------------------------------------------------------------
# Detailed per-algorithm table (for appendix / repository)
# ---------------------------------------------------------------------------

def generate_table(merged, test_col, caption, label):
    col_header = "G.test" if "group" in test_col else "T.test"

    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\smallskip")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}llrrrrcrc@{}}")
    lines.append(r"\toprule")
    lines.append(
        f"Algorithm & Strategy & CV mean & CI low & CI high & {col_header} "
        r"& $\Delta$\% & inCI \\"
    )
    lines.append(r"\midrule")

    for ds in DATASET_ORDER:
        ds_data = merged[merged["dataset"] == ds]
        if ds_data.empty:
            continue

        ds_label = DATASET_DETAIL.get(ds, ds)
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{ds_label}}}}} \\\\")
        lines.append(r"\addlinespace")

        for algo in ALGO_ORDER:
            algo_data = ds_data[ds_data["algorithm"] == algo]
            if algo_data.empty:
                continue

            first_row = True
            for strat in STRATEGY_ORDER:
                row = algo_data[algo_data["cv_strategy"] == strat]
                if row.empty:
                    continue
                row = row.iloc[0]

                test_val = row[test_col]
                pct_diff = (row["cv_mean"] - test_val) / test_val * 100
                in_ci = row["cv_ci_low"] <= test_val <= row["cv_ci_high"]

                algo_str = ALGO_LABELS.get(algo, algo) if first_row else ""
                strat_str = STRATEGY_LABELS.get(strat, strat)

                lines.append(
                    f"{algo_str:<13}& {strat_str:<16}"
                    f"& {fmt(row['cv_mean'])} & {fmt(row['cv_ci_low'])} "
                    f"& {fmt(row['cv_ci_high'])} & {fmt(test_val)} "
                    f"& {fmt_pct(pct_diff)} "
                    f"& {'Y' if in_ci else 'N'} \\\\"
                )
                first_row = False

        lines.append(r"\addlinespace")

    if lines[-1] == r"\addlinespace":
        lines.pop()

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scenario-specific summary table (one per holdout scenario)
# ---------------------------------------------------------------------------

def generate_scenario_summary(data, test_col, caption, label):
    """One row per dataset, 4 strategies × (Coverage, Δ%)."""

    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\smallskip")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}l" + "rr" * len(STRATEGY_ORDER) + r"@{}}")
    lines.append(r"\toprule")

    # Header row 1: strategy names.
    header1 = "& "
    for strat in STRATEGY_ORDER:
        header1 += r"\multicolumn{2}{c}{" + STRATEGY_LABELS[strat] + "} & "
    header1 = header1.rstrip("& ") + r" \\"
    lines.append(header1)

    # Cmidrules.
    cmi = ""
    for i in range(len(STRATEGY_ORDER)):
        col_start = 2 + i * 2
        cmi += f"\\cmidrule(lr){{{col_start}-{col_start+1}}} "
    lines.append(cmi)

    # Header row 2.
    header2 = "Dataset "
    for _ in STRATEGY_ORDER:
        header2 += r"& Cov. & $\Delta$\% "
    header2 += r"\\"
    lines.append(header2)
    lines.append(r"\midrule")

    for ds in DATASET_ORDER:
        ds_data = data[data["dataset"] == ds]
        if ds_data.empty:
            continue

        ds_short = DATASET_SHORT.get(ds, ds)
        row = f"{ds_short:<16} "

        row_stats = {}
        for strat in STRATEGY_ORDER:
            strat_data = ds_data[ds_data["cv_strategy"] == strat]
            if strat_data.empty:
                row_stats[strat] = None
                continue
            n_algos = len(strat_data)
            test_vals = strat_data[test_col]
            in_ci = (
                (test_vals.values >= strat_data["cv_ci_low"].values)
                & (test_vals.values <= strat_data["cv_ci_high"].values)
            )
            coverage = int(in_ci.sum())
            pct_diffs = (
                (strat_data["cv_mean"].values - test_vals.values)
                / test_vals.values
                * 100
            )
            mean_bias = pct_diffs.mean()
            row_stats[strat] = (coverage, n_algos, mean_bias)

        valid = [v for v in row_stats.values() if v is not None]
        covs = [v[0] for v in valid]
        abs_biases = [abs(v[2]) for v in valid]
        # Only bold when strategies actually differ on that metric.
        best_cov = max(covs) if valid and min(covs) != max(covs) else None
        best_abs_bias = (
            min(abs_biases)
            if valid and min(abs_biases) != max(abs_biases)
            else None
        )

        for strat in STRATEGY_ORDER:
            v = row_stats[strat]
            if v is None:
                row += "& -- & -- "
                continue
            coverage, n_algos, mean_bias = v
            cov_str = f"{coverage}/{n_algos}"
            bias_str = fmt_pct(mean_bias)
            if best_cov is not None and coverage == best_cov:
                cov_str = r"\textbf{" + cov_str + "}"
            if best_abs_bias is not None and abs(mean_bias) == best_abs_bias:
                bias_str = r"\textbf{" + bias_str + "}"
            row += f"& {cov_str} & {bias_str} "

        row += r"\\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CV-selected window table
# ---------------------------------------------------------------------------

def generate_cv_selection_table(cv, test_wide):
    """Table comparing test RMSE when window is selected by each CV strategy
    vs. the oracle (best test RMSE).

    For each (dataset, algo, strategy), pick the window minimising CV mean.
    Report the test RMSE at that window vs. the oracle-best window.
    """

    # Oracle: best window per (dataset, algo) for each scenario.
    oracle_group = (
        test_wide.groupby(["dataset", "algorithm"])
        .apply(lambda g: g.loc[g["test_group"].idxmin()], include_groups=False)
        [["window_size", "test_group"]]
        .rename(columns={"window_size": "oracle_w", "test_group": "oracle_rmse"})
        .reset_index()
    )
    oracle_temporal = (
        test_wide.groupby(["dataset", "algorithm"])
        .apply(lambda g: g.loc[g["test_temporal"].idxmin()], include_groups=False)
        [["window_size", "test_temporal"]]
        .rename(columns={"window_size": "oracle_w", "test_temporal": "oracle_rmse"})
        .reset_index()
    )

    # CV mean per (dataset, algo, strategy, window).
    cv_mean = (
        cv.groupby(["dataset", "algorithm", "cv_strategy", "window_size"])["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": "cv_mean"})
    )

    # Best window per CV strategy.
    cv_best = (
        cv_mean
        .sort_values("cv_mean")
        .groupby(["dataset", "algorithm", "cv_strategy"])
        .first()
        .reset_index()
        .rename(columns={"window_size": "cv_w"})
    )

    rows = []
    for scenario, oracle_df, test_col in [
        ("group", oracle_group, "test_group"),
        ("temporal", oracle_temporal, "test_temporal"),
    ]:
        for _, ob in oracle_df.iterrows():
            ds, algo = ob["dataset"], ob["algorithm"]
            for strat in STRATEGY_ORDER:
                cv_row = cv_best[
                    (cv_best["dataset"] == ds) &
                    (cv_best["algorithm"] == algo) &
                    (cv_best["cv_strategy"] == strat)
                ]
                if cv_row.empty:
                    continue
                cv_w = int(cv_row.iloc[0]["cv_w"])

                # Test RMSE at the CV-selected window.
                tw = test_wide[
                    (test_wide["dataset"] == ds) &
                    (test_wide["algorithm"] == algo) &
                    (test_wide["window_size"] == cv_w)
                ]
                if tw.empty:
                    continue
                test_at_cv_w = float(tw.iloc[0][test_col])

                rows.append({
                    "scenario": scenario,
                    "dataset": ds,
                    "algorithm": algo,
                    "cv_strategy": strat,
                    "cv_w": cv_w,
                    "oracle_w": int(ob["oracle_w"]),
                    "test_at_cv_w": test_at_cv_w,
                    "oracle_rmse": ob["oracle_rmse"],
                })

    sel = pd.DataFrame(rows)
    sel["regret_pct"] = (sel["test_at_cv_w"] - sel["oracle_rmse"]) / sel["oracle_rmse"] * 100

    # Summary: median regret per (scenario, dataset, strategy).
    summary = (
        sel.groupby(["scenario", "dataset", "cv_strategy"])["regret_pct"]
        .median()
        .reset_index()
    )

    # One table covering both scenarios side by side.  Strategy names are
    # abbreviated in the header so that eight numeric columns fit the width.
    abbrev = {
        "kfold": "KF",
        "group_kfold": "GKF",
        "group_shuffle": "GSS",
        "timeseries": "TS",
    }
    scenarios = [("temporal", "Temporal holdout"), ("group", "Group holdout")]
    n_strat = len(STRATEGY_ORDER)

    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Median regret (\%) of the CV-selected window.  "
        r"KF: KFold; GKF: GroupKFold; GSS: GroupShuffle; TS: TimeSeries.}"
    )
    lines.append(r"\label{tab:selection}")
    lines.append(r"\smallskip")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}l" + "r" * (n_strat * len(scenarios)) + r"@{}}")
    lines.append(r"\toprule")

    header1 = "& " + " & ".join(
        r"\multicolumn{" + str(n_strat) + r"}{c}{" + title + "}"
        for _, title in scenarios
    ) + r" \\"
    lines.append(header1)
    lines.append(" ".join(
        f"\\cmidrule(lr){{{2 + i * n_strat}-{1 + (i + 1) * n_strat}}}"
        for i in range(len(scenarios))
    ))

    header2 = "Dataset " + "".join(
        f"& {abbrev[strat]} " for _ in scenarios for strat in STRATEGY_ORDER
    ) + r"\\"
    lines.append(header2)
    lines.append(r"\midrule")

    for ds in DATASET_ORDER:
        row = f"{DATASET_SHORT.get(ds, ds):<16} "
        for scenario, _ in scenarios:
            sc = summary[summary["scenario"] == scenario]
            vals = {}
            for strat in STRATEGY_ORDER:
                val = sc[
                    (sc["dataset"] == ds) & (sc["cv_strategy"] == strat)
                ]["regret_pct"]
                vals[strat] = float(val.iloc[0]) if not val.empty else None
            numeric = [v for v in vals.values() if v is not None]
            # Bold the best strategy within each scenario block.
            best_regret = (
                min(numeric)
                if numeric and min(numeric) != max(numeric)
                else None
            )
            for strat in STRATEGY_ORDER:
                v = vals[strat]
                if v is None:
                    row += "& -- "
                    continue
                cell = f"{v:+.1f}"
                if best_regret is not None and v == best_regret:
                    cell = r"\textbf{" + cell + "}"
                row += f"& {cell} "
        row += r"\\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dataset scale table
# ---------------------------------------------------------------------------

def generate_scale_table():
    """Table showing dataset sizes and sliding-window sample counts."""

    rows = []
    for ds in DATASET_ORDER:
        path = DATA_DIR / f"{ds}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        n_groups = df["group"].nunique()
        grp_sizes = df.groupby("group").size()
        min_steps = int(grp_sizes.min())
        max_steps = int(grp_sizes.max())

        # Sliding window sample counts.
        samples_wmin = sum(max(0, sz - MIN_WINDOW) for sz in grp_sizes)
        samples_wmax = sum(max(0, sz - MAX_WINDOW) for sz in grp_sizes)

        if min_steps == max_steps:
            steps_str = fmt_count(min_steps)
        else:
            steps_str = f"{fmt_count(min_steps)}--{fmt_count(max_steps)}"

        rows.append({
            "ds": ds,
            "n_groups": n_groups,
            "steps_str": steps_str,
            "samples_wmin": samples_wmin,
            "samples_wmax": samples_wmax,
        })

    n_datasets = len(rows)
    n_algos = len(ALGO_ORDER)
    total_configs = n_datasets * N_WINDOWS * n_algos
    total_fits = total_configs * N_SEEDS * (N_CV_STRATEGIES * N_FOLDS + 1)

    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Scale of the experimental pipeline.}"
    )
    lines.append(r"\label{tab:scale}")
    lines.append(r"\smallskip")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}lrrrr@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"Dataset & Groups & Steps/group & "
        rf"Samples ($w\!=\!{MIN_WINDOW}$) & Samples ($w\!=\!{MAX_WINDOW}$) \\"
    )
    lines.append(r"\midrule")

    for r in rows:
        ds_short = DATASET_SHORT.get(r["ds"], r["ds"])
        lines.append(
            f"{ds_short:<16}"
            f"& {r['n_groups']} "
            f"& {r['steps_str']} "
            f"& {fmt_count(r['samples_wmin'])} "
            f"& {fmt_count(r['samples_wmax'])} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Autoregressive evaluation table
# ---------------------------------------------------------------------------

def generate_autoregressive_table(cv, test_wide):
    """Table showing how test RMSE changes with autoregressive horizon.

    Compares CV mean RMSE (one-step) with test RMSE at h=1, 5, 10 for both
    holdout scenarios.  Shows Delta% (CV bias) at each horizon.
    """
    ar_path = RESULTS_DIR / "autoregressive.csv"
    if not ar_path.exists():
        print(f"  Skipping autoregressive table ({ar_path} not found)")
        return None

    ar = pd.read_csv(ar_path)

    # Collapse across seeds (median) so every (dataset, window, algo, horizon,
    # scenario) maps to a single RMSE before joining with CV means.
    if "seed" in ar.columns:
        ar = (
            ar.groupby(
                ["dataset", "window_size", "algorithm", "horizon", "scenario"]
            )["rmse"]
            .median()
            .reset_index()
        )

    # Get CV mean at the same window sizes used in AR experiments.
    cv_mean = (
        cv.groupby(["dataset", "algorithm", "cv_strategy", "window_size"])["value"]
        .mean()
        .reset_index()
        .rename(columns={"value": "cv_mean"})
    )

    lines = []
    lines.append(r"\begin{table}")
    lines.append(r"\centering")
    lines.append(
        r"\caption{CV bias ($\Delta$\%) at increasing autoregressive horizons.  "
        r"CV mean is always one-step-ahead; test RMSE is evaluated at "
        r"horizon $h$.  Median across algorithms.}"
    )
    lines.append(r"\label{tab:autoregressive}")
    lines.append(r"\smallskip")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}llrrr|rrr@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"& & \multicolumn{3}{c}{Temporal holdout} "
        r"& \multicolumn{3}{c}{Group holdout} \\"
    )
    lines.append(r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}")
    lines.append(
        r"Dataset & Strategy & $h{=}1$ & $h{=}5$ & $h{=}10$ "
        r"& $h{=}1$ & $h{=}5$ & $h{=}10$ \\"
    )
    lines.append(r"\midrule")

    ar_datasets = sorted(ar["dataset"].unique(), key=lambda d: DATASET_ORDER.index(d) if d in DATASET_ORDER else 99)

    for ds in ar_datasets:
        ds_short = DATASET_SHORT.get(ds, ds)
        first_ds = True

        for strat in STRATEGY_ORDER:
            vals = []
            for scenario in ["temporal", "group"]:
                for h in [1, 5, 10]:
                    # Get AR test RMSE for this (ds, strat=N/A, h, scenario).
                    ar_sub = ar[
                        (ar["dataset"] == ds) &
                        (ar["horizon"] == h) &
                        (ar["scenario"] == scenario)
                    ]
                    if ar_sub.empty:
                        vals.append(None)
                        continue

                    # Get CV mean for same (ds, algo, strat) at same windows.
                    biases = []
                    for _, ar_row in ar_sub.iterrows():
                        algo = ar_row["algorithm"]
                        w = ar_row["window_size"]
                        test_rmse = ar_row["rmse"]
                        if np.isnan(test_rmse) or test_rmse == 0:
                            continue
                        cv_row = cv_mean[
                            (cv_mean["dataset"] == ds) &
                            (cv_mean["algorithm"] == algo) &
                            (cv_mean["cv_strategy"] == strat) &
                            (cv_mean["window_size"] == w)
                        ]
                        if cv_row.empty:
                            continue
                        cv_val = float(cv_row.iloc[0]["cv_mean"])
                        biases.append((cv_val - test_rmse) / test_rmse * 100)

                    if biases:
                        vals.append(float(np.median(biases)))
                    else:
                        vals.append(None)

            ds_label = ds_short if first_ds else ""
            strat_label = STRATEGY_LABELS.get(strat, strat)
            first_ds = False

            cells = []
            for v in vals:
                if v is None:
                    cells.append("--")
                else:
                    cells.append(fmt_pct(v))

            lines.append(
                f"{ds_label:<13}& {strat_label:<16}"
                f"& {cells[0]} & {cells[1]} & {cells[2]} "
                f"& {cells[3]} & {cells[4]} & {cells[5]} \\\\"
            )

        lines.append(r"\addlinespace")

    if lines[-1] == r"\addlinespace":
        lines.pop()

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_all_results()
    cv = df[df["metric"] == "val_rmse"]
    # Collapse test RMSE across seeds (median) before pivoting so each
    # (dataset, window, algorithm, cv_strategy) has a single value.
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

    # Detailed per-algorithm tables.
    group_data = build_table_data(cv, test_wide, "test_group")
    (OUT_DIR / "tab_group.tex").write_text(generate_table(
        group_data, "test_group",
        caption=(
            r"CV vs.\ test RMSE at best window for the group holdout scenario. "
            r"$\Delta$\%: signed percentage bias of CV mean relative to test. "
            r"inCI: test RMSE falls within the 95\% Nadeau--Bengio CI."
        ),
        label="tab:group",
    ))
    print(f"Saved {OUT_DIR / 'tab_group.tex'}")

    temporal_data = build_table_data(cv, test_wide, "test_temporal")
    (OUT_DIR / "tab_temporal.tex").write_text(generate_table(
        temporal_data, "test_temporal",
        caption=(
            r"CV vs.\ test RMSE at best window for the temporal holdout "
            r"scenario. Same notation as Table~\ref{tab:group}."
        ),
        label="tab:temporal",
    ))
    print(f"Saved {OUT_DIR / 'tab_temporal.tex'}")

    # Scenario-specific summary tables.
    (OUT_DIR / "tab_temporal_summary.tex").write_text(generate_scenario_summary(
        temporal_data, "test_temporal",
        caption=r"CV calibration for the temporal holdout scenario.",
        label="tab:temporal_summary",
    ))
    print(f"Saved {OUT_DIR / 'tab_temporal_summary.tex'}")

    (OUT_DIR / "tab_group_summary.tex").write_text(generate_scenario_summary(
        group_data, "test_group",
        caption=r"CV calibration for the group holdout scenario.",
        label="tab:group_summary",
    ))
    print(f"Saved {OUT_DIR / 'tab_group_summary.tex'}")

    # Dataset scale table.
    (OUT_DIR / "tab_scale.tex").write_text(generate_scale_table())
    print(f"Saved {OUT_DIR / 'tab_scale.tex'}")

    # CV-selected window tables.
    selection_path = OUT_DIR / "tab_selection.tex"
    selection_path.write_text(generate_cv_selection_table(cv, test_wide))
    print(f"Saved {selection_path}")

    # Autoregressive evaluation table.
    ar_tex = generate_autoregressive_table(cv, test_wide)
    if ar_tex is not None:
        (OUT_DIR / "tab_autoregressive.tex").write_text(ar_tex)
        print(f"Saved {OUT_DIR / 'tab_autoregressive.tex'}")


if __name__ == "__main__":
    main()
