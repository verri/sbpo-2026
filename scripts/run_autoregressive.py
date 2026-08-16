#!/usr/bin/env python3
"""Evaluate trained models with multi-step autoregressive prediction.

For a subset of datasets, retrain the final model at the oracle-best window
size and evaluate on temporal and group holdouts at horizons h=1, 5, 10.

Usage:
    python scripts/run_autoregressive.py

Produces results/autoregressive.csv with columns:
    dataset, window_size, algorithm, seed, horizon, scenario, rmse

The whole fit + AR evaluation is repeated for each seed in SEEDS so that
downstream analysis can aggregate across replications consistently with
the main experiment pipeline.
"""

import os
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.simplefilter("ignore")

import glob
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

# Reuse infrastructure from the main experiment.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiment import (
    ALGO_NAMES,
    DATA_DIR,
    RESULTS_DIR,
    SEEDS,
    WINDOWS,
    build_sliding_windows,
    create_holdouts,
    load_dataset,
    make_model,
)


def log(msg):
    print(msg, flush=True)


AR_DATASETS = [
    "electricity", "pamap2", "mhealth", "har",
    "pemsbay", "metrla", "gsod",
]
AR_ALGOS = list(ALGO_NAMES)
HORIZONS = [1, 5, 10]
OUT_PATH = RESULTS_DIR / "autoregressive.csv"


def find_best_window(dataset, algo, scenario):
    """Oracle-best window for (dataset, algo, scenario).

    Aggregates test RMSE across seeds (median) before selecting the window
    that minimises it, matching the main calibration pipeline.
    """
    test_key = "temporal_osa" if scenario == "temporal" else "group_osa"
    paths = sorted(glob.glob(str(RESULTS_DIR / f"{dataset}_w*_{algo}_s*.csv")))
    if not paths:
        return None
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    test = df[(df["cv_strategy"] == test_key) & (df["metric"] == "test_rmse")]
    if test.empty:
        return None
    # Median across seeds, then pick the window with the lowest median.
    agg = test.groupby("window_size")["value"].median()
    return int(agg.idxmin())


def autoregressive_evaluate(model, df, feature_cols, target_col,
                            holdout_mask, groups, time_steps,
                            window_size, mode, horizon):
    """Evaluate with h-step autoregressive prediction.

    For each group in the holdout:
    - Temporal mode: seed with the last w true values before the holdout boundary,
      then predict h steps feeding predictions back.
    - Group mode: use the first w true values of the held-out group as seed,
      then predict h steps.

    All rollouts across all groups/blocks advance in lockstep: at each step we
    call ``model.predict`` once on the full ``(n_rollouts, w)`` batch so that
    tree ensembles / KNN amortise their per-call overhead.

    Returns RMSE across all groups at the given horizon.
    """
    target_idx = feature_cols.index(target_col)
    holdout_groups = np.unique(groups[holdout_mask])

    seeds = []    # (w,) arrays — initial windows
    truths = []   # scalars — true target at the h-th step

    for g in holdout_groups:
        gdf = df[df["group"] == g].sort_values("time_step")
        values = gdf[feature_cols].values.astype(np.float32)
        T = len(values)

        if T <= window_size + horizon:
            continue

        if mode == "temporal":
            g_mask = groups == g
            g_holdout = holdout_mask & g_mask
            n_test = int(g_holdout.sum())
            if n_test < horizon:
                continue
            train_end = T - n_test
            if train_end < window_size:
                continue
            seq = values[train_end - window_size:]
        elif mode == "group":
            seq = values
        else:
            raise ValueError(f"Unknown mode: {mode}")

        n_available = len(seq) - window_size
        if n_available < horizon:
            continue

        n_blocks = n_available // horizon
        for b in range(n_blocks):
            start = window_size + b * horizon
            seed_start = start - window_size
            seeds.append(seq[seed_start:start, target_idx].copy())
            truths.append(float(seq[start + horizon - 1, target_idx]))

    if not seeds:
        return float("nan")

    windows = np.stack(seeds).astype(np.float32)
    for _ in range(horizon):
        preds = np.asarray(model.predict(windows), dtype=np.float32)
        windows = np.concatenate(
            [windows[:, 1:], preds.reshape(-1, 1)], axis=1
        )

    final_preds = windows[:, -1]
    truths_arr = np.asarray(truths, dtype=np.float64)
    return float(root_mean_squared_error(truths_arr, final_preds))


def _run_seed(dataset, algo, df, feature_cols, target_col,
              best_w, X, y, groups, time_steps, seed):
    """Fit one (ds, algo, seed) model and evaluate all (h, scenario) cells."""
    holdouts = create_holdouts(groups, time_steps, seed)
    train_mask = holdouts["train_mask"]

    model = make_model(algo, seed)
    model.fit(X[train_mask], y[train_mask])

    seed_rows = []
    for h in HORIZONS:
        for scenario, mask_key in [
            ("temporal", "temporal_test_mask"),
            ("group", "group_test_mask"),
        ]:
            rmse = autoregressive_evaluate(
                model, df, feature_cols, target_col,
                holdouts[mask_key], groups, time_steps,
                best_w, scenario, h,
            )
            seed_rows.append({
                "dataset": dataset,
                "window_size": best_w,
                "algorithm": algo,
                "seed": seed,
                "horizon": h,
                "scenario": scenario,
                "rmse": rmse,
            })
            log(f"    [{dataset}/{algo} seed={seed}] "
                f"h={h}, {scenario}: RMSE={rmse:.4f}")
    return seed_rows


def run_one_algo(dataset, algo, df, feature_cols, target_col):
    """Evaluate one (dataset, algo) combination across all seeds, horizons,
    and scenarios.  One oracle-best window is chosen per algorithm using the
    median test RMSE across seeds; the fit/AR-evaluation is then replicated
    per seed at that shared window.

    Seeds run in parallel only for MLP (single-threaded estimator); other
    algos already use n_jobs=-1 internally and would oversubscribe.
    """
    best_w = find_best_window(dataset, algo, "group")
    if best_w is None:
        return []

    log(f"  [{dataset}/{algo}] best window = {best_w}")

    X, y, groups, time_steps = build_sliding_windows(
        df, best_w, feature_cols, target_col
    )

    if algo == "mlp":
        from joblib import Parallel, delayed
        per_seed = Parallel(n_jobs=-1)(
            delayed(_run_seed)(
                dataset, algo, df, feature_cols, target_col,
                best_w, X, y, groups, time_steps, seed,
            )
            for seed in SEEDS
        )
    else:
        per_seed = [
            _run_seed(dataset, algo, df, feature_cols, target_col,
                      best_w, X, y, groups, time_steps, seed)
            for seed in SEEDS
        ]

    return [r for seed_rows in per_seed for r in seed_rows]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # Pre-load all datasets (I/O bound, done once).
    datasets = {}
    for ds in AR_DATASETS:
        log(f"Loading {ds} ...")
        datasets[ds] = load_dataset(ds)

    # Run sequentially: each underlying algo uses n_jobs=-1 internally, so
    # a layer of joblib parallelism above it would heavily oversubscribe
    # cores (confirmed on this machine — see CLAUDE.md parallelism note).
    tasks = [(ds, algo) for ds in AR_DATASETS for algo in AR_ALGOS]
    total = len(tasks)
    log(f"\nRunning {total} (dataset, algorithm) tasks sequentially ...")

    all_rows = []
    for i, (ds, algo) in enumerate(tasks, start=1):
        t_task = time.time()
        rows = run_one_algo(ds, algo, *datasets[ds])
        all_rows.extend(rows)
        elapsed = time.time() - t_start
        avg = elapsed / i
        remaining = (total - i) * avg
        log(f"  [{i}/{total}] {ds}/{algo} done in "
            f"{time.time() - t_task:.1f}s  "
            f"elapsed={elapsed:.0f}s  ETA={remaining:.0f}s")

    result_df = pd.DataFrame(all_rows)
    result_df.to_csv(OUT_PATH, index=False)
    log(f"\nSaved {OUT_PATH} ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
