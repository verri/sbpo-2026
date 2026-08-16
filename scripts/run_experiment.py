#!/usr/bin/env python3
"""Run all experiments for a single dataset across all window sizes and algorithms.

Usage:
    python scripts/run_experiment.py <dataset>

where <dataset> is one of: electricity, pamap2, mhealth, har, pemsbay, metrla.

For each (window_size, algorithm, seed) combination, the script:
1. Builds sliding windows from the preprocessed parquet.
2. Creates temporal (last 10%) and group (10% of groups) holdout sets,
   using `seed` for the group selection and for all stochastic splitters
   and estimators.
3. Runs 4 CV strategies (10 folds each) on the training data.
4. Trains a final model and evaluates on both holdouts (one-step-ahead).
5. Saves results to results/{dataset}_w{w}_{algo}_s{seed}.csv.

Each run is repeated across SEEDS to assess replication variability.
Existing result files are skipped (resume support at the per-seed level).
"""

# Suppress all warnings before any imports — this propagates to joblib workers
# via the env var, and to the main process via warnings.simplefilter.
import os
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.simplefilter("ignore")

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import as_strided
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    LeaveOneGroupOut,
    TimeSeriesSplit,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEEDS = [42, 43, 44]
WINDOWS = range(5, 16)
N_SPLITS = 10
ALGO_NAMES = ["ridge", "gbr", "mlp", "knn", "et"]
TEMPORAL_FRAC = 0.10
GROUP_FRAC = 0.10
RESULTS_DIR = Path("results")
DATA_DIR = Path("data/processed")

# Target feature for scoring. All features are used as inputs in the sliding
# window, but RMSE is computed only on this column. For single-feature datasets
# (electricity) the target is the only feature.
TARGET_COL = {
    "electricity": "power_mw",
    "pamap2": "heart_rate",
    "mhealth": "ecg_lead1",
    "har": "total_acc_x",
    "pemsbay": "speed_mph",
    "metrla": "speed_mph",
    "gsod": "temp_c",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_dataset(name):
    path = DATA_DIR / f"{name}.parquet"
    target_col = TARGET_COL[name]
    # Univariate: use only the target column as input and output.
    # Multivariate case (all features as input) is left as future work.
    df = pd.read_parquet(path, columns=["group", "time_step", target_col])
    feature_cols = [target_col]
    return df, feature_cols, target_col


# ---------------------------------------------------------------------------
# Sliding window construction
# ---------------------------------------------------------------------------


def build_sliding_windows(df, window_size, feature_cols, target_col):
    """Build X (w*d,) and y (scalar) arrays from all groups.

    All feature_cols are used as inputs in the window. The target is a single
    column (target_col) at the next time step.
    """
    d = len(feature_cols)
    target_idx = feature_cols.index(target_col)
    X_parts, y_parts, group_parts, tstep_parts = [], [], [], []

    for group_name, gdf in df.groupby("group"):
        values = gdf[feature_cols].values.astype(np.float32)
        T = len(values)
        n_samples = T - window_size
        if n_samples <= 0:
            continue

        # Use as_strided for zero-copy windowed view, then copy once.
        row_stride, col_stride = values.strides
        X_view = as_strided(
            values,
            shape=(n_samples, window_size, d),
            strides=(row_stride, row_stride, col_stride),
        )
        X_parts.append(X_view.reshape(n_samples, window_size * d).copy())
        # Single target column at the next time step.
        y_parts.append(values[window_size : window_size + n_samples, target_idx].copy())
        group_parts.append(np.full(n_samples, group_name, dtype=object))
        tstep_parts.append(gdf["time_step"].values[window_size : window_size + n_samples])

    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    groups = np.concatenate(group_parts)
    time_steps = np.concatenate(tstep_parts)
    return X, y, groups, time_steps


# ---------------------------------------------------------------------------
# Holdout creation
# ---------------------------------------------------------------------------


def create_holdouts(groups, time_steps, seed):
    """Create temporal and group holdout masks.

    Returns dict with:
        train_mask: bool array — training samples
        temporal_test_mask: bool array — last 10% of non-held-out groups
        group_test_mask: bool array — all samples from held-out groups
        holdout_groups: array of held-out group names
    """
    unique_groups = np.sort(np.unique(groups))
    rng = np.random.RandomState(seed)

    # Group holdout: 10% of groups.
    n_holdout = max(1, int(len(unique_groups) * GROUP_FRAC))
    holdout_groups = rng.choice(unique_groups, n_holdout, replace=False)
    group_test_mask = np.isin(groups, holdout_groups)

    # Temporal holdout: last 10% of time_steps per non-held-out group.
    temporal_test_mask = np.zeros(len(groups), dtype=bool)
    non_holdout_groups = unique_groups[~np.isin(unique_groups, holdout_groups)]
    for g in non_holdout_groups:
        g_mask = groups == g
        g_tsteps = time_steps[g_mask]
        cutoff = np.quantile(g_tsteps, 1 - TEMPORAL_FRAC)
        temporal_test_mask |= g_mask & (time_steps >= cutoff)

    # Training: not in either holdout.
    train_mask = ~temporal_test_mask & ~group_test_mask

    return {
        "train_mask": train_mask,
        "temporal_test_mask": temporal_test_mask,
        "group_test_mask": group_test_mask,
        "holdout_groups": holdout_groups,
    }


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def make_model(algo_name, seed):
    # Scaling matters for Ridge and MLP; harmless for tree-based methods.
    needs_scaling = algo_name in ("ridge", "knn", "mlp")

    estimators = {
        "ridge": Ridge(),
        "knn": KNeighborsRegressor(n_jobs=-1),
        "rf": RandomForestRegressor(
            n_estimators=10, random_state=seed, n_jobs=-1,
        ),
        "gbr": LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1),
        "mlp": MLPRegressor(random_state=seed, max_iter=50, tol=1e-3),
        "et": ExtraTreesRegressor(random_state=seed, n_jobs=-1),
    }
    estimator = estimators[algo_name]

    if needs_scaling:
        return Pipeline([("scaler", StandardScaler()), ("model", estimator)])
    return estimator


# ---------------------------------------------------------------------------
# CV strategies
# ---------------------------------------------------------------------------


def get_cv_splitters(seed):
    return {
        "kfold": KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed),
        "group_kfold": GroupKFold(n_splits=N_SPLITS),
        # TODO: re-enable LOGO once performance is acceptable.
        # "logo": LeaveOneGroupOut(),
        "group_shuffle": GroupShuffleSplit(
            n_splits=N_SPLITS, test_size=0.2, random_state=seed
        ),
        "timeseries": TimeSeriesSplit(n_splits=N_SPLITS),
    }


def run_all_cv(X_train, y_train, groups_train, time_steps_train,
               model_factory, seed, algo):
    """Run all CV strategies and return {strategy: [fold_scores]}.

    The only algo whose estimator is single-threaded is MLP, so we
    parallelise its CV folds (n_jobs=-1).  All other algos already use
    n_jobs=-1 internally; layering joblib on top would oversubscribe.
    """
    splitters = get_cv_splitters(seed)
    results = {}
    cv_n_jobs = -1 if algo == "mlp" else 1

    for name, splitter in splitters.items():
        model = model_factory()

        # Determine groups argument (some splitters need it).
        needs_groups = name in ("group_kfold", "logo", "group_shuffle")
        groups_arg = groups_train if needs_groups else None

        # For TimeSeriesSplit, sort by time_step so splits are temporal.
        if name == "timeseries":
            sort_idx = np.argsort(time_steps_train, kind="stable")
            X_cv, y_cv = X_train[sort_idx], y_train[sort_idx]
            groups_cv = groups_train[sort_idx] if groups_arg is not None else None
        else:
            X_cv, y_cv, groups_cv = X_train, y_train, groups_arg

        scores = cross_val_score(
            model,
            X_cv,
            y_cv,
            groups=groups_cv,
            cv=splitter,
            scoring="neg_root_mean_squared_error",
            n_jobs=cv_n_jobs,
        )
        fold_rmses = (-scores).tolist()
        results[name] = fold_rmses

        n_folds = len(fold_rmses)
        mean_rmse = np.mean(fold_rmses)
        std_rmse = np.std(fold_rmses)
        log(f"    CV {name} ({n_folds} folds): {mean_rmse:.6f} ± {std_rmse:.6f}")

    return results


# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------


# TODO: rolling window prediction (autoregressive) — disabled for now.
# To re-enable, implement rolling_window_predict() that feeds predictions
# back into the window, and call it per-group in evaluate_on_holdout().


def evaluate_on_holdout(model, df, feature_cols, target_col, holdout_mask,
                        groups, time_steps, window_size, mode):
    """Evaluate model on a holdout set (one-step-ahead, batched).

    mode: "temporal" — test portion is the tail of each non-held-out group.
          "group" — test is all data from held-out groups (after first w values).

    Returns RMSE (float) on the target column only.
    """
    n_features = len(feature_cols)
    target_idx = feature_cols.index(target_col)
    holdout_groups = np.unique(groups[holdout_mask])

    X_all, y_all = [], []

    for g in holdout_groups:
        gdf = df[df["group"] == g].sort_values("time_step")
        values = gdf[feature_cols].values.astype(np.float32)
        T = len(values)

        if T <= window_size:
            continue

        if mode == "temporal":
            g_mask = groups == g
            g_holdout = holdout_mask & g_mask
            n_test = int(g_holdout.sum())
            if n_test == 0:
                continue
            train_end = T - n_test
            if train_end < window_size:
                continue
            full_seq = values[train_end - window_size :]
        elif mode == "group":
            full_seq = values
        else:
            raise ValueError(f"Unknown mode: {mode}")

        n_osa = len(full_seq) - window_size
        if n_osa <= 0:
            continue

        row_stride, col_stride = full_seq.strides
        X_osa = as_strided(
            full_seq,
            shape=(n_osa, window_size, n_features),
            strides=(row_stride, row_stride, col_stride),
        ).reshape(n_osa, window_size * n_features).copy()
        y_osa = full_seq[window_size:, target_idx].copy()

        X_all.append(X_osa)
        y_all.append(y_osa)

    if not X_all:
        return float("nan")

    # Single batched predict call over all test groups.
    X_batch = np.concatenate(X_all)
    y_batch = np.concatenate(y_all)
    y_pred = model.predict(X_batch)
    return float(root_mean_squared_error(y_batch, y_pred))


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------


def result_path(dataset, window_size, algo, seed):
    return RESULTS_DIR / f"{dataset}_w{window_size}_{algo}_s{seed}.csv"


def save_results(dataset, window_size, algo, seed, cv_results, test_results):
    rows = []
    # CV fold scores.
    for strategy, fold_scores in cv_results.items():
        for i, score in enumerate(fold_scores):
            rows.append({
                "dataset": dataset,
                "window_size": window_size,
                "algorithm": algo,
                "seed": seed,
                "cv_strategy": strategy,
                "fold": i,
                "metric": "val_rmse",
                "value": score,
            })
    # Test scores.
    for key, value in test_results.items():
        rows.append({
            "dataset": dataset,
            "window_size": window_size,
            "algorithm": algo,
            "seed": seed,
            "cv_strategy": key,
            "fold": "final",
            "metric": "test_rmse",
            "value": value,
        })

    out = result_path(dataset, window_size, algo, seed)
    pd.DataFrame(rows).to_csv(out, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def fmt_duration(secs):
    """Format seconds as H:MM:SS (or M:SS for <1h)."""
    secs = int(max(0, secs))
    h, r = divmod(secs, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def main():
    valid_datasets = list(TARGET_COL.keys())
    if len(sys.argv) != 2 or sys.argv[1] not in valid_datasets:
        log(f"Usage: python scripts/run_experiment.py <{'|'.join(valid_datasets)}>")
        sys.exit(1)

    dataset = sys.argv[1]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log(f"=== Loading {dataset} ===")
    df, feature_cols, target_col = load_dataset(dataset)
    n_features = len(feature_cols)
    n_groups = df["group"].nunique()
    log(f"  {len(df):,} rows, {n_groups} groups, {n_features} features, target={target_col}")

    t_start = time.time()

    # Total unit count for ETA: one "unit" = one (window, algo, seed) result file.
    total_units = len(list(WINDOWS)) * len(ALGO_NAMES) * len(SEEDS)
    # Count already-done units upfront so ETA reflects remaining work.
    done_units = sum(
        1 for w in WINDOWS for algo in ALGO_NAMES for seed in SEEDS
        if result_path(dataset, w, algo, seed).exists()
    )
    completed_this_run = 0
    log(f"  Plan: {total_units} (window, algo, seed) units — "
        f"{done_units} already on disk, {total_units - done_units} to run.")

    def log_eta(elapsed_unit):
        # Running average over units completed in this invocation.
        nonlocal completed_this_run
        completed_this_run += 1
        avg = (time.time() - t_start) / completed_this_run
        remaining = total_units - done_units - completed_this_run
        eta = remaining * avg
        log(f"    [ETA] unit {completed_this_run}/{total_units - done_units} "
            f"done in {elapsed_unit:.1f}s; "
            f"avg {avg:.1f}s/unit; "
            f"remaining {remaining} → ~{fmt_duration(eta)} "
            f"(elapsed {fmt_duration(time.time() - t_start)}).")

    # Process largest windows first: they are slowest, so front-loading
    # them makes the running ETA estimate pessimistic early and more
    # accurate as the run progresses.
    for wi, w in enumerate(reversed(list(WINDOWS)), 1):
        log(f"\n--- Window size {w} ({wi}/{len(list(WINDOWS))}) ---")

        # Sliding windows and group arrays are seed-independent, so build once.
        X, y, groups, time_steps = build_sliding_windows(df, w, feature_cols, target_col)
        log(f"  Samples: {len(X):,} total")

        for algo in ALGO_NAMES:
            for seed in SEEDS:
                out_path = result_path(dataset, w, algo, seed)
                if out_path.exists():
                    log(f"  [{algo} seed={seed}] Already done, skipping.")
                    continue

                t_unit = time.time()

                # Holdouts depend on seed (group selection).
                holdouts = create_holdouts(groups, time_steps, seed)
                train_mask = holdouts["train_mask"]
                X_train = X[train_mask]
                y_train = y[train_mask]
                groups_train = groups[train_mask]
                time_steps_train = time_steps[train_mask]

                log(f"  [{algo} seed={seed}] train={len(X_train):,}, "
                    f"temporal_test={holdouts['temporal_test_mask'].sum():,}, "
                    f"group_test={holdouts['group_test_mask'].sum():,}")

                model_factory = lambda algo=algo, seed=seed: make_model(algo, seed)

                log(f"  [{algo} seed={seed}] Running CV ...")
                cv_results = run_all_cv(
                    X_train, y_train, groups_train, time_steps_train,
                    model_factory, seed, algo,
                )

                log(f"  [{algo} seed={seed}] Training final model ...")
                final_model = model_factory()
                final_model.fit(X_train, y_train)

                log(f"  [{algo} seed={seed}] Evaluating temporal holdout ...")
                temp_osa = evaluate_on_holdout(
                    final_model, df, feature_cols, target_col,
                    holdouts["temporal_test_mask"], groups, time_steps,
                    w, "temporal",
                )

                log(f"  [{algo} seed={seed}] Evaluating group holdout ...")
                group_osa = evaluate_on_holdout(
                    final_model, df, feature_cols, target_col,
                    holdouts["group_test_mask"], groups, time_steps,
                    w, "group",
                )

                test_results = {
                    "temporal_osa": temp_osa,
                    "group_osa": group_osa,
                }

                elapsed = time.time() - t_unit
                log(f"    Temporal RMSE={temp_osa:.6f}, Group RMSE={group_osa:.6f}")

                save_results(dataset, w, algo, seed, cv_results, test_results)
                log(f"    Saved {out_path} ({elapsed:.1f}s)")

                del final_model, X_train, y_train, groups_train, time_steps_train
                gc.collect()

                log_eta(elapsed)

        del X, y, groups, time_steps
        gc.collect()

    total = time.time() - t_start
    log(f"\n=== Done: {dataset} ({fmt_duration(total)} total) ===")


if __name__ == "__main__":
    main()
