#!/usr/bin/env python3
"""Preprocess UCI HAR Smartphones into tidy parquet.

Output format:
    group (str): "S{subject_id}_A{activity_id}"
    time_step (int): sequential index within the group
    <9 feature columns>: raw inertial signals

The raw data is pre-segmented into 128-sample windows with 50% overlap
at 50 Hz.  We reconstruct continuous time series per (subject, activity)
by taking the first 64 samples (non-overlapping portion) of each window,
plus all 128 samples of the last window.

Sensor channels (9):
    total_acc_x, total_acc_y, total_acc_z  (raw accelerometer, in g)
    body_acc_x, body_acc_y, body_acc_z     (gravity-subtracted accel)
    body_gyro_x, body_gyro_y, body_gyro_z  (gyroscope, rad/s)

We combine train and test splits (our experiment does its own splitting).
After reconstruction, each group is block-averaged every 10 consecutive
rows (50 Hz -> ~5 Hz) using mean aggregation.  Groups with fewer than
MIN_GROUP_LEN samples after block-averaging are dropped.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/har/UCI HAR Dataset")
OUT_FILE = Path("data/processed/har.parquet")

SIGNAL_FILES = [
    "total_acc_x", "total_acc_y", "total_acc_z",
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
]

BLOCK_SIZE = 10                # 50 Hz raw -> ~5 Hz aggregated
MIN_GROUP_LEN = 2 * 15 + 10    # 2 * max_window + horizon
WINDOW_SIZE = 128
STEP_SIZE = 64  # 50% overlap


def load_split(split_dir: Path):
    """Load one split (train or test) and return signals, subjects, activities."""
    subjects = np.loadtxt(split_dir / f"subject_{split_dir.name}.txt", dtype=int)
    activities = np.loadtxt(split_dir / f"y_{split_dir.name}.txt", dtype=int)

    signals = []
    sig_dir = split_dir / "Inertial Signals"
    for sig_name in SIGNAL_FILES:
        fname = f"{sig_name}_{split_dir.name}.txt"
        # Each row is 128 space-separated floats (one window).
        data = np.loadtxt(sig_dir / fname)
        signals.append(data)

    # signals: list of 9 arrays, each (n_windows, 128)
    return signals, subjects, activities


def reconstruct_continuous(window_indices, signals_list):
    """Reconstruct continuous time series from overlapping windows.

    Takes the first STEP_SIZE samples of each window except the last,
    from which all WINDOW_SIZE samples are taken.
    """
    n_windows = len(window_indices)
    if n_windows == 0:
        return np.empty((0, len(signals_list)))

    parts = []
    for i, wi in enumerate(window_indices):
        if i < n_windows - 1:
            # Take first STEP_SIZE samples (non-overlapping portion).
            row = np.column_stack([sig[wi, :STEP_SIZE] for sig in signals_list])
        else:
            # Last window: take all samples.
            row = np.column_stack([sig[wi, :] for sig in signals_list])
        parts.append(row)

    return np.vstack(parts)


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    all_groups = []

    for split_name in ["train", "test"]:
        split_dir = RAW_DIR / split_name
        print(f"  Loading {split_name} ...")
        signals, subjects, activities = load_split(split_dir)

        # Group by (subject, activity) and reconstruct.
        unique_pairs = set(zip(subjects, activities))
        for subj, act in sorted(unique_pairs):
            mask = (subjects == subj) & (activities == act)
            indices = np.where(mask)[0]

            values = reconstruct_continuous(indices, signals)
            # Block-average every BLOCK_SIZE rows (50 Hz -> ~5 Hz).
            n_blocks = len(values) // BLOCK_SIZE
            if n_blocks < MIN_GROUP_LEN:
                continue

            trimmed = values[:n_blocks * BLOCK_SIZE]
            aggregated = trimmed.reshape(n_blocks, BLOCK_SIZE, len(SIGNAL_FILES)).mean(axis=1)

            df = pd.DataFrame(aggregated, columns=SIGNAL_FILES)
            df["group"] = f"S{subj}_A{act}"
            df["time_step"] = range(len(df))
            all_groups.append(df)

    result = pd.concat(all_groups, ignore_index=True)
    result = result.dropna()

    feature_cols = [c for c in result.columns if c not in ("group", "time_step")]
    result = result[["group", "time_step"] + sorted(feature_cols)]

    result = result.astype({c: np.float32 for c in feature_cols})
    result.to_parquet(OUT_FILE, index=False)
    n_groups = result["group"].nunique()
    print(f"  Saved {len(result):,} rows, {n_groups} groups to {OUT_FILE}")


if __name__ == "__main__":
    main()
