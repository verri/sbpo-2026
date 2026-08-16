#!/usr/bin/env python3
"""Preprocess MHEALTH into tidy parquet.

Output format:
    group (str): "S{subject_id}_A{activity_id}"
    time_step (int): sequential index within the group
    <23 feature columns>: all sensor readings

Sensor layout per row (24 columns):
    0-2:   chest accelerometer (x, y, z)
    3-4:   ECG lead 1, lead 2
    5-7:   left-ankle accelerometer (x, y, z)
    8-10:  left-ankle gyroscope (x, y, z)
    11-13: left-ankle magnetometer (x, y, z)
    14-16: right-wrist accelerometer (x, y, z)
    17-19: right-wrist gyroscope (x, y, z)
    20-22: right-wrist magnetometer (x, y, z)
    23:    activity label (0 = no activity / transition)

Raw sampling rate is 50 Hz.  We block-average every 10 consecutive
samples within each (subject, activity) group to reduce the rate to
~5 Hz, applying mean-aggregation to every channel (including ECG)
without any anti-alias filter.

Groups with fewer than MIN_GROUP_LEN samples after aggregation are
dropped.  Activity label 0 (null/transition class) is excluded.
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/mhealth")
OUT_FILE = Path("data/processed/mhealth.parquet")

FEATURE_NAMES = [
    "chest_acc_x", "chest_acc_y", "chest_acc_z",
    "ecg_lead1", "ecg_lead2",
    "ankle_acc_x", "ankle_acc_y", "ankle_acc_z",
    "ankle_gyro_x", "ankle_gyro_y", "ankle_gyro_z",
    "ankle_mag_x", "ankle_mag_y", "ankle_mag_z",
    "wrist_acc_x", "wrist_acc_y", "wrist_acc_z",
    "wrist_gyro_x", "wrist_gyro_y", "wrist_gyro_z",
    "wrist_mag_x", "wrist_mag_y", "wrist_mag_z",
    "activity_label",
]

BLOCK_SIZE = 10               # 50 Hz raw → ~5 Hz aggregated
MIN_GROUP_LEN = 2 * 15 + 10   # 2 * max_window + horizon


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Find subject log files (mHealth_subject1.log, ..., mHealth_subject10.log).
    log_files = sorted(RAW_DIR.rglob("*.log"))
    print(f"Found {len(log_files)} subject files")

    all_groups = []

    for log_path in log_files:
        # Extract subject ID from filename.
        subject_id = int("".join(c for c in log_path.stem if c.isdigit()))
        print(f"  Processing subject {subject_id} ({log_path.name})")

        df = pd.read_csv(log_path, sep=r"\s+", header=None, names=FEATURE_NAMES)

        # Drop activity 0 (null class / transitions between activities).
        df = df[df["activity_label"] != 0]

        for activity_id, activity_df in df.groupby("activity_label"):
            features = activity_df.drop(columns=["activity_label"]).reset_index(drop=True)

            # Block-average every BLOCK_SIZE rows (50 Hz -> ~5 Hz).
            n_blocks = len(features) // BLOCK_SIZE
            if n_blocks < MIN_GROUP_LEN:
                continue

            trimmed = features.iloc[:n_blocks * BLOCK_SIZE]
            blocks = trimmed.index // BLOCK_SIZE
            aggregated = trimmed.groupby(blocks).mean().reset_index(drop=True)

            aggregated["group"] = f"S{subject_id}_A{int(activity_id)}"
            aggregated["time_step"] = range(len(aggregated))
            all_groups.append(aggregated)

    result = pd.concat(all_groups, ignore_index=True)

    # Drop any NaN rows.
    before = len(result)
    result = result.dropna()
    if len(result) < before:
        print(f"  Dropped {before - len(result)} rows with NaN")

    # Reorder: group, time_step, then features alphabetically.
    feature_cols = [c for c in result.columns if c not in ["group", "time_step"]]
    result = result[["group", "time_step"] + sorted(feature_cols)]

    result.to_parquet(OUT_FILE, index=False)
    n_groups = result["group"].nunique()
    print(f"  Saved {len(result)} rows, {n_groups} groups to {OUT_FILE}")


if __name__ == "__main__":
    main()
