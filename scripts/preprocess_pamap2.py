#!/usr/bin/env python3
"""Preprocess PAMAP2 into tidy parquet.

Output format:
    group (str): "S{subject_id}_A{activity_id}"
    time_step (int): sequential index within the group
    <feature columns>: all valid sensor readings

Sensor layout per row (54 columns):
    0: timestamp (s)
    1: activityID
    2: heart rate (bpm) — sampled at ~9 Hz
    3-19:  IMU hand    (17 cols: temp, acc16g×3, acc6g×3, gyro×3, mag×3, orient×4)
    20-36: IMU chest   (same layout)
    37-53: IMU ankle   (same layout)

Orientation columns (indices 16-19, 33-36, 50-53 within each IMU block) are
invalid and dropped.

Resampling strategy: We block-average every 100 consecutive raw rows
(100 Hz → ~1 Hz) within each (subject, activity) group, going directly
from the 100 Hz IMU rate to ~1 Hz. Heart rate (sampled at ~9 Hz in the
raw file) is forward-filled before averaging so each bin has a value.

Groups with fewer than MIN_GROUP_LEN samples after resampling are
dropped (transient activities or very short segments).
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/pamap2")
OUT_FILE = Path("data/processed/pamap2.parquet")

# Column names for the 54-column format.
IMU_SENSORS = ["temp", "acc16g_x", "acc16g_y", "acc16g_z",
               "acc6g_x", "acc6g_y", "acc6g_z",
               "gyro_x", "gyro_y", "gyro_z",
               "mag_x", "mag_y", "mag_z",
               "orient_1", "orient_2", "orient_3", "orient_4"]

COLUMN_NAMES = ["timestamp", "activityID", "heart_rate"]
for location in ["hand", "chest", "ankle"]:
    for sensor in IMU_SENSORS:
        COLUMN_NAMES.append(f"{location}_{sensor}")

# Orientation columns to drop (invalid in this dataset).
ORIENT_COLS = [c for c in COLUMN_NAMES if "orient" in c]

BLOCK_SIZE = 100  # 100 Hz raw → ~1 Hz aggregated
MIN_GROUP_LEN = 2 * 15 + 10  # 2 * max_window + horizon


def load_subject(dat_path: Path, subject_id: int) -> pd.DataFrame:
    """Load a single subject .dat file."""
    df = pd.read_csv(dat_path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    df["subject_id"] = subject_id
    return df


def resample_group(group_df: pd.DataFrame) -> pd.DataFrame:
    """Resample a single (subject, activity) group from 100 Hz to ~1 Hz.

    Strategy: heart rate has NaN where no reading exists (sensor reports
    at ~9 Hz among 100 Hz rows).  We forward-fill heart rate, then
    block-average every BLOCK_SIZE consecutive rows.
    """
    df = group_df.sort_values("timestamp").reset_index(drop=True)

    # Forward-fill heart rate NaNs.
    df["heart_rate"] = df["heart_rate"].ffill().bfill()

    n_blocks = len(df) // BLOCK_SIZE
    if n_blocks < 1:
        return pd.DataFrame()

    df_trimmed = df.iloc[:n_blocks * BLOCK_SIZE]
    feature_cols = [c for c in df.columns if c not in
                    ["timestamp", "activityID", "subject_id"] + ORIENT_COLS]

    blocks = np.arange(len(df_trimmed)) // BLOCK_SIZE
    resampled = df_trimmed[feature_cols].groupby(blocks).mean()

    return resampled.reset_index(drop=True)


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Use only Protocol directory (Optional has extra activities not shared
    # across all subjects, which would create incomplete groups).
    protocol_dir = RAW_DIR / "PAMAP2_Dataset" / "Protocol"
    dat_files = sorted(protocol_dir.glob("subject*.dat"))
    if not dat_files:
        # Fallback: search recursively but only in Protocol folders.
        dat_files = sorted(p for p in RAW_DIR.rglob("Protocol/subject*.dat"))
    print(f"Found {len(dat_files)} subject files")

    all_groups = []

    for dat_path in dat_files:
        # Extract subject ID from filename (e.g., "subject101.dat" → 101).
        subject_id = int("".join(c for c in dat_path.stem if c.isdigit()))
        print(f"  Processing subject {subject_id} ({dat_path.name})")

        df = load_subject(dat_path, subject_id)

        # Drop orientation columns and rows with activityID == 0 (transient).
        df = df.drop(columns=ORIENT_COLS)
        df = df[df["activityID"] != 0]

        # Process each (subject, activity) group.
        for activity_id, activity_df in df.groupby("activityID"):
            resampled = resample_group(activity_df)
            if len(resampled) < MIN_GROUP_LEN:
                continue

            resampled["group"] = f"S{subject_id}_A{int(activity_id)}"
            resampled["time_step"] = range(len(resampled))
            all_groups.append(resampled)

    result = pd.concat(all_groups, ignore_index=True)

    # Drop any remaining NaN rows (e.g., from sensors that never reported).
    before = len(result)
    result = result.dropna()
    if len(result) < before:
        print(f"  Dropped {before - len(result)} rows with NaN")

    # Reorder columns: group, time_step, then features.
    feature_cols = [c for c in result.columns if c not in ["group", "time_step"]]
    result = result[["group", "time_step"] + sorted(feature_cols)]

    result.to_parquet(OUT_FILE, index=False)
    n_groups = result["group"].nunique()
    print(f"  Saved {len(result)} rows, {n_groups} groups to {OUT_FILE}")


if __name__ == "__main__":
    main()
