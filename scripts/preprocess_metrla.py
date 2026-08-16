#!/usr/bin/env python3
"""Preprocess METR-LA traffic speed dataset into tidy parquet.

Output format:
    group (str): sensor ID
    time_step (int): sequential index within the group (one per 2-hour bin)
    speed_mph (float): mean traffic speed in mph per 2-hour bin

Reads from either .h5 (HDF5) or .csv depending on what was downloaded.
Missing values (0 or NaN) are forward-filled.  Sensors with >10%
missing data are excluded.  Samples are block-averaged to 2-hour means.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/metrla")
OUT_FILE = Path("data/processed/metrla.parquet")

MAX_MISSING_FRAC = 0.10

MIN_GROUP_LEN = 2 * 15 + 10  # 2 * max_window + horizon


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    csv_path = RAW_DIR / "metr-la.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No metr-la.csv in {RAW_DIR}")

    print(f"Reading {csv_path} ...")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    print(f"  Shape: {df.shape} ({df.shape[0]} timestamps × {df.shape[1]} sensors)")

    # Replace 0 with NaN.
    df = df.replace(0, np.nan)

    # Drop sensors with too many missing values.
    missing_frac = df.isna().mean()
    good_sensors = missing_frac[missing_frac <= MAX_MISSING_FRAC].index
    print(f"  Keeping {len(good_sensors)} sensors (dropped {len(df.columns) - len(good_sensors)} with >{MAX_MISSING_FRAC:.0%} missing)")
    df = df[good_sensors]

    df = df.ffill().bfill()

    # Aggregate to 2-hour means (24 × 5-min readings per 2-hour bin).
    print("  Aggregating to 2-hour means ...")
    df = df.resample("2h").mean()
    df = df.ffill().bfill()
    print(f"  After resampling: {df.shape[0]} timestamps × {df.shape[1]} sensors")

    df.index.name = "timestamp"
    long = df.reset_index().melt(id_vars="timestamp", var_name="group", value_name="speed_mph")
    long["group"] = long["group"].astype(str)

    long = long.sort_values(["group", "timestamp"]).reset_index(drop=True)
    long["time_step"] = long.groupby("group").cumcount()

    # Drop groups shorter than the minimum required for sliding windows.
    sizes = long.groupby("group").size()
    short = sizes[sizes < MIN_GROUP_LEN].index
    if len(short) > 0:
        print(f"  Dropping {len(short)} groups with < {MIN_GROUP_LEN} rows")
        long = long[~long["group"].isin(short)].copy()
        long["time_step"] = long.groupby("group").cumcount()

    long = long[["group", "time_step", "speed_mph"]]
    long.to_parquet(OUT_FILE, index=False)
    n_groups = long["group"].nunique()
    n_steps = long.groupby("group").size().iloc[0]
    print(f"  Saved {len(long):,} rows ({n_groups} sensors × {n_steps} bins) to {OUT_FILE}")


if __name__ == "__main__":
    main()
