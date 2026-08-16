#!/usr/bin/env python3
"""Preprocess ElectricityLoadDiagrams20112014 into tidy parquet.

Output format:
    group (str): client ID (e.g., "MT_001")
    time_step (int): sequential index within the group (one per week)
    power_mw (float): weekly mean power in MW

The raw file has timestamps as rows (15-min intervals) and clients as
columns (wide format). We aggregate to weekly means (7*96 raw readings
per bin) to make the dataset tractable for all algorithms, then melt
into long format.

Clients with constant zero consumption are dropped.  Groups with fewer
than MIN_GROUP_LEN rows after aggregation are dropped.
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw/electricity")
OUT_FILE = Path("data/processed/electricity.parquet")

MIN_GROUP_LEN = 2 * 15 + 10  # 2 * max_window + horizon


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # The file uses semicolons and commas as decimal separators.
    raw_path = RAW_DIR / "LD2011_2014.txt"
    print(f"Reading {raw_path} ...")
    df = pd.read_csv(raw_path, sep=";", decimal=",", index_col=0, parse_dates=True)

    # Drop clients that are all zeros (inactive meters).
    active = df.columns[df.sum() > 0]
    print(f"  {len(active)} active clients out of {len(df.columns)}")
    df = df[active]

    # Aggregate to weekly means (7 days * 96 15-min readings per bin).
    # resample("7D") uses fixed 7-day blocks anchored at the first timestamp,
    # so bins never cross clients (each client is an independent column).
    print("  Aggregating to 7-day (weekly) means ...")
    df = df.resample("7D").mean()
    print(f"  {len(df)} weeks × {len(df.columns)} clients")

    # Melt wide → long: each client becomes a group.
    df.index.name = "timestamp"
    long = df.reset_index().melt(id_vars="timestamp", var_name="group", value_name="power_mw")

    # Convert kW to MW.
    long["power_mw"] = long["power_mw"] / 1000.0

    # Add sequential time_step per group (preserving chronological order).
    long = long.sort_values(["group", "timestamp"]).reset_index(drop=True)
    long["time_step"] = long.groupby("group").cumcount()

    # Drop groups shorter than the minimum required for sliding windows.
    sizes = long.groupby("group").size()
    short = sizes[sizes < MIN_GROUP_LEN].index
    if len(short) > 0:
        print(f"  Dropping {len(short)} groups with < {MIN_GROUP_LEN} rows")
        long = long[~long["group"].isin(short)].copy()
        long["time_step"] = long.groupby("group").cumcount()

    # Keep only the standardized columns.
    long = long[["group", "time_step", "power_mw"]]
    long.to_parquet(OUT_FILE, index=False)
    n_groups = long["group"].nunique()
    n_steps = long.groupby("group").size().iloc[0]
    print(f"  Saved {len(long)} rows ({n_groups} clients × {n_steps} weeks) to {OUT_FILE}")


if __name__ == "__main__":
    main()
