#!/usr/bin/env python3
"""Preprocess NOAA GSOD weather data into tidy parquet.

Output format:
    group (str): station ID (e.g., "72594024229")
    time_step (int): sequential index within the group (one per week)
    temp_c (float): weekly mean temperature in degrees Celsius

Three years of data (2021-2023) are merged.  Stations not present in all
years or with >5% missing temperature values are excluded.  We subsample
to 300 stations for tractability.  Missing values (GSOD code 9999.9) are
linearly interpolated at the daily scale, then block-averaged to weekly
means (non-overlapping 7-day blocks per station).
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/gsod")
OUT_FILE = Path("data/processed/gsod.parquet")

YEARS = [2021, 2022, 2023]
MAX_MISSING_FRAC = 0.05
MAX_STATIONS = 300
SEED = 42

BLOCK_SIZE = 7                # days per weekly bin
MIN_GROUP_LEN = 2 * 15 + 10   # 2 * max_window + horizon


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Find stations present in all years.
    print("Scanning station files ...")
    station_sets = []
    for year in YEARS:
        year_dir = RAW_DIR / str(year)
        stations = {f.stem for f in year_dir.glob("*.csv")}
        station_sets.append(stations)
        print(f"  {year}: {len(stations)} stations")

    common = sorted(station_sets[0].intersection(*station_sets[1:]))
    print(f"  {len(common)} stations present in all {len(YEARS)} years")

    # Read all CSV files for common stations.
    print("Reading station files ...")
    frames = []
    for i, station_id in enumerate(common):
        if (i + 1) % 2000 == 0:
            print(f"  ... {i + 1}/{len(common)} stations")
        for year in YEARS:
            path = RAW_DIR / str(year) / f"{station_id}.csv"
            try:
                sdf = pd.read_csv(
                    path, usecols=["STATION", "DATE", "TEMP"],
                    dtype={"STATION": str},
                )
                frames.append(sdf)
            except Exception:
                continue

    df = pd.concat(frames, ignore_index=True)
    print(f"  Read {len(df):,} rows for {df['STATION'].nunique()} stations")

    # Parse dates and replace missing temperature code.
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["TEMP"] = pd.to_numeric(df["TEMP"], errors="coerce")
    df.loc[df["TEMP"] >= 9999, "TEMP"] = np.nan

    # Expected date range.
    full_range = pd.date_range(f"{YEARS[0]}-01-01", f"{YEARS[-1]}-12-31", freq="D")
    n_expected = len(full_range)
    print(f"  Expected {n_expected} days per station")

    # Compute coverage per station.
    coverage = (
        df.groupby("STATION")["TEMP"]
        .apply(lambda s: s.notna().sum() / n_expected)
    )
    good = coverage[coverage >= (1 - MAX_MISSING_FRAC)].index.tolist()
    print(f"  {len(good)} stations with >= {100 * (1 - MAX_MISSING_FRAC):.0f}% temperature coverage")

    # Subsample for tractability.
    if len(good) > MAX_STATIONS:
        rng = np.random.RandomState(SEED)
        good = list(rng.choice(good, MAX_STATIONS, replace=False))
        print(f"  Subsampled to {MAX_STATIONS} stations")

    df = df[df["STATION"].isin(good)].copy()

    # Fill missing dates and interpolate temperature per station.
    print("  Filling gaps and interpolating ...")
    result = []
    for station_id, sdf in df.groupby("STATION"):
        sdf = sdf.drop_duplicates("DATE").set_index("DATE").sort_index()
        sdf = sdf.reindex(full_range)
        sdf["STATION"] = station_id
        sdf["TEMP"] = sdf["TEMP"].interpolate(limit_direction="both")
        result.append(sdf)

    df = pd.concat(result).reset_index(names="DATE")

    # Convert Fahrenheit to Celsius.
    df["temp_c"] = (df["TEMP"] - 32) * 5 / 9

    # Create standardized columns.
    df["group"] = df["STATION"].astype(str)
    df = df.sort_values(["group", "DATE"]).reset_index(drop=True)
    df["day_idx"] = df.groupby("group").cumcount()

    # Block-average non-overlapping 7-day windows per station.
    print(f"  Block-averaging every {BLOCK_SIZE} days per station ...")
    df["_block"] = df["day_idx"] // BLOCK_SIZE
    weekly = (
        df.groupby(["group", "_block"])["temp_c"]
        .mean()
        .reset_index()
    )
    weekly["time_step"] = weekly.groupby("group").cumcount()

    # Drop groups shorter than the minimum required for sliding windows.
    sizes = weekly.groupby("group").size()
    short = sizes[sizes < MIN_GROUP_LEN].index
    if len(short) > 0:
        print(f"  Dropping {len(short)} groups with < {MIN_GROUP_LEN} rows")
        weekly = weekly[~weekly["group"].isin(short)].copy()
        weekly["time_step"] = weekly.groupby("group").cumcount()

    weekly = weekly[["group", "time_step", "temp_c"]]
    weekly.to_parquet(OUT_FILE, index=False)

    n_groups = weekly["group"].nunique()
    n_steps = weekly.groupby("group").size().iloc[0]
    size_mb = OUT_FILE.stat().st_size / 1e6
    print(f"  Saved {len(weekly):,} rows ({n_groups} stations x {n_steps} weeks) to {OUT_FILE}")
    print(f"  File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
