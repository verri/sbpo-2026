#!/usr/bin/env python3
"""Display summary statistics for each preprocessed dataset."""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")


def describe_dataset(name: str, path: Path):
    print(f"\n{'=' * 70}")
    print(f"  {name}  ({path})")
    print(f"{'=' * 70}")

    df = pd.read_parquet(path)

    feature_cols = [c for c in df.columns if c not in ("group", "time_step")]

    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    groups = df.groupby("group")["time_step"].count()
    print(f"\nGroups: {groups.shape[0]}")
    print(f"  Rows per group:")
    print(f"    min    = {groups.min():,}")
    print(f"    median = {int(groups.median()):,}")
    print(f"    max    = {groups.max():,}")
    print(f"    mean   = {groups.mean():,.1f}")
    print(f"    std    = {groups.std():,.1f}")

    print(f"\nGroup list ({len(groups)}):")
    for g, n in groups.items():
        print(f"    {g}: {n:,} rows")

    print(f"\nFeature statistics:")
    print(df[feature_cols].describe().to_string())


def main():
    datasets = [
        ("Electricity", PROCESSED_DIR / "electricity.parquet"),
        ("PAMAP2", PROCESSED_DIR / "pamap2.parquet"),
        ("MHEALTH", PROCESSED_DIR / "mhealth.parquet"),
    ]

    for name, path in datasets:
        if not path.exists():
            print(f"\n[SKIP] {name}: {path} not found")
            continue
        describe_dataset(name, path)


if __name__ == "__main__":
    main()
