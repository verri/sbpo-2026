#!/usr/bin/env python3
"""Download the ElectricityLoadDiagrams20112014 dataset from UCI."""

import io
import zipfile
from pathlib import Path

import requests

URL = "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
RAW_DIR = Path("data/raw/electricity")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    out_file = RAW_DIR / "LD2011_2014.txt"
    if out_file.exists():
        print(f"Already downloaded: {out_file}")
        return

    print(f"Downloading {URL} ...")
    resp = requests.get(URL, timeout=300)
    resp.raise_for_status()

    # The zip contains another zip or a text file; extract everything.
    with zipfile.ZipFile(io.BytesIO(resp.content)) as outer_zip:
        names = outer_zip.namelist()
        print(f"  Outer zip contents: {names}")
        outer_zip.extractall(RAW_DIR)

    # The outer zip may contain a nested zip.
    for f in RAW_DIR.glob("*.zip"):
        print(f"  Extracting nested zip: {f.name}")
        with zipfile.ZipFile(f) as inner_zip:
            inner_zip.extractall(RAW_DIR)
        f.unlink()

    # Verify the expected file exists.
    txt_files = list(RAW_DIR.glob("*.txt")) + list(RAW_DIR.glob("*.csv"))
    if not txt_files:
        raise FileNotFoundError(f"No data files found in {RAW_DIR}")
    print(f"  Done. Files: {[f.name for f in txt_files]}")


if __name__ == "__main__":
    main()
