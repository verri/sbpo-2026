#!/usr/bin/env python3
"""Download the PAMAP2 Physical Activity Monitoring dataset from UCI."""

import io
import zipfile
from pathlib import Path

import requests

URL = "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"
RAW_DIR = Path("data/raw/pamap2")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded (look for .dat files).
    if list(RAW_DIR.rglob("*.dat")):
        print(f"Already downloaded: {RAW_DIR}")
        return

    print(f"Downloading {URL} ...")
    resp = requests.get(URL, timeout=300)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as outer_zip:
        names = outer_zip.namelist()
        print(f"  Outer zip contents ({len(names)} entries)")
        outer_zip.extractall(RAW_DIR)

    # Handle nested zips.
    for f in RAW_DIR.rglob("*.zip"):
        print(f"  Extracting nested zip: {f.name}")
        with zipfile.ZipFile(f) as inner_zip:
            inner_zip.extractall(f.parent)
        f.unlink()

    dat_files = list(RAW_DIR.rglob("*.dat"))
    if not dat_files:
        raise FileNotFoundError(f"No .dat files found in {RAW_DIR}")
    print(f"  Done. Found {len(dat_files)} .dat files.")


if __name__ == "__main__":
    main()
