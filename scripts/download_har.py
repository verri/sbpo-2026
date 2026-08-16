#!/usr/bin/env python3
"""Download the UCI Human Activity Recognition Using Smartphones dataset."""

import io
import zipfile
from pathlib import Path

import requests

URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
RAW_DIR = Path("data/raw/har")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if (RAW_DIR / "UCI HAR Dataset").exists():
        print(f"Already downloaded: {RAW_DIR}")
        return

    print(f"Downloading {URL} ...")
    resp = requests.get(URL, timeout=300)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as outer_zip:
        print(f"  Outer zip contents ({len(outer_zip.namelist())} entries)")
        outer_zip.extractall(RAW_DIR)

    # Handle nested zip.
    for f in RAW_DIR.rglob("*.zip"):
        print(f"  Extracting nested zip: {f.name}")
        with zipfile.ZipFile(f) as inner_zip:
            inner_zip.extractall(RAW_DIR)
        f.unlink()

    if not (RAW_DIR / "UCI HAR Dataset").exists():
        raise FileNotFoundError("UCI HAR Dataset directory not found after extraction")
    print("  Done.")


if __name__ == "__main__":
    main()
