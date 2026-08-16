#!/usr/bin/env python3
"""Download the METR-LA traffic speed dataset (CSV from Zenodo)."""

from pathlib import Path

import requests

URL = "https://zenodo.org/records/5146275/files/METR-LA.csv"
RAW_DIR = Path("data/raw/metrla")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RAW_DIR / "metr-la.csv"

    if out_file.exists():
        print(f"Already downloaded: {out_file}")
        return

    print(f"Downloading {URL} ...")
    resp = requests.get(URL, timeout=600, stream=True)
    resp.raise_for_status()

    with open(out_file, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)

    print(f"  Done. {out_file} ({out_file.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
