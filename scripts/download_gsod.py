#!/usr/bin/env python3
"""Download NOAA Global Surface Summary of the Day (GSOD) data.

Downloads yearly archives (2021-2023) from NOAA NCEI and extracts the
per-station CSV files into data/raw/gsod/{year}/.
"""

import tarfile
from pathlib import Path

import requests

YEARS = [2021, 2022, 2023]
BASE_URL = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/archive"
RAW_DIR = Path("data/raw/gsod")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for year in YEARS:
        year_dir = RAW_DIR / str(year)
        if year_dir.exists() and any(year_dir.glob("*.csv")):
            n = len(list(year_dir.glob("*.csv")))
            print(f"Already downloaded: {year_dir} ({n} files)")
            continue

        url = f"{BASE_URL}/{year}.tar.gz"
        tar_path = RAW_DIR / f"{year}.tar.gz"

        print(f"Downloading {url} ...")
        resp = requests.get(url, timeout=600, stream=True)
        resp.raise_for_status()

        with open(tar_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        print(f"  Downloaded {tar_path.stat().st_size / 1e6:.1f} MB")

        # Extract CSV files into year subdirectory.
        year_dir.mkdir(exist_ok=True)
        print(f"  Extracting to {year_dir} ...")
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.isfile() or not member.name.endswith(".csv"):
                    continue
                content = tar.extractfile(member)
                if content is None:
                    continue
                out_path = year_dir / Path(member.name).name
                out_path.write_bytes(content.read())

        tar_path.unlink()
        csv_count = len(list(year_dir.glob("*.csv")))
        print(f"  Done. {csv_count} station files extracted.")

    print("All years downloaded.")


if __name__ == "__main__":
    main()
