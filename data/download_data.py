"""
Step 1 — Acquire the dataset.

We use the **AI4I 2020 Predictive Maintenance Dataset**. This is the exact same
10,000-row table that Kaggle publishes as "Machine Predictive Maintenance
Classification" — Kaggle mirrors it from the UCI Machine Learning Repository.

We pull from UCI instead of Kaggle because UCI needs no API token, so this
script runs on any machine with no setup. If you would rather cite Kaggle
directly, see the `--kaggle` note at the bottom of this file.

Why this dataset for a mechatronics project:
    It is *synthetic but physically grounded*. Each failure mode in it is
    generated from a real machine-physics rule (heat dissipation, power
    envelope, mechanical overstrain, tool wear). That means we can write
    threshold logic that a mechanical engineer can actually defend, rather than
    picking cut-offs out of thin air.

Usage:
    python data/download_data.py
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

# Repo root = parent of the data/ folder this file lives in
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_CSV = RAW_DIR / "ai4i2020.csv"

UCI_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)


def download() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_CSV.exists():
        print(f"[skip] {RAW_CSV.relative_to(ROOT)} already exists "
              f"({RAW_CSV.stat().st_size / 1024:.0f} KB)")
        return RAW_CSV

    print(f"[get ] {UCI_URL}")
    req = urllib.request.Request(UCI_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = resp.read()

    # The UCI download is a zip containing a single CSV.
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"No CSV inside the archive: {zf.namelist()}")
        with zf.open(csv_names[0]) as src:
            RAW_CSV.write_bytes(src.read())

    print(f"[ok  ] wrote {RAW_CSV.relative_to(ROOT)} "
          f"({RAW_CSV.stat().st_size / 1024:.0f} KB)")
    return RAW_CSV


def preview(path: Path) -> None:
    """Print the header and first two rows so you can eyeball the schema."""
    with path.open() as fh:
        for i, line in enumerate(fh):
            if i > 2:
                break
            print("      " + line.rstrip())


if __name__ == "__main__":
    try:
        csv_path = download()
    except Exception as exc:  # noqa: BLE001 - top-level script, show the cause
        print(f"[fail] {exc}", file=sys.stderr)
        print(
            "\nFallback: download manually from either source and save it as\n"
            f"  {RAW_CSV}\n"
            "  UCI    : https://archive.ics.uci.edu/dataset/601/"
            "ai4i+2020+predictive+maintenance+dataset\n"
            "  Kaggle : kaggle datasets download -d shivamb/"
            "machine-predictive-maintenance-classification",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print("\nSchema preview:")
    preview(csv_path)
