"""
Fetch the NAB files used by this study into `data/nab/`.

Run once:  python scripts/download_data.py

Downloads only the corpora listed in `itops.nab.CORPORA`, plus the label
file. Existing files are skipped, so re-running is cheap and the experiment
stays reproducible offline afterwards.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from itops.nab import CORPORA, DATA_URL, LABELS_URL  # noqa: E402


def fetch(url: str, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return False
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            destination.write_bytes(response.read())
    except urllib.error.URLError as error:
        print(f"  FAILED {url}: {error}")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "nab"))
    parser.add_argument("--corpora", nargs="*", default=list(CORPORA))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    labels_path = data_dir / "combined_windows.json"
    if fetch(LABELS_URL, labels_path):
        print(f"labels -> {labels_path}")
    else:
        print(f"labels already present: {labels_path}")

    labels = json.loads(labels_path.read_text())
    keys = [k for k in sorted(labels) if k.split("/")[0] in args.corpora]
    print(f"{len(keys)} series across {len(args.corpora)} corpora")

    downloaded = 0
    for key in keys:
        if fetch(f"{DATA_URL}/{key}", data_dir / key):
            downloaded += 1
            print(f"  + {key}")

    present = sum(1 for key in keys if (data_dir / key).exists())
    print(f"\ndownloaded {downloaded}, present {present}/{len(keys)} in {data_dir}")
    return 0 if present == len(keys) else 1


if __name__ == "__main__":
    raise SystemExit(main())
