"""Download the public Kaggle FIFA player dataset and save one season's CSV to
data/raw/kaggle_players.csv, ready for `fta load-kaggle`. No Kaggle account
is needed for this public dataset; kagglehub caches the archive under
~/.cache/kagglehub so re-runs are instant.

Usage: python scripts/download_kaggle_players.py [--edition 22]
Requires: pip install -e ".[kaggle]"
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DATASET = "stefanoleone992/fifa-22-complete-player-dataset"
ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "raw" / "kaggle_players.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edition", type=int, default=22, choices=range(15, 23),
                    help="FIFA edition (15-22); each is one season of ratings")
    args = ap.parse_args()

    try:
        import kagglehub
    except ImportError:
        raise SystemExit('kagglehub is not installed. Run: pip install -e ".[kaggle]"') from None

    src = Path(kagglehub.dataset_download(DATASET)) / f"players_{args.edition}.csv"
    if not src.exists():
        raise SystemExit(f"{src.name} not found in the downloaded dataset at {src.parent}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, OUT_PATH)
    print(f"Saved FIFA {args.edition} players -> {OUT_PATH}")
    print("Next: fta load-kaggle --csv data/raw/kaggle_players.csv")


if __name__ == "__main__":
    main()
