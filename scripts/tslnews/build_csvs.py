#!/usr/bin/env python3
# Adapted for sign representation training and evaluation.
"""
Builds Phoenix-style CSVs for TSL-News so we can reuse the
phoenix_video_dataset dataloader for Stage 1.

Input layout (already present):
  data/TSL-News/splitted-data/
    ├── train/
    │   ├── texts/*.txt
    │   └── videos/*.mp4
    ├── dev/
    │   ├── texts/*.txt
    │   └── videos/*.mp4
    └── test/
        ├── texts/*.txt
        └── videos/*.mp4

Output CSVs:
  data/TSL-News/annotations/TSL-News.<split>.corpus.csv

CSV columns (minimal):
  name|translation|orth
Where:
  - name: base file name (without extension)
  - translation: contents of the corresponding .txt
  - orth: dummy field (not used in Stage 1, but required by phoenix loader)
"""
import os
import argparse
import pandas as pd
from pathlib import Path


def build_split_csv(split_root: Path, out_csv: Path) -> int:
    texts_dir = split_root / "texts"
    videos_dir = split_root / "videos"

    rows = []
    for txt_path in texts_dir.glob("*.txt"):
        name = txt_path.stem
        vid_path = videos_dir / f"{name}.mp4"
        if not vid_path.exists():
            # Skip if the video is missing; LMDB creation follows videos
            continue
        with open(txt_path, "r", encoding="utf-8") as f:
            translation = f.read().strip()
        # `orth` is unused for Stage 1, but the phoenix dataloader expects the column
        rows.append({
            "name": name,
            "translation": translation,
            "orth": "na",
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, sep="|", index=False, encoding="utf-8")
    return len(df)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tsl_root", default=str(Path(__file__).resolve().parents[2] / "data/TSL-News"))
    args = p.parse_args()

    tsl_root = Path(args.tsl_root)
    split_root = tsl_root / "splitted-data"
    out_root = tsl_root / "annotations"

    totals = {}
    for split in ["train", "dev", "test"]:
        n = build_split_csv(split_root / split, out_root / f"TSL-News.{split}.corpus.csv")
        totals[split] = n
        print(f"Built {split} CSV with {n} rows → {out_root / f'TSL-News.{split}.corpus.csv'}")

    print("Summary:")
    for k, v in totals.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
