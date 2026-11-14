#!/usr/bin/env python3
# Adapted for sign representation training and evaluation.
"""
Create per-sequence LMDBs for TSL-News videos, matching the layout
expected by dataloaders.data_utils.LMDBUtility:

For each split (train/dev/test), iterate videos at:
  data/TSL-News/splitted-data/<split>/videos/<name>.mp4

Write LMDBs to:
  data/TSL-News/lmdb_videos/<split>/<name>

Each LMDB contains:
  - frame index keys: b"0", b"1", ... → raw JPEG bytes
  - b"details" → pickled dict with {"num_frames": N}
"""
import os
import lmdb
import cv2
import io
import pickle
import argparse
from pathlib import Path
from time import time


def build_lmdb_for_video(mp4_path: Path, env_dir: Path, commit_every: int = 200) -> int:
    if env_dir.exists() and any(env_dir.iterdir()):
        with lmdb.open(str(env_dir), readonly=True, lock=False) as existing:
            with existing.begin() as txn:
                raw_details = txn.get(b"details")
                if raw_details is None or int(pickle.loads(raw_details)["num_frames"]) <= 0:
                    raise RuntimeError(f"Incomplete LMDB at {env_dir}; use a fresh output directory.")
        return 0

    env_dir.mkdir(parents=True, exist_ok=True)

    # Big map_size to be safe (1 TB); LMDB grows sparingly but needs max map size
    env = lmdb.open(path=str(env_dir), map_size=2 ** 40)
    txn = env.begin(write=True)

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        txn.abort()
        cap.release()
        env.close()
        raise RuntimeError(f"Failed to open video: {mp4_path}")

    num_frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Convert BGR → RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Encode as JPEG into bytes
            is_ok, buf = cv2.imencode(".jpg", frame)
            if not is_ok:
                raise RuntimeError("JPEG encode failed")
            txn.put(key=str(num_frames).encode("ascii"), value=buf.tobytes())
            num_frames += 1

            if num_frames % commit_every == 0:
                txn.commit()
                txn = env.begin(write=True)

        # Store details
        txn.put(
            key=b"details",
            value=pickle.dumps({"num_frames": num_frames}, protocol=4),
        )
        txn.commit()
    finally:
        cap.release()
        env.close()

    return num_frames


def build_split(in_videos_root: Path, out_lmdb_root: Path, split: str):
    in_dir = in_videos_root / split / "videos"
    out_root = out_lmdb_root / split
    out_root.mkdir(parents=True, exist_ok=True)

    mp4s = sorted(in_dir.glob("*.mp4"))
    total = 0
    for v in mp4s:
        name = v.stem
        env_dir = out_root / name
        nf = build_lmdb_for_video(v, env_dir)
        total += 1 if nf > 0 else 0
        print(f"{split}: {name} → {nf} frames [{env_dir}]")
    print(f"{split}: processed {total}/{len(mp4s)} sequences")


def main():
    p = argparse.ArgumentParser()
    default_root = Path(__file__).resolve().parents[2] / "data/TSL-News"
    p.add_argument("--tsl_root", default=str(default_root))
    args = p.parse_args()

    tsl_root = Path(args.tsl_root)
    in_videos_root = tsl_root / "splitted-data"
    out_lmdb_root = tsl_root / "lmdb_videos"

    for split in ["train", "dev", "test"]:
        build_split(in_videos_root, out_lmdb_root, split)


if __name__ == "__main__":
    main()
