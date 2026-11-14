#!/usr/bin/env python3
# Adapted for sign representation training and evaluation.
import argparse, importlib, json
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lmdb
import pickle
import shutil
import pandas as pd

from augmentation.get_aug import get_aug
from dataloaders.get_dataset import get_dataset
from dataloaders.data_utils.file_utils import read_pickle
from models.get_models import get_model
from train_utils.checkpoint_helpers import get_best_checkpoint_details, get_latest_saved_file

# ---------------- Helpers ----------------
def load_cfg_and_model(cfg_path, checkpoint=None, load_best=True):
    from train_utils.release_helpers import load_config
    cfg = load_config(cfg_path)
    if checkpoint is None or not Path(checkpoint).is_file():
        raise FileNotFoundError("Provide --checkpoint with an existing paper checkpoint.")
    model = get_model(cfg.model_name, cfg.model_params.to_dict())
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model, cfg, checkpoint

def forward_one(model, frames, num_classes, device):
    frames = frames.to(device)
    out = model(frame_features=[frames], max_len=frames.shape[0])
    post = out["dict_post_output"]
    mask = post.get("mask", torch.ones(frames.shape[0], device=device, dtype=torch.bool))[0].bool()
    scores = post.get("softmax_scores")
    if scores is None:
        sim = post["time_res"]
        sim_btk = sim if sim.shape[1] == mask.shape[0] else sim.transpose(1, 2)
        sim_btk = sim_btk.masked_fill(~mask[None, :, None], float("-inf"))
        logP = torch.log_softmax(sim_btk, dim=1)
        logC = torch.log_softmax(sim_btk, dim=2)
        scores = torch.exp(logP + logC)
    else:
        if scores.shape[1] != mask.shape[0]:
            scores = scores.transpose(1, 2)
    scores = scores[0, : mask.sum(), :num_classes].detach().cpu()        # [T,K]
    hidden = out["enc_output"]["hidden_state"][0, : mask.sum()].detach().cpu()  # [T,dim]
    return scores, hidden, int(mask.sum().item())


def build_single_lmdb_from_mp4(mp4_path: Path, env_dir: Path, stride: int = 1, max_len: int = None):
    """
    Write frames from a single MP4 into an LMDB layout compatible with LMDBUtility.
    - mp4_path: source video
    - env_dir: target LMDB directory (will be created if needed)
    - stride: frame subsample factor (1 = every frame)
    - max_len: optional cap on number of stored frames (after subsample)
    Returns (env_dir, num_frames)
    """
    env_dir.mkdir(parents=True, exist_ok=True)
    if any(env_dir.iterdir()):
        with lmdb.open(str(env_dir), readonly=True, lock=False) as existing:
            with existing.begin() as txn:
                raw_details = txn.get(b"details")
                if raw_details is None:
                    raise RuntimeError(f"Incomplete cached LMDB at {env_dir}; use a fresh temporary directory.")
                num_frames = int(pickle.loads(raw_details)["num_frames"])
                if num_frames <= 0:
                    raise RuntimeError(f"Cached LMDB contains no frames: {env_dir}")
        return env_dir, num_frames

    env = lmdb.open(path=str(env_dir), map_size=2**32)
    txn = env.begin(write=True)
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        txn.abort()
        cap.release()
        env.close()
        raise RuntimeError(f"Failed to open video: {mp4_path}")

    idx = 0
    stored = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if stride > 1 and (idx % stride) != 0:
                idx += 1
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            is_ok, buf = cv2.imencode(".jpg", frame)
            if not is_ok:
                raise RuntimeError("JPEG encode failed")
            txn.put(key=str(stored).encode("ascii"), value=buf.tobytes())
            stored += 1
            idx += 1
            if max_len is not None and stored >= max_len:
                break
            if stored % 200 == 0:
                txn.commit()
                txn = env.begin(write=True)
        txn.put(key=b"details", value=pickle.dumps({"num_frames": stored}, protocol=4))
        txn.commit()
    finally:
        cap.release()
        env.close()
    return env_dir, stored

def write_single_csv(csv_path: Path, name: str):
    """
    Minimal CSV with columns expected by PhoenixVideoDataset.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("name|translation|orth|pseudo_gloss\n")
        f.write(f"{name}||| \n")

def compute_span(prob_row, threshold=0.2, smooth=3):
    row = prob_row
    if smooth and smooth > 1:
        kernel = np.ones(smooth, dtype=np.float32) / float(smooth)
        row = np.convolve(prob_row, kernel, mode="same")
    if not np.any(row > 0):
        return {"peak": None, "peak_value": 0.0, "start": None, "end": None}
    peak = int(row.argmax()); peak_val = float(row[peak]); rel = peak_val * threshold
    above = row >= rel
    segments = []
    in_seg = False; s = 0
    for t, v in enumerate(above):
        if v and not in_seg:
            in_seg = True; s = t
        elif not v and in_seg:
            in_seg = False; segments.append((s, t - 1))
    if in_seg: segments.append((s, len(row) - 1))
    chosen = next(((a, b) for (a, b) in segments if a <= peak <= b), None)
    if chosen is None and segments:
        chosen = max(segments, key=lambda x: x[1]-x[0])
    start, end = chosen if chosen else (None, None)
    return {"peak": peak, "peak_value": peak_val, "start": start, "end": end}

def sliding_cosine(sentence_hidden, iso_hidden, stride=1):
    sent = F.normalize(sentence_hidden, dim=1)
    iso = F.normalize(iso_hidden, dim=1)
    iso_vec = iso.mean(dim=0)
    win = iso.shape[0]
    sims = []
    T = sent.shape[0]
    if T < win:  # tek karşılaştırma
        sims.append(float(torch.dot(sent.mean(dim=0), iso_vec)))
        return np.array(sims)
    for t in range(0, T - win + 1, stride):
        chunk_vec = sent[t:t+win].mean(dim=0)
        sims.append(float(torch.dot(chunk_vec, iso_vec)))
    return np.array(sims)

def _normalize_text(s):
    return str(s).casefold()

def find_sample_index_by_gloss(csv_path: str, sep: str, gloss: str):
    df = pd.read_csv(csv_path, sep=sep, keep_default_na=False)
    if "name" not in df.columns:
        raise KeyError(f"Missing 'name' column in {csv_path}")

    gloss_norm = _normalize_text(gloss)
    match_field = None
    matches = None

    if "pseudo_gloss" in df.columns:
        mask = df["pseudo_gloss"].astype(str).apply(
            lambda s: gloss_norm in [_normalize_text(tok) for tok in s.split()]
        )
        if mask.any():
            matches = df[mask]
            match_field = "pseudo_gloss"

    if matches is None:
        if "translation" not in df.columns:
            raise KeyError(f"Missing 'translation' column in {csv_path}")
        mask = df["translation"].astype(str).apply(
            lambda s: gloss_norm in _normalize_text(s)
        )
        if mask.any():
            matches = df[mask]
            match_field = "translation"

    if matches is None or matches.empty:
        raise ValueError(f"No samples contain gloss '{gloss}' in {csv_path}")

    names_sorted = sorted(df["name"].astype(str).unique())
    name_to_idx = {name: i for i, name in enumerate(names_sorted)}
    match_names = matches["name"].astype(str).unique().tolist()
    chosen_name = min(match_names, key=lambda n: name_to_idx.get(n, 10**12))
    sample_index = name_to_idx[chosen_name]

    sample_translation = ""
    if "translation" in matches.columns:
        sample_translation = matches[matches["name"].astype(str) == chosen_name][
            "translation"
        ].iloc[0]
    return sample_index, chosen_name, match_field, sample_translation

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tslnews/tslnews_stage1_configs/TSL_example_s1_dyn_config.py")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", choices=["tid","test","dev","train"], default="tid")
    ap.add_argument("--sample_index", type=int, default=0)
    ap.add_argument("--sample_gloss", default=None, help="Pick first sentence containing this gloss")
    ap.add_argument("--tid_csv", default="data/TSL-News/annotations/TID.all.corpus.csv")
    ap.add_argument("--tid_lmdb_video_dir", default="data/TSL-News/lmdb_videos/TID")
    ap.add_argument("--vocab_pkl", default="data/TSL-News/6538_gloss/processed_words_map.tsl_pkl")
    ap.add_argument("--target_gloss",  default="cumhurbaşkanı")
    ap.add_argument("--isolated_video", default="data/TSL-News/splitted-data/TID/cumhurbaşkanı.mp4")
    ap.add_argument("--iso_tmp_root", default="outputs/analysis/tmp_iso", help="Where to materialize temporary LMDB/CSV for isolated video")
    ap.add_argument("--output_npz", default="outputs/analysis/verify_all.npz")
    ap.add_argument("--stride", type=int, default=1, help="sliding stride in feature steps")
    ap.add_argument("--iso_stride_frames", type=int, default=2, help="frame subsample stride for isolated video")
    ap.add_argument("--iso_max_len", type=int, default=12, help="max frames after subsample for isolated video")
    ap.add_argument("--plot", action="store_true", help="save sim_curve.png and prob_curve.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, ckpt = load_cfg_and_model(args.config, args.checkpoint)
    model.to(device)

    vocab = read_pickle(args.vocab_pkl)
    lem_to_id = vocab["dict_lem_to_id"]
    if args.target_gloss not in lem_to_id:
        raise KeyError(f"{args.target_gloss} not in vocab ({len(lem_to_id)})")
    gloss_id = lem_to_id[args.target_gloss]
    num_classes = len(lem_to_id)

    transform = get_aug(cfg.aug_name, cfg.aug_params)

    # dataloader bs=1 deterministic
    if args.split == "tid":
        dl_params = {
            "csv_dir": args.tid_csv,
            "pseudo_gloss_dir": args.vocab_pkl,
            "sep": "|",
            "name": "translation",
            "ds_params": {"lmdb_video_dir": args.tid_lmdb_video_dir, "isValid": True},
            "shuffle": False, "num_workers": 0, "bs": 1, "drop_last": False,
        }
    else:
        dl_params = getattr(cfg, f"{args.split}_ds_params").to_dict()
        dl_params.update({"bs":1,"num_workers":0,"shuffle":False,"drop_last":False})

    if args.sample_gloss:
        idx, name, field, translation = find_sample_index_by_gloss(
            dl_params["csv_dir"], dl_params.get("sep", "|"), args.sample_gloss
        )
        args.sample_index = idx
        print(
            f"[INFO] sample_gloss '{args.sample_gloss}' matched {field} "
            f"in name '{name}' (sample_index={idx})"
        )
        if translation:
            print(f"[INFO] matched translation: {translation}")

    dl, _ = get_dataset(cfg.test_ds_name, dl_params, transform=transform)

    import itertools
    batch = next(itertools.islice(dl, args.sample_index, None))
    frames = batch["frames"][0]

    sent_scores, sent_hidden, valid_T = forward_one(model, frames, num_classes, device)
    sent_probs = sent_scores.numpy().T  # [G,T]
    target_span = compute_span(sent_probs[gloss_id], threshold=0.2, smooth=3)

    # --- Isolated video: convert to temporary LMDB + CSV, then load with same dataloader pipeline ---
    iso_tmp_root = Path(args.iso_tmp_root)
    iso_tmp_root.mkdir(parents=True, exist_ok=True)
    iso_name = Path(args.isolated_video).stem
    iso_lmdb_dir = iso_tmp_root / "lmdb" / iso_name
    iso_csv = iso_tmp_root / f"{iso_name}.csv"
    build_single_lmdb_from_mp4(
        Path(args.isolated_video),
        iso_lmdb_dir,
        stride=max(1, args.iso_stride_frames),
        max_len=args.iso_max_len,
    )
    write_single_csv(iso_csv, iso_name)

    iso_params = {
        "csv_dir": str(iso_csv),
        "pseudo_gloss_dir": args.vocab_pkl,
        "sep": "|",
        "name": "translation",
        "ds_params": {"lmdb_video_dir": str(iso_lmdb_dir.parent), "isValid": True},
        "shuffle": False,
        "num_workers": 0,
        "bs": 1,
        "drop_last": False,
    }
    iso_dl, _ = get_dataset(cfg.test_ds_name, iso_params, transform=transform)
    iso_batch = next(iter(iso_dl))
    iso_frames = iso_batch["frames"][0]
    iso_scores, iso_hidden, iso_T = forward_one(model, iso_frames, num_classes, device)

    sim_curve = sliding_cosine(sent_hidden, iso_hidden, stride=args.stride or 1)

    out_path = Path(args.output_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             sim_curve=sim_curve,
             target_gloss=args.target_gloss,
             gloss_id=gloss_id,
             target_span=target_span,
             sent_probs=sent_probs,
             iso_len=iso_T,
             valid_T=valid_T,
             checkpoint=str(ckpt))
    summary = {
        "npz": str(out_path),
        "target_gloss": args.target_gloss,
        "gloss_id": gloss_id,
        "sim_peak": float(sim_curve.max()),
        "sim_peak_step": int(sim_curve.argmax()),
        "sim_peak_frame_est": int(sim_curve.argmax() * cfg.aug_params["stride"]),
        "model_span": target_span,
        "iso_len": iso_T,
        "sentence_valid_T": valid_T,
        "checkpoint": str(ckpt),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.plot:
        out_dir = out_path.parent
        plt.figure(figsize=(10,3))
        plt.plot(sim_curve, label="cosine")
        if target_span["start"] is not None:
            plt.axvspan(target_span["start"], target_span["end"], color="orange", alpha=0.25, label="model span")
        plt.axvline(sim_curve.argmax(), color="red", ls="--", label="sim peak")
        plt.xlabel("feature step (~frame = step*stride)")
        plt.ylabel("cosine sim")
        plt.title(f"Similarity vs time ({args.target_gloss})")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / f"sim_curve_{args.target_gloss}.png", dpi=200); plt.close()

        plt.figure(figsize=(10,3))
        plt.plot(sent_probs[gloss_id], label=f"prob({args.target_gloss})")
        plt.xlabel("time step"); plt.ylabel("probability")
        plt.title(f"Pseudo-gloss prob for {args.target_gloss} (id={gloss_id})")
        plt.tight_layout()
        plt.savefig(out_dir / f"prob_curve_{args.target_gloss}.png", dpi=200); plt.close()
        print(f"[INFO] Saved plots to {out_dir}")

    # Cleanup temporary ISO LMDB/CSV
    try:
        if iso_tmp_root.exists():
            shutil.rmtree(iso_tmp_root, ignore_errors=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
