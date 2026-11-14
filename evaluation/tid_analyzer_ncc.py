#!/usr/bin/env python3
# Adapted for sign representation training and evaluation.
"""
Variant of tid_gloss: uses per-frame normalized cross-correlation (NCC) over the window
instead of mean-pool cosine. NCC = weighted mean of per-frame cosines over the aligned window.
Optional: ignore leading/trailing frames of the isolated gloss, optional triangular weighting.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def sliding_ncc(sentence_hidden, iso_hidden, stride=1, ignore_frames=0, weighting="none"):
    """
    Compute mean cosine similarity over aligned, L2-normalized frame features.
    sentence_hidden: [T_sent, D]
    iso_hidden: [T_iso, D]
    Returns: np.array of NCC scores over sliding windows.
    - ignore_frames: drop that many frames from both start and end of iso (after norm)
    - weighting: "none" or "triangle" (higher weight in the middle of the window)
    """
    sent = F.normalize(sentence_hidden, dim=1)
    iso = F.normalize(iso_hidden, dim=1)

    if ignore_frames > 0 and iso.shape[0] > 2 * ignore_frames:
        iso = iso[ignore_frames: -ignore_frames]

    win = iso.shape[0]
    if win == 0:
        return np.array([])

    # weights over time
    w = torch.ones(win, device=sent.device)
    if weighting == "triangle" and win > 1:
        center = (win - 1) / 2.0
        pos = torch.arange(win, device=sent.device, dtype=torch.float32)
        w = 1.0 - torch.abs(pos - center) / (center + 1e-6)

    w_sum = w.sum()
    T = sent.shape[0]
    if T < win:
        # fallback: single NCC over min length
        eff = min(T, win)
        cos_per_frame = torch.einsum("td,td->t", sent[:eff], iso[:eff])
        score = float((w[:eff] * cos_per_frame).sum() / (w[:eff].sum()))
        return np.array([score])

    sims = []
    for t in range(0, T - win + 1, stride):
        chunk = sent[t : t + win]
        cos_per_frame = torch.einsum("td,td->t", chunk, iso)  # per-frame cosine
        score = float((w * cos_per_frame).sum() / w_sum)
        sims.append(score)
    return np.array(sims)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from augmentation.get_aug import get_aug
    from dataloaders.get_dataset import get_dataset
    from dataloaders.data_utils.file_utils import read_pickle
    from evaluation.tid_analyzer import (
        load_cfg_and_model,
        forward_one,
        build_single_lmdb_from_mp4,
        write_single_csv,
        find_sample_index_by_gloss,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tslnews/tslnews_stage1_configs/TSL_example_s1_dyn_config.py")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", choices=["tid","test","dev","train"], default="tid")
    ap.add_argument("--sample_index", type=int, default=0)
    ap.add_argument("--sample_gloss", default=None, help="Pick first sentence containing this gloss")
    ap.add_argument("--tid_csv", default="data/TSL-News/annotations/TID.all.corpus.csv")
    ap.add_argument("--tid_lmdb_video_dir", default="data/TSL-News/lmdb_videos/TID")
    ap.add_argument("--vocab_pkl", default="data/TSL-News/6538_gloss/processed_words_map.tsl_pkl")
    ap.add_argument("--target_gloss",  default="akşam", help="Single gloss (use --target_glosses for multiple)")
    ap.add_argument("--isolated_video", default="data/TSL-News/splitted-data/TID/akşam.mp4", help="Single iso mp4 (use --isolated_videos for multiple)")
    ap.add_argument("--target_glosses", action="append", help="Optional: multiple glosses (one per iso video), repeat flag")
    ap.add_argument("--isolated_videos", action="append", help="Optional: multiple iso videos, repeat flag")
    ap.add_argument("--iso_root_dir", default=None, help="Optional: root dir containing subfolders per gloss with mp4s inside")
    ap.add_argument("--iso_tmp_root", default="outputs/analysis/tmp_iso", help="Where to materialize temporary LMDB/CSV for isolated video")
    ap.add_argument("--output_npz", default="outputs/analysis/verify_ncc.npz")
    ap.add_argument("--stride", type=int, default=None, help="sliding stride in feature steps")
    ap.add_argument("--iso_stride_frames", type=int, default=3, help="frame subsample stride for isolated video (mp4->lmdb)")
    ap.add_argument("--iso_max_len", type=int, default=12, help="max frames after subsample for isolated video (mp4->lmdb)")
    ap.add_argument("--ncc_ignore_frames", type=int, default=0, help="ignore this many frames from start/end of iso before NCC")
    ap.add_argument("--ncc_weighting", choices=["none","triangle"], default="none", help="temporal weighting over iso frames in NCC")
    ap.add_argument("--plot", action="store_true", help="save sim_curve.png and prob_curve.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, ckpt = load_cfg_and_model(args.config, args.checkpoint)
    model.to(device)

    vocab = read_pickle(args.vocab_pkl)
    lem_to_id = vocab["dict_lem_to_id"]
    num_classes = len(lem_to_id)
    # resolve iso/gloss lists
    iso_list = []
    gloss_list = []
    if args.iso_root_dir:
        root = Path(args.iso_root_dir)
        for gloss_dir in sorted(root.iterdir()):
            if gloss_dir.is_dir():
                for mp4 in sorted(gloss_dir.glob("*.mp4")):
                    iso_list.append(str(mp4))
                    gloss_list.append(gloss_dir.name)
    else:
        iso_list = args.isolated_videos if args.isolated_videos else [args.isolated_video]
        gloss_list = args.target_glosses if args.target_glosses else [args.target_gloss]
        if len(gloss_list) == 1 and len(iso_list) > 1:
            gloss_list = gloss_list * len(iso_list)
    if len(gloss_list) != len(iso_list):
        raise ValueError("Number of target_gloss(es) must match isolated_videos")
    gloss_ids = []
    for g in gloss_list:
        if g not in lem_to_id:
            raise KeyError(f"{g} not in vocab ({len(lem_to_id)})")
        gloss_ids.append(lem_to_id[g])

    transform = get_aug(cfg.aug_name, cfg.aug_params)

    # Sentence loader (bs=1 deterministic)
    if args.split == "tid":
        dl_params = {
            "csv_dir": args.tid_csv,
            "pseudo_gloss_dir": args.vocab_pkl,
            "sep": "|",
            "name": "translation",
            "ds_params": {"lmdb_video_dir": args.tid_lmdb_video_dir, "isValid": True},
            "shuffle": False,
            "num_workers": 0,
            "bs": 1,
            "drop_last": False,
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

    # span via pseudo-gloss row (relative threshold)
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

    # Isolated: mp4 -> tmp lmdb -> loader
    iso_tmp_root = Path(args.iso_tmp_root)
    iso_tmp_root.mkdir(parents=True, exist_ok=True)

    for iso_path, gloss, gloss_id in zip(iso_list, gloss_list, gloss_ids):
        target_span = compute_span(sent_probs[gloss_id], threshold=0.2, smooth=3)

        iso_name = Path(iso_path).stem
        iso_lmdb_dir = iso_tmp_root / "lmdb" / iso_name
        iso_csv = iso_tmp_root / f"{iso_name}.csv"
        build_single_lmdb_from_mp4(
            Path(iso_path),
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

        sim_curve = sliding_ncc(
            sent_hidden,
            iso_hidden,
            stride=args.stride or 1,
            ignore_frames=args.ncc_ignore_frames,
            weighting=args.ncc_weighting,
        )

        # derive per-item output path
        out_path = Path(args.output_npz)
        if len(iso_list) > 1:
            out_path = out_path.with_name(f"{out_path.stem}_{iso_name}{out_path.suffix}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            out_path,
            sim_curve=sim_curve,
            target_gloss=gloss,
            gloss_id=gloss_id,
            target_span=target_span,
            sent_probs=sent_probs,
            iso_len=iso_T,
            valid_T=valid_T,
            checkpoint=str(ckpt),
        )
        summary = {
            "npz": str(out_path),
            "target_gloss": gloss,
            "gloss_id": gloss_id,
            "sim_peak": float(sim_curve.max()) if sim_curve.size > 0 else None,
            "sim_peak_step": int(sim_curve.argmax()) if sim_curve.size > 0 else None,
            "sim_peak_frame_est": int(sim_curve.argmax() * cfg.aug_params["stride"]) if sim_curve.size > 0 else None,
            "model_span": target_span,
            "iso_len": iso_T,
            "sentence_valid_T": valid_T,
            "checkpoint": str(ckpt),
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))

        if args.plot and sim_curve.size > 0:
            out_dir = out_path.parent
            plt.figure(figsize=(10,3))
            plt.plot(sim_curve, label="ncc")
            if target_span["start"] is not None:
                plt.axvspan(target_span["start"], target_span["end"], color="orange", alpha=0.25, label="model span")
            plt.axvline(sim_curve.argmax(), color="red", ls="--", label="sim peak")
            plt.xlabel("feature step (~frame = step*stride)")
            plt.ylabel("NCC (mean per-frame cosine)")
            plt.title(f"Similarity vs time (NCC) ({gloss})")
            plt.legend(); plt.tight_layout()
            plt.savefig(out_dir / f"sim_curve_ncc_{gloss}.png", dpi=200); plt.close()

            plt.figure(figsize=(10,3))
            plt.plot(sent_probs[gloss_id], label=f"prob({gloss})")
            plt.xlabel("time step"); plt.ylabel("probability")
            plt.title(f"Pseudo-gloss prob for {gloss} (id={gloss_id})")
            plt.tight_layout()
            plt.savefig(out_dir / f"prob_curve_{gloss}.png", dpi=200); plt.close()

    # Cleanup temporary ISO LMDB/CSV
    import shutil
    try:
        if iso_tmp_root.exists():
            shutil.rmtree(iso_tmp_root, ignore_errors=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
