#!/usr/bin/env python3
# Adapted for sign representation training and evaluation.
"""
All-vocabulary NCC evaluation for TID overlap annotations.

The older tid_overlap_ncc_eval.py evaluator searches the whole continuous
example video and only ranks isolated videos for the annotated target word. This
script changes both axes:

* For each annotated GT segment, search only the GT interval plus a small buffer
  by default, or search the whole example video with --search_scope full_video.
* Rank every usable isolated TID video whose word is present in the pseudo-gloss
  vocabulary.
* Keep one NCC peak per isolated video inside a local GT window. In full-video
  mode, keep multiple peaks per isolated video and reduce nearby duplicate
  windows with temporal NMS.
* Report top-1/3/5 retrieval accuracy for the target gloss.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import shutil
import statistics
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from evaluation.tid_overlap_ncc_eval import (
    DEFAULT_ANNOTATION_DIR,
    DEFAULT_EXAMPLE_MANIFEST,
    DEFAULT_TID_RAW_ROOTS,
    DEFAULT_WORD_MANIFEST,
    TOP_K_EVAL_VALUES,
    ascii_slug,
    build_word_aliases,
    build_word_variant_aliases,
    extract_video_id,
    find_example_for_annotation,
    hash_for_path,
    interval_iou_seconds,
    load_annotations_for_evaluation,
    load_example_manifest_lookup,
    load_tid_word_lookup,
    lookup_word_entry,
    read_csv_rows,
    resolve_existing_path,
    resolve_path,
    resolve_tid_raw_roots,
    write_csv_rows,
)


def log_status(message: str) -> None:
    print(f"[tid_overlap_local_all_vocab_ncc_eval] {message}", file=sys.stderr, flush=True)


def unique_values(values: Sequence[str]) -> List[str]:
    ordered: "OrderedDict[str, None]" = OrderedDict()
    for value in values:
        value = str(value or "").strip()
        if value:
            ordered[value] = None
    return list(ordered.keys())


def split_word_filter_terms(values: Sequence[str]) -> List[str]:
    terms: List[str] = []
    for value in values:
        normalized = str(value or "").replace(";", ",").replace("\n", ",").replace("\r", ",")
        for part in normalized.split(","):
            part = part.strip()
            if part:
                terms.append(part)
    return unique_values(terms)


def word_filter_aliases(word: str, word_dir: str = "") -> List[str]:
    candidates = [word, word_dir]
    candidates.extend(build_word_variant_aliases(word))
    candidates.extend(build_word_aliases(word, word_dir))
    aliases: List[str] = []
    for candidate in candidates:
        candidate = str(candidate or "").strip()
        if not candidate:
            continue
        aliases.append(candidate)
        aliases.append(candidate.casefold())
    return unique_values(aliases)


def load_word_filter_terms(
    cli_terms: Sequence[str],
    word_file: Optional[str],
    script_root: Path,
    *,
    label: str,
) -> List[str]:
    terms = split_word_filter_terms(cli_terms)
    if word_file:
        path = resolve_existing_path(word_file, script_root)
        if path is None:
            raise FileNotFoundError(f"{label} word filter file not found: {word_file}")
        file_terms: List[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                file_terms.append(line)
        terms.extend(split_word_filter_terms(file_terms))
    return unique_values(terms)


def load_candidate_word_terms(
    cli_terms: Sequence[str],
    word_file: Optional[str],
    script_root: Path,
) -> List[str]:
    return load_word_filter_terms(
        cli_terms,
        word_file,
        script_root,
        label="Candidate",
    )


def load_eval_word_terms(
    cli_terms: Sequence[str],
    word_file: Optional[str],
    script_root: Path,
) -> List[str]:
    return load_word_filter_terms(
        cli_terms,
        word_file,
        script_root,
        label="Evaluation",
    )


def candidate_filter_key_set(terms: Sequence[str]) -> Set[str]:
    keys: Set[str] = set()
    for term in terms:
        for alias in word_filter_aliases(term):
            keys.add(alias.casefold())
    return keys


def word_filter_selected_keys(terms: Sequence[str]) -> List[str]:
    keys: List[str] = []
    for term in terms:
        keys.extend(word_filter_aliases(term))
    return unique_values(keys)


def catalog_item_filter_keys(item: Dict[str, object]) -> Set[str]:
    keys: Set[str] = set()
    for alias in word_filter_aliases(
        str(item.get("word_name") or ""),
        str(item.get("word_dir") or ""),
    ):
        keys.add(alias.casefold())
    for gloss in item.get("glosses") or []:
        for alias in word_filter_aliases(str(gloss)):
            keys.add(alias.casefold())
    return keys


def evaluation_group_filter_keys(
    annotation_word_key: str,
    gt_rows: Sequence[Dict[str, object]],
) -> Set[str]:
    keys: Set[str] = set()
    for alias in word_filter_aliases(annotation_word_key):
        keys.add(alias.casefold())
    for row in gt_rows:
        for field in ("word_name", "annotation_word_key"):
            for alias in word_filter_aliases(str(row.get(field) or "")):
                keys.add(alias.casefold())
    return keys


def load_manifest_word_keys(word_manifest_path: Optional[Path]) -> List[str]:
    if word_manifest_path is None or not word_manifest_path.exists():
        return []
    keys: List[str] = []
    for row in read_csv_rows(word_manifest_path):
        word = row.get("word", "").strip()
        word_dir = row.get("word_dir", "").strip()
        keys.extend(build_word_aliases(word, word_dir))
        keys.extend(build_word_variant_aliases(word))
    return unique_values(keys)


def entry_glosses_in_vocab(
    entry: Dict[str, object],
    lem_to_id: Dict[str, int],
) -> List[str]:
    candidates: List[str] = []
    candidates.extend(str(token) for token in entry.get("matched_lemmas") or [])
    candidates.extend(
        build_word_variant_aliases(str(entry.get("word_name") or ""))
    )
    candidates.extend(
        build_word_aliases(
            str(entry.get("word_name") or ""),
            str(entry.get("word_dir") or ""),
        )
    )
    return unique_values([token for token in candidates if token in lem_to_id])


def unique_word_entries(word_lookup: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    entries: "OrderedDict[str, Dict[str, object]]" = OrderedDict()
    for entry in word_lookup.values():
        word_info_path = entry.get("word_info_path")
        key = str(word_info_path) if word_info_path else str(id(entry))
        entries.setdefault(key, entry)
    return list(entries.values())


def build_isolated_catalog(
    word_lookup: Dict[str, Dict[str, object]],
    lem_to_id: Dict[str, int],
    *,
    max_isolated: Optional[int] = None,
) -> List[Dict[str, object]]:
    catalog: List[Dict[str, object]] = []
    seen_paths = set()
    for entry in unique_word_entries(word_lookup):
        glosses = entry_glosses_in_vocab(entry, lem_to_id)
        if not glosses:
            continue

        for isolated in entry.get("isolated_videos") or []:
            iso_path_raw = isolated.get("path") if isinstance(isolated, dict) else None
            if not iso_path_raw:
                continue
            iso_path = Path(iso_path_raw)
            if not iso_path.exists():
                continue
            resolved = iso_path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            catalog.append(
                {
                    "catalog_index": len(catalog),
                    "word_name": str(entry.get("word_name") or ""),
                    "word_dir": str(entry.get("word_dir") or ""),
                    "glosses": glosses,
                    "isolated_video_id": str(
                        isolated.get("video_id") or extract_video_id(str(resolved))
                    ),
                    "isolated_video_path": resolved,
                    "isolated_relative_path": str(isolated.get("relative_path") or ""),
                    "word_info_path": str(entry.get("word_info_path") or ""),
                }
            )
            if max_isolated is not None and len(catalog) >= max_isolated:
                return catalog
    return catalog


def path_suffix_key(path: Path) -> str:
    parts = path.as_posix().split("/")
    if "sign2gpt" in parts:
        idx = parts.index("sign2gpt")
        return "/".join(parts[idx:])
    return path.as_posix()


def parse_optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        parsed = float(raw)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def load_movement_window_lookup(
    csv_path: Path,
) -> Tuple[Dict[str, Dict[str, str]], Dict[Tuple[str, str], Dict[str, str]], List[Dict[str, str]]]:
    rows = read_csv_rows(csv_path)
    by_path: Dict[str, Dict[str, str]] = {}
    by_role_video_id: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in rows:
        role = str(row.get("video_role") or "").strip()
        video_id = str(row.get("video_id") or row.get("source_video_id") or "").strip()
        if role and video_id:
            by_role_video_id.setdefault((role, video_id), row)
        for key in ("source_video_path", "source_video_path_key", "cropped_video_path", "cropped_video_path_key"):
            raw_path = str(row.get(key) or "").strip()
            if not raw_path:
                continue
            by_path.setdefault(raw_path, row)
            by_path.setdefault(path_suffix_key(Path(raw_path)), row)
            path = Path(raw_path)
            if path.exists():
                by_path.setdefault(str(path.resolve()), row)
    return by_path, by_role_video_id, rows


def movement_window_source_label(row: Dict[str, str]) -> str:
    for key in ("window_source", "movement_window_source", "window_method"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    if str(row.get("mediapipe_backend") or "").strip():
        return "mediapipe"
    return "movement_window"


def load_cfg_and_model_without_checkpoint(cfg_path: str):
    from models.get_models import get_model

    from train_utils.release_helpers import load_config
    cfg = load_config(cfg_path)
    model = get_model(cfg.model_name, cfg.model_params.to_dict())
    model.eval()
    return model, cfg, None


def evaluate_local_all_vocab_ncc(args: argparse.Namespace) -> Dict[str, object]:
    from train_utils.release_helpers import validate_evaluation_vocabulary
    validate_evaluation_vocabulary(args.config, args.vocab_pkl)
    log_status("importing model/video dependencies")
    try:
        import cv2
        import numpy as np
        import torch

        from augmentation.get_aug import get_aug
        from dataloaders.data_utils.file_utils import read_pickle
        from dataloaders.get_dataset import get_dataset
        from evaluation.tid_analyzer import (
            build_single_lmdb_from_mp4,
            load_cfg_and_model,
            write_single_csv,
        )
        from evaluation.tid_analyzer_ncc import sliding_ncc
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Local all-vocab NCC evaluation requires the Sign2GPT training/video "
            f"environment. Missing module: {exc.name}"
        ) from exc

    if args.no_checkpoint and args.checkpoint:
        raise ValueError("Use either --checkpoint or --no_checkpoint, not both.")

    script_root = Path(__file__).resolve().parents[1]
    annotation_dir = resolve_existing_path(args.annotation_dir, script_root)
    word_manifest_path = resolve_existing_path(args.word_manifest, script_root)
    example_manifest_path = resolve_existing_path(args.example_manifest, script_root)
    candidate_word_terms = load_candidate_word_terms(
        args.candidate_words or [],
        args.candidate_word_file,
        script_root,
    )
    candidate_word_keys = candidate_filter_key_set(candidate_word_terms)
    eval_word_terms = load_eval_word_terms(
        args.eval_words or [],
        args.eval_word_file,
        script_root,
    )
    eval_word_keys = candidate_filter_key_set(eval_word_terms)
    tid_raw_roots = resolve_tid_raw_roots(args.tid_raw_roots, script_root)
    movement_windows_path = (
        resolve_existing_path(args.movement_windows_csv, script_root)
        if args.movement_windows_csv
        else None
    )
    output_dir = resolve_path(args.output_dir, script_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    if annotation_dir is None:
        raise FileNotFoundError(f"Annotation directory not found: {args.annotation_dir}")
    if args.movement_windows_csv and movement_windows_path is None:
        raise FileNotFoundError(f"Movement-window CSV not found: {args.movement_windows_csv}")

    movement_windows_by_path: Dict[str, Dict[str, str]] = {}
    movement_windows_by_role_video_id: Dict[Tuple[str, str], Dict[str, str]] = {}
    movement_window_rows: List[Dict[str, str]] = []
    if movement_windows_path is not None and args.movement_window_apply != "none":
        (
            movement_windows_by_path,
            movement_windows_by_role_video_id,
            movement_window_rows,
        ) = load_movement_window_lookup(movement_windows_path)
        log_status(f"loaded {len(movement_window_rows)} movement-window rows from {movement_windows_path}")

    vocab_resolved = resolve_existing_path(args.vocab_pkl, script_root)
    if vocab_resolved is None:
        raise FileNotFoundError(f"Vocab pickle not found: {args.vocab_pkl}")
    log_status(f"loading vocab pickle {vocab_resolved}")
    vocab = read_pickle(str(vocab_resolved))
    lem_to_id = vocab["dict_lem_to_id"]
    num_classes = len(lem_to_id)
    log_status(f"vocab has {num_classes} classes")

    effective_candidate_terms: List[str] = []
    effective_candidate_keys: Set[str] = set()
    if candidate_word_terms:
        effective_candidate_terms = candidate_word_terms
        effective_candidate_keys = candidate_word_keys
    elif eval_word_terms and args.eval_candidate_scope == "eval_words":
        effective_candidate_terms = eval_word_terms
        effective_candidate_keys = eval_word_keys

    selected_word_keys = load_manifest_word_keys(word_manifest_path)
    restrict_metadata_to_eval_request = bool(eval_word_terms) and (
        bool(effective_candidate_terms) or args.eval_candidate_scope == "eval_words"
    )
    if restrict_metadata_to_eval_request:
        selected_word_keys = unique_values(
            word_filter_selected_keys(eval_word_terms)
            + word_filter_selected_keys(effective_candidate_terms)
        )
    if selected_word_keys:
        log_status(f"loading TID metadata for {len(selected_word_keys)} manifest aliases")
    else:
        log_status("loading TID metadata by scanning raw roots")
    word_lookup = load_tid_word_lookup(
        word_manifest_path,
        tid_raw_roots,
        script_root,
        selected_word_keys=selected_word_keys or None,
    )
    unique_entry_count = len(unique_word_entries(word_lookup))
    log_status(f"loaded {unique_entry_count} unique TID word entries")

    labels_by_example = load_annotations_for_evaluation(
        annotation_dir,
        word_manifest_path,
        word_lookup,
    )
    example_lookup = (
        load_example_manifest_lookup(example_manifest_path)
        if example_manifest_path is not None
        else {}
    )
    log_status(f"loaded {len(labels_by_example)} annotated example groups")

    isolated_catalog = build_isolated_catalog(
        word_lookup,
        lem_to_id,
        max_isolated=None if effective_candidate_keys else args.max_isolated,
    )
    if effective_candidate_keys:
        unfiltered_count = len(isolated_catalog)
        isolated_catalog = [
            item
            for item in isolated_catalog
            if catalog_item_filter_keys(item).intersection(effective_candidate_keys)
        ]
        log_status(
            "isolated candidate filter kept "
            f"{len(isolated_catalog)}/{unfiltered_count} isolated videos "
            f"for {len(effective_candidate_terms)} requested terms"
        )
        if args.max_isolated is not None:
            isolated_catalog = isolated_catalog[: max(0, int(args.max_isolated))]
            log_status(
                f"debug --max_isolated cap kept {len(isolated_catalog)} filtered videos"
            )
    if not isolated_catalog:
        raise RuntimeError("No isolated videos with in-vocab gloss labels were found.")
    log_status(
        f"isolated catalog has {len(isolated_catalog)} videos "
        f"from {len({item['word_info_path'] for item in isolated_catalog})} word entries"
    )

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = None
    if args.checkpoint:
        checkpoint_resolved = resolve_existing_path(args.checkpoint, script_root)
        checkpoint_path = str(checkpoint_resolved) if checkpoint_resolved is not None else args.checkpoint

    if args.no_checkpoint:
        log_status(f"building model from config without checkpoint ({args.config})")
        model, cfg, ckpt = load_cfg_and_model_without_checkpoint(args.config)
        checkpoint_mode = "config_init"
    else:
        log_status(f"loading model config/checkpoint ({args.config})")
        model, cfg, ckpt = load_cfg_and_model(args.config, checkpoint_path)
        checkpoint_mode = "checkpoint"
    model.to(device)
    model.eval()
    checkpoint_label = str(ckpt) if ckpt is not None else ""

    ncc_embedding_source = args.ncc_embedding_source
    if ncc_embedding_source == "auto":
        ncc_embedding_source = "spatial" if args.no_checkpoint else "encoder"
    log_status(
        f"model loaded on {device}; checkpoint_mode={checkpoint_mode}; "
        f"embedding_source={ncc_embedding_source}"
    )

    transform = get_aug(cfg.aug_name, cfg.aug_params)
    plt = None
    plot_dir: Optional[Path] = None
    if not args.no_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt_module

        plt = plt_module
        plot_dir = output_dir / args.plot_dir
        plot_dir.mkdir(parents=True, exist_ok=True)

    temp_base = resolve_path(args.tmp_dir, script_root) if args.tmp_dir else Path(tempfile.gettempdir())
    temp_base.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix="tid_local_all_vocab_ncc_", dir=str(temp_base)))
    log_status(f"temporary LMDB root is {temp_root}")

    sentence_cache: Dict[Tuple[Path, int, Optional[int]], Dict[str, object]] = {}
    hidden_cache: Dict[Tuple[Path, int, Optional[int]], Dict[str, object]] = {}
    plot_frame_cache: Dict[Tuple[str, int], Optional[object]] = {}
    plots_written = 0
    movement_stats = {
        "example_lookup_hits": 0,
        "example_windows_applied": 0,
        "isolated_lookup_hits": 0,
        "isolated_windows_applied": 0,
    }

    def forward_one_for_eval(frames):
        frames = frames.to(device)
        with torch.no_grad():
            out = model(frame_features=[frames], max_len=frames.shape[0])
        post = out["dict_post_output"]
        mask = post.get("mask", torch.ones(frames.shape[0], device=device, dtype=torch.bool))[0].bool()
        scores = post.get("softmax_scores")
        if scores is None:
            sim = post["time_res"]
            sim_btk = sim if sim.shape[1] == mask.shape[0] else sim.transpose(1, 2)
            sim_btk = sim_btk.masked_fill(~mask[None, :, None], float("-inf"))
            log_p = torch.log_softmax(sim_btk, dim=1)
            log_c = torch.log_softmax(sim_btk, dim=2)
            scores = torch.exp(log_p + log_c)
        else:
            if scores.shape[1] != mask.shape[0]:
                scores = scores.transpose(1, 2)

        valid_encoder_steps = int(mask.sum().item())
        scores = scores[0, :valid_encoder_steps, :num_classes].detach().cpu()

        if ncc_embedding_source == "spatial":
            spatial_features = out.get("dict_feat", {}).get("list_of_original_features")
            if spatial_features is None:
                raise RuntimeError(
                    "NCC embedding source 'spatial' was requested, but the model output "
                    "does not contain dict_feat['list_of_original_features']."
                )
            if isinstance(spatial_features, (list, tuple)):
                if not spatial_features:
                    raise RuntimeError("NCC spatial feature list is empty.")
                spatial_features = spatial_features[0]
            if spatial_features.dim() == 3:
                spatial_features = spatial_features[0]
            hidden = spatial_features[: frames.shape[0]].float().detach().cpu()
            valid_t = int(hidden.shape[0])
        else:
            hidden = out["enc_output"]["hidden_state"][0, :valid_encoder_steps].float().detach().cpu()
            valid_t = valid_encoder_steps
        return scores, hidden, valid_t

    def video_timing(video_path: Path) -> Dict[str, float]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video for duration: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        cap.release()
        if fps <= 0.0 or frame_count <= 0.0:
            raise RuntimeError(f"Invalid fps/frame_count for video: {video_path}")
        return {
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": frame_count / fps,
        }

    def frame_indices_for_span(
        timing: Dict[str, float],
        start_sec: float,
        end_sec: float,
        *,
        count: int,
    ) -> List[int]:
        if count <= 0:
            return []
        fps = float(timing["fps"])
        frame_count = int(timing["frame_count"])
        duration = float(timing["duration_sec"])
        if fps <= 0.0 or frame_count <= 0 or duration <= 0.0:
            return []

        start = max(0.0, min(float(start_sec), duration))
        end = max(0.0, min(float(end_sec), duration))
        if end <= start:
            return []

        start_idx = int(math.floor(start * fps))
        end_idx = int(math.ceil(end * fps)) - 1
        start_idx = max(0, min(frame_count - 1, start_idx))
        end_idx = max(start_idx, min(frame_count - 1, end_idx))

        available_count = end_idx - start_idx + 1
        sample_count = min(count, available_count)
        if sample_count <= 0:
            return []
        if sample_count == 1:
            return [start_idx]

        sampled = np.linspace(start_idx, end_idx, num=sample_count)
        indices: List[int] = []
        for value in sampled:
            index = max(start_idx, min(end_idx, int(round(float(value)))))
            if index not in indices:
                indices.append(index)
        return indices

    def blank_frame_cell(*, height: int = 96, width: int = 144, fill: int = 245):
        return np.full((height, width, 3), fill, dtype=np.uint8)

    def frame_to_cell(frame, *, height: int = 96, width: int = 144):
        cell = blank_frame_cell(height=height, width=width, fill=255)
        if frame is None:
            return cell
        frame_height, frame_width = frame.shape[:2]
        if frame_height <= 0 or frame_width <= 0:
            return cell
        scale = min(width / float(frame_width), height / float(frame_height))
        resized_width = max(1, int(round(frame_width * scale)))
        resized_height = max(1, int(round(frame_height * scale)))
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        x0 = (width - resized_width) // 2
        y0 = (height - resized_height) // 2
        cell[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
        return cell

    def read_frame_cell(video_path: Path, frame_index: int):
        try:
            resolved = str(video_path.resolve())
        except OSError:
            resolved = str(video_path)
        cache_key = (resolved, int(frame_index))
        if cache_key in plot_frame_cache:
            cached = plot_frame_cache[cache_key]
            return cached if cached is not None else blank_frame_cell()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            plot_frame_cache[cache_key] = None
            return blank_frame_cell()
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            plot_frame_cache[cache_key] = None
            return blank_frame_cell()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cell = frame_to_cell(frame_rgb)
        plot_frame_cache[cache_key] = cell
        return cell

    def sample_frame_cells(
        video_path: Path,
        start_sec: float,
        end_sec: float,
        *,
        count: int,
    ) -> List[object]:
        if count <= 0:
            return []
        try:
            timing = video_timing(video_path)
        except Exception:
            return []
        indices = frame_indices_for_span(timing, start_sec, end_sec, count=count)
        return [read_frame_cell(video_path, index) for index in indices]

    def template_effective_video_span(template: Dict[str, object]) -> Tuple[float, float]:
        duration = float(template.get("isolated_duration_seconds") or 0.0)
        start = max(0.0, float(template.get("iso_trim_start_seconds") or 0.0))
        end = duration - max(0.0, float(template.get("iso_trim_end_seconds") or 0.0))
        end = max(start, min(duration, end))

        trimmed_steps = int(template.get("isolated_trimmed_steps") or 0)
        ignore_frames = max(0, int(args.ncc_ignore_frames))
        if ignore_frames > 0 and trimmed_steps > 2 * ignore_frames and end > start:
            step_sec = (end - start) / float(trimmed_steps)
            start += ignore_frames * step_sec
            end -= ignore_frames * step_sec
        return start, max(start, end)

    def short_plot_label(value: object, *, limit: int = 34) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)].rstrip() + "…"

    def frame_rows_to_montage(
        frame_rows: Sequence[Dict[str, object]],
        *,
        frame_sample_count: int,
        cell_height: int = 96,
        cell_width: int = 144,
    ):
        row_images = []
        for row in frame_rows:
            cells = list(row.get("cells") or [])
            while len(cells) < frame_sample_count:
                cells.append(blank_frame_cell(height=cell_height, width=cell_width))
            row_images.append(np.concatenate(cells[:frame_sample_count], axis=1))
        return np.concatenate(row_images, axis=0)

    def draw_frame_rows_axis(
        frame_ax,
        frame_rows: Sequence[Dict[str, object]],
        *,
        frame_sample_count: int,
        title: str,
    ) -> None:
        cell_height = 96
        cell_width = 144
        montage = frame_rows_to_montage(
            frame_rows,
            frame_sample_count=frame_sample_count,
            cell_height=cell_height,
            cell_width=cell_width,
        )
        frame_ax.imshow(montage, aspect="auto")
        frame_ax.set_title(title, fontsize=9)
        frame_ax.set_xticks(
            [(idx + 0.5) * cell_width for idx in range(frame_sample_count)]
        )
        frame_ax.set_xticklabels(
            [f"{idx + 1}" for idx in range(frame_sample_count)],
            fontsize=7,
        )
        frame_ax.set_yticks(
            [(idx + 0.5) * cell_height for idx in range(len(frame_rows))]
        )
        frame_ax.set_yticklabels(
            [str(row["label"]) for row in frame_rows],
            fontsize=7,
        )
        for tick_label, row in zip(frame_ax.get_yticklabels(), frame_rows):
            tick_label.set_color(str(row.get("color") or "#202124"))
        for idx in range(frame_sample_count + 1):
            frame_ax.axvline(idx * cell_width - 0.5, color="#E0E0E0", lw=0.5)
        for idx in range(len(frame_rows) + 1):
            frame_ax.axhline(idx * cell_height - 0.5, color="#E0E0E0", lw=0.5)
        frame_ax.set_xlim(-0.5, frame_sample_count * cell_width - 0.5)
        frame_ax.set_ylim(len(frame_rows) * cell_height - 0.5, -0.5)
        frame_ax.set_xlabel(
            "evenly sampled frames; blanks mean the span exposed fewer readable frames",
            fontsize=7,
        )
        frame_ax.tick_params(axis="both", length=0)
        for spine in ("top", "right", "left", "bottom"):
            frame_ax.spines[spine].set_visible(False)

    def save_frame_rows_plot(
        frame_plot_path: Path,
        frame_rows: Sequence[Dict[str, object]],
        *,
        frame_sample_count: int,
        title: str,
    ) -> None:
        if plt is None or not frame_rows or frame_sample_count <= 0:
            return
        fig_height = max(2.2, 0.55 * float(len(frame_rows)) + 0.9)
        fig = plt.figure(figsize=(10.5, fig_height))
        frame_ax = fig.add_subplot(1, 1, 1)
        draw_frame_rows_axis(
            frame_ax,
            frame_rows,
            frame_sample_count=frame_sample_count,
            title=title,
        )
        frame_plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(frame_plot_path, dpi=180)
        plt.close(fig)

    def loaded_duration_seconds(
        timing: Dict[str, float],
        *,
        stored_frames: int,
        lmdb_stride: int,
        max_len: Optional[int],
    ) -> float:
        if stored_frames <= 0:
            return 0.0
        if max_len is None and lmdb_stride <= 1:
            return float(timing["duration_sec"])
        sampled_source_frames = ((stored_frames - 1) * max(1, lmdb_stride)) + 1
        sampled_source_frames = min(float(sampled_source_frames), float(timing["frame_count"]))
        return sampled_source_frames / float(timing["fps"])

    def load_hidden(
        video_path: Path,
        *,
        lmdb_stride: int,
        max_len: Optional[int],
        cache: Dict[Tuple[Path, int, Optional[int]], Dict[str, object]],
    ) -> Dict[str, object]:
        cache_key = (video_path.resolve(), lmdb_stride, max_len)
        if cache_key in cache:
            return cache[cache_key]

        safe_name = f"{ascii_slug(video_path.stem) or 'video'}_{hash_for_path(video_path)}"
        env_dir = temp_root / "lmdb" / safe_name
        csv_path = temp_root / f"{safe_name}.csv"
        try:
            _, stored_frames = build_single_lmdb_from_mp4(
                video_path,
                env_dir,
                stride=max(1, lmdb_stride),
                max_len=max_len,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to build temporary LMDB at {env_dir} from video {video_path}. "
                "If this is an LMDB mdb_txn_commit I/O error, pass --tmp_dir pointing "
                "to a writable native Linux scratch directory such as /tmp."
            ) from exc

        write_single_csv(csv_path, env_dir.name)
        dl_params = {
            "csv_dir": str(csv_path),
            "pseudo_gloss_dir": str(vocab_resolved),
            "sep": "|",
            "name": "translation",
            "ds_params": {"lmdb_video_dir": str(env_dir.parent), "isValid": True},
            "shuffle": False,
            "num_workers": 0,
            "bs": 1,
            "drop_last": False,
        }
        dl, _ = get_dataset(cfg.test_ds_name, dl_params, transform=transform)
        batch = next(iter(dl))
        frames = batch["frames"][0]
        scores, hidden, valid_t = forward_one_for_eval(frames)
        timing = video_timing(video_path)
        item = {
            "scores": scores,
            "hidden": hidden,
            "valid_T": valid_t,
            "duration_sec": loaded_duration_seconds(
                timing,
                stored_frames=int(stored_frames),
                lmdb_stride=max(1, lmdb_stride),
                max_len=max_len,
            ),
            "source_duration_sec": timing["duration_sec"],
            "stored_frames": int(stored_frames),
        }
        cache[cache_key] = item
        return item

    def trim_hidden_by_seconds(
        hidden,
        *,
        duration_sec: float,
        trim_start_sec: float,
        trim_end_sec: float,
    ):
        steps = int(hidden.shape[0])
        if steps <= 0 or duration_sec <= 0.0:
            return hidden, 0, 0, 0.0, 0.0
        step_sec = duration_sec / float(steps)
        start_steps = max(0, int(round(trim_start_sec / step_sec)))
        end_steps = max(0, int(round(trim_end_sec / step_sec)))
        if start_steps + end_steps >= steps:
            return hidden, 0, 0, 0.0, 0.0
        end_idx = steps - end_steps if end_steps > 0 else steps
        return (
            hidden[start_steps:end_idx],
            start_steps,
            end_steps,
            start_steps * step_sec,
            end_steps * step_sec,
        )

    def strict_hidden_window_by_seconds(
        hidden,
        *,
        duration_sec: float,
        start_sec: float,
        end_sec: float,
    ):
        steps = int(hidden.shape[0])
        if steps <= 0 or duration_sec <= 0.0:
            return hidden, 0, 0, 0.0, 0.0
        step_sec = duration_sec / float(steps)
        start_steps = max(0, int(math.ceil((max(0.0, start_sec) / step_sec) - 1e-9)))
        end_idx = min(steps, int(math.floor((min(float(duration_sec), end_sec) / step_sec) + 1e-9)))
        if end_idx <= start_steps:
            return hidden, 0, 0, 0.0, 0.0
        end_steps = steps - end_idx
        return (
            hidden[start_steps:end_idx],
            start_steps,
            end_steps,
            start_steps * step_sec,
            end_steps * step_sec,
        )

    def time_window_to_steps(
        *,
        start_time: float,
        end_time: float,
        step_sec: float,
        total_steps: int,
        strict_start: bool,
        strict_end: bool,
    ) -> Tuple[int, int]:
        if strict_start:
            start_step = int(math.ceil((start_time / step_sec) - 1e-9))
        else:
            start_step = int(math.floor(start_time / step_sec))
        if strict_end:
            end_step = int(math.floor((end_time / step_sec) + 1e-9))
        else:
            end_step = int(math.ceil(end_time / step_sec))
        start_step = max(0, min(start_step, total_steps - 1))
        end_step = max(start_step + 1, min(end_step, total_steps))
        return start_step, end_step

    def movement_apply_enabled(role: str) -> bool:
        if args.movement_window_apply == "both":
            return True
        if args.movement_window_apply == "examples" and role == "example":
            return True
        if args.movement_window_apply == "isolated" and role == "isolated":
            return True
        return False

    def movement_row_for_video(
        video_path: Path,
        *,
        role: str,
        video_id: str = "",
    ) -> Optional[Dict[str, str]]:
        if not movement_apply_enabled(role):
            return None
        candidates = [
            str(video_path),
            str(video_path.resolve()),
            path_suffix_key(video_path),
            path_suffix_key(video_path.resolve()),
        ]
        for key in candidates:
            row = movement_windows_by_path.get(key)
            if row is not None:
                return row
        if video_id:
            return movement_windows_by_role_video_id.get((role, str(video_id)))
        return None

    def movement_bounds_for_video(
        video_path: Path,
        *,
        role: str,
        duration_sec: float,
        video_id: str = "",
    ) -> Optional[Tuple[float, float, Dict[str, str]]]:
        row = movement_row_for_video(video_path, role=role, video_id=video_id)
        if row is None:
            return None
        movement_found = str(row.get("movement_found") or "").strip().lower()
        if movement_found in {"0", "false", "no"}:
            return None
        start = parse_optional_float(row.get("movement_start_time"))
        end = parse_optional_float(row.get("movement_end_time"))
        if start is None or end is None:
            return None
        buffer_sec = max(0.0, float(args.movement_window_buffer_seconds))
        start = max(0.0, start - buffer_sec)
        end = min(float(duration_sec), end + buffer_sec)
        min_duration = max(0.0, float(args.movement_window_min_duration_seconds))
        if end - start < min_duration:
            return None
        if end <= start:
            return None
        if role == "example":
            movement_stats["example_lookup_hits"] += 1
        elif role == "isolated":
            movement_stats["isolated_lookup_hits"] += 1
        return start, end, row

    def trim_hidden_by_absolute_window(
        hidden,
        *,
        duration_sec: float,
        start_sec: float,
        end_sec: float,
    ):
        return strict_hidden_window_by_seconds(
            hidden,
            duration_sec=duration_sec,
            start_sec=start_sec,
            end_sec=end_sec,
        )

    def movement_trimmed_hidden_for_video(
        hidden,
        *,
        video_path: Path,
        role: str,
        duration_sec: float,
        video_id: str = "",
    ):
        bounds = movement_bounds_for_video(
            video_path,
            role=role,
            duration_sec=duration_sec,
            video_id=video_id,
        )
        if bounds is None:
            return None
        start_sec, end_sec, row = bounds
        (
            trimmed_hidden,
            start_steps,
            end_steps,
            applied_start,
            applied_end,
        ) = trim_hidden_by_absolute_window(
            hidden,
            duration_sec=duration_sec,
            start_sec=start_sec,
            end_sec=end_sec,
        )
        if int(trimmed_hidden.shape[0]) <= 0:
            return None
        if role == "example":
            movement_stats["example_windows_applied"] += 1
        elif role == "isolated":
            movement_stats["isolated_windows_applied"] += 1
        return {
            "hidden": trimmed_hidden,
            "start_steps": start_steps,
            "end_steps": end_steps,
            "start_time": applied_start,
            "end_trim_seconds": applied_end,
            "end_time": float(duration_sec) - applied_end,
            "row": row,
        }

    def iso_trim_bounds_seconds(duration: float) -> Tuple[float, float]:
        trim_start = (
            args.iso_trim_start_seconds
            if args.iso_trim_start_seconds is not None
            else args.iso_trim_seconds
        )
        trim_end = (
            args.iso_trim_end_seconds
            if args.iso_trim_end_seconds is not None
            else args.iso_trim_seconds
        )
        trim_start = max(0.0, float(trim_start))
        trim_end = max(0.0, float(trim_end))
        if duration <= 0.0 or trim_start + trim_end >= duration:
            return 0.0, 0.0
        return trim_start, trim_end

    def load_isolated_templates() -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
        templates: List[Dict[str, object]] = []
        skipped: List[Dict[str, object]] = []
        total = len(isolated_catalog)
        for idx, item in enumerate(isolated_catalog, start=1):
            if idx == 1 or idx % max(1, args.log_every) == 0 or idx == total:
                log_status(f"encoding isolated template {idx}/{total}")
            iso_path = Path(item["isolated_video_path"])
            try:
                iso_item = load_hidden(
                    iso_path,
                    lmdb_stride=args.iso_stride_frames,
                    max_len=args.iso_max_len,
                    cache=hidden_cache,
                )
                movement_trim = movement_trimmed_hidden_for_video(
                    iso_item["hidden"],
                    video_path=iso_path,
                    role="isolated",
                    duration_sec=float(iso_item["duration_sec"]),
                    video_id=str(item.get("isolated_video_id") or ""),
                )
                if movement_trim is not None:
                    iso_hidden = movement_trim["hidden"]
                    iso_trim_start_steps = int(movement_trim["start_steps"])
                    iso_trim_end_steps = int(movement_trim["end_steps"])
                    applied_iso_trim_start = float(movement_trim["start_time"])
                    applied_iso_trim_end = float(movement_trim["end_trim_seconds"])
                    movement_window_source = movement_window_source_label(movement_trim["row"])
                    movement_window_start_time = float(movement_trim["start_time"])
                    movement_window_end_time = float(movement_trim["end_time"])
                else:
                    iso_trim_start, iso_trim_end = iso_trim_bounds_seconds(float(iso_item["duration_sec"]))
                    (
                        iso_hidden,
                        iso_trim_start_steps,
                        iso_trim_end_steps,
                        applied_iso_trim_start,
                        applied_iso_trim_end,
                    ) = trim_hidden_by_seconds(
                        iso_item["hidden"],
                        duration_sec=float(iso_item["duration_sec"]),
                        trim_start_sec=iso_trim_start,
                        trim_end_sec=iso_trim_end,
                    )
                    movement_window_source = "fixed_iso_trim"
                    movement_window_start_time = 0.0
                    movement_window_end_time = float(iso_item["duration_sec"])
            except Exception as exc:
                row = {
                    "word_name": item["word_name"],
                    "word_dir": item["word_dir"],
                    "glosses": " ".join(item["glosses"]),
                    "isolated_video_id": item["isolated_video_id"],
                    "isolated_video_path": str(iso_path),
                    "reason": f"template_load_failed: {exc}",
                }
                skipped.append(row)
                if args.fail_on_template_error:
                    raise
                continue

            if int(iso_hidden.shape[0]) <= 0:
                skipped.append(
                    {
                        "word_name": item["word_name"],
                        "word_dir": item["word_dir"],
                        "glosses": " ".join(item["glosses"]),
                        "isolated_video_id": item["isolated_video_id"],
                        "isolated_video_path": str(iso_path),
                        "reason": "empty_trimmed_template",
                    }
                )
                continue

            effective_steps = int(iso_hidden.shape[0])
            if args.ncc_ignore_frames > 0 and effective_steps > 2 * args.ncc_ignore_frames:
                effective_steps -= 2 * args.ncc_ignore_frames
            if effective_steps <= 0:
                skipped.append(
                    {
                        "word_name": item["word_name"],
                        "word_dir": item["word_dir"],
                        "glosses": " ".join(item["glosses"]),
                        "isolated_video_id": item["isolated_video_id"],
                        "isolated_video_path": str(iso_path),
                        "reason": "empty_template_after_ncc_ignore_frames",
                    }
                )
                continue

            template = dict(item)
            template.update(
                {
                    "hidden": iso_hidden,
                    "isolated_duration_seconds": float(iso_item["duration_sec"]),
                    "isolated_source_duration_seconds": float(iso_item["source_duration_sec"]),
                    "isolated_steps": int(iso_item["hidden"].shape[0]),
                    "isolated_trimmed_steps": int(iso_hidden.shape[0]),
                    "effective_isolated_steps": effective_steps,
                    "iso_trim_start_seconds": applied_iso_trim_start,
                    "iso_trim_end_seconds": applied_iso_trim_end,
                    "iso_trim_start_steps": iso_trim_start_steps,
                    "iso_trim_end_steps": iso_trim_end_steps,
                    "movement_window_source": movement_window_source,
                    "movement_window_start_time": movement_window_start_time,
                    "movement_window_end_time": movement_window_end_time,
                }
            )
            templates.append(template)
        return templates, skipped

    def local_hidden_window(
        sentence_item: Dict[str, object],
        gt_start_sec: float,
        gt_end_sec: float,
        *,
        video_path: Path,
        video_id: str = "",
    ) -> Optional[Dict[str, object]]:
        sentence_steps = int(sentence_item["valid_T"])
        sentence_duration = float(sentence_item["duration_sec"])
        if sentence_steps <= 0 or sentence_duration <= 0.0:
            return None

        step_sec = sentence_duration / float(sentence_steps)
        buffer_sec = max(0.0, float(args.gt_buffer_seconds))
        requested_start = max(0.0, gt_start_sec - buffer_sec)
        requested_end = min(sentence_duration, gt_end_sec + buffer_sec)
        strict_start = False
        strict_end = False
        movement_source = "none"
        movement_start = ""
        movement_end = ""
        movement_bounds = movement_bounds_for_video(
            video_path,
            role="example",
            duration_sec=sentence_duration,
            video_id=video_id,
        )
        if movement_bounds is not None:
            movement_start_sec, movement_end_sec, movement_row = movement_bounds
            movement_source_label = movement_window_source_label(movement_row)
            movement_start = f"{movement_start_sec:.6f}"
            movement_end = f"{movement_end_sec:.6f}"
            clipped_start = max(requested_start, movement_start_sec)
            clipped_end = min(requested_end, movement_end_sec)
            if clipped_end > clipped_start:
                strict_start = clipped_start > requested_start + 1e-9
                strict_end = clipped_end < requested_end - 1e-9
                requested_start = clipped_start
                requested_end = clipped_end
                movement_source = movement_source_label
                movement_stats["example_windows_applied"] += 1
            else:
                movement_source = f"{movement_source_label}_no_overlap"
        if requested_end <= requested_start:
            return None

        start_step, end_step = time_window_to_steps(
            start_time=requested_start,
            end_time=requested_end,
            step_sec=step_sec,
            total_steps=sentence_steps,
            strict_start=strict_start,
            strict_end=strict_end,
        )
        actual_start = start_step * step_sec
        actual_end = min(sentence_duration, end_step * step_sec)
        return {
            "hidden": sentence_item["hidden"][start_step:end_step],
            "start_step": start_step,
            "end_step": end_step,
            "start_time": actual_start,
            "end_time": actual_end,
            "step_sec": step_sec,
            "sentence_steps": sentence_steps,
            "sentence_duration": sentence_duration,
            "movement_window_source": movement_source,
            "movement_window_start_time": movement_start,
            "movement_window_end_time": movement_end,
        }

    def full_video_hidden_window(
        sentence_item: Dict[str, object],
        *,
        video_path: Path,
        video_id: str = "",
    ) -> Optional[Dict[str, object]]:
        sentence_steps = int(sentence_item["valid_T"])
        sentence_duration = float(sentence_item["duration_sec"])
        if sentence_steps <= 0 or sentence_duration <= 0.0:
            return None
        step_sec = sentence_duration / float(sentence_steps)
        start_time = 0.0
        end_time = sentence_duration
        movement_source = "none"
        movement_start = ""
        movement_end = ""
        strict_movement_window = False
        movement_bounds = movement_bounds_for_video(
            video_path,
            role="example",
            duration_sec=sentence_duration,
            video_id=video_id,
        )
        if movement_bounds is not None:
            start_time, end_time, movement_row = movement_bounds
            movement_source = movement_window_source_label(movement_row)
            movement_start = f"{start_time:.6f}"
            movement_end = f"{end_time:.6f}"
            strict_movement_window = True
            movement_stats["example_windows_applied"] += 1
        start_step, end_step = time_window_to_steps(
            start_time=start_time,
            end_time=end_time,
            step_sec=step_sec,
            total_steps=sentence_steps,
            strict_start=strict_movement_window,
            strict_end=strict_movement_window,
        )
        actual_start = start_step * step_sec
        actual_end = min(sentence_duration, end_step * step_sec)
        return {
            "hidden": sentence_item["hidden"][start_step:end_step],
            "start_step": start_step,
            "end_step": end_step,
            "start_time": actual_start,
            "end_time": actual_end,
            "step_sec": step_sec,
            "sentence_steps": sentence_steps,
            "sentence_duration": sentence_duration,
            "movement_window_source": movement_source,
            "movement_window_start_time": movement_start,
            "movement_window_end_time": movement_end,
        }

    def temporal_hit(iou: float) -> bool:
        threshold = max(0.0, float(args.hit_iou_threshold))
        if args.search_scope == "local_gt" and threshold <= 0.0:
            return True
        if threshold <= 0.0:
            return iou > 0.0
        return iou >= threshold

    def ncc_window_steps_from_peak(
        *,
        search_start_step: int,
        search_end_step: int,
        peak_idx: int,
        window_steps: int,
        stride_steps: int,
    ) -> Tuple[int, int]:
        if search_end_step <= search_start_step:
            return search_start_step, search_end_step
        if search_end_step - search_start_step < window_steps:
            return search_start_step, search_end_step

        peak_step = search_start_step + (int(peak_idx) * stride_steps)
        if args.ncc_peak_position == "center":
            start_step = int(math.floor(float(peak_step) - (float(window_steps) / 2.0) + 0.5))
            min_start = search_start_step
            max_start = max(search_start_step, search_end_step - window_steps)
            start_step = min(max(start_step, min_start), max_start)
            end_step = min(search_end_step, start_step + window_steps)
            return start_step, end_step

        start_step = peak_step
        end_step = min(search_end_step, start_step + window_steps)
        return start_step, end_step

    def candidate_from_peak(
        search_window: Dict[str, object],
        template: Dict[str, object],
        *,
        peak_idx: int,
        peak_score: float,
        nms_scope: str,
    ) -> Dict[str, object]:
        effective_iso_steps = int(template["effective_isolated_steps"])
        stride_steps = max(1, args.stride)
        start_step, end_step = ncc_window_steps_from_peak(
            search_start_step=int(search_window["start_step"]),
            search_end_step=int(search_window["end_step"]),
            peak_idx=peak_idx,
            window_steps=effective_iso_steps,
            stride_steps=stride_steps,
        )

        step_sec = float(search_window["step_sec"])
        pred_start = start_step * step_sec
        pred_end = min(float(search_window["sentence_duration"]), end_step * step_sec)
        return {
            "rank": 0,
            "sort_index": int(template["catalog_index"]),
            "template_catalog_index": int(template["catalog_index"]),
            "word_name": template["word_name"],
            "word_dir": template["word_dir"],
            "glosses": list(template["glosses"]),
            "predicted_gloss": " ".join(template["glosses"]),
            "isolated_video_id": template["isolated_video_id"],
            "isolated_video_path": str(template["isolated_video_path"]),
            "isolated_duration_seconds": float(template["isolated_duration_seconds"]),
            "isolated_source_duration_seconds": float(template["isolated_source_duration_seconds"]),
            "isolated_steps": int(template["isolated_steps"]),
            "isolated_trimmed_steps": int(template["isolated_trimmed_steps"]),
            "effective_isolated_steps": effective_iso_steps,
            "iso_trim_start_seconds": float(template["iso_trim_start_seconds"]),
            "iso_trim_end_seconds": float(template["iso_trim_end_seconds"]),
            "template_movement_window_source": str(template.get("movement_window_source") or ""),
            "template_movement_window_start_time": float(template.get("movement_window_start_time") or 0.0),
            "template_movement_window_end_time": float(template.get("movement_window_end_time") or 0.0),
            "peak_score": float(peak_score),
            "peak_index": int(peak_idx),
            "start_step": start_step,
            "end_step": end_step,
            "start_time": pred_start,
            "end_time": pred_end,
            "time_interval": f"[{pred_start:.6f}, {pred_end:.6f}]",
            "ncc_stride": stride_steps,
            "ncc_peak_position": args.ncc_peak_position,
            "nms_scope": nms_scope,
            "nms_merged_count": 1,
        }

    def annotate_candidate_for_gt(
        candidate: Dict[str, object],
        target_glosses: Sequence[str],
        gt_start_sec: float,
        gt_end_sec: float,
    ) -> Dict[str, object]:
        item = dict(candidate)
        target_set = set(target_glosses)
        candidate_set = set(candidate.get("glosses") or [])
        matched_target_glosses = sorted(target_set.intersection(candidate_set))
        iou = interval_iou_seconds(
            float(candidate["start_time"]),
            float(candidate["end_time"]),
            gt_start_sec,
            gt_end_sec,
        )
        is_target_gloss = bool(matched_target_glosses)
        item.update(
            {
                "matched_target_glosses": " ".join(matched_target_glosses),
                "is_target_gloss": is_target_gloss,
                "is_correct": bool(is_target_gloss and temporal_hit(iou)),
                "iou": iou,
            }
        )
        return item

    def sim_curve_for_template(
        search_window: Dict[str, object],
        template: Dict[str, object],
    ):
        return sliding_ncc(
            search_window["hidden"],
            template["hidden"],
            stride=max(1, args.stride),
            ignore_frames=args.ncc_ignore_frames,
            weighting=args.ncc_weighting,
        )

    def candidate_interval_iou(left: Dict[str, object], right: Dict[str, object]) -> float:
        return interval_iou_seconds(
            float(left["start_time"]),
            float(left["end_time"]),
            float(right["start_time"]),
            float(right["end_time"]),
        )

    def candidate_center_distance(left: Dict[str, object], right: Dict[str, object]) -> float:
        left_center = (float(left["start_time"]) + float(left["end_time"])) / 2.0
        right_center = (float(right["start_time"]) + float(right["end_time"])) / 2.0
        return abs(left_center - right_center)

    def should_merge_or_suppress(left: Dict[str, object], right: Dict[str, object]) -> bool:
        if candidate_interval_iou(left, right) >= max(0.0, float(args.nms_iou_threshold)):
            return True
        center_threshold = max(0.0, float(args.nms_center_distance_seconds))
        return center_threshold > 0.0 and candidate_center_distance(left, right) <= center_threshold

    def merge_detection_cluster(cluster: Sequence[Dict[str, object]]) -> Dict[str, object]:
        best = max(
            cluster,
            key=lambda item: (float(item["peak_score"]), -int(item["sort_index"])),
        )
        merged = dict(best)
        if args.nms_mode == "merge" and len(cluster) > 1:
            start_time = min(float(item["start_time"]) for item in cluster)
            end_time = max(float(item["end_time"]) for item in cluster)
            start_step = min(int(item["start_step"]) for item in cluster)
            end_step = max(int(item["end_step"]) for item in cluster)
            merged.update(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "time_interval": f"[{start_time:.6f}, {end_time:.6f}]",
                    "start_step": start_step,
                    "end_step": end_step,
                }
            )
        merged["nms_merged_count"] = len(cluster)
        return merged

    def temporal_nms(detections: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        remaining = sorted(
            detections,
            key=lambda item: (-float(item["peak_score"]), int(item["peak_index"])),
        )
        kept: List[Dict[str, object]] = []
        while remaining:
            best = remaining.pop(0)
            cluster = [best]
            next_remaining: List[Dict[str, object]] = []
            for item in remaining:
                if should_merge_or_suppress(best, item):
                    cluster.append(item)
                else:
                    next_remaining.append(item)
            kept.append(merge_detection_cluster(cluster))
            remaining = next_remaining
        kept.sort(key=lambda item: (-float(item["peak_score"]), int(item["peak_index"])))
        return kept

    def peak_indices_for_full_video_curve(sim_curve) -> List[int]:
        if sim_curve.size == 0:
            return []
        finite_mask = np.isfinite(sim_curve)
        if not np.any(finite_mask):
            return []

        finite_scores = np.where(finite_mask, sim_curve, -np.inf)
        global_peak = int(np.argmax(finite_scores))
        max_score = float(finite_scores[global_peak])
        score_threshold = float(args.full_video_min_peak_score)
        relative_threshold = max(0.0, float(args.full_video_relative_peak_threshold))
        if relative_threshold > 0.0 and max_score > 0.0:
            score_threshold = max(score_threshold, max_score * relative_threshold)

        peaks: List[int] = []
        for idx in np.where(finite_mask)[0]:
            idx = int(idx)
            score = float(sim_curve[idx])
            if score < score_threshold:
                continue
            left_ok = (
                idx == 0
                or not bool(finite_mask[idx - 1])
                or score >= float(sim_curve[idx - 1])
            )
            right_ok = (
                idx == sim_curve.size - 1
                or not bool(finite_mask[idx + 1])
                or score >= float(sim_curve[idx + 1])
            )
            if left_ok and right_ok:
                peaks.append(idx)

        if not peaks and max_score >= score_threshold:
            peaks = [global_peak]

        peaks = sorted(peaks, key=lambda idx: (-float(sim_curve[idx]), idx))
        pre_limit = max(1, int(args.full_video_pre_nms_peak_limit))
        return peaks[:pre_limit]

    def detections_for_template_full_video(
        search_window: Dict[str, object],
        template: Dict[str, object],
    ) -> List[Dict[str, object]]:
        sim_curve = sim_curve_for_template(search_window, template)
        peak_indices = peak_indices_for_full_video_curve(sim_curve)
        if not peak_indices:
            return []

        raw_detections = [
            candidate_from_peak(
                search_window,
                template,
                peak_idx=peak_idx,
                peak_score=float(sim_curve[peak_idx]),
                nms_scope="per_isolated_video_full_video_temporal_nms",
            )
            for peak_idx in peak_indices
        ]
        kept = temporal_nms(raw_detections)
        return kept[: max(1, int(args.full_video_max_peaks_per_isolated))]

    def rank_templates_for_full_video(
        search_window: Dict[str, object],
        templates: Sequence[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        ranked: List[Dict[str, object]] = []
        for template in templates:
            ranked.extend(detections_for_template_full_video(search_window, template))
        ranked.sort(
            key=lambda item: (
                -float(item["peak_score"]),
                float(item["start_time"]),
                int(item["sort_index"]),
                int(item["peak_index"]),
            )
        )
        for rank, item in enumerate(ranked, start=1):
            item["rank"] = rank
        return ranked

    def annotate_ranked_for_gt(
        ranked: Sequence[Dict[str, object]],
        target_glosses: Sequence[str],
        gt_start_sec: float,
        gt_end_sec: float,
    ) -> List[Dict[str, object]]:
        return [
            annotate_candidate_for_gt(item, target_glosses, gt_start_sec, gt_end_sec)
            for item in ranked
        ]

    def one_peak_for_template(
        local_window: Dict[str, object],
        template: Dict[str, object],
        target_glosses: Sequence[str],
        gt_start_sec: float,
        gt_end_sec: float,
    ) -> Optional[Dict[str, object]]:
        local_hidden = local_window["hidden"]
        if int(local_hidden.shape[0]) <= 0:
            return None

        sim_curve = sliding_ncc(
            local_hidden,
            template["hidden"],
            stride=max(1, args.stride),
            ignore_frames=args.ncc_ignore_frames,
            weighting=args.ncc_weighting,
        )
        if sim_curve.size == 0:
            return None

        finite_mask = np.isfinite(sim_curve)
        if not np.any(finite_mask):
            return None

        masked_curve = np.where(finite_mask, sim_curve, -np.inf)
        peak_idx = int(np.argmax(masked_curve))
        peak_score = float(masked_curve[peak_idx])
        if not np.isfinite(peak_score):
            return None

        candidate = candidate_from_peak(
            local_window,
            template,
            peak_idx=peak_idx,
            peak_score=peak_score,
            nms_scope="one_peak_per_isolated_video_per_gt_window",
        )
        return annotate_candidate_for_gt(candidate, target_glosses, gt_start_sec, gt_end_sec)

    def rank_templates_for_gt(
        local_window: Dict[str, object],
        templates: Sequence[Dict[str, object]],
        target_glosses: Sequence[str],
        gt_start_sec: float,
        gt_end_sec: float,
    ) -> List[Dict[str, object]]:
        ranked: List[Dict[str, object]] = []
        for template in templates:
            item = one_peak_for_template(
                local_window,
                template,
                target_glosses,
                gt_start_sec,
                gt_end_sec,
            )
            if item is not None:
                ranked.append(item)
        ranked.sort(key=lambda item: (-float(item["peak_score"]), int(item["sort_index"])))
        for rank, item in enumerate(ranked, start=1):
            item["rank"] = rank
        return ranked

    def resolve_group_entry(annotation_word_key: str) -> Optional[Dict[str, object]]:
        group = lookup_word_entry(annotation_word_key, word_lookup)
        if group is None:
            group = lookup_word_entry(annotation_word_key, example_lookup)
        return group

    def top_candidate_summary(candidates: Sequence[Dict[str, object]], limit: int) -> str:
        parts: List[str] = []
        for item in candidates[:limit]:
            correct = "*" if item.get("is_correct") else ""
            parts.append(
                f"{item['rank']}:{item['predicted_gloss']}"
                f"{correct}:{float(item['peak_score']):.6f}"
            )
        return "; ".join(parts)

    def save_detection_plot(
        plot_path: Path,
        *,
        word_name: str,
        source_video_id: str,
        example_video_number: int,
        example_video_path: Path,
        frame_plot_path: Optional[Path],
        search_window: Dict[str, object],
        ranked: Sequence[Dict[str, object]],
        gt_rows: Sequence[Dict[str, object]],
        current_gt_segment_index: int,
    ) -> bool:
        if plt is None:
            return False

        colors = ["#174EA6", "#D93025", "#F29900", "#9334E6", "#00ACC1"]
        plot_items = list(ranked[: max(1, int(args.plot_top_k))])
        current_gt = gt_rows[current_gt_segment_index - 1]
        frame_sample_count = max(0, int(args.plot_frame_count))
        frame_rows: List[Dict[str, object]] = []
        frame_mode = str(args.plot_frame_mode)
        wants_frame_rows = frame_sample_count > 0 and frame_mode != "none"
        if wants_frame_rows:
            gt_start = float(current_gt["start_time"])
            gt_end = float(current_gt["end_time"])
            gt_cells = sample_frame_cells(
                example_video_path,
                gt_start,
                gt_end,
                count=frame_sample_count,
            )
            frame_rows.append(
                {
                    "label": f"GT {gt_start:.2f}-{gt_end:.2f}s ({len(gt_cells)}/{frame_sample_count})",
                    "cells": gt_cells,
                    "color": "#188038",
                }
            )
            for idx, item in enumerate(plot_items):
                template = templates_by_catalog_index.get(int(item["template_catalog_index"]))
                if template is None:
                    continue
                iso_start, iso_end = template_effective_video_span(template)
                iso_path = Path(str(item["isolated_video_path"]))
                iso_cells = sample_frame_cells(
                    iso_path,
                    iso_start,
                    iso_end,
                    count=frame_sample_count,
                )
                label = (
                    f"#{int(item['rank'])} "
                    f"{short_plot_label(item['predicted_gloss'], limit=24)} "
                    f"{float(item['peak_score']):.2f} ({len(iso_cells)}/{frame_sample_count})"
                )
                frame_rows.append(
                    {
                        "label": label,
                        "cells": iso_cells,
                        "color": colors[idx % len(colors)],
                    }
                )

        embed_frame_rows = bool(frame_rows) and frame_mode in {"embedded", "both"}
        save_separate_frame_rows = (
            bool(frame_rows)
            and frame_plot_path is not None
            and frame_mode in {"separate", "both"}
        )

        if embed_frame_rows:
            frame_panel_height = max(1.6, 0.46 * float(len(frame_rows)) + 0.55)
            fig = plt.figure(figsize=(12.0, 6.2 + frame_panel_height))
            grid = fig.add_gridspec(
                3,
                1,
                height_ratios=[3.2, 1.7, frame_panel_height],
                hspace=0.18,
            )
            ax = fig.add_subplot(grid[0])
            region_ax = fig.add_subplot(grid[1], sharex=ax)
            frame_ax = fig.add_subplot(grid[2])
        else:
            fig = plt.figure(figsize=(12.0, 6.2))
            grid = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.7], hspace=0.12)
            ax = fig.add_subplot(grid[0])
            region_ax = fig.add_subplot(grid[1], sharex=ax)
            frame_ax = None

        step_sec = float(search_window["step_sec"])
        search_start = float(search_window["start_time"])
        search_end = float(search_window["end_time"])
        duration = float(search_window["sentence_duration"])

        for idx, item in enumerate(plot_items):
            template = templates_by_catalog_index.get(int(item["template_catalog_index"]))
            if template is None:
                continue
            sim_curve = sim_curve_for_template(search_window, template)
            if sim_curve.size == 0:
                continue
            curve_offset_steps = (
                float(template["effective_isolated_steps"]) / 2.0
                if args.ncc_peak_position == "center"
                else 0.0
            )
            x_seconds = search_start + (
                (
                    np.arange(sim_curve.size, dtype=np.float32) * max(1, int(args.stride))
                    + curve_offset_steps
                )
                * step_sec
            )
            color = colors[idx % len(colors)]
            label = (
                f"#{int(item['rank'])} {item['predicted_gloss']} "
                f"{float(item['peak_score']):.3f}"
            )
            if item.get("is_correct"):
                label += " hit"
            ax.plot(x_seconds, sim_curve, color=color, lw=1.5, alpha=0.9, label=label)
            peak_time = search_start + (
                (int(item["peak_index"]) * max(1, int(args.stride)) + curve_offset_steps)
                * step_sec
            )
            ax.scatter(
                [peak_time],
                [float(item["peak_score"])],
                color=color,
                s=24,
                zorder=5,
            )

        ax.axvspan(search_start, search_end, color="#9AA0A6", alpha=0.08, label="search window")
        for idx, gt in enumerate(gt_rows, start=1):
            gt_start = float(gt["start_time"])
            gt_end = float(gt["end_time"])
            is_current = idx == current_gt_segment_index
            ax.axvspan(
                gt_start,
                gt_end,
                color="#188038",
                alpha=0.26 if is_current else 0.13,
                label="current GT" if is_current else ("other GT" if idx == 1 else None),
            )

        for idx, item in enumerate(plot_items):
            color = colors[idx % len(colors)]
            ax.axvspan(
                float(item["start_time"]),
                float(item["end_time"]),
                color=color,
                alpha=0.12,
            )

        scope_label = "full video" if args.search_scope == "full_video" else "GT local"
        ax.set_title(
            f"{word_name} | source={source_video_id or example_video_number} | "
            f"{scope_label} | top1={int(ranked[0]['is_correct']) if ranked else 0}"
        )
        ax.set_ylabel("NCC score")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best", fontsize=7)
        ax.tick_params(axis="x", labelbottom=False)

        gt_y = 0.0
        region_ax.broken_barh(
            [(float(current_gt["start_time"]), float(current_gt["end_time"]) - float(current_gt["start_time"]))],
            (gt_y - 0.28, 0.56),
            facecolors="#188038",
            alpha=0.55,
        )
        region_rows: List[Tuple[float, str, str, float, float]] = []
        for idx, item in enumerate(plot_items):
            start = float(item["start_time"])
            end = float(item["end_time"])
            if end <= start:
                continue
            hit_suffix = " hit" if item.get("is_correct") else ""
            label = (
                f"#{int(item['rank'])} IoU={float(item['iou']):.2f} "
                f"{float(item['peak_score']):.2f}{hit_suffix}"
            )
            region_rows.append(
                (
                    float(len(region_rows) + 1),
                    label,
                    colors[idx % len(colors)],
                    start,
                    end,
                )
            )

        for y, label, color, start, end in region_rows:
            region_ax.broken_barh(
                [(start, end - start)],
                (y - 0.28, 0.56),
                facecolors=color,
                alpha=0.62,
            )
            region_ax.text(
                min(end, duration),
                y,
                f" {start:.2f}-{end:.2f}s",
                va="center",
                ha="left",
                fontsize=7,
                color="#202124",
            )

        region_ax.set_yticks([gt_y] + [row[0] for row in region_rows])
        region_ax.set_yticklabels(["GT"] + [row[1] for row in region_rows], fontsize=8)
        region_ax.set_xlabel("time in example video (seconds)")
        region_ax.set_ylabel("top-k")
        region_ax.grid(True, axis="x", alpha=0.2)
        for spine in ("top", "right"):
            region_ax.spines[spine].set_visible(False)

        if frame_ax is not None and frame_sample_count > 0:
            draw_frame_rows_axis(
                frame_ax,
                frame_rows,
                frame_sample_count=frame_sample_count,
                title="GT and top-k isolated frame samples",
            )

        if args.search_scope == "local_gt":
            pad = max(0.1, float(args.gt_buffer_seconds) * 0.5)
            x_start = max(0.0, search_start - pad)
            x_end = min(duration, search_end + pad)
        else:
            x_start = 0.0
            x_end = duration
        ax.set_xlim(x_start, x_end)
        region_ax.set_xlim(x_start, x_end)
        region_ax.set_ylim(-0.75, max(1.0, float(len(region_rows)) + 0.75))

        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        if save_separate_frame_rows and frame_plot_path is not None:
            save_frame_rows_plot(
                frame_plot_path,
                frame_rows,
                frame_sample_count=frame_sample_count,
                title=(
                    f"{word_name} | source={source_video_id or example_video_number} | "
                    "GT and top-k isolated frame samples"
                ),
            )
            return True
        return False

    try:
        templates, skipped_template_rows = load_isolated_templates()
        if not templates:
            raise RuntimeError("All isolated templates failed to load.")
        log_status(
            f"using {len(templates)} encoded isolated templates; "
            f"skipped {len(skipped_template_rows)}"
        )
        templates_by_catalog_index = {
            int(template["catalog_index"]): template
            for template in templates
        }

        evaluation_rows: List[Dict[str, object]] = []
        topk_rows: List[Dict[str, object]] = []
        full_ranking_rows: List[Dict[str, object]] = []
        skipped_occurrence_rows: List[Dict[str, object]] = []
        topk_hits: Dict[int, List[int]] = {top_k: [] for top_k in TOP_K_EVAL_VALUES}
        correct_ranks: List[int] = []
        reciprocal_ranks: List[float] = []
        target_candidate_counts: List[int] = []
        top1_ious: List[float] = []
        best_correct_ious: List[float] = []

        sorted_label_items = sorted(labels_by_example.items())
        if eval_word_keys:
            unfiltered_group_count = len(sorted_label_items)
            sorted_label_items = [
                item
                for item in sorted_label_items
                if evaluation_group_filter_keys(item[0][0], item[1]).intersection(eval_word_keys)
            ]
            log_status(
                "evaluation word filter kept "
                f"{len(sorted_label_items)}/{unfiltered_group_count} annotated groups "
                f"for {len(eval_word_terms)} requested terms"
            )
        if args.max_examples is not None:
            sorted_label_items = sorted_label_items[: args.max_examples]

        occurrence_index = 0
        for group_index, ((annotation_word_key, source_video_id, example_video_number), gt_rows) in enumerate(sorted_label_items, start=1):
            if group_index == 1 or group_index % max(1, args.log_every) == 0 or group_index == len(sorted_label_items):
                log_status(
                    f"evaluating annotated group {group_index}/{len(sorted_label_items)} "
                    f"{annotation_word_key} source={source_video_id or example_video_number}"
                )

            group = resolve_group_entry(annotation_word_key)
            if group is None:
                skipped_occurrence_rows.append(
                    {
                        "annotation_word_key": annotation_word_key,
                        "source_video_id": source_video_id,
                        "example_video_number": example_video_number,
                        "gt_segment_index": "",
                        "reason": "metadata_missing",
                    }
                )
                continue

            target_glosses = entry_glosses_in_vocab(group, lem_to_id)
            if not target_glosses:
                skipped_occurrence_rows.append(
                    {
                        "annotation_word_key": annotation_word_key,
                        "source_video_id": source_video_id,
                        "example_video_number": example_video_number,
                        "gt_segment_index": "",
                        "reason": "target_gloss_not_in_vocab",
                    }
                )
                continue

            target_set = set(target_glosses)
            target_template_count = sum(
                1 for template in templates
                if target_set.intersection(set(template["glosses"]))
            )
            if target_template_count <= 0:
                skipped_occurrence_rows.append(
                    {
                        "annotation_word_key": annotation_word_key,
                        "source_video_id": source_video_id,
                        "example_video_number": example_video_number,
                        "gt_segment_index": "",
                        "reason": "no_target_isolated_template",
                    }
                )
                continue

            local_example = find_example_for_annotation(group, source_video_id, example_video_number)
            example_video_path = None
            if local_example is not None and local_example.get("path"):
                example_video_path = Path(local_example["path"])
            if example_video_path is None or not example_video_path.exists():
                skipped_occurrence_rows.append(
                    {
                        "annotation_word_key": annotation_word_key,
                        "source_video_id": source_video_id,
                        "example_video_number": example_video_number,
                        "gt_segment_index": "",
                        "reason": "example_video_missing",
                    }
                )
                continue

            sentence_item = load_hidden(
                example_video_path,
                lmdb_stride=1,
                max_len=None,
                cache=sentence_cache,
            )
            full_video_search_window = None
            full_video_ranked_base: Optional[List[Dict[str, object]]] = None
            if args.search_scope == "full_video":
                full_video_search_window = full_video_hidden_window(
                    sentence_item,
                    video_path=example_video_path,
                    video_id=source_video_id,
                )
                if full_video_search_window is None:
                    skipped_occurrence_rows.append(
                        {
                            "annotation_word_key": annotation_word_key,
                            "source_video_id": source_video_id,
                            "example_video_number": example_video_number,
                            "gt_segment_index": "",
                            "reason": "empty_full_video_window",
                        }
                    )
                    continue
                full_video_ranked_base = rank_templates_for_full_video(
                    full_video_search_window,
                    templates,
                )
                if not full_video_ranked_base:
                    skipped_occurrence_rows.append(
                        {
                            "annotation_word_key": annotation_word_key,
                            "source_video_id": source_video_id,
                            "example_video_number": example_video_number,
                            "gt_segment_index": "",
                            "reason": "no_full_video_ranked_candidates",
                        }
                    )
                    continue

            for gt_segment_index, gt in enumerate(gt_rows, start=1):
                occurrence_index += 1
                gt_start = float(gt["start_time"])
                gt_end = float(gt["end_time"])
                if args.search_scope == "full_video":
                    search_window = full_video_search_window
                    ranked = annotate_ranked_for_gt(
                        full_video_ranked_base or [],
                        target_glosses,
                        gt_start,
                        gt_end,
                    )
                else:
                    search_window = local_hidden_window(
                        sentence_item,
                        gt_start,
                        gt_end,
                        video_path=example_video_path,
                        video_id=source_video_id,
                    )
                    if search_window is None:
                        skipped_occurrence_rows.append(
                            {
                                "annotation_word_key": annotation_word_key,
                                "source_video_id": source_video_id,
                                "example_video_number": example_video_number,
                                "gt_segment_index": gt_segment_index,
                                "reason": "empty_local_window",
                            }
                        )
                        continue

                    ranked = rank_templates_for_gt(
                        search_window,
                        templates,
                        target_glosses,
                        gt_start,
                        gt_end,
                    )
                if not ranked:
                    skipped_occurrence_rows.append(
                        {
                            "annotation_word_key": annotation_word_key,
                            "source_video_id": source_video_id,
                            "example_video_number": example_video_number,
                            "gt_segment_index": gt_segment_index,
                            "reason": "no_ranked_candidates",
                        }
                    )
                    continue

                first_correct = next((item for item in ranked if item["is_correct"]), None)
                correct_rank = int(first_correct["rank"]) if first_correct is not None else None
                if correct_rank is not None:
                    correct_ranks.append(correct_rank)
                    reciprocal_ranks.append(1.0 / float(correct_rank))
                    best_correct_ious.append(float(first_correct["iou"]))
                else:
                    reciprocal_ranks.append(0.0)
                    best_correct_ious.append(0.0)

                top1 = ranked[0]
                top1_ious.append(float(top1["iou"]))
                target_candidate_counts.append(target_template_count)

                topk_fields: Dict[str, object] = {}
                for top_k in TOP_K_EVAL_VALUES:
                    hit = any(item["is_correct"] for item in ranked[:top_k])
                    topk_hits[top_k].append(1 if hit else 0)
                    topk_fields[f"top{top_k}_hit"] = int(hit)
                    topk_fields[f"top{top_k}_candidate_count"] = min(top_k, len(ranked))

                for item in ranked[: max(TOP_K_EVAL_VALUES)]:
                    topk_rows.append(
                        {
                            "occurrence_index": occurrence_index,
                            "search_scope": args.search_scope,
                            "rank": item["rank"],
                            "is_target_gloss": int(item["is_target_gloss"]),
                            "is_correct": int(item["is_correct"]),
                            "matched_target_glosses": item["matched_target_glosses"],
                            "prediction_word_name": item["word_name"],
                            "predicted_gloss": item["predicted_gloss"],
                            "isolated_video_id": item["isolated_video_id"],
                            "isolated_video_path": item["isolated_video_path"],
                            "template_movement_window_source": item.get("template_movement_window_source", ""),
                            "template_movement_window_start_time": f"{float(item.get('template_movement_window_start_time') or 0.0):.6f}",
                            "template_movement_window_end_time": f"{float(item.get('template_movement_window_end_time') or 0.0):.6f}",
                            "peak_score": f"{float(item['peak_score']):.6f}",
                            "predicted_time_interval": item["time_interval"],
                            "iou": f"{float(item['iou']):.6f}",
                            "ncc_peak_position": item.get("ncc_peak_position", args.ncc_peak_position),
                            "nms_merged_count": item.get("nms_merged_count", 1),
                        }
                    )

                if args.write_full_rankings:
                    for item in ranked:
                        full_ranking_rows.append(
                            {
                                "occurrence_index": occurrence_index,
                                "search_scope": args.search_scope,
                                "rank": item["rank"],
                                "is_target_gloss": int(item["is_target_gloss"]),
                                "is_correct": int(item["is_correct"]),
                                "matched_target_glosses": item["matched_target_glosses"],
                                "prediction_word_name": item["word_name"],
                                "prediction_word_dir": item["word_dir"],
                                "predicted_gloss": item["predicted_gloss"],
                                "isolated_video_id": item["isolated_video_id"],
                                "isolated_video_path": item["isolated_video_path"],
                                "template_movement_window_source": item.get("template_movement_window_source", ""),
                                "template_movement_window_start_time": f"{float(item.get('template_movement_window_start_time') or 0.0):.6f}",
                                "template_movement_window_end_time": f"{float(item.get('template_movement_window_end_time') or 0.0):.6f}",
                                "peak_score": f"{float(item['peak_score']):.6f}",
                                "predicted_time_interval": item["time_interval"],
                                "iou": f"{float(item['iou']):.6f}",
                                "ncc_peak_position": item.get("ncc_peak_position", args.ncc_peak_position),
                                "nms_merged_count": item.get("nms_merged_count", 1),
                            }
                        )

                plot_path = ""
                frame_plot_path = ""
                if plot_dir is not None and (
                    args.max_plots is None or plots_written < int(args.max_plots)
                ):
                    safe_word = ascii_slug(str(gt_rows[0]["word_name"])) or "word"
                    safe_source = ascii_slug(source_video_id) or f"example_{example_video_number}"
                    plot_file = (
                        f"{safe_word}_ex{example_video_number}_{safe_source}_"
                        f"seg{gt_segment_index}_{args.search_scope}.png"
                    )
                    full_plot_path = plot_dir / plot_file
                    full_frame_plot_path = full_plot_path.with_name(
                        f"{full_plot_path.stem}_frames{full_plot_path.suffix}"
                    )
                    frame_plot_written = save_detection_plot(
                        full_plot_path,
                        word_name=str(gt_rows[0]["word_name"]),
                        source_video_id=source_video_id,
                        example_video_number=example_video_number,
                        example_video_path=example_video_path,
                        frame_plot_path=full_frame_plot_path,
                        search_window=search_window,
                        ranked=ranked,
                        gt_rows=gt_rows,
                        current_gt_segment_index=gt_segment_index,
                    )
                    plot_path = str(full_plot_path)
                    if frame_plot_written:
                        frame_plot_path = str(full_frame_plot_path)
                    plots_written += 1

                evaluation_rows.append(
                    {
                        "occurrence_index": occurrence_index,
                        "search_scope": args.search_scope,
                        "word_name": gt_rows[0]["word_name"],
                        "annotation_word_key": annotation_word_key,
                        "example_video_number": example_video_number,
                        "source_video_id": source_video_id,
                        "example_video_path": str(example_video_path),
                        "gt_segment_index": gt_segment_index,
                        "gt_time_interval": f"[{gt_start:.6f}, {gt_end:.6f}]",
                        "gt_start_time": f"{gt_start:.6f}",
                        "gt_end_time": f"{gt_end:.6f}",
                        "search_window_interval": (
                            f"[{float(search_window['start_time']):.6f}, "
                            f"{float(search_window['end_time']):.6f}]"
                        ),
                        "search_window_steps": int(search_window["end_step"]) - int(search_window["start_step"]),
                        "example_movement_window_source": search_window.get("movement_window_source", ""),
                        "example_movement_window_start_time": search_window.get("movement_window_start_time", ""),
                        "example_movement_window_end_time": search_window.get("movement_window_end_time", ""),
                        "local_window_interval": (
                            f"[{float(search_window['start_time']):.6f}, "
                            f"{float(search_window['end_time']):.6f}]"
                        ),
                        "local_window_steps": int(search_window["end_step"]) - int(search_window["start_step"]),
                        "gt_buffer_seconds": f"{float(args.gt_buffer_seconds):.6f}",
                        "hit_iou_threshold": f"{float(args.hit_iou_threshold):.6f}",
                        "target_glosses": " ".join(target_glosses),
                        "target_isolated_candidate_count": target_template_count,
                        "all_candidate_count": len(ranked),
                        "correct_rank": correct_rank if correct_rank is not None else "",
                        "reciprocal_rank": (
                            f"{(1.0 / float(correct_rank)):.6f}"
                            if correct_rank is not None
                            else "0.000000"
                        ),
                        "best_correct_peak_score": (
                            f"{float(first_correct['peak_score']):.6f}"
                            if first_correct is not None
                            else ""
                        ),
                        "best_correct_iou": (
                            f"{float(first_correct['iou']):.6f}"
                            if first_correct is not None
                            else "0.000000"
                        ),
                        "top1_word_name": top1["word_name"],
                        "top1_predicted_gloss": top1["predicted_gloss"],
                        "top1_is_target_gloss": int(top1["is_target_gloss"]),
                        "top1_is_correct": int(top1["is_correct"]),
                        "top1_isolated_video_id": top1["isolated_video_id"],
                        "top1_isolated_video_path": top1["isolated_video_path"],
                        "top1_template_movement_window_source": top1.get("template_movement_window_source", ""),
                        "top1_template_movement_window_start_time": f"{float(top1.get('template_movement_window_start_time') or 0.0):.6f}",
                        "top1_template_movement_window_end_time": f"{float(top1.get('template_movement_window_end_time') or 0.0):.6f}",
                        "top1_peak_score": f"{float(top1['peak_score']):.6f}",
                        "top1_predicted_time_interval": top1["time_interval"],
                        "top1_iou": f"{float(top1['iou']):.6f}",
                        "top3_candidates": top_candidate_summary(ranked, 3),
                        "top5_candidates": top_candidate_summary(ranked, 5),
                        **topk_fields,
                        "ncc_stride": args.stride,
                        "ncc_peak_position": args.ncc_peak_position,
                        "ncc_weighting": args.ncc_weighting,
                        "ncc_ignore_frames": args.ncc_ignore_frames,
                        "nms_scope": top1["nms_scope"],
                        "ncc_plot_path": plot_path,
                        "ncc_frame_plot_path": frame_plot_path,
                    }
                )

        output_prefix = (
            "full_video_all_vocab_ncc"
            if args.search_scope == "full_video"
            else "local_all_vocab_ncc"
        )
        eval_csv = output_dir / f"{output_prefix}_eval.csv"
        topk_csv = output_dir / f"{output_prefix}_topk.csv"
        full_rankings_csv = output_dir / f"{output_prefix}_full_rankings.csv"
        skipped_csv = output_dir / f"{output_prefix}_skipped.csv"
        summary_json = output_dir / f"{output_prefix}_summary.json"

        eval_fields = [
            "occurrence_index",
            "search_scope",
            "word_name",
            "annotation_word_key",
            "example_video_number",
            "source_video_id",
            "example_video_path",
            "gt_segment_index",
            "gt_time_interval",
            "gt_start_time",
            "gt_end_time",
            "search_window_interval",
            "search_window_steps",
            "example_movement_window_source",
            "example_movement_window_start_time",
            "example_movement_window_end_time",
            "local_window_interval",
            "local_window_steps",
            "gt_buffer_seconds",
            "hit_iou_threshold",
            "target_glosses",
            "target_isolated_candidate_count",
            "all_candidate_count",
            "correct_rank",
            "reciprocal_rank",
            "best_correct_peak_score",
            "best_correct_iou",
            "top1_word_name",
            "top1_predicted_gloss",
            "top1_is_target_gloss",
            "top1_is_correct",
            "top1_isolated_video_id",
            "top1_isolated_video_path",
            "top1_template_movement_window_source",
            "top1_template_movement_window_start_time",
            "top1_template_movement_window_end_time",
            "top1_peak_score",
            "top1_predicted_time_interval",
            "top1_iou",
            "top3_candidates",
            "top5_candidates",
            "top1_hit",
            "top1_candidate_count",
            "top3_hit",
            "top3_candidate_count",
            "top5_hit",
            "top5_candidate_count",
            "ncc_stride",
            "ncc_peak_position",
            "ncc_weighting",
            "ncc_ignore_frames",
            "nms_scope",
            "ncc_plot_path",
            "ncc_frame_plot_path",
        ]
        write_csv_rows(eval_csv, evaluation_rows, eval_fields)
        write_csv_rows(
            topk_csv,
            topk_rows,
            [
                "occurrence_index",
                "search_scope",
                "rank",
                "is_target_gloss",
                "is_correct",
                "matched_target_glosses",
                "prediction_word_name",
                "predicted_gloss",
                "isolated_video_id",
                "isolated_video_path",
                "template_movement_window_source",
                "template_movement_window_start_time",
                "template_movement_window_end_time",
                "peak_score",
                "predicted_time_interval",
                "iou",
                "ncc_peak_position",
                "nms_merged_count",
            ],
        )
        if args.write_full_rankings:
            write_csv_rows(
                full_rankings_csv,
                full_ranking_rows,
                [
                    "occurrence_index",
                    "search_scope",
                    "rank",
                    "is_target_gloss",
                    "is_correct",
                    "matched_target_glosses",
                    "prediction_word_name",
                    "prediction_word_dir",
                    "predicted_gloss",
                    "isolated_video_id",
                    "isolated_video_path",
                    "template_movement_window_source",
                    "template_movement_window_start_time",
                    "template_movement_window_end_time",
                    "peak_score",
                    "predicted_time_interval",
                    "iou",
                    "ncc_peak_position",
                    "nms_merged_count",
                ],
            )
        all_skipped_rows = skipped_template_rows + skipped_occurrence_rows
        if all_skipped_rows:
            skipped_fields = [
                "word_name",
                "word_dir",
                "glosses",
                "isolated_video_id",
                "isolated_video_path",
                "annotation_word_key",
                "source_video_id",
                "example_video_number",
                "gt_segment_index",
                "reason",
            ]
            normalized_skipped = []
            for row in all_skipped_rows:
                normalized_skipped.append({field: row.get(field, "") for field in skipped_fields})
            write_csv_rows(skipped_csv, normalized_skipped, skipped_fields)

        topk_summary: Dict[str, object] = {
            "top_k_eval_values": list(TOP_K_EVAL_VALUES),
        }
        for top_k in TOP_K_EVAL_VALUES:
            values = topk_hits[top_k]
            topk_summary[f"top{top_k}_correct"] = int(sum(values))
            topk_summary[f"top{top_k}_accuracy"] = (
                float(sum(values)) / float(len(values)) if values else 0.0
            )

        summary = {
            "annotation_dir": str(annotation_dir),
            "word_manifest": str(word_manifest_path) if word_manifest_path is not None else "",
            "example_manifest": str(example_manifest_path) if example_manifest_path is not None else "",
            "tid_raw_roots": [str(root) for root in tid_raw_roots],
            "vocab_pkl": str(vocab_resolved),
            "num_vocab_classes": num_classes,
            "checkpoint": checkpoint_label,
            "checkpoint_mode": checkpoint_mode,
            "ncc_embedding_source": ncc_embedding_source,
            "search_scope": args.search_scope,
            "movement_windows_csv": str(movement_windows_path) if movement_windows_path is not None else "",
            "movement_window_apply": args.movement_window_apply,
            "movement_window_buffer_seconds": args.movement_window_buffer_seconds,
            "movement_window_min_duration_seconds": args.movement_window_min_duration_seconds,
            "movement_window_row_count": len(movement_window_rows),
            **movement_stats,
            "gt_buffer_seconds": args.gt_buffer_seconds,
            "hit_iou_threshold": args.hit_iou_threshold,
            "nms_scope": (
                "per_isolated_video_full_video_temporal_nms"
                if args.search_scope == "full_video"
                else "one_peak_per_isolated_video_per_gt_window"
            ),
            "nms_mode": args.nms_mode,
            "nms_iou_threshold": args.nms_iou_threshold,
            "nms_center_distance_seconds": args.nms_center_distance_seconds,
            "full_video_max_peaks_per_isolated": args.full_video_max_peaks_per_isolated,
            "full_video_pre_nms_peak_limit": args.full_video_pre_nms_peak_limit,
            "full_video_min_peak_score": args.full_video_min_peak_score,
            "full_video_relative_peak_threshold": args.full_video_relative_peak_threshold,
            "isolated_catalog_size": len(isolated_catalog),
            "encoded_isolated_template_count": len(templates),
            "skipped_template_count": len(skipped_template_rows),
            "loaded_annotated_group_count": len(labels_by_example),
            "annotated_group_count": len(sorted_label_items),
            "evaluated_occurrence_count": len(evaluation_rows),
            "skipped_occurrence_count": len(skipped_occurrence_rows),
            "mean_correct_rank": (
                float(statistics.fmean(correct_ranks)) if correct_ranks else 0.0
            ),
            "median_correct_rank": (
                float(statistics.median(correct_ranks)) if correct_ranks else 0.0
            ),
            "mean_reciprocal_rank": (
                float(statistics.fmean(reciprocal_ranks)) if reciprocal_ranks else 0.0
            ),
            "mean_target_isolated_candidate_count": (
                float(statistics.fmean(target_candidate_counts))
                if target_candidate_counts
                else 0.0
            ),
            "mean_top1_iou": float(statistics.fmean(top1_ious)) if top1_ious else 0.0,
            "mean_best_correct_iou": (
                float(statistics.fmean(best_correct_ious)) if best_correct_ious else 0.0
            ),
            "eval_csv": str(eval_csv),
            "topk_csv": str(topk_csv),
            "full_rankings_csv": str(full_rankings_csv) if args.write_full_rankings else "",
            "skipped_csv": str(skipped_csv) if all_skipped_rows else "",
            "tmp_dir": str(temp_base),
            "plot_dir": str(plot_dir) if plot_dir is not None else "",
            "plots_written": plots_written,
            "plot_top_k": args.plot_top_k,
            "plot_frame_count": args.plot_frame_count,
            "plot_frame_mode": args.plot_frame_mode,
            "eval_word_filter_terms": eval_word_terms,
            "eval_word_filter_term_count": len(eval_word_terms),
            "eval_candidate_scope": args.eval_candidate_scope,
            "effective_candidate_filter_terms": effective_candidate_terms,
            "effective_candidate_filter_term_count": len(effective_candidate_terms),
            "candidate_word_filter_terms": candidate_word_terms,
            "candidate_word_filter_term_count": len(candidate_word_terms),
            "iso_stride_frames": args.iso_stride_frames,
            "iso_max_len": args.iso_max_len,
            "iso_trim_seconds": args.iso_trim_seconds,
            "iso_trim_start_seconds": (
                args.iso_trim_start_seconds
                if args.iso_trim_start_seconds is not None
                else args.iso_trim_seconds
            ),
            "iso_trim_end_seconds": (
                args.iso_trim_end_seconds
                if args.iso_trim_end_seconds is not None
                else args.iso_trim_seconds
            ),
            "stride": args.stride,
            "ncc_ignore_frames": args.ncc_ignore_frames,
            "ncc_peak_position": args.ncc_peak_position,
            "ncc_weighting": args.ncc_weighting,
        }
        summary.update(topk_summary)

        with summary_json.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        log_status(f"wrote summary to {summary_json}")
        return summary
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all-vocabulary NCC retrieval for TID overlap annotations."
        )
    )
    parser.add_argument(
        "--annotation_dir",
        default=DEFAULT_ANNOTATION_DIR,
        help="Directory containing VIA *_export.csv files.",
    )
    parser.add_argument(
        "--word_manifest",
        default=DEFAULT_WORD_MANIFEST,
        help="Word manifest that maps TID words to pseudo-gloss vocabulary items.",
    )
    parser.add_argument(
        "--example_manifest",
        default=DEFAULT_EXAMPLE_MANIFEST,
        help="Fallback sampled example manifest for legacy metadata.",
    )
    parser.add_argument(
        "--tid_raw_roots",
        default=DEFAULT_TID_RAW_ROOTS,
        help="Comma-separated roots containing TID word directories with word_info.json.",
    )
    parser.add_argument(
        "--config",
        default="configs/tslnews/tslnews_stage1_configs/TSL_example_s1_dyn_config.py",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Stage-1 checkpoint to load. If omitted, the shared loader keeps its "
            "legacy behavior and selects the best checkpoint from cfg.save_dir."
        ),
    )
    parser.add_argument(
        "--no_checkpoint",
        action="store_true",
        help="Instantiate the model from config and use NCC without loading a Sign2GPT checkpoint.",
    )
    parser.add_argument("--vocab_pkl", default="data/TSL-News/6538_gloss/processed_words_map.tsl_pkl")
    parser.add_argument("--output_dir", default="outputs/spotting/local_retrieval")
    parser.add_argument(
        "--tmp_dir",
        default=None,
        help="Directory for temporary per-video LMDBs. Defaults to the system temp directory.",
    )
    parser.add_argument(
        "--movement_windows_csv",
        default=None,
        help=(
            "CSV produced by scripts/tid/build_movement_windows.py or "
            "scripts/tid/build_visual_motion_windows.py. When enabled, "
            "example and/or isolated hidden sequences are trimmed to "
            "movement_start_time/movement_end_time."
        ),
    )
    parser.add_argument(
        "--movement_window_apply",
        choices=["none", "examples", "isolated", "both"],
        default="none",
        help="Which sides of NCC should use movement-window CSV bounds.",
    )
    parser.add_argument(
        "--movement_window_buffer_seconds",
        type=float,
        default=0.0,
        help="Expand movement-window CSV bounds by this many seconds on each side.",
    )
    parser.add_argument(
        "--movement_window_min_duration_seconds",
        type=float,
        default=0.05,
        help="Ignore movement windows shorter than this duration.",
    )
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto")
    parser.add_argument(
        "--ncc_embedding_source",
        choices=["auto", "encoder", "spatial"],
        default="auto",
        help=(
            "Feature sequence used by NCC. auto uses trained encoder hidden states "
            "for checkpointed runs and raw spatial backbone features for --no_checkpoint."
        ),
    )
    parser.add_argument(
        "--gt_buffer_seconds",
        type=float,
        default=0.25,
        help="Seconds added before/after each GT interval for the local NCC search window.",
    )
    parser.add_argument(
        "--search_scope",
        choices=["local_gt", "full_video"],
        default="local_gt",
        help=(
            "local_gt searches only GT +/- buffer and keeps one peak per isolated video. "
            "full_video searches the whole example and applies temporal NMS per isolated video."
        ),
    )
    parser.add_argument(
        "--hit_iou_threshold",
        type=float,
        default=0.0,
        help=(
            "Minimum temporal IoU for a target-gloss detection to count as correct. "
            "0 preserves target-only scoring in local_gt and means any positive "
            "overlap in full_video."
        ),
    )
    parser.add_argument("--stride", type=int, default=1, help="Sliding stride in feature steps.")
    parser.add_argument(
        "--iso_stride_frames",
        type=int,
        default=3,
        help="Frame subsample stride for isolated video LMDB building.",
    )
    parser.add_argument(
        "--iso_max_len",
        type=int,
        default=12,
        help="Max frames after subsample for isolated videos.",
    )
    parser.add_argument(
        "--iso_trim_seconds",
        type=float,
        default=0.2,
        help="Ignore this many seconds at both start and end of each isolated video before NCC.",
    )
    parser.add_argument(
        "--iso_trim_start_seconds",
        type=float,
        default=None,
        help="Override --iso_trim_seconds for only the start of each isolated video.",
    )
    parser.add_argument(
        "--iso_trim_end_seconds",
        type=float,
        default=None,
        help="Override --iso_trim_seconds for only the end of each isolated video.",
    )
    parser.add_argument(
        "--ncc_ignore_frames",
        type=int,
        default=0,
        help="Ignore this many frames from both ends of the isolated hidden sequence inside sliding_ncc.",
    )
    parser.add_argument(
        "--ncc_peak_position",
        choices=["start", "center"],
        default="start",
        help=(
            "How to convert an NCC peak index into a prediction window. "
            "start treats the peak as the sliding-window start; center treats it "
            "as the center of the predicted window."
        ),
    )
    parser.add_argument(
        "--ncc_weighting",
        choices=["none", "triangle"],
        default="none",
        help="Temporal weighting used in NCC over isolated video frames.",
    )
    parser.add_argument(
        "--nms_mode",
        choices=["merge", "suppress"],
        default="merge",
        help=(
            "For --search_scope full_video, merge nearby same-isolated-video windows "
            "or keep the highest scoring one and suppress the rest."
        ),
    )
    parser.add_argument(
        "--nms_iou_threshold",
        type=float,
        default=0.5,
        help="For --search_scope full_video, temporal IoU threshold for per-isolated-video NMS.",
    )
    parser.add_argument(
        "--nms_center_distance_seconds",
        type=float,
        default=0.0,
        help=(
            "For --search_scope full_video, also merge/suppress same-isolated-video "
            "windows whose centers are this close. 0 disables this rule."
        ),
    )
    parser.add_argument(
        "--full_video_max_peaks_per_isolated",
        type=int,
        default=5,
        help="For --search_scope full_video, max post-NMS detections kept per isolated video.",
    )
    parser.add_argument(
        "--full_video_pre_nms_peak_limit",
        type=int,
        default=50,
        help="For --search_scope full_video, max local maxima considered before NMS per isolated video.",
    )
    parser.add_argument(
        "--full_video_min_peak_score",
        type=float,
        default=-1.0,
        help="For --search_scope full_video, minimum NCC peak score considered before NMS.",
    )
    parser.add_argument(
        "--full_video_relative_peak_threshold",
        type=float,
        default=0.0,
        help=(
            "For --search_scope full_video, optional fraction of each isolated video's "
            "best NCC score used as a pre-NMS peak threshold."
        ),
    )
    parser.add_argument(
        "--plot_dir",
        default="all_vocab_ncc_plots",
        help="Directory, relative to --output_dir, for per-occurrence NCC plots.",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Disable per-occurrence NCC plots.",
    )
    parser.add_argument(
        "--plot_top_k",
        type=int,
        default=5,
        help="Number of ranked candidates to draw in each plot.",
    )
    parser.add_argument(
        "--plot_frame_count",
        type=int,
        default=5,
        help=(
            "Number of evenly sampled frames to show for the current GT span "
            "and each plotted isolated candidate. Set 0 to hide frame strips."
        ),
    )
    parser.add_argument(
        "--plot_frame_mode",
        choices=["separate", "embedded", "both", "none"],
        default="separate",
        help=(
            "How to write frame-strip plots: separate saves *_frames.png beside "
            "the NCC plot, embedded puts them under the NCC/span plot, both does both."
        ),
    )
    parser.add_argument(
        "--max_plots",
        type=int,
        default=None,
        help="Optional cap on the number of plots written.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Debug limit on annotated example groups.",
    )
    parser.add_argument(
        "--max_isolated",
        type=int,
        default=None,
        help="Debug limit on isolated videos loaded from the all-vocab catalog.",
    )
    parser.add_argument(
        "--eval_words",
        nargs="*",
        default=[],
        help=(
            "Optional comma-separated and/or space-separated GT word terms. "
            "Only annotated examples for these words are evaluated; candidate "
            "isolated videos default to the same words unless "
            "--eval_candidate_scope full_vocab or --candidate_words is set."
        ),
    )
    parser.add_argument(
        "--eval_word_file",
        default=None,
        help=(
            "Optional text file with GT word terms to evaluate, one per line. "
            "Lines may include comments after '#'."
        ),
    )
    parser.add_argument(
        "--eval_candidate_scope",
        choices=["eval_words", "full_vocab"],
        default="eval_words",
        help=(
            "When --eval_words/--eval_word_file is set and --candidate_words is not, "
            "choose whether isolated candidates are limited to those eval words "
            "or loaded from the full vocabulary."
        ),
    )
    parser.add_argument(
        "--candidate_words",
        nargs="*",
        default=[],
        help=(
            "Optional comma-separated and/or space-separated candidate word/gloss terms. "
            "Only isolated videos matching these terms are encoded and ranked."
        ),
    )
    parser.add_argument(
        "--candidate_word_file",
        default=None,
        help=(
            "Optional text file with candidate word/gloss terms, one per line. "
            "Lines may include comments after '#'."
        ),
    )
    parser.add_argument(
        "--write_full_rankings",
        action="store_true",
        help="Write every candidate rank for every occurrence. This can be very large.",
    )
    parser.add_argument(
        "--fail_on_template_error",
        action="store_true",
        help="Raise immediately if an isolated template cannot be encoded.",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=100,
        help="Progress logging interval for template encoding and example evaluation.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    script_root = Path(__file__).resolve().parents[1]
    explicit_word_manifest = any(
        arg == "--word_manifest" or arg.startswith("--word_manifest=")
        for arg in sys.argv[1:]
    )
    annotation_dir = resolve_existing_path(args.annotation_dir, script_root)
    word_manifest_path = resolve_existing_path(args.word_manifest, script_root)
    if annotation_dir is None:
        parser.error(f"Annotation directory not found: {args.annotation_dir}")
    if explicit_word_manifest and word_manifest_path is None:
        parser.error(f"Word manifest not found: {args.word_manifest}")

    summary = evaluate_local_all_vocab_ncc(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
