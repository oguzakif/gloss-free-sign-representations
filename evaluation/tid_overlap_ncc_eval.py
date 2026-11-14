#!/usr/bin/env python3
# Adapted for sign representation training and evaluation.
"""
Convert VIA temporal annotations for TID overlap examples into flat label CSVs and
optionally evaluate NCC-based localization against those labels with IoU.

The VIA exports under data/TID/overlap_tslnews/annotation/ contain URLs such as
.../144-05_cr_0.2.mp4. Updated TID word_info.json files store local continuous
videos under example_videos/144-05_cr_0.5.mp4 and isolated word videos under
isolated_videos/. The evaluator matches by the stable video id before the final
quality suffix, e.g. 144-05_cr.

If a word has not yet migrated to the new metadata layout, the script falls back
to meaning order and legacy word_representation_*.mp4 files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import re
import shutil
import statistics
import sys
import tempfile
import unicodedata
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_ANNOTATION_DIR = "data/TID/overlap_tslnews/annotation"
DEFAULT_LABEL_DIR = "data/TID/overlap_tslnews/labels"
DEFAULT_WORD_MANIFEST = "data/TID/overlap_tslnews/overlap_words.csv"
DEFAULT_EXAMPLE_MANIFEST = (
    "data/TID/overlap_tslnews/samples/stratified-log_n300_seed42/"
    "sampled_example_videos.csv"
)
DEFAULT_TID_RAW_ROOTS = (
    "data/TID/raw"
)
TOP_K_EVAL_VALUES = (1, 3, 5)
WER_SEGMENT_RELATIVE_THRESHOLD = 0.2
WER_SEGMENT_ABSOLUTE_THRESHOLD = 0.05
WER_SEGMENT_SMOOTH_WINDOW = 3


def log_status(message: str) -> None:
    print(f"[tid_overlap_ncc_eval] {message}", file=sys.stderr, flush=True)


def normalize_unicode_text(text: str) -> str:
    return unicodedata.normalize("NFC", str(text))


TURKISH_ASCII_MAP = str.maketrans(
    {
        "c": "c",
        "g": "g",
        "i": "i",
        "o": "o",
        "s": "s",
        "u": "u",
        "C": "C",
        "G": "G",
        "I": "I",
        "O": "O",
        "S": "S",
        "U": "U",
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
        "Ç": "C",
        "Ğ": "G",
        "İ": "I",
        "Ö": "O",
        "Ş": "S",
        "Ü": "U",
    }
)
TURKISH_TOKEN_RE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü']+")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def casefold_text(text: str) -> str:
    return normalize_unicode_text(text).strip().casefold()


def turkish_lower(text: str) -> str:
    return normalize_unicode_text(text).replace("I", "ı").replace("İ", "i").lower()


def tokenize_turkish(text: str) -> List[str]:
    return [tok for tok in TURKISH_TOKEN_RE.findall(turkish_lower(text or "")) if tok]


def slugify_text(text: str, ascii_only: bool = False) -> str:
    value = casefold_text(text)
    if ascii_only:
        value = value.translate(TURKISH_ASCII_MAP)
        value = "".join(
            ch for ch in unicodedata.normalize("NFKD", value)
            if not unicodedata.combining(ch)
        )
    chars: List[str] = []
    prev_underscore = False
    for ch in value:
        if ch.isalnum():
            chars.append(ch)
            prev_underscore = False
        else:
            if not prev_underscore:
                chars.append("_")
                prev_underscore = True
    return "".join(chars).strip("_")


def build_word_aliases(word: str, word_dir: str = "") -> List[str]:
    aliases = OrderedDict()
    for value in (word, word_dir):
        if not value:
            continue
        for alias in (
            casefold_text(value),
            slugify_text(value, ascii_only=False),
            slugify_text(value, ascii_only=True),
        ):
            if alias:
                aliases[alias] = None
    return list(aliases.keys())


def split_word_variants(word: str) -> List[str]:
    cleaned = re.sub(r"[()]", ",", word or "")
    return [part.strip() for part in re.split(r"[,;/]+", cleaned) if part.strip()]


def build_word_variant_aliases(word: str) -> List[str]:
    aliases = OrderedDict()
    for variant in split_word_variants(word):
        for alias in build_word_aliases(variant):
            aliases[alias] = None
    return list(aliases.keys())


def parse_time_interval(temporal_raw: str) -> Optional[Tuple[float, float]]:
    if not temporal_raw:
        return None
    try:
        coords = json.loads(temporal_raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(coords, list) or len(coords) != 2:
        return None
    start = float(coords[0])
    end = float(coords[1])
    if end < start:
        start, end = end, start
    return start, end


def extract_video_id(value: str) -> str:
    """Normalize TID video names/URLs like 144-05_cr_0.2.mp4 -> 144-05_cr."""
    if not value:
        return ""
    basename = Path(str(value).split("?", 1)[0]).name
    if basename.endswith(".mp4"):
        basename = basename[:-4]
    match = re.match(r"^(.+?_cr)(?:_[0-9]+(?:\.[0-9]+)?)?$", basename)
    return match.group(1) if match else ""


def split_path_list(raw_value: str) -> List[str]:
    return [item.strip() for item in str(raw_value).split(",") if item.strip()]


def load_display_word_lookup(word_manifest_path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    lookup: Dict[str, Dict[str, str]] = {}
    if word_manifest_path is None or not word_manifest_path.exists():
        return lookup

    for row in read_csv_rows(word_manifest_path):
        word = row.get("word", "").strip()
        word_dir = row.get("word_dir", "").strip()
        if not word:
            continue
        for alias in build_word_aliases(word, word_dir):
            lookup.setdefault(alias, row)
    return lookup


def annotation_stem_to_display_name(annotation_stem: str, display_lookup: Dict[str, Dict[str, str]]) -> str:
    for alias in (
        annotation_stem,
        casefold_text(annotation_stem),
        slugify_text(annotation_stem, ascii_only=False),
        slugify_text(annotation_stem, ascii_only=True),
    ):
        row = display_lookup.get(alias)
        if row:
            return row.get("word", annotation_stem)
    return annotation_stem.replace("_", " ")


def iter_annotation_exports(annotation_dir: Path) -> Iterable[Path]:
    return sorted(annotation_dir.glob("*_export.csv"))


def annotation_word_keys(annotation_dir: Path) -> List[str]:
    keys: List[str] = []
    for path in iter_annotation_exports(annotation_dir):
        stem = path.stem
        keys.append(stem[: -len("_export")] if stem.endswith("_export") else stem)
    return keys


def convert_annotation_file(
    annotation_path: Path,
    display_lookup: Dict[str, Dict[str, str]],
    word_lookup: Optional[Dict[str, Dict[str, object]]] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    annotation_stem = annotation_path.stem
    if annotation_stem.endswith("_export"):
        annotation_word_key = annotation_stem[: -len("_export")]
    else:
        annotation_word_key = annotation_stem

    word_entry = lookup_word_entry(annotation_word_key, word_lookup or {})
    word_name = (
        str(word_entry.get("word_name"))
        if word_entry is not None and word_entry.get("word_name")
        else annotation_stem_to_display_name(annotation_word_key, display_lookup)
    )
    source_order: "OrderedDict[str, int]" = OrderedDict()
    label_rows: List[Dict[str, object]] = []
    skipped_rows: List[Dict[str, object]] = []
    raw_indices: List[int] = []
    matched_source_video_ids = set()

    with annotation_path.open("r", encoding="utf-8", newline="") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            if not raw_line.strip() or raw_line.startswith("#"):
                continue
            row = next(csv.reader([raw_line]))
            if len(row) != 6:
                raise ValueError(
                    f"Unexpected VIA row width in {annotation_path} line {line_no}: {len(row)}"
                )

            metadata_id, file_list_raw, _, temporal_raw, _, _ = row
            try:
                file_list = json.loads(file_list_raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Could not parse file_list in {annotation_path} line {line_no}"
                ) from exc

            source_video_url = ""
            if isinstance(file_list, list) and file_list:
                source_video_url = str(file_list[0])
            source_video_id = extract_video_id(source_video_url)

            if source_video_url not in source_order:
                source_order[source_video_url] = len(source_order) + 1
            fallback_example_number = source_order[source_video_url]

            local_example = find_example_for_annotation(
                word_entry,
                source_video_id,
                fallback_example_number,
            )
            if local_example is not None:
                matched_source_video_ids.add(source_video_id or str(fallback_example_number))
            example_video_number = int(
                local_example.get("meaning_number") or fallback_example_number
            ) if local_example is not None else fallback_example_number

            raw_example_index = None
            raw_prefix = metadata_id.split("_", 1)[0]
            if raw_prefix.isdigit():
                raw_example_index = int(raw_prefix)
                raw_indices.append(raw_example_index)

            interval = parse_time_interval(temporal_raw)
            base_row = {
                "word_name": word_name,
                "annotation_word_key": annotation_word_key,
                "example_video_number": example_video_number,
                "source_video_id": source_video_id,
                "annotation_file": annotation_path.name,
                "source_video_url": source_video_url,
                "local_example_video_path": (
                    str(local_example.get("path")) if local_example is not None and local_example.get("path") else ""
                ),
                "local_example_video_id": (
                    str(local_example.get("video_id")) if local_example is not None and local_example.get("video_id") else ""
                ),
                "local_meaning_number": (
                    local_example.get("meaning_number") if local_example is not None else ""
                ),
                "raw_metadata_id": metadata_id,
                "raw_example_index": raw_example_index,
                "raw_temporal_coordinates": temporal_raw,
            }

            if interval is None:
                skipped_rows.append(dict(base_row))
                continue

            start_time, end_time = interval
            detailed = dict(base_row)
            detailed.update(
                {
                    "start_time": f"{start_time:.6f}",
                    "end_time": f"{end_time:.6f}",
                    "time_interval": f"[{start_time:.6f}, {end_time:.6f}]",
                }
            )
            label_rows.append(detailed)

    non_dense_raw = False
    if raw_indices:
        unique_raw = sorted(set(raw_indices))
        expected = list(range(min(unique_raw), max(unique_raw) + 1))
        non_dense_raw = unique_raw != expected

    summary = {
        "annotation_file": annotation_path.name,
        "annotation_word_key": annotation_word_key,
        "word_name": word_name,
        "source_video_count": len(source_order),
        "matched_local_source_video_count": len(matched_source_video_ids),
        "segment_count": len(label_rows),
        "skipped_non_segment_row_count": len(skipped_rows),
        "multi_segment_example_count": len(
            {
                row["example_video_number"]
                for row in label_rows
                if sum(
                    1
                    for other in label_rows
                    if other["example_video_number"] == row["example_video_number"]
                )
                > 1
            }
        ),
        "raw_example_indices_non_dense": non_dense_raw,
    }
    return label_rows, skipped_rows, summary


def export_labels(
    annotation_dir: Path,
    label_dir: Path,
    word_manifest_path: Optional[Path],
    tid_raw_roots: Optional[Sequence[Path]] = None,
    script_root: Optional[Path] = None,
) -> Dict[str, object]:
    annotation_paths = list(iter_annotation_exports(annotation_dir))
    log_status(
        f"export_labels: {len(annotation_paths)} annotation CSVs from {annotation_dir}"
    )
    display_lookup = load_display_word_lookup(word_manifest_path)
    word_lookup: Dict[str, Dict[str, object]] = {}
    if script_root is not None:
        log_status("export_labels: loading TID word metadata")
        word_lookup = load_tid_word_lookup(
            word_manifest_path,
            tid_raw_roots or [],
            script_root,
            selected_word_keys=[
                path.stem[: -len("_export")]
                if path.stem.endswith("_export")
                else path.stem
                for path in annotation_paths
            ],
        )
        log_status(f"export_labels: loaded {len({id(value) for value in word_lookup.values()})} word metadata entries")

    combined_simple_rows: List[Dict[str, object]] = []
    combined_detailed_rows: List[Dict[str, object]] = []
    combined_skipped_rows: List[Dict[str, object]] = []
    per_file_summary_rows: List[Dict[str, object]] = []

    by_word_simple: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for file_index, annotation_path in enumerate(annotation_paths, start=1):
        if file_index == 1 or file_index % 25 == 0 or file_index == len(annotation_paths):
            log_status(
                f"export_labels: converting annotation {file_index}/{len(annotation_paths)}: {annotation_path.name}"
            )
        detailed_rows, skipped_rows, summary = convert_annotation_file(
            annotation_path,
            display_lookup,
            word_lookup,
        )
        per_file_summary_rows.append(summary)
        combined_detailed_rows.extend(detailed_rows)
        combined_skipped_rows.extend(skipped_rows)

        simple_rows = [
            {
                "word_name": row["word_name"],
                "example_video_number": row["example_video_number"],
                "time_interval": row["time_interval"],
            }
            for row in detailed_rows
        ]
        combined_simple_rows.extend(simple_rows)

        word_key = summary["annotation_word_key"]
        by_word_simple[word_key].extend(simple_rows)

    combined_simple_path = label_dir / "tid_overlap_labels.csv"
    combined_detailed_path = label_dir / "tid_overlap_labels_detailed.csv"
    summary_csv_path = label_dir / "annotation_export_summary.csv"
    summary_json_path = label_dir / "annotation_export_summary.json"
    skipped_path = label_dir / "tid_overlap_non_segment_rows.csv"
    per_word_dir = label_dir / "by_word"

    write_csv_rows(
        combined_simple_path,
        combined_simple_rows,
        fieldnames=["word_name", "example_video_number", "time_interval"],
    )
    write_csv_rows(
        combined_detailed_path,
        combined_detailed_rows,
        fieldnames=[
            "word_name",
            "annotation_word_key",
            "example_video_number",
            "start_time",
            "end_time",
            "time_interval",
            "source_video_id",
            "local_example_video_path",
            "local_example_video_id",
            "local_meaning_number",
            "annotation_file",
            "source_video_url",
            "raw_metadata_id",
            "raw_example_index",
            "raw_temporal_coordinates",
        ],
    )
    write_csv_rows(
        summary_csv_path,
        per_file_summary_rows,
        fieldnames=[
            "annotation_file",
            "annotation_word_key",
            "word_name",
            "source_video_count",
            "matched_local_source_video_count",
            "segment_count",
            "skipped_non_segment_row_count",
            "multi_segment_example_count",
            "raw_example_indices_non_dense",
        ],
    )

    if combined_skipped_rows:
        write_csv_rows(
            skipped_path,
            combined_skipped_rows,
            fieldnames=[
                "word_name",
                "annotation_word_key",
                "example_video_number",
                "source_video_id",
                "annotation_file",
                "source_video_url",
                "local_example_video_path",
                "local_example_video_id",
                "local_meaning_number",
                "raw_metadata_id",
                "raw_example_index",
                "raw_temporal_coordinates",
            ],
        )

    for word_key, rows in sorted(by_word_simple.items()):
        write_csv_rows(
            per_word_dir / f"{word_key}.csv",
            rows,
            fieldnames=["word_name", "example_video_number", "time_interval"],
        )

    export_summary = {
        "annotation_dir": str(annotation_dir),
        "label_dir": str(label_dir),
        "annotation_file_count": len(per_file_summary_rows),
        "segment_row_count": len(combined_detailed_rows),
        "simple_label_row_count": len(combined_simple_rows),
        "non_segment_row_count": len(combined_skipped_rows),
        "metadata_word_count": len({id(value) for value in word_lookup.values()}),
        "files_with_multi_segment_examples": sum(
            1 for row in per_file_summary_rows if int(row["multi_segment_example_count"]) > 0
        ),
        "files_with_non_dense_raw_indices": sum(
            1 for row in per_file_summary_rows if str(row["raw_example_indices_non_dense"]).lower() == "true"
        ),
        "combined_simple_csv": str(combined_simple_path),
        "combined_detailed_csv": str(combined_detailed_path),
        "per_word_dir": str(per_word_dir),
    }

    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(export_summary, handle, indent=2, ensure_ascii=False)

    return export_summary


def load_annotations_for_evaluation(
    annotation_dir: Path,
    word_manifest_path: Optional[Path],
    word_lookup: Optional[Dict[str, Dict[str, object]]] = None,
) -> Dict[Tuple[str, str, int], List[Dict[str, object]]]:
    display_lookup = load_display_word_lookup(word_manifest_path)
    word_lookup = word_lookup or {}
    grouped: Dict[Tuple[str, str, int], List[Dict[str, object]]] = defaultdict(list)
    for annotation_path in iter_annotation_exports(annotation_dir):
        detailed_rows, _, _ = convert_annotation_file(annotation_path, display_lookup, word_lookup)
        for row in detailed_rows:
            key = (
                str(row["annotation_word_key"]),
                str(row.get("source_video_id") or ""),
                int(row["example_video_number"]),
            )
            grouped[key].append(row)
    return grouped


def load_example_manifest_lookup(example_manifest_path: Path) -> Dict[str, Dict[str, object]]:
    grouped_by_primary: Dict[str, Dict[str, object]] = OrderedDict()
    alias_lookup: Dict[str, Dict[str, object]] = {}

    for row in read_csv_rows(example_manifest_path):
        word = row.get("word", "").strip()
        word_dir = row.get("word_dir", "").strip()
        if not word:
            continue
        primary_key = word_dir or slugify_text(word, ascii_only=True)
        info = grouped_by_primary.get(primary_key)
        if info is None:
            matched_lemmas = [tok for tok in row.get("matched_lemmas", "").split() if tok]
            info = {
                "word_name": word,
                "word_dir": word_dir,
                "matched_lemmas": matched_lemmas,
                "rows": [],
            }
            grouped_by_primary[primary_key] = info
            for alias in build_word_aliases(word, word_dir):
                alias_lookup[alias] = info
        info["rows"].append(dict(row))
    return alias_lookup


def resolve_existing_path(raw_path: str, script_root: Path) -> Optional[Path]:
    candidate = Path(raw_path)
    candidates: List[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        repo_root = script_root.parent
        candidates.extend(
            [
                Path.cwd() / candidate,
                script_root / candidate,
                repo_root / candidate,
                repo_root / "sign2gpt" / candidate,
            ]
        )

    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def resolve_path(raw_path: str, script_root: Path) -> Path:
    existing = resolve_existing_path(raw_path, script_root)
    if existing is not None:
        return existing

    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate

    repo_root = script_root.parent
    if candidate.parts and candidate.parts[0] == "sign2gpt":
        return repo_root / candidate
    return Path.cwd() / candidate


def resolve_tid_raw_roots(raw_roots: str, script_root: Path) -> List[Path]:
    roots: List[Path] = []
    for raw_root in split_path_list(raw_roots):
        root = resolve_existing_path(raw_root, script_root)
        if root is not None and root.is_dir():
            roots.append(root)
    return roots


def resolve_word_relative_path(word_dir: Path, raw_path: str, script_root: Path) -> Optional[Path]:
    if not raw_path:
        return None
    candidate = Path(raw_path)
    candidates: List[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        repo_root = script_root.parent
        candidates.extend(
            [
                word_dir / candidate,
                Path.cwd() / candidate,
                script_root / candidate,
                repo_root / candidate,
                repo_root / "sign2gpt" / candidate,
            ]
        )
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def lookup_word_entry(
    annotation_word_key: str,
    word_lookup: Dict[str, Dict[str, object]],
) -> Optional[Dict[str, object]]:
    for alias in (
        annotation_word_key,
        casefold_text(annotation_word_key),
        slugify_text(annotation_word_key, ascii_only=False),
        slugify_text(annotation_word_key, ascii_only=True),
    ):
        entry = word_lookup.get(alias)
        if entry is not None:
            return entry
    return None


def candidate_word_info_paths_for_key(
    annotation_word_key: str,
    tid_raw_roots: Sequence[Path],
) -> List[Path]:
    aliases = OrderedDict()
    for alias in build_word_aliases(annotation_word_key, annotation_word_key):
        aliases[alias] = None

    candidates: List[Path] = []
    seen = set()
    for root in tid_raw_roots:
        for alias in aliases:
            for path in root.glob(f"*/{alias}/word_info.json"):
                resolved = path.resolve()
                if resolved in seen or not resolved.exists():
                    continue
                seen.add(resolved)
                candidates.append(resolved)
    return candidates


def find_example_for_annotation(
    word_entry: Optional[Dict[str, object]],
    source_video_id: str,
    example_video_number: int,
) -> Optional[Dict[str, object]]:
    if word_entry is None:
        return None
    examples_by_id = word_entry.get("examples_by_id", {})
    if source_video_id and isinstance(examples_by_id, dict):
        match = examples_by_id.get(source_video_id)
        if match is not None:
            return match
    examples = word_entry.get("examples", [])
    if isinstance(examples, list) and 1 <= example_video_number <= len(examples):
        return examples[example_video_number - 1]
    return None


def load_manifest_rows_by_alias(word_manifest_path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    rows_by_alias: Dict[str, Dict[str, str]] = {}
    if word_manifest_path is None or not word_manifest_path.exists():
        return rows_by_alias
    for row in read_csv_rows(word_manifest_path):
        word = row.get("word", "").strip()
        word_dir = row.get("word_dir", "").strip()
        for alias in build_word_aliases(word, word_dir):
            rows_by_alias.setdefault(alias, row)
    return rows_by_alias


def candidate_word_info_paths(
    manifest_row: Dict[str, str],
    tid_raw_roots: Sequence[Path],
    script_root: Path,
) -> List[Path]:
    candidates: List[Path] = []
    for key in ("output_dir", "source_dir"):
        raw_path = manifest_row.get(key, "")
        resolved = resolve_existing_path(raw_path, script_root) if raw_path else None
        if resolved is not None:
            candidates.append(resolved / "word_info.json")

    letter = manifest_row.get("letter", "")
    word_dir = manifest_row.get("word_dir", "")
    for root in tid_raw_roots:
        if letter and word_dir:
            candidates.append(root / letter / word_dir / "word_info.json")
        elif word_dir:
            candidates.extend(root.glob(f"*/{word_dir}/word_info.json"))

    unique: List[Path] = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def build_tid_word_entry(
    word_info_path: Path,
    manifest_row: Optional[Dict[str, str]],
    script_root: Path,
) -> Dict[str, object]:
    word_dir = word_info_path.parent
    info = json.loads(word_info_path.read_text(encoding="utf-8"))
    manifest_row = manifest_row or {}
    word_name = info.get("word") or manifest_row.get("word") or word_dir.name
    word_dir_name = info.get("sanitized_word") or manifest_row.get("word_dir") or word_dir.name
    matched_lemmas = [tok for tok in manifest_row.get("matched_lemmas", "").split() if tok]

    examples: List[Dict[str, object]] = []
    meanings = info.get("meanings")
    if isinstance(meanings, list) and meanings:
        for order_index, meaning in enumerate(meanings, start=1):
            videos = meaning.get("videos", {}) if isinstance(meaning.get("videos"), dict) else {}
            rel_path = (
                meaning.get("example_video")
                or videos.get("example_video")
                or meaning.get("example_video_filename")
                or ""
            )
            video_id = (
                meaning.get("example_video_id")
                or videos.get("example_video_id")
                or extract_video_id(rel_path)
            )
            path = resolve_word_relative_path(word_dir, str(rel_path), script_root)
            examples.append(
                {
                    "meaning_number": int(meaning.get("meaning_number") or order_index),
                    "video_id": str(video_id or ""),
                    "path": path,
                    "relative_path": rel_path,
                    "translation_text": meaning.get("translation_text", ""),
                    "transcription_text": meaning.get("transcription_text", ""),
                }
            )
    else:
        for meaning_dir in sorted(word_dir.glob("meaning_*")):
            data_path = meaning_dir / "data.json"
            meaning_data: Dict[str, object] = {}
            if data_path.exists():
                meaning_data = json.loads(data_path.read_text(encoding="utf-8"))
            raw_num = meaning_data.get("meaning_number") or meaning_dir.name.rsplit("_", 1)[-1]
            try:
                meaning_number = int(raw_num)
            except (TypeError, ValueError):
                meaning_number = len(examples) + 1
            videos = meaning_data.get("videos", {}) if isinstance(meaning_data.get("videos"), dict) else {}
            rel_path = videos.get("example_video") or "example_video.mp4"
            path = resolve_word_relative_path(meaning_dir, str(rel_path), script_root)
            examples.append(
                {
                    "meaning_number": meaning_number,
                    "video_id": extract_video_id(str(rel_path)),
                    "path": path,
                    "relative_path": str(meaning_dir.relative_to(word_dir) / str(rel_path)),
                    "translation_text": meaning_data.get("translation_text", ""),
                    "transcription_text": meaning_data.get("transcription_text", ""),
                }
            )

    isolated_videos: List[Dict[str, object]] = []

    def add_isolated(raw_path: str, video_id: str = "") -> None:
        path = resolve_word_relative_path(word_dir, raw_path, script_root)
        if path is None:
            return
        isolated_videos.append(
            {
                "video_id": video_id or extract_video_id(raw_path),
                "path": path,
                "relative_path": raw_path,
            }
        )

    for item in info.get("isolated_videos", []) if isinstance(info.get("isolated_videos"), list) else []:
        if isinstance(item, dict):
            add_isolated(str(item.get("path") or item.get("filename") or ""), str(item.get("video_id") or ""))
    for raw_path in info.get("word_representation_video_paths", []) if isinstance(info.get("word_representation_video_paths"), list) else []:
        add_isolated(str(raw_path))
    for raw_path in info.get("word_representation_videos", []) if isinstance(info.get("word_representation_videos"), list) else []:
        add_isolated(str(raw_path))
    for path in sorted(word_dir.glob("word_representation*.mp4")) + sorted((word_dir / "isolated_videos").glob("*.mp4")):
        isolated_videos.append(
            {
                "video_id": extract_video_id(path.name),
                "path": path.resolve(),
                "relative_path": str(path.relative_to(word_dir)),
            }
        )

    unique_isolated: List[Dict[str, object]] = []
    seen_iso = set()
    for item in isolated_videos:
        path = item.get("path")
        if path is None:
            continue
        resolved = Path(path).resolve()
        if resolved in seen_iso:
            continue
        seen_iso.add(resolved)
        item = dict(item)
        item["path"] = resolved
        unique_isolated.append(item)

    examples_by_id = {
        str(item["video_id"]): item
        for item in examples
        if item.get("video_id")
    }
    return {
        "word_name": str(word_name),
        "word_dir": str(word_dir_name),
        "word_info_path": word_info_path,
        "word_dir_path": word_dir,
        "matched_lemmas": matched_lemmas,
        "examples": examples,
        "examples_by_id": examples_by_id,
        "isolated_videos": unique_isolated,
    }


def build_lookup_from_word_entries(
    entries: Sequence[Dict[str, object]],
) -> Dict[str, Dict[str, object]]:
    lookup: Dict[str, Dict[str, object]] = {}
    variant_candidates: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for entry in entries:
        for alias in build_word_aliases(str(entry.get("word_name", "")), str(entry.get("word_dir", ""))):
            lookup.setdefault(alias, entry)
        for alias in build_word_variant_aliases(str(entry.get("word_name", ""))):
            variant_candidates[alias].append(entry)

    for alias, candidates in variant_candidates.items():
        unique_candidates = {
            str(candidate.get("word_info_path")): candidate
            for candidate in candidates
        }
        if len(unique_candidates) == 1:
            lookup.setdefault(alias, next(iter(unique_candidates.values())))
    return lookup


def load_tid_word_lookup(
    word_manifest_path: Optional[Path],
    tid_raw_roots: Sequence[Path],
    script_root: Path,
    selected_word_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, object]]:
    manifest_rows_by_alias = load_manifest_rows_by_alias(word_manifest_path)
    entries: List[Dict[str, object]] = []
    seen_word_infos = set()
    selected_aliases = set()
    for key in selected_word_keys or []:
        selected_aliases.update(build_word_aliases(key, key))

    primary_manifest_rows = {
        id(row): row for row in manifest_rows_by_alias.values()
    }.values()
    primary_manifest_rows = list(primary_manifest_rows)
    if word_manifest_path is not None:
        log_status(
            f"metadata: reading {len(primary_manifest_rows)} manifest rows from {word_manifest_path}"
        )
    else:
        log_status("metadata: no word manifest resolved; scanning TID raw roots")

    for row_index, row in enumerate(primary_manifest_rows, start=1):
        if row_index == 1 or row_index % 100 == 0 or row_index == len(primary_manifest_rows):
            log_status(f"metadata: resolving manifest row {row_index}/{len(primary_manifest_rows)}")
        if selected_aliases:
            row_aliases = set(build_word_aliases(row.get("word", ""), row.get("word_dir", "")))
            row_aliases.update(build_word_variant_aliases(row.get("word", "")))
            if row_aliases.isdisjoint(selected_aliases):
                continue
        for word_info_path in candidate_word_info_paths(row, tid_raw_roots, script_root):
            resolved = word_info_path.resolve()
            if resolved in seen_word_infos:
                continue
            seen_word_infos.add(resolved)
            entries.append(build_tid_word_entry(resolved, row, script_root))
            break

    # If no manifest exists, fall back to scanning all roots.
    if not manifest_rows_by_alias:
        for root in tid_raw_roots:
            log_status(f"metadata: scanning {root} for word_info.json files")
            for word_info_path in sorted(root.glob("*/*/word_info.json")):
                resolved = word_info_path.resolve()
                if resolved in seen_word_infos:
                    continue
                seen_word_infos.add(resolved)
                entries.append(build_tid_word_entry(resolved, None, script_root))

    lookup = build_lookup_from_word_entries(entries)
    if manifest_rows_by_alias and selected_word_keys:
        missing_keys = [
            key for key in selected_word_keys
            if lookup_word_entry(key, lookup) is None
        ]
        if missing_keys:
            log_status(
                f"metadata: {len(missing_keys)} selected annotation words are absent "
                "from the manifest; trying raw TID fallback"
            )
        fallback_added = 0
        for key in missing_keys:
            for word_info_path in candidate_word_info_paths_for_key(key, tid_raw_roots):
                resolved = word_info_path.resolve()
                if resolved in seen_word_infos:
                    continue
                seen_word_infos.add(resolved)
                entries.append(build_tid_word_entry(resolved, None, script_root))
                fallback_added += 1
                break
        if fallback_added:
            log_status(f"metadata: added {fallback_added} raw TID fallback entries")
            lookup = build_lookup_from_word_entries(entries)
    return lookup


def ascii_slug(text: str) -> str:
    return slugify_text(text, ascii_only=True)


def find_isolated_video_path(args: argparse.Namespace, script_root: Path, gloss: str) -> Path:
    candidates: List[str] = []
    gloss_aliases = [gloss, slugify_text(gloss, ascii_only=False), ascii_slug(gloss)]

    if args.isolated_video_template:
        for alias in gloss_aliases:
            candidates.append(args.isolated_video_template.format(gloss=alias))

    if args.isolated_video_dir:
        for alias in gloss_aliases:
            candidates.append(str(Path(args.isolated_video_dir) / f"{alias}.mp4"))

    if args.iso_root_dir:
        root = resolve_existing_path(args.iso_root_dir, script_root)
        if root is not None:
            for alias in gloss_aliases:
                folder = root / alias
                if folder.is_dir():
                    matches = sorted(folder.glob("*.mp4"))
                    if matches:
                        return matches[0]

    for raw in candidates:
        resolved = resolve_existing_path(raw, script_root)
        if resolved is not None:
            return resolved

    raise FileNotFoundError(
        f"Could not resolve an isolated video for gloss '{gloss}'. "
        "Use --isolated_video_template, --isolated_video_dir, or --iso_root_dir."
    )


def interval_iou_seconds(
    pred_start: float,
    pred_end: float,
    gt_start: float,
    gt_end: float,
) -> float:
    pred_len = max(0.0, pred_end - pred_start)
    gt_len = max(0.0, gt_end - gt_start)
    if pred_len <= 0.0 or gt_len <= 0.0:
        return 0.0
    inter = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = pred_len + gt_len - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def word_edit_distance(reference_tokens: Sequence[str], hypothesis_tokens: Sequence[str]) -> int:
    if not reference_tokens:
        return len(hypothesis_tokens)
    if not hypothesis_tokens:
        return len(reference_tokens)

    previous_row = list(range(len(hypothesis_tokens) + 1))
    for ref_idx, ref_token in enumerate(reference_tokens, start=1):
        current_row = [ref_idx]
        for hyp_idx, hyp_token in enumerate(hypothesis_tokens, start=1):
            substitution_cost = 0 if ref_token == hyp_token else 1
            current_row.append(
                min(
                    previous_row[hyp_idx] + 1,
                    current_row[hyp_idx - 1] + 1,
                    previous_row[hyp_idx - 1] + substitution_cost,
                )
            )
        previous_row = current_row
    return previous_row[-1]


def hash_for_path(path: Path) -> str:
    return hashlib.md5(str(path).encode("utf-8")).hexdigest()[:10]


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


def evaluate_ncc(args: argparse.Namespace) -> Dict[str, object]:
    from train_utils.release_helpers import validate_evaluation_vocabulary
    validate_evaluation_vocabulary(args.config, args.vocab_pkl)
    log_status("evaluate: importing model/video dependencies")
    try:
        import cv2
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import torch

        from augmentation.get_aug import get_aug
        from dataloaders.get_dataset import get_dataset
        from dataloaders.data_utils.file_utils import read_pickle
        from models.get_models import get_model
        from evaluation.tid_analyzer import (
            build_single_lmdb_from_mp4,
            load_cfg_and_model,
            write_single_csv,
        )
        from evaluation.tid_analyzer_ncc import sliding_ncc
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Evaluation mode requires the model/video dependencies from the training environment. "
            f"Missing module: {exc.name}"
        ) from exc

    def load_cfg_and_model_without_checkpoint(cfg_path: str):
        from train_utils.release_helpers import load_config
        cfg = load_config(cfg_path)
        model = get_model(cfg.model_name, cfg.model_params.to_dict())
        model.eval()
        return model, cfg, None

    if args.no_checkpoint and args.checkpoint:
        raise ValueError("Use either --checkpoint or --no_checkpoint, not both.")
    if args.no_checkpoint and args.prediction_method != "ncc":
        raise ValueError(
            "--no_checkpoint is only supported with --prediction_method ncc. "
            "The model_scores method evaluates pseudo-gloss class scores and needs a trained checkpoint."
        )

    script_root = Path(__file__).resolve().parents[1]
    annotation_dir = resolve_existing_path(args.annotation_dir, script_root)
    example_manifest_path = resolve_existing_path(args.example_manifest, script_root)
    word_manifest_path = resolve_existing_path(args.word_manifest, script_root)
    tid_raw_roots = resolve_tid_raw_roots(args.tid_raw_roots, script_root)
    movement_windows_path = (
        resolve_existing_path(args.movement_windows_csv, script_root)
        if args.movement_windows_csv
        else None
    )
    output_dir = resolve_path(args.output_dir, script_root)

    if annotation_dir is None:
        raise FileNotFoundError(f"Annotation directory not found: {args.annotation_dir}")
    if args.movement_windows_csv and movement_windows_path is None:
        raise FileNotFoundError(f"Movement-window CSV not found: {args.movement_windows_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    movement_windows_by_path: Dict[str, Dict[str, str]] = {}
    movement_windows_by_role_video_id: Dict[Tuple[str, str], Dict[str, str]] = {}
    movement_window_rows: List[Dict[str, str]] = []
    if movement_windows_path is not None and args.movement_window_apply != "none":
        (
            movement_windows_by_path,
            movement_windows_by_role_video_id,
            movement_window_rows,
        ) = load_movement_window_lookup(movement_windows_path)
        log_status(f"evaluate: loaded {len(movement_window_rows)} movement-window rows from {movement_windows_path}")

    vocab_path = args.vocab_pkl
    vocab_resolved = resolve_existing_path(vocab_path, script_root)
    if vocab_resolved is None:
        raise FileNotFoundError(f"Vocab pickle not found: {vocab_path}")
    log_status(f"evaluate: loading vocab pickle {vocab_resolved}")
    vocab = read_pickle(str(vocab_resolved))
    lem_to_id = vocab["dict_lem_to_id"]
    num_classes = len(lem_to_id)
    log_status(f"evaluate: vocab has {num_classes} classes")

    log_status("evaluate: loading TID word metadata")
    word_lookup = load_tid_word_lookup(
        word_manifest_path,
        tid_raw_roots,
        script_root,
        selected_word_keys=annotation_word_keys(annotation_dir),
    )
    log_status(f"evaluate: loaded {len({id(value) for value in word_lookup.values()})} word metadata entries")
    labels_by_example = load_annotations_for_evaluation(annotation_dir, word_manifest_path, word_lookup)
    log_status(f"evaluate: loaded {len(labels_by_example)} annotated examples")
    example_lookup = (
        load_example_manifest_lookup(example_manifest_path)
        if example_manifest_path is not None
        else {}
    )

    def candidate_glosses_for_group(
        group: Dict[str, object],
        gt_rows: Sequence[Dict[str, object]],
    ) -> List[str]:
        candidate_glosses = list(group.get("matched_lemmas") or [])
        if not candidate_glosses:
            candidate_glosses = [str(gt_rows[0]["word_name"]).casefold()]
        return candidate_glosses

    skipped_out_of_vocab_rows: List[Dict[str, object]] = []
    filtered_labels_by_example: Dict[Tuple[str, str, int], List[Dict[str, object]]] = {}
    for key, gt_rows in sorted(labels_by_example.items()):
        annotation_word_key, source_video_id, example_video_number = key
        group = lookup_word_entry(annotation_word_key, word_lookup)
        if group is None:
            group = lookup_word_entry(annotation_word_key, example_lookup)
        if group is None:
            fallback_candidates = build_word_aliases(annotation_word_key, annotation_word_key)
            if not any(candidate in lem_to_id for candidate in fallback_candidates):
                skipped_out_of_vocab_rows.append(
                    {
                        "word_name": gt_rows[0]["word_name"],
                        "annotation_word_key": annotation_word_key,
                        "source_video_id": source_video_id,
                        "example_video_number": example_video_number,
                        "candidate_glosses": " ".join(fallback_candidates),
                        "reason": "metadata_missing_and_no_annotation_alias_in_vocab",
                    }
                )
                continue
            raise KeyError(f"No metadata entry matches annotation word '{annotation_word_key}'.")

        candidate_glosses = candidate_glosses_for_group(group, gt_rows)
        glosses_in_vocab = [gloss for gloss in candidate_glosses if gloss in lem_to_id]
        if not glosses_in_vocab:
            skipped_out_of_vocab_rows.append(
                {
                    "word_name": gt_rows[0]["word_name"],
                    "annotation_word_key": annotation_word_key,
                    "source_video_id": source_video_id,
                    "example_video_number": example_video_number,
                    "candidate_glosses": " ".join(candidate_glosses),
                    "reason": "candidate_glosses_not_in_vocab",
                }
            )
            continue
        filtered_labels_by_example[key] = gt_rows

    if skipped_out_of_vocab_rows:
        log_status(
            f"evaluate: skipped {len(skipped_out_of_vocab_rows)} annotated examples "
            "with no candidate gloss in the vocab"
        )
    labels_by_example = filtered_labels_by_example
    log_status(f"evaluate: kept {len(labels_by_example)} annotated examples after vocab filtering")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = None
    if args.checkpoint:
        checkpoint_resolved = resolve_existing_path(args.checkpoint, script_root)
        checkpoint_path = str(checkpoint_resolved) if checkpoint_resolved is not None else args.checkpoint

    if args.no_checkpoint:
        log_status(f"evaluate: building model from config without checkpoint ({args.config})")
        model, cfg, ckpt = load_cfg_and_model_without_checkpoint(args.config)
        checkpoint_mode = "config_init"
    else:
        log_status(f"evaluate: loading model config/checkpoint ({args.config})")
        model, cfg, ckpt = load_cfg_and_model(args.config, checkpoint_path)
        checkpoint_mode = "checkpoint"
    model.to(device)
    model.eval()
    checkpoint_label = str(ckpt) if ckpt is not None else ""
    log_status(
        f"evaluate: model loaded on {device}; "
        f"checkpoint_mode={checkpoint_mode}; checkpoint={checkpoint_label or 'none'}"
    )
    ncc_embedding_source = args.ncc_embedding_source
    if ncc_embedding_source == "auto":
        ncc_embedding_source = "spatial" if args.no_checkpoint else "encoder"
    log_status(f"evaluate: NCC embedding source is {ncc_embedding_source}")

    transform = get_aug(cfg.aug_name, cfg.aug_params)
    temp_base = resolve_path(args.tmp_dir, script_root) if args.tmp_dir else Path(tempfile.gettempdir())
    temp_base.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix="tid_overlap_ncc_eval_", dir=str(temp_base)))
    log_status(f"evaluate: temporary LMDB root is {temp_root}")

    sentence_cache: Dict[Path, Dict[str, object]] = {}
    isolated_cache: Dict[Tuple[Path, int, Optional[int]], Dict[str, object]] = {}
    plot_dir = output_dir / args.plot_dir
    if not args.no_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)
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
                    "NCC embedding source 'spatial' was requested, but the model output does not "
                    "contain dict_feat['list_of_original_features']."
                )
            if isinstance(spatial_features, (list, tuple)):
                if not spatial_features:
                    raise RuntimeError("NCC spatial feature list is empty.")
                spatial_features = spatial_features[0]
            if spatial_features.dim() == 3:
                spatial_features = spatial_features[0]
            hidden = spatial_features[: frames.shape[0]].float().detach().cpu()
            valid_T = int(hidden.shape[0])
        else:
            hidden = out["enc_output"]["hidden_state"][0, :valid_encoder_steps].float().detach().cpu()
            valid_T = valid_encoder_steps
        return scores, hidden, valid_T

    def trim_bounds_seconds(duration: float) -> Tuple[float, float, float]:
        trim_start = (
            args.trim_start_seconds
            if args.trim_start_seconds is not None
            else args.trim_seconds
        )
        trim_end = (
            args.trim_end_seconds
            if args.trim_end_seconds is not None
            else args.trim_seconds
        )
        trim_start = max(0.0, float(trim_start))
        trim_end = max(0.0, float(trim_end))
        if duration <= 0.0:
            return 0.0, 0.0, 0.0
        if trim_start + trim_end >= duration:
            # Avoid eliminating the whole search space for very short examples.
            return 0.0, 0.0, duration
        return trim_start, trim_end, duration - trim_end

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

    def transcription_tokens_in_vocab(transcription_text: str) -> List[str]:
        return [token for token in tokenize_turkish(transcription_text) if token in lem_to_id]

    def decode_reference_vocab_sequence(
        sentence_item: Dict[str, object],
        reference_tokens: Sequence[str],
    ) -> List[str]:
        if not reference_tokens:
            return []
        raw_scores = sentence_item.get("scores")
        if raw_scores is None:
            return []
        if hasattr(raw_scores, "detach"):
            score_matrix = raw_scores.detach().cpu().numpy()
        else:
            score_matrix = np.asarray(raw_scores)
        if score_matrix.ndim != 2 or score_matrix.size == 0:
            return []

        unique_reference_tokens = list(OrderedDict.fromkeys(str(token) for token in reference_tokens if token))
        if not unique_reference_tokens:
            return []

        events: List[Dict[str, object]] = []
        if WER_SEGMENT_SMOOTH_WINDOW > 1:
            kernel = (
                np.ones(WER_SEGMENT_SMOOTH_WINDOW, dtype=np.float32)
                / float(WER_SEGMENT_SMOOTH_WINDOW)
            )
        else:
            kernel = None

        for token in unique_reference_tokens:
            gloss_id = lem_to_id.get(token)
            if gloss_id is None or gloss_id < 0 or gloss_id >= score_matrix.shape[1]:
                continue

            row = np.asarray(score_matrix[:, int(gloss_id)], dtype=np.float32)
            if row.size == 0:
                continue
            if kernel is not None and row.size > 1:
                row = np.convolve(row, kernel, mode="same")

            peak_value = float(row.max())
            if peak_value < WER_SEGMENT_ABSOLUTE_THRESHOLD:
                continue

            segment_threshold = max(
                peak_value * WER_SEGMENT_RELATIVE_THRESHOLD,
                WER_SEGMENT_ABSOLUTE_THRESHOLD,
            )
            above = row >= segment_threshold
            seg_start: Optional[int] = None
            for step_idx, active in enumerate(above):
                if active and seg_start is None:
                    seg_start = step_idx
                    continue
                if active or seg_start is None:
                    continue

                seg_end = step_idx - 1
                segment = row[seg_start : seg_end + 1]
                peak_offset = int(segment.argmax())
                events.append(
                    {
                        "token": token,
                        "start": seg_start,
                        "peak": seg_start + peak_offset,
                        "score": float(segment[peak_offset]),
                    }
                )
                seg_start = None

            if seg_start is not None:
                segment = row[seg_start:]
                peak_offset = int(segment.argmax())
                events.append(
                    {
                        "token": token,
                        "start": seg_start,
                        "peak": seg_start + peak_offset,
                        "score": float(segment[peak_offset]),
                    }
                )

        events.sort(
            key=lambda item: (
                int(item["start"]),
                int(item["peak"]),
                -float(item["score"]),
                str(item["token"]),
            )
        )
        return [str(item["token"]) for item in events]

    def trim_hidden_by_seconds(
        hidden,
        *,
        duration_sec: float,
        trim_start_sec: float,
        trim_end_sec: float,
    ) -> Tuple[object, int, int, float, float]:
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
    ) -> Tuple[object, int, int, float, float]:
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

    def example_trim_bounds_seconds(
        duration: float,
        *,
        video_path: Path,
        video_id: str = "",
    ) -> Tuple[float, float, float, str, str, str]:
        trim_start, trim_end, usable_end = trim_bounds_seconds(duration)
        movement_source = "none"
        movement_start = ""
        movement_end = ""
        bounds = movement_bounds_for_video(
            video_path,
            role="example",
            duration_sec=duration,
            video_id=video_id,
        )
        if bounds is not None:
            movement_start_sec, movement_end_sec, movement_row = bounds
            movement_source_label = movement_window_source_label(movement_row)
            movement_start = f"{movement_start_sec:.6f}"
            movement_end = f"{movement_end_sec:.6f}"
            clipped_start = max(trim_start, movement_start_sec)
            clipped_end = min(usable_end, movement_end_sec)
            if clipped_end > clipped_start:
                trim_start = clipped_start
                usable_end = clipped_end
                trim_end = max(0.0, duration - usable_end)
                movement_source = movement_source_label
                movement_stats["example_windows_applied"] += 1
            else:
                movement_source = f"{movement_source_label}_no_overlap"
        return trim_start, trim_end, usable_end, movement_source, movement_start, movement_end

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

    def load_hidden(
        video_path: Path,
        *,
        lmdb_stride: int,
        max_len: Optional[int],
        cache: Dict[object, Dict[str, object]],
    ) -> Dict[str, object]:
        cache_key = (video_path, lmdb_stride, max_len)
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
        scores, hidden, valid_T = forward_one_for_eval(frames)
        item = {
            "scores": scores,
            "hidden": hidden,
            "valid_T": valid_T,
            "embedding_source": ncc_embedding_source,
        }
        timing = video_timing(video_path)
        item["duration_sec"] = loaded_duration_seconds(
            timing,
            stored_frames=int(stored_frames),
            lmdb_stride=max(1, lmdb_stride),
            max_len=max_len,
        )
        item["source_duration_sec"] = timing["duration_sec"]
        item["stored_frames"] = int(stored_frames)
        cache[cache_key] = item
        return item

    def ranked_predictions_for_example(
        sentence_item: Dict[str, object],
        candidate_glosses: Sequence[str],
        isolated_candidates: Sequence[Dict[str, object]],
        *,
        example_video_path: Path,
        source_video_id: str = "",
    ) -> List[Dict[str, object]]:
        ranked_results: List[Dict[str, object]] = []
        top_k_limit = max(TOP_K_EVAL_VALUES)
        sentence_hidden = sentence_item["hidden"]
        sentence_steps = int(sentence_item["valid_T"])
        sentence_duration = float(sentence_item["duration_sec"])
        if sentence_steps <= 0:
            raise RuntimeError("Sentence hidden states are empty.")
        (
            trim_start,
            trim_end,
            usable_end,
            example_movement_source,
            example_movement_start,
            example_movement_end,
        ) = example_trim_bounds_seconds(
            sentence_duration,
            video_path=example_video_path,
            video_id=source_video_id,
        )
        usable_duration = max(0.0, usable_end - trim_start)

        glosses_in_vocab = [gloss for gloss in candidate_glosses if gloss in lem_to_id]
        predicted_gloss = " ".join(glosses_in_vocab or candidate_glosses)

        def ncc_curve_windows(
            curve_size: int,
            *,
            effective_iso_len: int,
            stride_steps: int,
        ) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
            if sentence_steps < effective_iso_len:
                start_steps = np.zeros(curve_size, dtype=np.int64)
                end_steps = np.full(curve_size, sentence_steps, dtype=np.int64)
                anchor_steps = np.zeros(curve_size, dtype=np.float32)
                return start_steps, end_steps, anchor_steps

            peak_steps = np.arange(curve_size, dtype=np.int64) * int(stride_steps)
            if args.ncc_peak_position == "center":
                start_steps = np.floor(
                    peak_steps.astype(np.float32)
                    - (float(effective_iso_len) / 2.0)
                    + 0.5
                ).astype(np.int64)
                max_start = max(0, sentence_steps - effective_iso_len)
                start_steps = np.clip(start_steps, 0, max_start)
                anchor_steps = peak_steps.astype(np.float32)
            else:
                start_steps = peak_steps
                anchor_steps = start_steps.astype(np.float32)
            end_steps = np.minimum(sentence_steps, start_steps + effective_iso_len)
            return start_steps, end_steps, anchor_steps

        candidates = list(isolated_candidates)
        if not candidates:
            for gloss in candidate_glosses:
                try:
                    iso_path = find_isolated_video_path(args, script_root, gloss)
                except FileNotFoundError:
                    continue
                candidates.append({"path": iso_path, "video_id": "", "relative_path": str(iso_path)})

        for isolated in candidates:
            iso_path_raw = isolated.get("path")
            if not iso_path_raw:
                continue
            iso_path = Path(iso_path_raw)
            if not iso_path.exists():
                continue
            iso_item = load_hidden(
                iso_path,
                lmdb_stride=args.iso_stride_frames,
                max_len=args.iso_max_len,
                cache=isolated_cache,
            )
            iso_raw_hidden = iso_item["hidden"]
            movement_trim = movement_trimmed_hidden_for_video(
                iso_raw_hidden,
                video_path=iso_path,
                role="isolated",
                duration_sec=float(iso_item["duration_sec"]),
                video_id=str(isolated.get("video_id") or extract_video_id(str(iso_path))),
            )
            if movement_trim is not None:
                iso_hidden = movement_trim["hidden"]
                iso_trim_start_steps = int(movement_trim["start_steps"])
                iso_trim_end_steps = int(movement_trim["end_steps"])
                applied_iso_trim_start = float(movement_trim["start_time"])
                applied_iso_trim_end = float(movement_trim["end_trim_seconds"])
                template_movement_source = movement_window_source_label(movement_trim["row"])
                template_movement_start = float(movement_trim["start_time"])
                template_movement_end = float(movement_trim["end_time"])
            else:
                iso_trim_start, iso_trim_end = iso_trim_bounds_seconds(float(iso_item["duration_sec"]))
                (
                    iso_hidden,
                    iso_trim_start_steps,
                    iso_trim_end_steps,
                    applied_iso_trim_start,
                    applied_iso_trim_end,
                ) = trim_hidden_by_seconds(
                    iso_raw_hidden,
                    duration_sec=float(iso_item["duration_sec"]),
                    trim_start_sec=iso_trim_start,
                    trim_end_sec=iso_trim_end,
                )
                template_movement_source = "fixed_iso_trim"
                template_movement_start = 0.0
                template_movement_end = float(iso_item["duration_sec"])
            if int(iso_hidden.shape[0]) <= 0:
                continue
            sim_curve = sliding_ncc(
                sentence_hidden,
                iso_hidden,
                stride=max(1, args.stride),
                ignore_frames=args.ncc_ignore_frames,
                weighting=args.ncc_weighting,
            )
            if sim_curve.size == 0:
                continue

            effective_iso_len = int(iso_hidden.shape[0])
            if args.ncc_ignore_frames > 0 and effective_iso_len > 2 * args.ncc_ignore_frames:
                effective_iso_len -= 2 * args.ncc_ignore_frames
            if effective_iso_len <= 0:
                continue

            stride_steps = max(1, args.stride)
            curve_start_steps, curve_end_steps, curve_anchor_steps = ncc_curve_windows(
                int(sim_curve.shape[0]),
                effective_iso_len=effective_iso_len,
                stride_steps=stride_steps,
            )
            curve_start_seconds = curve_start_steps * (sentence_duration / float(sentence_steps))
            curve_end_seconds = curve_end_steps * (sentence_duration / float(sentence_steps))
            curve_anchor_seconds = curve_anchor_steps * (sentence_duration / float(sentence_steps))
            valid_mask = (curve_start_seconds >= trim_start) & (curve_end_seconds <= usable_end)
            if not np.any(valid_mask):
                continue

            valid_score_mask = valid_mask & np.isfinite(sim_curve)
            if not np.any(valid_score_mask):
                continue

            masked_curve = np.where(valid_score_mask, sim_curve, -np.inf)
            peak_count = min(top_k_limit, int(np.count_nonzero(valid_score_mask)))
            if peak_count <= 0:
                continue
            peak_indices = np.argpartition(masked_curve, -peak_count)[-peak_count:]
            peak_indices = sorted(
                (int(idx) for idx in peak_indices),
                key=lambda idx: (-float(masked_curve[idx]), idx),
            )

            for peak_idx in peak_indices:
                peak_score = float(sim_curve[peak_idx])
                start_step = int(curve_start_steps[peak_idx])
                end_step = int(curve_end_steps[peak_idx])

                step_sec = sentence_duration / float(sentence_steps)
                original_start_sec = start_step * step_sec
                original_end_sec = end_step * step_sec
                start_sec = max(0.0, original_start_sec - trim_start)
                end_sec = max(0.0, original_end_sec - trim_start)

                ranked_results.append(
                    {
                        "rank": 0,
                        "sort_index": len(ranked_results),
                        "predicted_gloss": predicted_gloss,
                        "matched_lemmas_in_vocab": " ".join(glosses_in_vocab),
                        "isolated_video_id": str(isolated.get("video_id") or extract_video_id(str(iso_path))),
                        "isolated_video_path": str(iso_path),
                        "isolated_duration_seconds": float(iso_item["duration_sec"]),
                        "isolated_source_duration_seconds": float(iso_item["source_duration_sec"]),
                        "isolated_steps": int(iso_raw_hidden.shape[0]),
                        "isolated_trimmed_steps": int(iso_hidden.shape[0]),
                        "iso_trim_start_seconds": applied_iso_trim_start,
                        "iso_trim_end_seconds": applied_iso_trim_end,
                        "iso_trim_start_steps": iso_trim_start_steps,
                        "iso_trim_end_steps": iso_trim_end_steps,
                        "example_movement_window_source": example_movement_source,
                        "example_movement_window_start_time": example_movement_start,
                        "example_movement_window_end_time": example_movement_end,
                        "template_movement_window_source": template_movement_source,
                        "template_movement_window_start_time": template_movement_start,
                        "template_movement_window_end_time": template_movement_end,
                        "peak_score": peak_score,
                        "start_step": start_step,
                        "end_step": end_step,
                        "start_time": start_sec,
                        "end_time": end_sec,
                        "original_start_time": original_start_sec,
                        "original_end_time": original_end_sec,
                        "original_time_interval": f"[{original_start_sec:.6f}, {original_end_sec:.6f}]",
                        "time_interval": f"[{start_sec:.6f}, {end_sec:.6f}]",
                        "sim_curve": sim_curve.copy(),
                        "sentence_duration": sentence_duration,
                        "sentence_steps": sentence_steps,
                        "step_sec": step_sec,
                        "ncc_stride": stride_steps,
                        "ncc_peak_position": args.ncc_peak_position,
                        "curve_time_original": curve_anchor_seconds.copy(),
                        "curve_valid_mask": valid_mask.copy(),
                        "trim_start_seconds": trim_start,
                        "trim_end_seconds": trim_end,
                        "usable_end_seconds": usable_end,
                        "usable_duration_seconds": usable_duration,
                    }
                )

        if not ranked_results:
            raise RuntimeError(
                f"No usable isolated videos were found for candidate glosses: {candidate_glosses}"
            )
        ranked_results.sort(
            key=lambda item: (
                -float(item["peak_score"]),
                int(item["sort_index"]),
            )
        )
        for rank, item in enumerate(ranked_results, start=1):
            item["rank"] = rank
        return ranked_results

    def ranked_model_score_predictions_for_example(
        sentence_item: Dict[str, object],
        candidate_glosses: Sequence[str],
        *,
        example_video_path: Path,
        source_video_id: str = "",
    ) -> List[Dict[str, object]]:
        raw_scores = sentence_item.get("scores")
        if raw_scores is None:
            raise RuntimeError("Sentence pseudo-gloss scores are missing.")
        if hasattr(raw_scores, "detach"):
            score_matrix = raw_scores.detach().cpu().numpy()
        else:
            score_matrix = np.asarray(raw_scores)
        if score_matrix.ndim != 2 or score_matrix.size == 0:
            raise RuntimeError("Sentence pseudo-gloss score matrix is empty.")

        sentence_steps = int(sentence_item["valid_T"])
        sentence_duration = float(sentence_item["duration_sec"])
        if sentence_steps <= 0 or sentence_duration <= 0.0:
            raise RuntimeError("Sentence timing is empty.")

        (
            trim_start,
            trim_end,
            usable_end,
            example_movement_source,
            example_movement_start,
            example_movement_end,
        ) = example_trim_bounds_seconds(
            sentence_duration,
            video_path=example_video_path,
            video_id=source_video_id,
        )
        usable_duration = max(0.0, usable_end - trim_start)
        step_sec = sentence_duration / float(sentence_steps)
        curve_start_seconds = np.arange(sentence_steps, dtype=np.float32) * step_sec
        curve_end_seconds = curve_start_seconds + step_sec
        valid_mask = (curve_start_seconds >= trim_start) & (curve_end_seconds <= usable_end)
        if not np.any(valid_mask):
            valid_mask = np.ones(sentence_steps, dtype=bool)

        smooth_window = max(1, int(args.model_score_smooth_window))
        kernel = (
            np.ones(smooth_window, dtype=np.float32) / float(smooth_window)
            if smooth_window > 1
            else None
        )
        relative_threshold = max(0.0, float(args.model_score_relative_threshold))
        absolute_threshold = max(0.0, float(args.model_score_absolute_threshold))
        min_segment_steps = max(1, int(args.model_score_min_segment_steps))

        glosses_in_vocab = [gloss for gloss in candidate_glosses if gloss in lem_to_id]
        ranked_results: List[Dict[str, object]] = []
        for gloss in glosses_in_vocab:
            gloss_id = lem_to_id.get(gloss)
            if gloss_id is None or gloss_id < 0 or gloss_id >= score_matrix.shape[1]:
                continue

            score_curve = np.asarray(score_matrix[:, int(gloss_id)], dtype=np.float32)
            if score_curve.size != sentence_steps:
                score_curve = score_curve[:sentence_steps]
            if score_curve.size == 0:
                continue
            if kernel is not None and score_curve.size > 1:
                score_curve = np.convolve(score_curve, kernel, mode="same")

            finite_mask = np.isfinite(score_curve)
            valid_score_mask = valid_mask[: score_curve.size] & finite_mask
            if not np.any(valid_score_mask):
                continue

            valid_scores = np.where(valid_score_mask, score_curve, -np.inf)
            peak_value = float(np.max(valid_scores))
            if not np.isfinite(peak_value):
                continue
            threshold = max(peak_value * relative_threshold, absolute_threshold)
            above = (score_curve >= threshold) & valid_score_mask

            seg_start: Optional[int] = None
            for step_idx, active in enumerate(above):
                if active and seg_start is None:
                    seg_start = step_idx
                    continue
                if active or seg_start is None:
                    continue

                seg_end = step_idx
                if seg_end - seg_start >= min_segment_steps:
                    segment = score_curve[seg_start:seg_end]
                    peak_offset = int(np.argmax(segment))
                    peak_step = seg_start + peak_offset
                    ranked_results.append(
                        {
                            "rank": 0,
                            "sort_index": len(ranked_results),
                            "predicted_gloss": gloss,
                            "matched_lemmas_in_vocab": " ".join(glosses_in_vocab),
                            "isolated_video_id": "",
                            "isolated_video_path": "",
                            "isolated_duration_seconds": 0.0,
                            "isolated_source_duration_seconds": 0.0,
                            "isolated_steps": 0,
                            "isolated_trimmed_steps": 0,
                            "iso_trim_start_seconds": 0.0,
                            "iso_trim_end_seconds": 0.0,
                            "iso_trim_start_steps": 0,
                            "iso_trim_end_steps": 0,
                            "example_movement_window_source": example_movement_source,
                            "example_movement_window_start_time": example_movement_start,
                            "example_movement_window_end_time": example_movement_end,
                            "template_movement_window_source": "",
                            "template_movement_window_start_time": 0.0,
                            "template_movement_window_end_time": 0.0,
                            "peak_score": float(segment[peak_offset]),
                            "start_step": seg_start,
                            "end_step": seg_end,
                            "start_time": max(0.0, seg_start * step_sec - trim_start),
                            "end_time": max(0.0, seg_end * step_sec - trim_start),
                            "original_start_time": seg_start * step_sec,
                            "original_end_time": seg_end * step_sec,
                            "original_time_interval": f"[{seg_start * step_sec:.6f}, {seg_end * step_sec:.6f}]",
                            "time_interval": (
                                f"[{max(0.0, seg_start * step_sec - trim_start):.6f}, "
                                f"{max(0.0, seg_end * step_sec - trim_start):.6f}]"
                            ),
                            "sim_curve": score_curve.copy(),
                            "sentence_duration": sentence_duration,
                            "sentence_steps": sentence_steps,
                            "step_sec": step_sec,
                            "ncc_stride": 1,
                            "curve_time_original": curve_start_seconds[: score_curve.size].copy(),
                            "curve_valid_mask": valid_score_mask.copy(),
                            "trim_start_seconds": trim_start,
                            "trim_end_seconds": trim_end,
                            "usable_end_seconds": usable_end,
                            "usable_duration_seconds": usable_duration,
                            "model_score_threshold": threshold,
                            "model_score_gloss_id": int(gloss_id),
                        }
                    )
                seg_start = None

            if seg_start is not None:
                seg_end = int(score_curve.size)
                if seg_end - seg_start >= min_segment_steps:
                    segment = score_curve[seg_start:seg_end]
                    peak_offset = int(np.argmax(segment))
                    ranked_results.append(
                        {
                            "rank": 0,
                            "sort_index": len(ranked_results),
                            "predicted_gloss": gloss,
                            "matched_lemmas_in_vocab": " ".join(glosses_in_vocab),
                            "isolated_video_id": "",
                            "isolated_video_path": "",
                            "isolated_duration_seconds": 0.0,
                            "isolated_source_duration_seconds": 0.0,
                            "isolated_steps": 0,
                            "isolated_trimmed_steps": 0,
                            "iso_trim_start_seconds": 0.0,
                            "iso_trim_end_seconds": 0.0,
                            "iso_trim_start_steps": 0,
                            "iso_trim_end_steps": 0,
                            "example_movement_window_source": example_movement_source,
                            "example_movement_window_start_time": example_movement_start,
                            "example_movement_window_end_time": example_movement_end,
                            "template_movement_window_source": "",
                            "template_movement_window_start_time": 0.0,
                            "template_movement_window_end_time": 0.0,
                            "peak_score": float(segment[peak_offset]),
                            "start_step": seg_start,
                            "end_step": seg_end,
                            "start_time": max(0.0, seg_start * step_sec - trim_start),
                            "end_time": max(0.0, seg_end * step_sec - trim_start),
                            "original_start_time": seg_start * step_sec,
                            "original_end_time": seg_end * step_sec,
                            "original_time_interval": f"[{seg_start * step_sec:.6f}, {seg_end * step_sec:.6f}]",
                            "time_interval": (
                                f"[{max(0.0, seg_start * step_sec - trim_start):.6f}, "
                                f"{max(0.0, seg_end * step_sec - trim_start):.6f}]"
                            ),
                            "sim_curve": score_curve.copy(),
                            "sentence_duration": sentence_duration,
                            "sentence_steps": sentence_steps,
                            "step_sec": step_sec,
                            "ncc_stride": 1,
                            "curve_time_original": curve_start_seconds[: score_curve.size].copy(),
                            "curve_valid_mask": valid_score_mask.copy(),
                            "trim_start_seconds": trim_start,
                            "trim_end_seconds": trim_end,
                            "usable_end_seconds": usable_end,
                            "usable_duration_seconds": usable_duration,
                            "model_score_threshold": threshold,
                            "model_score_gloss_id": int(gloss_id),
                        }
                    )

        if not ranked_results:
            raise RuntimeError(
                f"No pseudo-gloss score segments were found for candidate glosses: {candidate_glosses}"
            )
        ranked_results.sort(
            key=lambda item: (
                -float(item["peak_score"]),
                int(item["sort_index"]),
            )
        )
        for rank, item in enumerate(ranked_results, start=1):
            item["rank"] = rank
        return ranked_results

    def save_ncc_plot(
        plot_path: Path,
        *,
        word_name: str,
        source_video_id: str,
        prediction: Dict[str, object],
        topk_plot_items: Sequence[Dict[str, object]],
        topk_best: Dict[int, Dict[str, object]],
        gt_rows: Sequence[Dict[str, object]],
        best_iou: float,
    ) -> None:
        sim_curve = np.asarray(prediction["sim_curve"], dtype=np.float32)
        x_seconds = np.asarray(prediction["curve_time_original"], dtype=np.float32)
        valid_mask = np.asarray(prediction["curve_valid_mask"], dtype=bool)
        score_label = (
            "pseudo-gloss score"
            if args.prediction_method == "model_scores"
            else "NCC score"
        )
        search_label = (
            "pseudo-gloss search region"
            if args.prediction_method == "model_scores"
            else "NCC search region"
        )
        peak_label = (
            "score peak"
            if args.prediction_method == "model_scores"
            else "NCC peak"
        )

        fig, (ax, region_ax) = plt.subplots(
            2,
            1,
            figsize=(11.5, 5.8),
            sharex=True,
            gridspec_kw={"height_ratios": [3.2, 1.6]},
        )
        ax.plot(x_seconds, sim_curve, color="#9AA0A6", lw=1.0, label=score_label)
        if np.any(valid_mask):
            ax.plot(
                x_seconds[valid_mask],
                sim_curve[valid_mask],
                color="#174EA6",
                lw=1.9,
                label=search_label,
            )

        pred_start = float(prediction["original_start_time"])
        pred_end = float(prediction["original_end_time"])
        ax.axvspan(pred_start, pred_end, color="#D93025", alpha=0.22, label="predicted span")
        ax.axvline(pred_start, color="#D93025", lw=1.0, ls="--")

        peak_idx = int(sim_curve.argmax()) if sim_curve.size else 0
        if np.any(valid_mask):
            peak_idx = int(np.where(valid_mask, sim_curve, -np.inf).argmax())
        peak_time = float(x_seconds[peak_idx]) if sim_curve.size else pred_start
        peak_score = float(prediction["peak_score"])
        ax.scatter([peak_time], [peak_score], color="#D93025", s=28, zorder=5, label=peak_label)

        for idx, gt in enumerate(gt_rows):
            gt_start = float(gt["start_time"])
            gt_end = float(gt["end_time"])
            ax.axvspan(
                gt_start,
                gt_end,
                color="#188038",
                alpha=0.18,
                label="GT span" if idx == 0 else None,
            )

        duration = float(prediction["sentence_duration"])
        trim_start = float(prediction["trim_start_seconds"])
        usable_end = float(prediction["usable_end_seconds"])
        if trim_start > 0:
            ax.axvspan(0.0, trim_start, color="#F29900", alpha=0.15, label="ignored trim")
        if usable_end < duration:
            ax.axvspan(usable_end, duration, color="#F29900", alpha=0.15)
        if duration > 0:
            ax.set_xlim(0.0, duration)
        ax.set_ylabel(score_label)
        topk_title = " | ".join(
            f"top{top_k}={float(topk_best[top_k]['iou']):.3f}"
            for top_k in TOP_K_EVAL_VALUES
            if top_k in topk_best
        )
        if not topk_title:
            topk_title = f"top1={best_iou:.3f}"
        if args.prediction_method == "model_scores":
            source_label = f"model={prediction['predicted_gloss']}"
        else:
            source_label = f"iso={prediction['isolated_video_id']}"
        ax.set_title(
            f"{word_name} | {source_video_id} | {source_label} | "
            f"peak={peak_score:.3f} | {topk_title} | trim={trim_start:.2f}s"
        )
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best", fontsize=8)
        ax.tick_params(axis="x", labelbottom=False)

        topk_colors = ["#D93025", "#F29900", "#9334E6", "#00ACC1", "#5F6368"]
        best_ranks_by_k = {
            top_k: int(item["prediction"]["rank"])
            for top_k, item in topk_best.items()
            if item.get("prediction") is not None
        }
        region_rows: List[Tuple[float, str, str, float, float, float]] = []
        for idx, item in enumerate(topk_plot_items):
            candidate_prediction = item.get("prediction")
            if candidate_prediction is None:
                continue
            candidate_rank = int(candidate_prediction["rank"])
            candidate_start = float(candidate_prediction["original_start_time"])
            candidate_end = float(candidate_prediction["original_end_time"])
            if candidate_end <= candidate_start:
                continue
            tags = [
                f"top{top_k}"
                for top_k, rank in best_ranks_by_k.items()
                if rank == candidate_rank
            ]
            tag_suffix = f" {'/'.join(tags)}" if tags else ""
            label = (
                f"#{candidate_rank} IoU={float(item['iou']):.3f} "
                f"score={float(candidate_prediction['peak_score']):.3f}{tag_suffix}"
            )
            color = topk_colors[idx % len(topk_colors)]
            region_rows.append(
                (
                    float(len(region_rows) + 1),
                    label,
                    color,
                    candidate_start,
                    candidate_end,
                    float(candidate_prediction["peak_score"]),
                )
            )

        gt_y = 0.0
        for idx, gt in enumerate(gt_rows):
            gt_start = float(gt["start_time"])
            gt_end = float(gt["end_time"])
            if gt_end <= gt_start:
                continue
            region_ax.broken_barh(
                [(gt_start, gt_end - gt_start)],
                (gt_y - 0.28, 0.56),
                facecolors="#188038",
                alpha=0.45,
                label="GT spans" if idx == 0 else None,
            )

        for y, label, color, start, end, _ in region_rows:
            region_ax.broken_barh(
                [(start, end - start)],
                (y - 0.28, 0.56),
                facecolors=color,
                alpha=0.62,
            )
            region_ax.axvline(start, color=color, lw=0.8, ls="--", alpha=0.8)
            region_ax.text(
                min(end, duration) if duration > 0 else end,
                y,
                f" {start:.2f}-{end:.2f}s",
                va="center",
                ha="left",
                fontsize=7,
                color="#202124",
            )

        region_ax.set_yticks([gt_y] + [row[0] for row in region_rows])
        region_ax.set_yticklabels(["GT"] + [row[1] for row in region_rows], fontsize=8)
        if duration > 0:
            region_ax.set_xlim(0.0, duration)
        region_ax.set_ylim(-0.75, max(1.0, float(len(region_rows)) + 0.75))
        region_ax.set_xlabel("original time in example video (seconds)")
        region_ax.set_ylabel("top-k")
        region_ax.grid(True, axis="x", alpha=0.2)
        for spine in ("top", "right"):
            region_ax.spines[spine].set_visible(False)
        fig.tight_layout()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)

    try:
        evaluation_rows: List[Dict[str, object]] = []
        predicted_label_rows: List[Dict[str, object]] = []
        ious: List[float] = []
        topk_ious: Dict[int, List[float]] = {top_k: [] for top_k in TOP_K_EVAL_VALUES}
        wer_values: List[float] = []
        wer_total_reference_words = 0
        wer_total_hypothesis_words = 0
        wer_total_edits = 0
        wer_examples_with_reference = 0
        wer_examples_without_reference = 0

        sorted_label_items = sorted(labels_by_example.items())
        for eval_index, ((annotation_word_key, source_video_id, example_video_number), gt_rows) in enumerate(sorted_label_items, start=1):
            if eval_index == 1 or eval_index % 10 == 0 or eval_index == len(sorted_label_items):
                log_status(
                    f"evaluate: example {eval_index}/{len(sorted_label_items)} "
                    f"{annotation_word_key} source={source_video_id or example_video_number}"
                )
            group = lookup_word_entry(annotation_word_key, word_lookup)
            if group is None:
                group = lookup_word_entry(annotation_word_key, example_lookup)
            if group is None:
                raise KeyError(f"No metadata entry matches annotation word '{annotation_word_key}'.")

            local_example = find_example_for_annotation(group, source_video_id, example_video_number)
            example_video_path = None
            if local_example is not None and local_example.get("path"):
                example_video_path = Path(local_example["path"])

            if example_video_path is None or not example_video_path.exists():
                # Backward-compatible fallback for old sampled manifests.
                example_rows = group.get("rows", []) if isinstance(group, dict) else []
                if example_rows and 1 <= example_video_number <= len(example_rows):
                    example_row = example_rows[example_video_number - 1]
                    for key in ("output_example_video", "source_example_video"):
                        example_video_path = resolve_existing_path(example_row.get(key, ""), script_root)
                        if example_video_path is not None:
                            break
            if example_video_path is None or not example_video_path.exists():
                raise FileNotFoundError(
                    f"Could not resolve local example video for '{annotation_word_key}' "
                    f"source_video_id='{source_video_id}' example #{example_video_number}."
                )

            sentence_item = load_hidden(
                example_video_path,
                lmdb_stride=1,
                max_len=None,
                cache=sentence_cache,
            )
            reference_tokens = transcription_tokens_in_vocab(
                str(local_example.get("transcription_text", "")) if local_example is not None else ""
            )
            if reference_tokens:
                hypothesis_tokens = decode_reference_vocab_sequence(sentence_item, reference_tokens)
                edit_distance = word_edit_distance(reference_tokens, hypothesis_tokens)
                wer_value = edit_distance / float(len(reference_tokens))
                wer_values.append(wer_value)
                wer_total_reference_words += len(reference_tokens)
                wer_total_hypothesis_words += len(hypothesis_tokens)
                wer_total_edits += edit_distance
                wer_examples_with_reference += 1
            else:
                hypothesis_tokens = []
                edit_distance = 0
                wer_value = None
                wer_examples_without_reference += 1

            candidate_glosses = [
                gloss for gloss in candidate_glosses_for_group(group, gt_rows)
                if gloss in lem_to_id
            ]
            if args.prediction_method == "model_scores":
                ranked_predictions = ranked_model_score_predictions_for_example(
                    sentence_item,
                    candidate_glosses,
                    example_video_path=example_video_path,
                    source_video_id=source_video_id,
                )
            else:
                isolated_candidates = list(group.get("isolated_videos") or [])
                ranked_predictions = ranked_predictions_for_example(
                    sentence_item,
                    candidate_glosses,
                    isolated_candidates,
                    example_video_path=example_video_path,
                    source_video_id=source_video_id,
                )
            prediction = ranked_predictions[0]

            normalized_gt_rows: List[Dict[str, object]] = []
            gt_intervals: List[str] = []
            gt_original_intervals: List[str] = []
            trim_start = float(prediction["trim_start_seconds"])
            trim_end = float(prediction["trim_end_seconds"])
            usable_end = float(prediction["usable_end_seconds"])
            usable_duration = float(prediction["usable_duration_seconds"])
            for gt in gt_rows:
                original_gt_start = float(gt["start_time"])
                original_gt_end = float(gt["end_time"])
                gt_original_intervals.append(f"[{original_gt_start:.6f}, {original_gt_end:.6f}]")
                clipped_gt_start = max(original_gt_start, trim_start)
                clipped_gt_end = min(original_gt_end, usable_end)
                if clipped_gt_end > clipped_gt_start:
                    gt_start = clipped_gt_start - trim_start
                    gt_end = clipped_gt_end - trim_start
                else:
                    gt_start = max(0.0, min(original_gt_start - trim_start, usable_duration))
                    gt_end = gt_start
                gt_time_interval = f"[{gt_start:.6f}, {gt_end:.6f}]"
                gt_original_time_interval = f"[{original_gt_start:.6f}, {original_gt_end:.6f}]"
                gt_intervals.append(gt_time_interval)
                normalized_gt_rows.append(
                    {
                        "gt": gt,
                        "start_time": gt_start,
                        "end_time": gt_end,
                        "time_interval": gt_time_interval,
                        "original_time_interval": gt_original_time_interval,
                    }
                )

            def best_iou_for_prediction(
                candidate_prediction: Dict[str, object],
            ) -> Tuple[float, Optional[Dict[str, object]]]:
                best_candidate_iou = -1.0
                best_candidate_gt = None
                for normalized_gt in normalized_gt_rows:
                    current_iou = interval_iou_seconds(
                        float(candidate_prediction["start_time"]),
                        float(candidate_prediction["end_time"]),
                        float(normalized_gt["start_time"]),
                        float(normalized_gt["end_time"]),
                    )
                    if current_iou > best_candidate_iou:
                        best_candidate_iou = current_iou
                        best_candidate_gt = normalized_gt
                return best_candidate_iou, best_candidate_gt

            topk_best: Dict[int, Dict[str, object]] = {}
            for top_k in TOP_K_EVAL_VALUES:
                best_topk_iou = -1.0
                best_topk_prediction: Optional[Dict[str, object]] = None
                best_topk_gt: Optional[Dict[str, object]] = None
                for candidate_prediction in ranked_predictions[:top_k]:
                    candidate_iou, candidate_gt = best_iou_for_prediction(candidate_prediction)
                    if candidate_iou > best_topk_iou:
                        best_topk_iou = candidate_iou
                        best_topk_prediction = candidate_prediction
                        best_topk_gt = candidate_gt
                topk_best[top_k] = {
                    "iou": best_topk_iou,
                    "prediction": best_topk_prediction,
                    "gt": best_topk_gt,
                    "candidate_count": min(top_k, len(ranked_predictions)),
                }
                topk_ious[top_k].append(best_topk_iou)

            top1_best = topk_best[1]
            best_iou = float(top1_best["iou"])
            best_gt = top1_best["gt"]
            best_gt_time_interval = str(best_gt["time_interval"]) if best_gt is not None else ""
            best_gt_original_time_interval = (
                str(best_gt["original_time_interval"]) if best_gt is not None else ""
            )
            topk_plot_items = []
            for candidate_prediction in ranked_predictions[: max(TOP_K_EVAL_VALUES)]:
                candidate_iou, candidate_gt = best_iou_for_prediction(candidate_prediction)
                topk_plot_items.append(
                    {
                        "prediction": candidate_prediction,
                        "iou": candidate_iou,
                        "gt": candidate_gt,
                    }
                )

            ious.append(best_iou)
            plot_path = ""
            if not args.no_plots:
                safe_word = ascii_slug(str(gt_rows[0]["word_name"])) or "word"
                safe_source = ascii_slug(source_video_id) or f"example_{example_video_number}"
                safe_iso = ascii_slug(
                    str(
                        prediction.get("isolated_video_id")
                        or prediction.get("predicted_gloss")
                        or args.prediction_method
                    )
                )
                plot_file = (
                    f"{safe_word}_ex{example_video_number}_{safe_source}_"
                    f"{args.prediction_method}_{safe_iso}.png"
                )
                full_plot_path = plot_dir / plot_file
                save_ncc_plot(
                    full_plot_path,
                    word_name=str(gt_rows[0]["word_name"]),
                    source_video_id=source_video_id,
                    prediction=prediction,
                    topk_plot_items=topk_plot_items,
                    topk_best=topk_best,
                    gt_rows=gt_rows,
                    best_iou=best_iou,
                )
                plot_path = str(full_plot_path)

            topk_eval_fields: Dict[str, object] = {}
            for top_k in (3, 5):
                topk_prediction = topk_best[top_k]["prediction"]
                prefix = f"top{top_k}"
                topk_eval_fields[f"{prefix}_iou"] = f"{float(topk_best[top_k]['iou']):.6f}"
                topk_eval_fields[f"{prefix}_candidate_count"] = topk_best[top_k]["candidate_count"]
                topk_eval_fields[f"{prefix}_best_rank"] = (
                    topk_prediction["rank"] if topk_prediction is not None else ""
                )
                topk_eval_fields[f"{prefix}_best_peak_score"] = (
                    f"{float(topk_prediction['peak_score']):.6f}" if topk_prediction is not None else ""
                )
                topk_eval_fields[f"{prefix}_best_time_interval"] = (
                    topk_prediction["time_interval"] if topk_prediction is not None else ""
                )
                topk_eval_fields[f"{prefix}_best_original_time_interval"] = (
                    topk_prediction["original_time_interval"] if topk_prediction is not None else ""
                )

            eval_row = {
                "prediction_method": args.prediction_method,
                "ncc_embedding_source": ncc_embedding_source,
                "ncc_peak_position": prediction.get("ncc_peak_position", ""),
                "word_name": gt_rows[0]["word_name"],
                "annotation_word_key": annotation_word_key,
                "example_video_number": example_video_number,
                "source_video_id": source_video_id,
                "example_video_path": str(example_video_path),
                "matched_lemmas": " ".join(candidate_glosses),
                "matched_lemmas_in_vocab": prediction.get("matched_lemmas_in_vocab", ""),
                "predicted_gloss": prediction["predicted_gloss"],
                "isolated_video_id": prediction["isolated_video_id"],
                "isolated_video_path": prediction["isolated_video_path"],
                "isolated_duration_seconds": f"{float(prediction['isolated_duration_seconds']):.6f}",
                "isolated_source_duration_seconds": f"{float(prediction['isolated_source_duration_seconds']):.6f}",
                "isolated_steps": prediction["isolated_steps"],
                "isolated_trimmed_steps": prediction["isolated_trimmed_steps"],
                "iso_trim_start_seconds": f"{float(prediction['iso_trim_start_seconds']):.6f}",
                "iso_trim_end_seconds": f"{float(prediction['iso_trim_end_seconds']):.6f}",
                "iso_trim_start_steps": prediction["iso_trim_start_steps"],
                "iso_trim_end_steps": prediction["iso_trim_end_steps"],
                "example_movement_window_source": prediction.get("example_movement_window_source", ""),
                "example_movement_window_start_time": prediction.get("example_movement_window_start_time", ""),
                "example_movement_window_end_time": prediction.get("example_movement_window_end_time", ""),
                "template_movement_window_source": prediction.get("template_movement_window_source", ""),
                "template_movement_window_start_time": f"{float(prediction.get('template_movement_window_start_time') or 0.0):.6f}",
                "template_movement_window_end_time": f"{float(prediction.get('template_movement_window_end_time') or 0.0):.6f}",
                "predicted_time_interval": prediction["time_interval"],
                "predicted_original_time_interval": prediction["original_time_interval"],
                "predicted_start_time": f"{float(prediction['start_time']):.6f}",
                "predicted_end_time": f"{float(prediction['end_time']):.6f}",
                "predicted_original_start_time": f"{float(prediction['original_start_time']):.6f}",
                "predicted_original_end_time": f"{float(prediction['original_end_time']):.6f}",
                "peak_score": f"{float(prediction['peak_score']):.6f}",
                "gt_time_intervals": "; ".join(gt_intervals),
                "gt_original_time_intervals": "; ".join(gt_original_intervals),
                "best_gt_time_interval": best_gt_time_interval if best_gt is not None else "",
                "best_gt_original_time_interval": best_gt_original_time_interval if best_gt is not None else "",
                "best_iou": f"{best_iou:.6f}",
                "top1_iou": f"{float(topk_best[1]['iou']):.6f}",
                **topk_eval_fields,
                "trim_start_seconds": f"{trim_start:.6f}",
                "trim_end_seconds": f"{trim_end:.6f}",
                "ncc_plot_path": plot_path,
                "gt_segment_count": len(gt_rows),
                "checkpoint": checkpoint_label,
            }
            evaluation_rows.append(eval_row)
            predicted_label_rows.append(
                {
                    "prediction_method": args.prediction_method,
                    "ncc_peak_position": prediction.get("ncc_peak_position", ""),
                    "word_name": gt_rows[0]["word_name"],
                    "example_video_number": example_video_number,
                    "source_video_id": source_video_id,
                    "time_interval": prediction["time_interval"],
                    "original_time_interval": prediction["original_time_interval"],
                    "predicted_gloss": prediction["predicted_gloss"],
                    "isolated_video_id": prediction["isolated_video_id"],
                    "iso_trim_start_seconds": f"{float(prediction['iso_trim_start_seconds']):.6f}",
                    "iso_trim_end_seconds": f"{float(prediction['iso_trim_end_seconds']):.6f}",
                    "example_movement_window_source": prediction.get("example_movement_window_source", ""),
                    "example_movement_window_start_time": prediction.get("example_movement_window_start_time", ""),
                    "example_movement_window_end_time": prediction.get("example_movement_window_end_time", ""),
                    "template_movement_window_source": prediction.get("template_movement_window_source", ""),
                    "template_movement_window_start_time": f"{float(prediction.get('template_movement_window_start_time') or 0.0):.6f}",
                    "template_movement_window_end_time": f"{float(prediction.get('template_movement_window_end_time') or 0.0):.6f}",
                    "peak_score": f"{float(prediction['peak_score']):.6f}",
                    "trim_start_seconds": f"{trim_start:.6f}",
                    "trim_end_seconds": f"{trim_end:.6f}",
                    "ncc_plot_path": plot_path,
                }
            )

        eval_csv = output_dir / "ncc_iou_eval.csv"
        pred_csv = output_dir / "ncc_predicted_labels.csv"
        skipped_vocab_csv = output_dir / "ncc_skipped_out_of_vocab.csv"
        summary_json = output_dir / "ncc_iou_summary.json"

        write_csv_rows(
            eval_csv,
            evaluation_rows,
            fieldnames=[
                "prediction_method",
                "ncc_embedding_source",
                "ncc_peak_position",
                "word_name",
                "annotation_word_key",
                "example_video_number",
                "source_video_id",
                "example_video_path",
                "matched_lemmas",
                "matched_lemmas_in_vocab",
                "predicted_gloss",
                "isolated_video_id",
                "isolated_video_path",
                "isolated_duration_seconds",
                "isolated_source_duration_seconds",
                "isolated_steps",
                "isolated_trimmed_steps",
                "iso_trim_start_seconds",
                "iso_trim_end_seconds",
                "iso_trim_start_steps",
                "iso_trim_end_steps",
                "example_movement_window_source",
                "example_movement_window_start_time",
                "example_movement_window_end_time",
                "template_movement_window_source",
                "template_movement_window_start_time",
                "template_movement_window_end_time",
                "predicted_time_interval",
                "predicted_original_time_interval",
                "predicted_start_time",
                "predicted_end_time",
                "predicted_original_start_time",
                "predicted_original_end_time",
                "peak_score",
                "gt_time_intervals",
                "gt_original_time_intervals",
                "best_gt_time_interval",
                "best_gt_original_time_interval",
                "best_iou",
                "top1_iou",
                "top3_iou",
                "top3_candidate_count",
                "top3_best_rank",
                "top3_best_peak_score",
                "top3_best_time_interval",
                "top3_best_original_time_interval",
                "top5_iou",
                "top5_candidate_count",
                "top5_best_rank",
                "top5_best_peak_score",
                "top5_best_time_interval",
                "top5_best_original_time_interval",
                "trim_start_seconds",
                "trim_end_seconds",
                "ncc_plot_path",
                "gt_segment_count",
                "checkpoint",
            ],
        )
        write_csv_rows(
            pred_csv,
            predicted_label_rows,
            fieldnames=[
                "prediction_method",
                "ncc_peak_position",
                "word_name",
                "example_video_number",
                "source_video_id",
                "time_interval",
                "original_time_interval",
                "predicted_gloss",
                "isolated_video_id",
                "iso_trim_start_seconds",
                "iso_trim_end_seconds",
                "example_movement_window_source",
                "example_movement_window_start_time",
                "example_movement_window_end_time",
                "template_movement_window_source",
                "template_movement_window_start_time",
                "template_movement_window_end_time",
                "peak_score",
                "trim_start_seconds",
                "trim_end_seconds",
                "ncc_plot_path",
            ],
        )
        if skipped_out_of_vocab_rows:
            write_csv_rows(
                skipped_vocab_csv,
                skipped_out_of_vocab_rows,
                fieldnames=[
                    "word_name",
                    "annotation_word_key",
                    "source_video_id",
                    "example_video_number",
                    "candidate_glosses",
                    "reason",
                ],
            )

        topk_summary: Dict[str, object] = {
            "top_k_eval_values": list(TOP_K_EVAL_VALUES),
        }
        for top_k in TOP_K_EVAL_VALUES:
            values = topk_ious[top_k]
            prefix = f"top{top_k}"
            topk_summary[f"{prefix}_mean_iou"] = float(statistics.fmean(values)) if values else 0.0
            topk_summary[f"{prefix}_median_iou"] = float(statistics.median(values)) if values else 0.0
            topk_summary[f"{prefix}_max_iou"] = float(max(values)) if values else 0.0
            topk_summary[f"{prefix}_min_iou"] = float(min(values)) if values else 0.0
            topk_summary[f"{prefix}_iou_at_0_10"] = sum(1 for value in values if value >= 0.10)
            topk_summary[f"{prefix}_iou_at_0_25"] = sum(1 for value in values if value >= 0.25)
            topk_summary[f"{prefix}_iou_at_0_50"] = sum(1 for value in values if value >= 0.50)

        summary = {
            "annotation_dir": str(annotation_dir),
            "example_manifest": str(example_manifest_path) if example_manifest_path is not None else "",
            "tid_raw_roots": [str(root) for root in tid_raw_roots],
            "checkpoint": checkpoint_label,
            "checkpoint_mode": checkpoint_mode,
            "prediction_method": args.prediction_method,
            "ncc_embedding_source": ncc_embedding_source,
            "ncc_peak_position": (
                args.ncc_peak_position if args.prediction_method == "ncc" else ""
            ),
            "movement_windows_csv": str(movement_windows_path) if movement_windows_path is not None else "",
            "movement_window_apply": args.movement_window_apply,
            "movement_window_buffer_seconds": args.movement_window_buffer_seconds,
            "movement_window_min_duration_seconds": args.movement_window_min_duration_seconds,
            "movement_window_row_count": len(movement_window_rows),
            **movement_stats,
            "num_examples": len(evaluation_rows),
            "mean_iou": float(statistics.fmean(ious)) if ious else 0.0,
            "median_iou": float(statistics.median(ious)) if ious else 0.0,
            "max_iou": float(max(ious)) if ious else 0.0,
            "min_iou": float(min(ious)) if ious else 0.0,
            "trim_seconds": args.trim_seconds,
            "trim_start_seconds": args.trim_start_seconds if args.trim_start_seconds is not None else args.trim_seconds,
            "trim_end_seconds": args.trim_end_seconds if args.trim_end_seconds is not None else args.trim_seconds,
            "iso_trim_seconds": args.iso_trim_seconds,
            "iso_trim_start_seconds": args.iso_trim_start_seconds if args.iso_trim_start_seconds is not None else args.iso_trim_seconds,
            "iso_trim_end_seconds": args.iso_trim_end_seconds if args.iso_trim_end_seconds is not None else args.iso_trim_seconds,
            "model_score_relative_threshold": args.model_score_relative_threshold,
            "model_score_absolute_threshold": args.model_score_absolute_threshold,
            "model_score_smooth_window": args.model_score_smooth_window,
            "model_score_min_segment_steps": args.model_score_min_segment_steps,
            "iou_at_0_10": sum(1 for value in ious if value >= 0.10),
            "iou_at_0_25": sum(1 for value in ious if value >= 0.25),
            "iou_at_0_50": sum(1 for value in ious if value >= 0.50),
            "evaluation_csv": str(eval_csv),
            "predicted_labels_csv": str(pred_csv),
            "skipped_out_of_vocab_count": len(skipped_out_of_vocab_rows),
            "skipped_out_of_vocab_csv": (
                str(skipped_vocab_csv) if skipped_out_of_vocab_rows else ""
            ),
            "ncc_plot_dir": "" if args.no_plots else str(plot_dir),
            "wer": {
                "scope": "transcription_text tokens present in the pseudo-gloss vocab",
                "decode_method": "temporal segments over reference-vocab score rows",
                "segment_relative_threshold": WER_SEGMENT_RELATIVE_THRESHOLD,
                "segment_absolute_threshold": WER_SEGMENT_ABSOLUTE_THRESHOLD,
                "smooth_window": WER_SEGMENT_SMOOTH_WINDOW,
                "num_examples_scored": wer_examples_with_reference,
                "num_examples_skipped_empty_reference": wer_examples_without_reference,
                "total_reference_words": wer_total_reference_words,
                "total_hypothesis_words": wer_total_hypothesis_words,
                "total_edit_distance": wer_total_edits,
                "corpus_wer": (
                    wer_total_edits / float(wer_total_reference_words)
                    if wer_total_reference_words > 0
                    else 0.0
                ),
                "mean_example_wer": float(statistics.fmean(wer_values)) if wer_values else 0.0,
                "median_example_wer": float(statistics.median(wer_values)) if wer_values else 0.0,
                "max_example_wer": float(max(wer_values)) if wer_values else 0.0,
                "min_example_wer": float(min(wer_values)) if wer_values else 0.0,
            },
        }
        summary.update(topk_summary)

        with summary_json.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        return summary
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export TID overlap annotations as labels and optionally evaluate NCC predictions with IoU."
    )
    parser.add_argument(
        "--mode",
        choices=["export_labels", "evaluate", "both"],
        default="export_labels",
        help="Run only label export, only evaluation, or both in sequence.",
    )
    parser.add_argument(
        "--annotation_dir",
        default=DEFAULT_ANNOTATION_DIR,
        help="Directory containing VIA *_export.csv files.",
    )
    parser.add_argument(
        "--label_dir",
        default=DEFAULT_LABEL_DIR,
        help="Output directory for normalized label CSVs.",
    )
    parser.add_argument(
        "--word_manifest",
        default=DEFAULT_WORD_MANIFEST,
        help="Optional word manifest used to recover display names from annotation stems.",
    )
    parser.add_argument(
        "--example_manifest",
        default=DEFAULT_EXAMPLE_MANIFEST,
        help="Fallback CSV that maps sampled words to example videos when word_info metadata is incomplete.",
    )
    parser.add_argument(
        "--tid_raw_roots",
        default=DEFAULT_TID_RAW_ROOTS,
        help="Comma-separated roots containing TID word directories with word_info.json.",
    )

    parser.add_argument("--config", default="configs/tslnews/tslnews_stage1_configs/TSL_example_s1_dyn_config.py")
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
        help=(
            "For --prediction_method ncc only: instantiate the model from the "
            "config and do not load any Sign2GPT checkpoint."
        ),
    )
    parser.add_argument("--vocab_pkl", default="data/TSL-News/6538_gloss/processed_words_map.tsl_pkl")
    parser.add_argument("--output_dir", default="outputs/spotting/ncc")
    parser.add_argument(
        "--tmp_dir",
        default=None,
        help=(
            "Directory for temporary per-video LMDBs. Defaults to the system temp "
            "directory instead of --output_dir to avoid LMDB issues on bind mounts."
        ),
    )
    parser.add_argument(
        "--movement_windows_csv",
        default=None,
        help=(
            "CSV produced by scripts/tid/build_movement_windows.py or "
            "scripts/tid/build_visual_motion_windows.py. When enabled, "
            "example and/or isolated hidden sequences are restricted to "
            "movement_start_time/movement_end_time."
        ),
    )
    parser.add_argument(
        "--movement_window_apply",
        choices=["none", "examples", "isolated", "both"],
        default="none",
        help="Which sides of evaluation should use movement-window CSV bounds.",
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
        "--isolated_video_template",
        default=None,
        help="Fallback template for isolated gloss videos when local word_info has no isolated_videos.",
    )
    parser.add_argument(
        "--isolated_video_dir",
        default=None,
        help="Fallback flat directory containing isolated videos as <gloss>.mp4.",
    )
    parser.add_argument(
        "--iso_root_dir",
        default=None,
        help="Fallback root directory containing one subfolder per gloss with mp4 files inside.",
    )

    parser.add_argument(
        "--prediction_method",
        choices=["ncc", "model_scores"],
        default="ncc",
        help=(
            "Use isolated-video NCC localization, or segment the stage-1 "
            "pseudo-gloss score curves directly without isolated videos."
        ),
    )
    parser.add_argument("--stride", type=int, default=1, help="Sliding stride in feature steps.")
    parser.add_argument(
        "--ncc_embedding_source",
        choices=["auto", "encoder", "spatial"],
        default="auto",
        help=(
            "Feature sequence used by NCC. auto uses trained encoder hidden states "
            "for checkpointed runs and raw spatial backbone features for --no_checkpoint."
        ),
    )
    parser.add_argument("--iso_stride_frames", type=int, default=3, help="Frame subsample stride for isolated video LMDB building.")
    parser.add_argument("--iso_max_len", type=int, default=12, help="Max frames after subsample for isolated videos.")
    parser.add_argument("--ncc_ignore_frames", type=int, default=0, help="Ignore this many frames from both ends of the isolated hidden sequence.")
    parser.add_argument(
        "--ncc_peak_position",
        choices=["start", "center"],
        default="start",
        help=(
            "For --prediction_method ncc, how to convert an NCC peak index into "
            "a prediction window. start treats the peak as the sliding-window "
            "start; center treats it as the center of the predicted window."
        ),
    )
    parser.add_argument(
        "--model_score_relative_threshold",
        type=float,
        default=0.2,
        help="For --prediction_method model_scores, keep score segments above this fraction of the class peak.",
    )
    parser.add_argument(
        "--model_score_absolute_threshold",
        type=float,
        default=0.0,
        help="For --prediction_method model_scores, minimum absolute score threshold before forming segments.",
    )
    parser.add_argument(
        "--model_score_smooth_window",
        type=int,
        default=3,
        help="For --prediction_method model_scores, moving-average window over each pseudo-gloss score curve.",
    )
    parser.add_argument(
        "--model_score_min_segment_steps",
        type=int,
        default=1,
        help="For --prediction_method model_scores, discard score segments shorter than this many model time steps.",
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
        "--trim_seconds",
        type=float,
        default=0.5,
        help="Ignore this many seconds at both start and end of each continuous example video.",
    )
    parser.add_argument(
        "--trim_start_seconds",
        type=float,
        default=None,
        help="Override --trim_seconds for only the start of each continuous example video.",
    )
    parser.add_argument(
        "--trim_end_seconds",
        type=float,
        default=None,
        help="Override --trim_seconds for only the end of each continuous example video.",
    )
    parser.add_argument(
        "--plot_dir",
        default="ncc_plots",
        help="Directory, relative to --output_dir, where per-example NCC-over-time PNG plots are saved.",
    )
    parser.add_argument(
        "--no_plots",
        action="store_true",
        help="Disable saving per-example NCC-over-time PNG plots.",
    )
    parser.add_argument(
        "--ncc_weighting",
        choices=["none", "triangle"],
        default="none",
        help="Temporal weighting used in NCC over isolated video frames.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    explicit_word_manifest = any(
        arg == "--word_manifest" or arg.startswith("--word_manifest=")
        for arg in sys.argv[1:]
    )

    script_root = Path(__file__).resolve().parents[1]
    annotation_dir = resolve_existing_path(args.annotation_dir, script_root)
    label_dir = resolve_path(args.label_dir, script_root)
    word_manifest_path = resolve_existing_path(args.word_manifest, script_root)
    tid_raw_roots = resolve_tid_raw_roots(args.tid_raw_roots, script_root)

    if annotation_dir is None:
        parser.error(f"Annotation directory not found: {args.annotation_dir}")
    if explicit_word_manifest and word_manifest_path is None:
        parser.error(f"Word manifest not found: {args.word_manifest}")

    outputs: Dict[str, object] = {}
    if args.mode in {"export_labels", "both"}:
        export_summary = export_labels(
            annotation_dir,
            label_dir,
            word_manifest_path,
            tid_raw_roots=tid_raw_roots,
            script_root=script_root,
        )
        outputs["export_labels"] = export_summary

    if args.mode in {"evaluate", "both"}:
        eval_summary = evaluate_ncc(args)
        outputs["evaluate"] = eval_summary

    print(json.dumps(outputs, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
