#!/usr/bin/env python3
"""
Copy TID word folders whose vocabulary overlaps with the TSL-News pseudo-gloss vocab.

Default behavior:
  1. Read `pseudo_gloss` tokens from:
       data/TSL-News/annotations/TSL-News.{train,dev,test}.corpus.csv
  2. Filter a conservative Turkish stopword list from the pseudo-gloss vocab.
  3. Scan TID raw entries under:
       data/TID/raw/<LETTER>/<word_dir>/
  4. Keep entries whose TID headword or single-token synonym variants overlap
     with the TSL-News pseudo-gloss vocabulary.
  5. Copy the matched TID word folders into:
       data/TID/overlap_tslnews/raw/<LETTER>/<word_dir>/
  6. Write manifests for later sampling / annotation.

This script intentionally uses a conservative default match rule
(`single-token-or-synonym`) to avoid noisy phrase-token matches such as
`bir daha` -> `bir`.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


TOKEN_RE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü']+")
DEFAULT_STOPWORDS = {
    "ama",
    "bir",
    "bu",
    "çünkü",
    "da",
    "dahi",
    "de",
    "diye",
    "en",
    "et",
    "fakat",
    "gibi",
    "hem",
    "her",
    "iç",
    "ile",
    "ise",
    "kadar",
    "ki",
    "mı",
    "mi",
    "mu",
    "mü",
    "ne",
    "o",
    "ol",
    "önce",
    "sonra",
    "şu",
    "ve",
    "veya",
    "ya",
    "yahut",
}


def tr_lower(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


def tokenize(text: str) -> List[str]:
    return [tok for tok in TOKEN_RE.findall(tr_lower(text or "")) if tok]


def split_word_variants(word: str) -> List[str]:
    cleaned = re.sub(r"[()]", ",", word or "")
    parts = [part.strip() for part in re.split(r"[,;/]+", cleaned) if part.strip()]
    return parts or [word.strip()]


def parse_meaning_index(meaning_dir: Path, data: Dict) -> int:
    idx = data.get("meaning_number")
    if isinstance(idx, int):
        return idx
    if isinstance(idx, str) and idx.isdigit():
        return int(idx)
    match = re.search(r"meaning_(\d+)", meaning_dir.name)
    return int(match.group(1)) if match else 0


def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_word_dirs(tid_root: Path) -> Iterable[Tuple[str, Path]]:
    for letter_dir in sorted(tid_root.iterdir()):
        if not letter_dir.is_dir():
            continue
        for word_dir in sorted(letter_dir.iterdir()):
            if word_dir.is_dir():
                yield letter_dir.name, word_dir


def resolve_video_path(meaning_dir: Path, dataset_type: str, videos: Dict) -> Optional[Path]:
    if dataset_type == "example":
        key = "example_video"
        default_name = "example_video.mp4"
        hint = "example"
    else:
        key = "meaning_video"
        default_name = "meaning_video.mp4"
        hint = "meaning"

    candidate = videos.get(key) if isinstance(videos, dict) else None
    if isinstance(candidate, str) and candidate.strip():
        cand_path = meaning_dir / candidate
        if cand_path.exists():
            return cand_path

    fallback = meaning_dir / default_name
    if fallback.exists():
        return fallback

    mp4s = sorted(meaning_dir.glob("*.mp4"))
    if not mp4s:
        return None

    hinted = [path for path in mp4s if hint in path.name.lower()]
    if hinted:
        return hinted[0]
    if len(mp4s) == 1:
        return mp4s[0]
    return None


def map_output_video_path(source_dir: Path, output_dir: Path, source_video: Optional[Path]) -> Optional[Path]:
    if source_video is None:
        return None
    try:
        return output_dir / source_video.relative_to(source_dir)
    except ValueError:
        return output_dir / source_video.name


def load_tsl_pseudo_gloss_vocab(tsl_root: Path, splits: Sequence[str]) -> Tuple[Counter, int]:
    ann_root = tsl_root / "annotations"
    if not ann_root.exists():
        ann_root = tsl_root
    lemma_counter: Counter = Counter()
    row_count = 0

    for split in splits:
        csv_path = ann_root / f"TSL-News.{split}.corpus.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing TSL-News annotation CSV: {csv_path}")

        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="|")
            if "pseudo_gloss" not in (reader.fieldnames or []):
                raise ValueError(
                    f"'pseudo_gloss' column not found in {csv_path}. "
                    "Run scripts/tslnews/pseudo_gloss_tr.py first."
                )

            for row in reader:
                row_count += 1
                lemma_counter.update(tok for tok in (row.get("pseudo_gloss") or "").split() if tok)

    return lemma_counter, row_count


def load_stopwords(
    stopword_files: Sequence[str],
    extra_stopwords: Sequence[str],
    use_default_stopwords: bool,
) -> set[str]:
    stopwords = set(DEFAULT_STOPWORDS if use_default_stopwords else [])

    for token in extra_stopwords:
        stopwords.update(tokenize(token))

    for stopword_file in stopword_files:
        path = Path(stopword_file)
        if not path.exists():
            raise FileNotFoundError(f"Stopword file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                stopwords.update(tokenize(line))

    return stopwords


def filter_lemma_counter(lemma_counter: Counter, stopwords: set[str]) -> Tuple[Counter, Dict]:
    if not stopwords:
        return lemma_counter, {
            "stopword_filter_enabled": False,
            "active_stopwords": 0,
            "removed_stopword_unique": 0,
            "removed_stopword_occurrences": 0,
            "top_removed_stopwords": [],
        }

    removed = Counter({lemma: count for lemma, count in lemma_counter.items() if lemma in stopwords})
    filtered = Counter({lemma: count for lemma, count in lemma_counter.items() if lemma not in stopwords})
    top_removed = [lemma for lemma, _ in removed.most_common(15)]

    return filtered, {
        "stopword_filter_enabled": True,
        "active_stopwords": len(stopwords),
        "removed_stopword_unique": len(removed),
        "removed_stopword_occurrences": sum(removed.values()),
        "top_removed_stopwords": top_removed,
    }


def build_match_candidates(word: str, match_mode: str) -> Tuple[List[str], List[str]]:
    variants = split_word_variants(word)
    all_tokens = sorted({tok for variant in variants for tok in tokenize(variant)})

    if match_mode == "single-token":
        head_tokens = tokenize(word)
        candidates = [head_tokens[0]] if len(head_tokens) == 1 else []
    elif match_mode == "single-token-or-synonym":
        candidates = sorted(
            {
                tokens[0]
                for tokens in (tokenize(variant) for variant in variants)
                if len(tokens) == 1
            }
        )
    elif match_mode == "any-token":
        candidates = all_tokens
    else:
        raise ValueError(f"Unknown match mode: {match_mode}")

    return candidates, all_tokens


def relpath_str(base: Path, path: Optional[Path]) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def collect_matches(
    tid_root: Path,
    tsl_freq: Counter,
    output_root: Path,
    base_root: Path,
    match_mode: str,
    limit: Optional[int],
) -> Tuple[List[Dict], Dict]:
    matches: List[Dict] = []
    stats = {
        "scanned_word_dirs": 0,
        "matched_word_dirs": 0,
        "matched_word_dirs_before_limit": 0,
        "skipped_missing_word_info": 0,
        "skipped_no_overlap": 0,
    }

    for letter, word_dir in iter_word_dirs(tid_root):
        stats["scanned_word_dirs"] += 1
        word_info = load_json(word_dir / "word_info.json")
        if word_info is None:
            stats["skipped_missing_word_info"] += 1
            word_info = {}

        word = str(word_info.get("word", word_dir.name)).strip() or word_dir.name
        candidates, all_tokens = build_match_candidates(word, match_mode)
        matched_lemmas = sorted(tok for tok in candidates if tok in tsl_freq)
        if not matched_lemmas:
            stats["skipped_no_overlap"] += 1
            continue

        meaning_dirs = [path for path in sorted(word_dir.glob("meaning_*")) if path.is_dir()]
        relative_source_dir = word_dir.relative_to(tid_root)
        output_dir = output_root / "raw" / relative_source_dir
        matched_counts = {lemma: int(tsl_freq[lemma]) for lemma in matched_lemmas}

        matches.append(
            {
                "letter": letter,
                "word": word,
                "word_dir": word_dir.name,
                "sanitized_word": str(word_info.get("sanitized_word", word_dir.name)),
                "source_url": str(word_info.get("source_url", "")),
                "relative_source_dir": relative_source_dir.as_posix(),
                "relative_output_dir": relpath_str(base_root, output_dir),
                "source_dir": relpath_str(base_root, word_dir),
                "candidate_tokens": candidates,
                "all_tokens": all_tokens,
                "matched_lemmas": matched_lemmas,
                "matched_counts": matched_counts,
                "max_tsl_frequency": max(matched_counts.values()),
                "sum_tsl_frequency": sum(matched_counts.values()),
                "meaning_count": len(meaning_dirs),
            }
        )

    matches.sort(
        key=lambda item: (
            -item["max_tsl_frequency"],
            -item["sum_tsl_frequency"],
            tr_lower(item["word"]),
            item["word_dir"],
        )
    )

    stats["matched_word_dirs_before_limit"] = len(matches)
    if limit is not None:
        matches = matches[:limit]

    stats["matched_word_dirs"] = len(matches)
    return matches, stats


def build_manifest_rows(
    matches: Sequence[Dict],
    tid_root: Path,
    output_root: Path,
    base_root: Path,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    word_rows: List[Dict] = []
    example_rows: List[Dict] = []
    meaning_rows: List[Dict] = []

    for match in matches:
        source_dir = tid_root / match["relative_source_dir"]
        output_dir = output_root / "raw" / match["relative_source_dir"]

        counts_str = ";".join(
            f"{lemma}:{match['matched_counts'][lemma]}" for lemma in match["matched_lemmas"]
        )
        word_rows.append(
            {
                "word": match["word"],
                "word_dir": match["word_dir"],
                "letter": match["letter"],
                "matched_lemmas": " ".join(match["matched_lemmas"]),
                "matched_tsl_counts": counts_str,
                "max_tsl_frequency": match["max_tsl_frequency"],
                "sum_tsl_frequency": match["sum_tsl_frequency"],
                "candidate_tokens": " ".join(match["candidate_tokens"]),
                "all_word_tokens": " ".join(match["all_tokens"]),
                "meaning_count": match["meaning_count"],
                "source_url": match["source_url"],
                "source_dir": relpath_str(base_root, source_dir),
                "output_dir": relpath_str(base_root, output_dir),
            }
        )

        for meaning_dir in sorted(path for path in source_dir.glob("meaning_*") if path.is_dir()):
            data = load_json(meaning_dir / "data.json") or {}
            videos = data.get("videos", {}) if isinstance(data.get("videos"), dict) else {}
            meaning_number = parse_meaning_index(meaning_dir, data)

            source_example = resolve_video_path(meaning_dir, "example", videos)
            source_meaning = resolve_video_path(meaning_dir, "meaning", videos)

            output_meaning_dir = output_dir / meaning_dir.name
            output_example = map_output_video_path(source_dir, output_dir, source_example)
            output_meaning = map_output_video_path(source_dir, output_dir, source_meaning)

            common = {
                "word": match["word"],
                "word_dir": match["word_dir"],
                "letter": match["letter"],
                "meaning_number": meaning_number,
                "matched_lemmas": " ".join(match["matched_lemmas"]),
                "matched_tsl_counts": counts_str,
                "max_tsl_frequency": match["max_tsl_frequency"],
                "sum_tsl_frequency": match["sum_tsl_frequency"],
                "source_dir": relpath_str(base_root, source_dir),
                "output_dir": relpath_str(base_root, output_dir),
                "source_meaning_dir": relpath_str(base_root, meaning_dir),
                "output_meaning_dir": relpath_str(base_root, output_meaning_dir),
                "synonym_title": str(data.get("synonym_title", "")),
            }

            example_rows.append(
                {
                    **common,
                    "example_video_exists": bool(source_example),
                    "source_example_video": relpath_str(base_root, source_example),
                    "output_example_video": relpath_str(base_root, output_example),
                    "subtitle": str(
                        data.get("translation_text")
                        or data.get("transcription_text")
                        or data.get("word")
                        or match["word"]
                    ),
                    "transcription_text": str(data.get("transcription_text", "")),
                }
            )

            meaning_rows.append(
                {
                    **common,
                    "meaning_video_exists": bool(source_meaning),
                    "source_meaning_video": relpath_str(base_root, source_meaning),
                    "output_meaning_video": relpath_str(base_root, output_meaning),
                    "label_text": str(data.get("word") or match["word"]),
                    "translation_text": str(data.get("translation_text", "")),
                }
            )

    return word_rows, example_rows, meaning_rows


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def copy_match_dirs(
    matches: Sequence[Dict],
    tid_root: Path,
    output_root: Path,
    copy_mode: str,
) -> Dict:
    stats = {"copied": 0, "linked": 0, "skipped_existing": 0}
    raw_root = output_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    for match in matches:
        src = tid_root / match["relative_source_dir"]
        dst = raw_root / match["relative_source_dir"]
        if dst.exists():
            stats["skipped_existing"] += 1
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        if copy_mode == "copy":
            shutil.copytree(src, dst)
            stats["copied"] += 1
        elif copy_mode == "symlink":
            dst.symlink_to(src, target_is_directory=True)
            stats["linked"] += 1
        else:
            raise ValueError(f"Unsupported copy mode: {copy_mode}")

    return stats


def build_summary(
    matches: Sequence[Dict],
    scan_stats: Dict,
    tsl_freq: Counter,
    tsl_rows: int,
    tsl_unique_before_stopwords: int,
    splits: Sequence[str],
    match_mode: str,
    output_root: Path,
    copy_mode: str,
    word_rows: Sequence[Dict],
    example_rows: Sequence[Dict],
    meaning_rows: Sequence[Dict],
    copy_stats: Optional[Dict],
    stopword_info: Dict,
) -> Dict:
    summary = {
        "tsl_rows_scanned": tsl_rows,
        "tsl_unique_pseudo_glosses_before_stopwords": tsl_unique_before_stopwords,
        "tsl_unique_pseudo_glosses": len(tsl_freq),
        "tsl_splits": list(splits),
        "match_mode": match_mode,
        "copy_mode": copy_mode,
        "output_root": output_root.as_posix(),
        **stopword_info,
        **scan_stats,
        "manifest_word_rows": len(word_rows),
        "manifest_example_rows": len(example_rows),
        "manifest_meaning_rows": len(meaning_rows),
        "top_matches": [
            {
                "word": match["word"],
                "matched_lemmas": match["matched_lemmas"],
                "max_tsl_frequency": match["max_tsl_frequency"],
            }
            for match in matches[:10]
        ],
    }
    if copy_stats is not None:
        summary.update(copy_stats)
    return summary


def print_preview(matches: Sequence[Dict], preview: int) -> None:
    print(f"[preview] showing top {min(preview, len(matches))} overlaps")
    for match in matches[:preview]:
        counts = ", ".join(
            f"{lemma}:{match['matched_counts'][lemma]}" for lemma in match["matched_lemmas"]
        )
        print(
            f"  - {match['word']} ({match['relative_source_dir']}) "
            f"-> [{counts}] | meanings={match['meaning_count']}"
        )


def main() -> None:
    base_root = Path(__file__).resolve().parents[2]
    default_tsl_root = base_root / "data" / "TSL-News"
    default_tid_root = base_root / "data" / "TID" / "raw"
    default_output_root = base_root / "data" / "TID" / "overlap_tslnews"

    parser = argparse.ArgumentParser(
        description="Copy overlapping TID vocabulary based on TSL-News pseudo-gloss vocab."
    )
    parser.add_argument("--tsl_root", default=str(default_tsl_root))
    parser.add_argument("--tid_root", default=str(default_tid_root))
    parser.add_argument("--output_root", default=str(default_output_root))
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev", "test"],
        help="TSL-News splits to scan for pseudo_gloss vocab.",
    )
    parser.add_argument(
        "--match_mode",
        choices=["single-token", "single-token-or-synonym", "any-token"],
        default="single-token-or-synonym",
        help=(
            "How to derive TID lexical candidates. Default keeps only single-token "
            "headwords or single-token synonym variants."
        ),
    )
    parser.add_argument(
        "--copy_mode",
        choices=["copy", "symlink", "none"],
        default="copy",
        help="How to materialize the matched TID folders under output_root.",
    )
    parser.add_argument(
        "--disable_default_stopwords",
        action="store_true",
        help="Disable the built-in Turkish stopword filter.",
    )
    parser.add_argument(
        "--stopword_file",
        action="append",
        default=[],
        help="Optional text file with extra stopwords, one lemma per line.",
    )
    parser.add_argument(
        "--extra_stopword",
        action="append",
        default=[],
        help="Add one extra stopword lemma. Repeat the flag to pass multiple tokens.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit matched word dirs.")
    parser.add_argument("--preview", type=int, default=10, help="How many matches to print.")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Scan and print summary only. No manifests and no copied output.",
    )
    args = parser.parse_args()

    tsl_root = Path(args.tsl_root)
    tid_root = Path(args.tid_root)
    output_root = Path(args.output_root)
    if not tsl_root.exists():
        raise FileNotFoundError(f"TSL-News root not found: {tsl_root}")
    if not tid_root.exists():
        raise FileNotFoundError(f"TID root not found: {tid_root}")

    tsl_freq, tsl_rows = load_tsl_pseudo_gloss_vocab(tsl_root, args.splits)
    tsl_unique_before_stopwords = len(tsl_freq)
    stopwords = load_stopwords(
        stopword_files=args.stopword_file,
        extra_stopwords=args.extra_stopword,
        use_default_stopwords=not args.disable_default_stopwords,
    )
    tsl_freq, stopword_info = filter_lemma_counter(tsl_freq, stopwords)
    if not tsl_freq:
        raise ValueError("No pseudo-gloss lemmas left after stopword filtering.")

    print(
        f"[stopwords] enabled={stopword_info['stopword_filter_enabled']} | "
        f"active={stopword_info['active_stopwords']} | "
        f"removed_unique={stopword_info['removed_stopword_unique']} | "
        f"remaining_unique={len(tsl_freq)}"
    )
    if stopword_info["top_removed_stopwords"]:
        print(
            "[stopwords] top removed: "
            + ", ".join(stopword_info["top_removed_stopwords"][:10])
        )

    matches, scan_stats = collect_matches(
        tid_root=tid_root,
        tsl_freq=tsl_freq,
        output_root=output_root,
        base_root=base_root,
        match_mode=args.match_mode,
        limit=args.limit,
    )

    match_summary = f"matches={scan_stats['matched_word_dirs_before_limit']}"
    if scan_stats["matched_word_dirs_before_limit"] != scan_stats["matched_word_dirs"]:
        match_summary += f" | limited_output={scan_stats['matched_word_dirs']}"
    print(
        f"[scan] TSL-News rows={tsl_rows} | unique pseudo_gloss={len(tsl_freq)} | "
        f"TID word dirs scanned={scan_stats['scanned_word_dirs']} | {match_summary}"
    )
    print(
        f"[scan] skipped missing word_info={scan_stats['skipped_missing_word_info']} | "
        f"no overlap={scan_stats['skipped_no_overlap']}"
    )
    if matches:
        print_preview(matches, args.preview)
    else:
        print("[scan] no overlaps found")

    if args.dry_run:
        return

    word_rows, example_rows, meaning_rows = build_manifest_rows(
        matches=matches,
        tid_root=tid_root,
        output_root=output_root,
        base_root=base_root,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "overlap_words.csv", word_rows)
    write_csv(output_root / "overlap_example_videos.csv", example_rows)
    write_csv(output_root / "overlap_meaning_videos.csv", meaning_rows)

    copy_stats = None
    if args.copy_mode != "none":
        copy_stats = copy_match_dirs(
            matches=matches,
            tid_root=tid_root,
            output_root=output_root,
            copy_mode=args.copy_mode,
        )
        print(
            f"[copy] copied={copy_stats['copied']} | linked={copy_stats['linked']} | "
            f"skipped_existing={copy_stats['skipped_existing']}"
        )
    else:
        print("[copy] skipped because --copy_mode none")

    summary = build_summary(
        matches=matches,
        scan_stats=scan_stats,
        tsl_freq=tsl_freq,
        tsl_rows=tsl_rows,
        tsl_unique_before_stopwords=tsl_unique_before_stopwords,
        splits=args.splits,
        match_mode=args.match_mode,
        output_root=output_root,
        copy_mode=args.copy_mode,
        word_rows=word_rows,
        example_rows=example_rows,
        meaning_rows=meaning_rows,
        copy_stats=copy_stats,
        stopword_info=stopword_info,
    )
    summary_path = output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"[write] wrote manifests to {output_root}")
    print(f"[write] summary: {summary_path}")


if __name__ == "__main__":
    main()
