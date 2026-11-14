#!/usr/bin/env python3
import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
RESOURCE_DIR = SCRIPT_DIR / "resources"
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "TSL-News" / "annotations"
DEFAULT_MAPPING = RESOURCE_DIR / "word_to_gloss_llm_full_v4.merged.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "TSL-News" / "6538_gloss"
DEFAULT_SPLITS = ["train", "dev", "test"]
TOKEN_RE = re.compile(r"[0-9A-Za-zÇĞİÖŞÜçğıöşüÂÎÛâîû]+")


def parse_args():
    parser = argparse.ArgumentParser(
        description="TSL-News corpus dosyalarında pseudo_gloss alanını mapping ile yeniden üretir."
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        required=True,
        help="Path to an externally supplied word-to-gloss mapping JSON (not bundled).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ANNOTATIONS_DIR,
        help=f"Girdi annotation klasörü. Varsayılan: {ANNOTATIONS_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Çıktı klasörü. Varsayılan: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--empty-policy",
        choices=["keep", "drop", "empty"],
        default="drop",
        help="Mapping değeri boşsa ne yapılacağı. keep=orijinal tokeni koru, drop=tokeni at, empty=boş string yaz.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        help="TSL-News splits to process, or explicit CSV filenames.",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffix inserted before .csv. Default writes training-ready names.",
    )
    return parser.parse_args()


def tr_lower(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


def normalize_text(text: str) -> str:
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = tr_lower(text)
    text = text.replace("i̇", "i")
    text = text.replace("\u0307", "")
    text = unicodedata.normalize("NFKC", text)
    return text


def map_token(token: str, mapping: dict[str, str], empty_policy: str) -> str:
    mapped = mapping.get(token, "")
    if mapped is None:
        mapped = ""
    mapped = str(mapped).strip()

    if mapped:
        return mapped
    if empty_policy == "keep":
        return token
    if empty_policy == "drop":
        return ""
    return ""


def transform_tokens(source_tokens: list[str], mapping: dict[str, str], empty_policy: str):
    target_tokens = []
    unresolved = []

    for token in source_tokens:
        mapped = map_token(token, mapping, empty_policy)
        if mapped:
            target_tokens.append(mapped)
        else:
            unresolved.append(token)

        original_value = mapping.get(token, "")
        if not str(original_value).strip():
            unresolved.append(token)

    return " ".join(target_tokens), unresolved


def build_output_name(input_name: str, output_suffix: str) -> str:
    if input_name.endswith(".csv"):
        return input_name[:-4] + output_suffix + ".csv"
    return input_name + output_suffix


def split_to_input_name(split_or_name: str) -> str:
    if split_or_name.endswith(".csv"):
        return split_or_name
    return f"TSL-News.{split_or_name}.corpus.csv"


def extract_source_tokens(row: dict[str, str]) -> list[str]:
    sentence = normalize_text(row.get("translation", ""))
    tokens = TOKEN_RE.findall(sentence)
    out = []
    for token in tokens:
        token = normalize_text(token).strip()
        if token:
            out.append(token)
    return out


def process_file(input_path: Path, output_path: Path, mapping: dict[str, str], empty_policy: str):
    unresolved_counter = Counter()
    rows = []

    with input_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        fieldnames = list(reader.fieldnames or ["name", "translation", "orth"])
        if "pseudo_gloss" not in fieldnames:
            fieldnames.append("pseudo_gloss")

        for row in reader:
            tokens = extract_source_tokens(row)
            mapped_pseudo_gloss, unresolved = transform_tokens(
                tokens,
                mapping,
                empty_policy,
            )
            row["pseudo_gloss"] = mapped_pseudo_gloss
            rows.append(row)
            unresolved_counter.update(unresolved)

    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="|")
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), unresolved_counter


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError("Mapping dosyası JSON object olmalı.")

    total_rows = 0
    total_unresolved = Counter()

    for split_or_name in args.splits:
        input_name = split_to_input_name(split_or_name)
        input_path = args.input_dir / input_name
        output_path = args.output_dir / build_output_name(input_name, args.output_suffix)

        row_count, unresolved_counter = process_file(
            input_path=input_path,
            output_path=output_path,
            mapping=mapping,
            empty_policy=args.empty_policy,
        )
        total_rows += row_count
        total_unresolved.update(unresolved_counter)

        print(f"{input_name} -> {output_path.name} ({row_count} satır)")

    report_path = args.output_dir / "TSL-News.mapped.unresolved_tokens.txt"
    report_lines = [f"{token}\t{count}" for token, count in total_unresolved.most_common()]
    report_path.write_text("\n".join(report_lines) + ("\n" if report_lines else ""), encoding="utf-8")

    print(f"Toplam satır: {total_rows}")
    print(f"Unresolved token türü: {len(total_unresolved)}")
    print(f"Unresolved raporu: {report_path}")


if __name__ == "__main__":
    main()
