#!/usr/bin/env python3
import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
RESOURCE_DIR = SCRIPT_DIR / "resources"
DEFAULT_CSVS = [
    PROJECT_ROOT / "data" / "TSL-News" / "annotations" / f"TSL-News.{split}.corpus.csv"
    for split in ("train", "dev", "test")
]
TOKEN_RE = re.compile(r"[0-9A-Za-zÇĞİÖŞÜçğışüÂÎÛâîû]+")


def tr_lower(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = tr_lower(text)
    text = text.replace("i̇", "i").replace("\u0307", "")
    return unicodedata.normalize("NFKC", text)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract unique normalized translation tokens for LLM word-to-gloss mapping."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        action="append",
        default=None,
        help="Input TSL-News corpus CSV containing a translation column. Repeat for multiple files.",
    )
    parser.add_argument(
        "--words-output",
        type=Path,
        default=RESOURCE_DIR / "unique_translation_words_clean_test.txt",
        help="Output txt file, one unique token per line.",
    )
    parser.add_argument(
        "--mapping-output",
        type=Path,
        default=RESOURCE_DIR / "translation_word_to_gloss_clean_test.json",
        help="Initial empty JSON mapping output.",
    )
    parser.add_argument(
        "--debug-token",
        action="append",
        default=[],
        help="Optional token substring to print matches for; may be repeated.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    unique_words = set()
    csv_paths = args.csv or DEFAULT_CSVS
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, sep="|")
        if "translation" not in df.columns:
            raise ValueError(f"{csv_path} has no translation column")
        for sentence in df["translation"].fillna(""):
            sentence = normalize_text(sentence)
            for token in TOKEN_RE.findall(sentence):
                token = normalize_text(token).strip()
                if token:
                    unique_words.add(token)

    unique_words = sorted(unique_words)
    args.words_output.parent.mkdir(parents=True, exist_ok=True)
    args.mapping_output.parent.mkdir(parents=True, exist_ok=True)

    args.words_output.write_text("\n".join(unique_words) + "\n", encoding="utf-8")
    args.mapping_output.write_text(
        json.dumps({word: "" for word in unique_words}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("CSV paths:")
    for csv_path in csv_paths:
        print(f"  {csv_path}")
    print(f"Unique word count: {len(unique_words)}")
    print(f"Saved txt: {args.words_output.resolve()}")
    print(f"Saved mapping json: {args.mapping_output.resolve()}")

    for token in args.debug_token:
        found = [word for word in unique_words if token in word]
        print(f"{token}: {found[:20]}")


if __name__ == "__main__":
    main()
