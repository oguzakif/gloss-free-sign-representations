#!/usr/bin/env python3
# Adapted for sign representation training and evaluation.
import argparse
import csv
import pickle
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
RESOURCE_DIR = SCRIPT_DIR / "resources"
DEFAULT_CORPUS_DIR = PROJECT_ROOT / "data" / "TSL-News" / "6538_gloss"
DEFAULT_OUT = DEFAULT_CORPUS_DIR / "processed_words_map.tsl_pkl"


def read_vocab(path: Path):
    vocab = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        vocab.append(token)
    return vocab


def parse_pseudo_gloss(value):
    if not isinstance(value, str):
        return []
    seen = set()
    out = []
    for token in value.split():
        token = token.strip()
        if token and token not in seen:
            seen.add(token)
            out.append(token)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Build TSL-News processed_words pkl from CSV pseudo_gloss columns."
    )
    parser.add_argument(
        "--tsl_root",
        type=Path,
        default=None,
        help="Deprecated compatibility option: TSL-News root containing annotations/.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_CORPUS_DIR,
        help="Directory containing TSL-News.<split>.corpus.csv files.",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        required=True,
        help="External ordered vocabulary, one label per line. This becomes dict_lem_to_id.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output pickle path used by the stage1 config.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev", "test"],
        help="Splits to read from annotations/TSL-News.<split>.corpus.csv",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a CSV pseudo_gloss token is missing from the vocab.",
    )
    args = parser.parse_args()

    ann_root = args.tsl_root / "annotations" if args.tsl_root else args.input_dir
    vocab = read_vocab(args.vocab)
    vocab_set = set(vocab)

    dict_sentence = {}
    counter = Counter()
    missing = Counter()

    for split in args.splits:
        csv_path = ann_root / f"TSL-News.{split}.corpus.csv"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="|")
            if not reader.fieldnames or "translation" not in reader.fieldnames:
                raise ValueError(f"{csv_path} has no translation column")
            if "pseudo_gloss" not in reader.fieldnames:
                raise ValueError(f"{csv_path} has no pseudo_gloss column")

            for row in reader:
                sentence = str(row.get("translation", ""))
                raw_pseudo = row.get("pseudo_gloss", "")
                lemmas = []
                for lemma in parse_pseudo_gloss(raw_pseudo):
                    if lemma in vocab_set:
                        lemmas.append(lemma)
                    else:
                        missing[lemma] += 1

                dict_sentence[sentence] = lemmas
                counter.update(lemmas)

    if missing and args.strict:
        sample = ", ".join(f"{k}({v})" for k, v in missing.most_common(20))
        raise ValueError(f"pseudo_gloss tokens missing from vocab: {sample}")

    dict_lem_to_id = {lemma: idx for idx, lemma in enumerate(vocab)}
    payload = {
        "dict_lem_counter": dict(counter),
        "dict_sentence": dict_sentence,
        "dict_lem_to_id": dict_lem_to_id,
    }

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(payload, f)

    print(f"Wrote: {out_path}")
    print(f"Vocab/classes: {len(dict_lem_to_id)}")
    print(f"Sentences: {len(dict_sentence)}")
    print(f"Observed pseudo glosses: {len(counter)}")
    print(f"Missing vocab tokens ignored: {sum(missing.values())}")
    if missing:
        print("Top missing:", ", ".join(f"{k}({v})" for k, v in missing.most_common(20)))


if __name__ == "__main__":
    main()
