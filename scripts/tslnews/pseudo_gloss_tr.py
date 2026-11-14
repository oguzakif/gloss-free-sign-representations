#!/usr/bin/env python3
# Adapted for sign representation training and evaluation.
# -*- coding: utf-8 -*-
"""
TSL-News corpus processing script.

1) For each split (train/dev/test), this script reads the `translation`
   column and extracts lemma lists using turkish_morphology, then writes
   them back to the same CSV as a new column: `pseudo_gloss`.

2) It also creates a combined processed_words pickle file:

   processed_words.tsl3.pkl:

   {
     "dict_lem_counter": Counter of all lemmas,
     "dict_sentence": {translation_str: [lemma1, lemma2, ...]},
     "dict_lem_to_id": {lemma: idx}
   }

Rules:
- Lemmas are extracted using turkish_morphology.
- Each lemma appears at most once per sentence (dedup).
- Single-character lemmas (a, e, i, etc.) are removed.
"""

import argparse
import pickle
from collections import Counter
from pathlib import Path
import re
from typing import List, Dict, Optional

import pandas as pd
from tqdm import tqdm

from turkish_morphology import analyze, decompose, analysis_pb2


def tr_lower(s: str) -> str:
    """Lowercasing helper for Turkish-specific characters."""
    return s.replace("I", "ı").replace("İ", "i").lower()


TOKEN_RE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü']+")


def tm_analyze_token(token: str) -> List[analysis_pb2.Analysis]:
    """Return turkish_morphology analyses for a token."""
    human_list = analyze.surface_form(token, use_proper_feature=False)
    return [decompose.human_readable_analysis(h) for h in human_list]


def get_best_lemma_from_analyses(analyses: List[analysis_pb2.Analysis]) -> Optional[str]:
    """
    Select the most common lemma among all morphological analyses.
    Returns None if no root is found.
    """
    roots = []
    for a in analyses:
        if a.ig and a.ig[0].root.morpheme:
            roots.append(tr_lower(a.ig[0].root.morpheme))

    if not roots:
        return None

    root_counts = Counter(roots)
    best_root, _ = root_counts.most_common(1)[0]
    return best_root


def get_sentence_lemmas(sentence: str, cache: Dict[str, Optional[str]]) -> List[str]:
    """
    Extract lemmas in surface order:
    - Uses cache for speed.
    - Deduplicates within the same sentence.
    - Skips None values.
    - Removes one-letter lemmas.
    """
    tokens = TOKEN_RE.findall(sentence)
    seen, out = set(), []

    for t in tokens:
        if t not in cache:
            analyses = tm_analyze_token(t)
            lem = get_best_lemma_from_analyses(analyses)
            cache[t] = lem

        lem = cache[t]
        if lem and len(lem) > 1 and lem not in seen:
            seen.add(lem)
            out.append(lem)

    return out


def sanity_check():
    """Quick diagnostic: print sample analyses."""
    probe = ["Akdeniz'de", "bugün", "şiddetli", "rüzgar", "var"]
    print("[sanity] quick analyzer check:")
    for w in probe:
        ans = analyze.surface_form(w, use_proper_feature=False)
        print(f"  {w:>12} -> {ans[:2]}")
    print()


def main():
    p = argparse.ArgumentParser()
    default_root = Path(__file__).resolve().parents[2] / "data" / "TSL-News"
    p.add_argument("--tsl_root", default=str(default_root),
                   help="Path to TSL-News root (must contain annotations/)")
    args = p.parse_args()

    tsl_root = Path(args.tsl_root)
    ann_root = tsl_root / "annotations"
    out_pkl = tsl_root / "processed_words.tsl3_pkl"

    sanity_check()

    splits = ["train", "dev", "test"]

    cache: Dict[str, Optional[str]] = {}
    dict_sentence: Dict[str, List[str]] = {}
    all_tokens: List[str] = []

    for split in splits:
        csv_path = ann_root / f"TSL-News.{split}.corpus.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing: {csv_path}")

        print(f"\n[split] Processing {csv_path.name}")
        df = pd.read_csv(csv_path, sep="|")

        sentences = df["translation"].astype(str).tolist()
        pseudo_gloss_col: List[str] = []

        for sent in tqdm(sentences,
                         desc=f"  -> building lemmas for {split}",
                         unit="sent"):
            lemmas = get_sentence_lemmas(sent, cache)
            pseudo_gloss_col.append(" ".join(lemmas))

            dict_sentence[sent] = lemmas
            all_tokens.extend(lemmas)

        df["pseudo_gloss"] = pseudo_gloss_col
        df.to_csv(csv_path, sep="|", index=False)
        print(f"[split] Wrote updated CSV with 'pseudo_gloss' column: {csv_path}")

    dict_lem_to_id = {tok: i for i, tok in enumerate(sorted(set(all_tokens)))}
    dict_processed_words = {
        "dict_lem_counter": dict(Counter(all_tokens)),
        "dict_sentence": dict(dict_sentence),
        "dict_lem_to_id": dict_lem_to_id,
    }

    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(dict_processed_words, f)

    print(f"\n[done] Wrote processed words to {out_pkl} "
          f"with {len(dict_lem_to_id)} unique lemmas.")
    print(f"Tokens processed: {len(all_tokens)} | "
          f"Unique lemmas: {len(dict_lem_to_id)}")


if __name__ == "__main__":
    main()
