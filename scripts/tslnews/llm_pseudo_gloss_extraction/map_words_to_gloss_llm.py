#!/usr/bin/env python3
# Adapted for sign representation training and evaluation.
import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from rapidfuzz import process, fuzz

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, total=None, desc=None):
        if desc:
            print(desc)
        return iterable


SERVER_URL = "http://localhost:8080/v1/chat/completions"
MODEL = "qwen"
SCRIPT_DIR = Path(__file__).resolve().parent
RESOURCE_DIR = SCRIPT_DIR / "resources"

SYSTEM_PROMPT = (
    "Turkce kelimeyi pseudo_gloss ile eslestir. "
    "Yalniz ADAYLAR listesinden sec; listede olmayan gloss yazma. "
    "Yeni kelime uretme. "
    "Kelime cekimli fiil veya ek almis isim olabilir. Kok/anlam olarak dusun. "
    "Adaylarda kelimenin kokune benzeyen bir gloss varsa MUTLAKA onu sec. "
    "Ornek: gosterdi->göstermek, gosterecek->göstermek, gosteren->göstermek. "
    "Ornek: geldiniz->gel, artirdik->artir, cikarilacak->cikar, bultenine->bulten. "
    "NONE sadece adaylarin hicbiri kelimeyle anlam veya kok olarak ilgili degilse kullanilir. "
    'Sadece JSON dondur: {"sonuclar":[{"kelime":"...","gloss":"..."}]}'
)



def tr_lower(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


def normalize_for_match(text: str) -> str:
    table = str.maketrans({
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
        "'": "",
    })
    return tr_lower(text).translate(table)


def gloss_match_key(word: str) -> str:
    word = normalize_for_match(word)
    if word.endswith("mak") or word.endswith("mek"):
        word = word[:-3]
    return word


def read_vocab(path: Path):
    items = []
    seen = set()

    for line in path.read_text(encoding="utf-8").splitlines():
        item = tr_lower(line.strip())
        if item and item not in seen:
            seen.add(item)
            items.append(item)

    return items, seen

def build_candidates_for_all(words, gloss_words, candidate_limit, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_gloss_choices = [gloss_match_key(g) for g in gloss_words]

    result = {}

    for word in tqdm(words, total=len(words), desc="Candidate building"):
        candidates = build_candidates_for_word(
            word=word,
            gloss_words=gloss_words,
            normalized_gloss_choices=normalized_gloss_choices,
            candidate_limit=candidate_limit
        )

        result[word] = candidates

        if len(result) % 500 == 0:
            output_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            print(f"Saved candidates: {len(result)}")

    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Candidate JSON yazıldı: {output_path}")
    print(f"Toplam word: {len(result)}")

def parse_json_object(text: str):
    if "</think>" in text:
        text = text[text.rfind("</think>") + len("</think>"):].strip()

    start = text.find("{")
    end = text.rfind("}") + 1

    if start == -1 or end <= start:
        return {}

    return json.loads(text[start:end])


def call_llm(batch_items, server_url, model, timeout):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "KELIMELER VE ADAYLAR:\n"
                    + json.dumps(batch_items, ensure_ascii=False)
                    + "\nSadece aday listesinden sec."
                ),
            },
        ],
        "temperature": 0.0,
        "max_tokens": max(300, len(batch_items) * 60),
    }

    try:
        response = requests.post(server_url, json=payload, timeout=timeout)
        response.raise_for_status()

        message = response.json()["choices"][0]["message"]
        content = message.get("content", "") or message.get("reasoning_content", "")

        data = parse_json_object(content.strip())
        results = data.get("sonuclar", [])

        out = {}

        for item in results:
            kelime = tr_lower(str(item.get("kelime", "")).strip())
            gloss = tr_lower(str(item.get("gloss", "")).strip())

            if gloss == "none":
                gloss = ""

            out[kelime] = gloss

        for item in batch_items:
            kelime = item["kelime"]
            if kelime not in out:
                out[kelime] = ""

        return out

    except Exception as exc:
        print(f"\nLLM hata: {exc}")
        return {item["kelime"]: "" for item in batch_items}


def chunks(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def write_review_csv(path: Path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["word", "selected_gloss", "status", "candidates"]
        )
        writer.writeheader()
        writer.writerows(rows)


def run_llm_mapping(candidate_path, gloss_set, args):
    candidates_map = json.loads(candidate_path.read_text(encoding="utf-8"))

    output_path = Path(args.output)
    review_path = Path(args.review_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.parent.mkdir(parents=True, exist_ok=True)

    final_mapping = {}

    # Eğer output varsa onu temel al
    if output_path.exists():
        final_mapping = json.loads(output_path.read_text(encoding="utf-8"))
        print(f"Existing output mapping loaded: {len(final_mapping)}")

    # Eski jsondan sadece empty olanları tekrar işle
    only_empty_words = None
    if args.only_empty_from:
        old_mapping_path = Path(args.only_empty_from)
        old_mapping = json.loads(old_mapping_path.read_text(encoding="utf-8"))

        only_empty_words = {
            word for word, gloss in old_mapping.items()
            if not str(gloss).strip()
        }

        print(f"Only-empty source loaded: {old_mapping_path}")
        print(f"Empty words to retry: {len(only_empty_words)}")

    review_rows = []
    llm_items = []

    for word, candidates in candidates_map.items():
        # Eğer sadece empty retry modundaysak, eski jsonda boş olmayanları atla
        if only_empty_words is not None and word not in only_empty_words:
            continue

        # Bu run içinde zaten doluysa tekrar işlemeyelim
        if word in final_mapping and str(final_mapping[word]).strip():
            continue


        if word in gloss_set:
            final_mapping[word] = word
            review_rows.append({
                "word": word,
                "selected_gloss": word,
                "status": "exact",
                "candidates": word,
            })
            continue

        llm_items.append({
            "kelime": word,
            "candidates": candidates,
        })

    print(f"Candidate words : {len(candidates_map)}")
    print(f"Current mapping  : {len(final_mapping)}")
    print(f"LLM needed      : {len(llm_items)}")

    batches = list(chunks(llm_items, args.batch_size))
    completed_batches = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                call_llm,
                batch,
                args.server_url,
                args.model,
                args.timeout,
            ): batch
            for batch in batches
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="LLM empty retry"):
            batch = futures[future]
            result = future.result()

            for item in batch:
                word = item["kelime"]
                candidates = item["candidates"]
                candidate_set = set(candidates)

                selected = result.get(word, "")

                if selected and selected in candidate_set and selected in gloss_set:
                    final_mapping[word] = selected
                    status = "llm_retry"
                else:
                    final_mapping[word] = ""
                    status = "none_retry"

                review_rows.append({
                    "word": word,
                    "selected_gloss": final_mapping[word],
                    "status": status,
                    "candidates": " | ".join(candidates),
                })

            completed_batches += 1

            output_path.write_text(
                json.dumps(final_mapping, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            write_review_csv(review_path, review_rows)

            if completed_batches % 10 == 0:
                mapped = sum(1 for v in final_mapping.values() if v)
                empty = sum(1 for v in final_mapping.values() if not v)
                print(f"Progress saved | total={len(final_mapping)} mapped={mapped} empty={empty}")

    output_path.write_text(
        json.dumps(final_mapping, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    write_review_csv(review_path, review_rows)

    mapped = sum(1 for v in final_mapping.values() if v)
    empty = sum(1 for v in final_mapping.values() if not v)

    print("\nTamamlandı")
    print(f"Mapped : {mapped}")
    print(f"Empty  : {empty}")
    print(f"Output : {output_path}")
    print(f"Review : {review_path}")


def word_prefix_variants(word_key: str, min_len: int = 3):
    variants = []

    for end in range(len(word_key), min_len - 1, -1):
        part = word_key[:end]
        if part not in variants:
            variants.append(part)

    return variants


def add_candidate(candidates, seen, gloss):
    if gloss not in seen:
        seen.add(gloss)
        candidates.append(gloss)


def build_candidates_for_word(word, gloss_words, normalized_gloss_choices, candidate_limit):
    word_key = normalize_for_match(word)

    candidates = []
    seen = set()

    variants = word_prefix_variants(word_key, min_len=3)

    # 1) Exact match
    for gloss, gloss_key in zip(gloss_words, normalized_gloss_choices):
        if word_key == gloss_key:
            add_candidate(candidates, seen, gloss)

    # 2) Sondan kısaltarak root/prefix match
    # bağımlılığının -> bağımlılığın -> ... -> bağımlı
    prefix_matches = []

    for variant_index, variant in enumerate(variants):
        for gloss, gloss_key in zip(gloss_words, normalized_gloss_choices):
            if gloss in seen:
                continue

            if variant == gloss_key:
                # daha az kırpılan varyant daha yüksek öncelik
                score = 10000 - variant_index
                prefix_matches.append((score, len(gloss_key), gloss))

            elif len(gloss_key) >= 3 and variant.startswith(gloss_key):
                score = 9000 - variant_index
                prefix_matches.append((score, len(gloss_key), gloss))

    prefix_matches.sort(key=lambda x: (x[0], x[1]), reverse=True)

    for _, _, gloss in prefix_matches:
        add_candidate(candidates, seen, gloss)
        if len(candidates) >= candidate_limit:
            return candidates

    # 3) Fuzzy fallback
    # Ama artık fuzzy sadece destekleyici olacak.
    fuzzy_matches = []

    for variant_index, variant in enumerate(variants[:8]):
        matches = process.extract(
            variant,
            normalized_gloss_choices,
            scorer=fuzz.WRatio,
            limit=candidate_limit
        )

        for _, score, idx in matches:
            gloss = gloss_words[idx]
            gloss_key = normalized_gloss_choices[idx]

            # Çok alakasız kısa kelimeleri engelle
            if score < 75:
                continue

            # Uzun köke yakın olanı öne al
            final_score = score - variant_index
            fuzzy_matches.append((final_score, len(gloss_key), gloss))

    fuzzy_matches.sort(key=lambda x: (x[0], x[1]), reverse=True)

    for _, _, gloss in fuzzy_matches:
        if gloss in seen:
            continue

        add_candidate(candidates, seen, gloss)

        if len(candidates) >= candidate_limit:
            break

    return candidates

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--words", type=Path, required=True, help="Translation-word list, for example the --words-output from extract_unique_translation_words.py.")
    parser.add_argument("--glosses", type=Path, required=True, help="External vocabulary inventory (not bundled).")
    parser.add_argument("--candidates-json", type=Path, default=RESOURCE_DIR / "word_candidates.json")
    parser.add_argument("--output", type=Path, default=RESOURCE_DIR / "word_to_gloss_llm.json")
    parser.add_argument("--review-csv", type=Path, default=RESOURCE_DIR / "word_to_gloss_review.csv")

    parser.add_argument("--build-candidates", action="store_true")
    parser.add_argument("--run-llm", action="store_true")

    parser.add_argument("--server-url", default=SERVER_URL)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=15)
    parser.add_argument(
        "--only-empty-from",
        type=Path,
        default=None,
        help="Only remap words whose value is empty in this mapping json.",
    )

    args = parser.parse_args()

    words, _ = read_vocab(args.words)
    gloss_words, gloss_set = read_vocab(args.glosses)

    print(f"Words : {len(words)}")
    print(f"Gloss : {len(gloss_words)}")

    candidate_path = args.candidates_json

    if args.build_candidates:
        build_candidates_for_all(
            words=words,
            gloss_words=gloss_words,
            candidate_limit=args.candidate_limit,
            output_path=candidate_path,
        )

    if args.run_llm:
        if not candidate_path.exists():
            raise FileNotFoundError(f"Candidate JSON bulunamadı: {candidate_path}")
        run_llm_mapping(candidate_path, gloss_set, args)

    if not args.build_candidates and not args.run_llm:
        print("Bir mod seç: --build-candidates veya --run-llm")


if __name__ == "__main__":
    main()
