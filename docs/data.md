# Data and pretrained inputs

TSL-News and TSL-SB, the TID-derived spotting benchmark used in this paper, have not yet been publicly released. This repository provides training and evaluation code. Reproducing the reported experiments requires the datasets and their corresponding splits, annotations, and vocabulary artifacts, which are not included here. No public dataset download links are currently available.

The layout and commands below describe the inputs required to run the code once those inputs are available. Generated vocabulary inventories, LLM mapping JSONs, pseudo-gloss labels, corpus CSVs, videos, annotation exports, PKLs, checkpoints, and fastText weights are not distributed in this repository. Store local inputs outside the repository or in the ignored `data/`, `artifacts/`, or `scripts/tslnews/llm_pseudo_gloss_extraction/resources/` directories.

## Expected prepared layout

```text
data/
  TSL-News/
    annotations/TSL-News.{train,dev,test}.corpus.csv
    processed_words.tsl3_pkl
    6538_gloss/
      TSL-News.{train,dev,test}.corpus.csv
      processed_words_map.tsl_pkl
    lmdb_videos/{train,dev,test}/<sequence_name>/
  TID/
    raw/<word_folder>/word_info.json
    raw/<word_folder>/example_videos/*.mp4
    raw/<word_folder>/isolated_videos/*.mp4
    overlap_tslnews_6538/overlap_words.csv
    overlap_tslnews_6538/annotated_24_6_26_total/*_export.csv
    overlap_tslnews_4802/overlap_words.csv
```

Set `SIGN_SPOTTING_DATA_ROOT` to replace `data/`. `SIGN_SPOTTING_LMDB_ROOT` can point to separate node-local storage containing `TSL-News/lmdb_videos/`. `SIGN_SPOTTING_ANNOTATIONS` overrides the selected corpus CSV directory, and `SIGN_SPOTTING_VOCAB` overrides the vocabulary PKL path. Explicit evaluator arguments specify the benchmark inputs.

Training CSVs use `|` separators and contain `name`, `translation`, and `orth`. LMDB keys are ASCII frame indices and `details`, a pickled dictionary containing `num_frames`. Each vocabulary PKL contains `dict_lem_to_id`, `dict_sentence`, and `dict_lem_counter`. Token IDs must be contiguous from zero; the release rejects invalid or missing vocabularies instead of silently constructing a 3,000-class model. Load PKLs/checkpoints only from sources you trust.

## Rebuild inputs from your authorized local data

The existing corpus/LMDB utilities expect `TSL-News/splitted-data/{train,dev,test}/{texts,videos}/`:

```bash
python scripts/tslnews/build_csvs.py --tsl_root /path/to/TSL-News
python scripts/tslnews/lmdb_creator.py --tsl_root /path/to/TSL-News
```

Morphology labels:

```bash
python -m pip install -r requirements-preprocessing.txt
python scripts/tslnews/pseudo_gloss_tr.py --tsl_root /path/to/TSL-News
```

This updates the annotation CSVs with `pseudo_gloss` and writes `processed_words.tsl3_pkl`. Use separate corpus copies when rebuilding alternative pseudo-gloss strategies. The morphology package is optional; reproducing the 4,802-class vocabulary requires matching corpus and analyzer settings, rather than assuming a new extraction produces identical IDs.

LLM lexical-rule preparation requires your own external mapping JSON and ordered vocabulary inventory:

```bash
python scripts/tslnews/llm_pseudo_gloss_extraction/build_mapped_tslnews_corpora.py \
  --mapping /path/to/external/word_to_gloss.json \
  --input-dir /path/to/TSL-News/annotations \
  --output-dir /path/to/TSL-News/6538_gloss
python scripts/tslnews/llm_pseudo_gloss_extraction/build_pseudo_pkl_from_csv.py \
  --vocab /path/to/external/ordered_vocabulary.txt \
  --input-dir /path/to/TSL-News/6538_gloss \
  --out /path/to/TSL-News/6538_gloss/processed_words_map.tsl_pkl
```

For exact reproduction, use the paper's final prepared corpus and exact ordered vocabulary/PKL. The paper LLM inventory has 6,539 IDs and the morphology inventory has 4,802; neither inventory is bundled. Use the ordered token IDs that match your checkpoint. Reordering or regenerating a vocabulary can break checkpoint compatibility even if its size is unchanged. `--strict` rejects labels missing from the supplied inventory.

To prepare a translation-word list for candidate generation:

```bash
python scripts/tslnews/llm_pseudo_gloss_extraction/extract_unique_translation_words.py \
  --csv /path/to/TSL-News/annotations/TSL-News.train.corpus.csv \
  --words-output /path/to/external/translation_words.txt \
  --mapping-output /path/to/external/initial_mapping.json
python scripts/tslnews/llm_pseudo_gloss_extraction/map_words_to_gloss_llm.py \
  --words /path/to/external/translation_words.txt \
  --glosses /path/to/external/ordered_vocabulary.txt --build-candidates
```

Repeat `--csv` to include additional corpus files under your chosen vocabulary protocol.

Candidate generation and constrained LLM mapping code remains available in `map_words_to_gloss_llm.py --help`; provide your own `--glosses` inventory, server, and model. Its default generated outputs go into the ignored resources directory. Test fixtures use tiny synthetic mappings created in temporary directories and contain no paper vocabulary or LLM-generated resources.

For benchmark construction, `scripts/tid/copy_overlapping_tid_vocab.py --help` exposes the retained vocabulary-matching utility. Human temporal annotations and the final benchmark membership are external inputs; regenerating an overlap list does not recreate the published benchmark by itself.

## Pretrained weights

Provide Turkish fastText `cc.tr.300.bin` through `SIGN_SPOTTING_FASTTEXT`; there is no implicit fastText download during training. DINOv2 ViT-S/14 base weights can be provided through `SIGN_SPOTTING_DINOV2_WEIGHTS` or downloaded by the model loader. Translation also needs the `facebook/xglm-1.7B` tokenizer and weights, resolved by Hugging Face. Each upstream asset retains its own terms.
