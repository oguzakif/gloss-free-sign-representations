# Paper spotting evaluation

Run from the repository root. `python -m evaluation.evaluate_spotting --help` lists the underlying flags. The underlying evaluators are available as `python -m evaluation.tid_overlap_ncc_eval` and `python -m evaluation.tid_overlap_local_all_vocab_ncc_eval`.

Prepare matching configuration, vocabulary PKL, overlap word manifest, temporal annotations, and dictionary videos. Set `SIGN_SPOTTING_VOCAB` to the same file passed as `--vocab_pkl` when using an external vocabulary location. Config/model and evaluation vocabularies are compared by their full token-to-ID mappings. Checkpoints must also have been trained with that mapping; equal class counts alone do not establish compatibility.

## Main target-known NCC localization

```bash
export SIGN_SPOTTING_VOCAB=/path/to/processed_words_map.tsl_pkl
python -m evaluation.evaluate_spotting --method ncc \
  --config configs/tslnews/tslnews_stage1_configs/TSL_example_s1_dyn_config.py \
  --checkpoint /path/to/stage1_checkpoint.pt \
  --vocab_pkl "$SIGN_SPOTTING_VOCAB" \
  --annotation_dir /path/to/annotated_24_6_26_total \
  --word_manifest /path/to/overlap_words.csv \
  --tid_raw_roots /path/to/TID/raw \
  --trim_seconds 0.5 --iso_trim_seconds 0.2 \
  --iso_stride_frames 3 --iso_max_len 12 \
  --stride 1 --ncc_ignore_frames 0 --ncc_weighting none \
  --ncc_peak_position start --movement_window_apply none \
  --output_dir outputs/ncc_llm --no_plots
```

For morphology, change the config to `TSL_example_s1_dyn_4802_config.py`, vocabulary to `processed_words.tsl3_pkl`, and word manifest to the morphology overlap manifest. Use the same benchmark annotation set. The evaluator searches the sentence with templates of the annotated target word, then reports best temporal IoU among the top 1/3/5 ranked peaks. This is target-known localization, not vocabulary-wide recognition.

The paper benchmark has 1,842 annotated example groups. Recorded LLM/spatial runs evaluate 1,817 and skip 25; morphology evaluates 1,159 and skips 683. Preserve skipped examples and the evaluation denominator when comparing results. A shared-subset analysis must match groups between both runs rather than compare different coverage sets.

Outputs include `ncc_iou_summary.json`, `ncc_iou_eval.csv`, predicted intervals, and out-of-vocabulary skips. The old WER diagnostic uses reference-vocabulary information and is not an unconstrained recognition score.

## Spatial and classifier diagnostics

Reuse the main command's inputs with `--method spatial` and omit `--checkpoint` for the raw pretrained DINOv2 spatial-feature baseline. It does not evaluate the random temporal encoder or random classifier as a baseline. The shared model construction still needs the vocabulary and fastText input.

Use `--method model_scores` with a trained checkpoint to localize from pseudo-gloss class-score curves. No isolated templates are needed for that method. Its segmentation thresholds retain the reference defaults; it is a separate diagnostic from NCC.

## All-vocabulary local retrieval

```bash
python -m evaluation.evaluate_spotting --method retrieval \
  --config configs/tslnews/tslnews_stage1_configs/TSL_example_s1_dyn_config.py \
  --checkpoint /path/to/stage1_checkpoint.pt \
  --vocab_pkl "$SIGN_SPOTTING_VOCAB" \
  --annotation_dir /path/to/annotated_24_6_26_total \
  --word_manifest /path/to/overlap_words.csv \
  --tid_raw_roots /path/to/TID/raw \
  --search_scope local_gt --gt_buffer_seconds 0.25 \
  --iso_stride_frames 3 --iso_max_len 12 --iso_trim_seconds 0.2 \
  --ncc_peak_position start --movement_window_apply none \
  --output_dir outputs/local_retrieval --no_plots
```

This ranks all eligible dictionary templates within a ground-truth interval plus a 0.25-second buffer. Its top-k accuracy measures lexical retrieval under a supplied temporal region. It is not the main whole-sentence localization metric. `--search_scope full_video` changes that protocol and should be reported separately.

Fixed trimming is the paper setting. Movement-window preprocessing and its optional dependencies are deliberately excluded from this focused release. Existing advanced evaluator flags are retained for compatibility, but movement CSV inputs would define a different experiment.

The NCC implementation L2-normalizes each frame feature and averages cosine similarities across aligned template and search-window frames.
