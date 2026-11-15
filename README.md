# Gloss-Free Representation Learning for Cross-Dataset Sign Spotting

Official implementation of [Gloss-Free Representation Learning for Cross-Dataset Sign Spotting](https://arxiv.org/abs/2608.11332), by Oğuz Akif Tüfekcioğlu, Ezgi Ekin, Mustafa Kaan Çevik, and Hacer Yalim Keles (2026).

We learn a reusable sign encoder from TSL-News video/transcript pairs using sentence-level pseudo-gloss supervision, then evaluate transfer to the TSL Spotting Benchmark. This repository contains the paper's DINOv2-based training and evaluation code, including both Turkish pseudo-gloss strategies and the downstream translation check.

## Installation

Training and evaluation use a Linux CUDA environment, available through Docker on Windows or Linux with an NVIDIA GPU. See [Docker setup](docs/docker.md) for host requirements, external-data mounts, and commands.

```bash
docker build -t gloss-free-sign-spotting .
```

For a native Linux installation, use Conda. The pinned environment retains the xFormers components required by the model:

```bash
conda env create -f environment.yml
conda activate gloss-free-sign-spotting
```

For optional pseudo-gloss reconstruction: `python -m pip install -r requirements-preprocessing.txt`.
For optional Weights & Biases logging: `python -m pip install -r requirements-logging.txt`, then export `SIGN_SPOTTING_WANDB=1` and your own `WANDB_API_KEY`. Console logging is enabled by default.

## Data and pretrained inputs

TSL-News and TSL-SB, the TID-derived spotting benchmark used in this paper, have not yet been publicly released. This repository provides training and evaluation code. Reproducing the reported experiments requires the datasets and their corresponding splits, annotations, and vocabulary artifacts, which are not included here. No public dataset download links are currently available.

Generated vocabularies, LLM mappings, pseudo-gloss labels, pretrained weights, and model checkpoints are not bundled. See [data preparation](docs/data.md) for the expected input layout and preparation commands. The code license does not grant rights to datasets or pretrained assets.

Set `SIGN_SPOTTING_DATA_ROOT` to your prepared data directory and `SIGN_SPOTTING_FASTTEXT` to your Turkish `cc.tr.300.bin`. Optionally set `SIGN_SPOTTING_DINOV2_WEIGHTS` to a local DINOv2 ViT-S/14 pretrained weight file. Otherwise DINOv2 weights are downloaded to the PyTorch cache on first use.

## Training

Run commands from the repository root. Check your inputs and required local weights before training:

```bash
python scripts/check_setup.py --config configs/tslnews/tslnews_stage1_configs/TSL_example_s1_dyn_config.py --training
python main.py --config configs/tslnews/tslnews_stage1_configs/TSL_example_s1_dyn_config.py
```

Use `TSL_example_s1_dyn_4802_config.py` for morphology-based pretraining. The `6538_gloss` vocabulary contains **6,539 classes**; morphology contains **4,802 classes**. Vocabulary IDs must match the checkpoint, not just the vocabulary size.

See [training](docs/training.md) for batch settings, checkpoint selection, and downstream translation training/evaluation.

## Spotting evaluation

```bash
python -m evaluation.evaluate_spotting --method ncc \
  --config configs/tslnews/tslnews_stage1_configs/TSL_example_s1_dyn_config.py \
  --checkpoint /path/to/stage1_checkpoint.pt \
  --vocab_pkl /path/to/processed_words_map.tsl_pkl \
  --annotation_dir /path/to/annotated_24_6_26_total \
  --word_manifest /path/to/overlap_words.csv \
  --tid_raw_roots /path/to/TID/raw \
  --output_dir outputs/ncc_llm --no_plots
```

For an external vocabulary location, also set `SIGN_SPOTTING_VOCAB` to the exact file passed as `--vocab_pkl`.

Use `--method spatial` without a checkpoint for the unadapted spatial baseline; `--method model_scores` evaluates pseudo-gloss scores; `--method retrieval` runs all-vocabulary local NCC. See [evaluation](docs/evaluation.md) for protocol details, vocabulary alignment, and complete commands.

## Published reference results

Results reported in the paper:

| Representation | Evaluated examples | Top-1 mean IoU | Top-3 mean IoU | Top-5 mean IoU |
| --- | ---: | ---: | ---: | ---: |
| Spatial features | 1,817 | 0.124 | 0.192 | 0.235 |
| Morphology-lemma encoder | 1,159 | 0.173 | 0.293 | 0.363 |
| LLM lexical-rule encoder | 1,817 | 0.271 | 0.403 | 0.465 |

The vocabularies cover different subsets. See the paper for its shared-subset comparison. Local all-vocabulary retrieval uses a ground-truth temporal crop and is a separate diagnostic from full-video localization.

## Tests

Run `python -m unittest discover -s tests -v`. The tests use synthetic inputs and require no datasets, model checkpoints, or generated vocabularies.

## Acknowledgements

This work is supported by the Scientific and Technological Research Council of Türkiye (TÜBİTAK) under the 1001 Scientific and Technological Research Projects Funding Program (Project No. 124E618). We acknowledge the EuroHPC Joint Undertaking for awarding us access to Vega at IZUM, Slovenia, through Development Access allocation 2025D08-090.

## Citation

If you use this code or build on this work, please cite our paper:

```bibtex
@misc{tufekcioglu2026glossfree,
  title={Gloss-Free Representation Learning for Cross-Dataset Sign Spotting},
  author={Oğuz Akif Tüfekcioğlu and Ezgi Ekin and Mustafa Kaan Çevik and Hacer Yalim Keles},
  year={2026},
  eprint={2608.11332},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2608.11332}
}
```

The implementation builds on Sign2GPT, DINOv2, xFormers, and Hugging Face Transformers. Please also acknowledge upstream work where appropriate. See [third-party notices](THIRD_PARTY_NOTICES.md) and [LICENSES](LICENSES/) for retained licenses. The root Apache-2.0 license applies subject to those notices.
