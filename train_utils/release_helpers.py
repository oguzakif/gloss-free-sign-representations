"""Portable input checks shared by training and paper evaluation."""
import importlib
import importlib.util
import os
import json
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read_vocabulary(path):
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Missing pseudo-gloss vocabulary: {path}. See docs/data.md or set SIGN_SPOTTING_VOCAB.")
    with path.open("rb") as stream:
        data = pickle.load(stream)
    mapping = data.get("dict_lem_to_id", {})
    if not mapping or sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError(f"Vocabulary IDs must be unique and contiguous from zero: {path}")
    for key in ("dict_sentence", "dict_lem_counter"):
        if key not in data:
            raise ValueError(f"Missing vocabulary field {key!r}: {path}")
    return data

def load_config(value):
    value = str(value)
    if value.endswith(".py") or "/" in value or "\\" in value:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        relative = path.resolve().relative_to(ROOT)
        module = ".".join(relative.with_suffix("").parts)
    else:
        module = value
    return importlib.import_module(module).get_config()

def apply_dataloader_settings(cfg):
    """Propagate top-level CLI overrides to the parameters consumed by loaders."""
    batch_size = int(cfg.bs)
    num_workers = int(cfg.num_workers)
    if batch_size < 1 or num_workers < 0:
        raise ValueError("Batch size must be positive and num_workers must be non-negative.")
    for params in (cfg.train_ds_params, cfg.valid_ds_params, cfg.test_ds_params):
        params.bs = batch_size
        params.num_workers = num_workers


def validate_model_inputs(cfg, check_logging=False):
    stage2 = cfg.model_name.endswith("test_stage2_model")
    params = cfg.model_params.stage1_params if stage2 else cfg.model_params
    lang = params.post_params.emb_lang
    fasttext = Path(os.environ.get("SIGN_SPOTTING_FASTTEXT", f"cc.{lang}.300.bin")).expanduser()
    if not fasttext.is_file():
        raise FileNotFoundError(f"Missing fastText vectors: {fasttext}. Set SIGN_SPOTTING_FASTTEXT.")
    dinov2 = os.environ.get("SIGN_SPOTTING_DINOV2_WEIGHTS")
    if dinov2 and not Path(dinov2).expanduser().is_file():
        raise FileNotFoundError(f"SIGN_SPOTTING_DINOV2_WEIGHTS is not a file: {dinov2}")
    if check_logging and os.environ.get("SIGN_SPOTTING_WANDB") == "1" and importlib.util.find_spec("wandb") is None:
        raise RuntimeError("W&B logging requires requirements-logging.txt, or build Docker with INSTALL_LOGGING=true.")


def validate_config(cfg, training=False):
    if cfg is None:
        raise ValueError("Provide --config with a paper configuration.")
    apply_dataloader_settings(cfg)
    vocab_path = cfg.train_ds_params.pseudo_gloss_dir
    vocabulary = read_vocabulary(vocab_path)
    for ds in (cfg.valid_ds_params, cfg.test_ds_params):
        if read_vocabulary(ds.pseudo_gloss_dir)["dict_lem_to_id"] != vocabulary["dict_lem_to_id"]:
            raise ValueError("Train/dev/test must use the same ordered pseudo-gloss vocabulary.")
    stage2 = cfg.model_name.endswith("test_stage2_model")
    params = cfg.model_params.stage1_params if stage2 else cfg.model_params
    if params.post_params.num_classes != len(vocabulary["dict_lem_to_id"]):
        raise ValueError("Model class count does not match the vocabulary.")
    head_vocabulary = read_vocabulary(params.post_params.emb_pkl_dir)
    if head_vocabulary["dict_lem_to_id"] != vocabulary["dict_lem_to_id"]:
        raise ValueError("Prototype head and dataset must use the same token-to-ID mapping.")
    if training:
        for ds in (cfg.train_ds_params, cfg.valid_ds_params, cfg.test_ds_params):
            for path in (ds.csv_dir, ds.ds_params.lmdb_video_dir):
                if not Path(path).exists():
                    raise FileNotFoundError(f"Missing training input: {path}")
        if stage2 and not cfg.get("translation_from_scratch", False):
            checkpoint = cfg.model_params.stage1_ckpt
            if not checkpoint or not Path(checkpoint).is_file():
                raise FileNotFoundError("Set SIGN_SPOTTING_STAGE1_CKPT to the exact Stage-1 checkpoint, or explicitly set it to 'none' for the no-pretraining translation baseline.")
        validate_model_inputs(cfg, check_logging=True)
    return {"config": cfg.name, "vocabulary_classes": len(vocabulary["dict_lem_to_id"]),
            "vocabulary": str(vocab_path), "stage": 2 if stage2 else 1}

def validate_evaluation_vocabulary(cfg_path, vocab_path):
    cfg = load_config(cfg_path)
    expected = read_vocabulary(cfg.model_params.post_params.emb_pkl_dir)["dict_lem_to_id"]
    actual = read_vocabulary(vocab_path)["dict_lem_to_id"]
    if expected != actual:
        raise ValueError("Evaluation and model vocabularies have different IDs. Use matching inputs or set SIGN_SPOTTING_VOCAB.")
    return cfg
