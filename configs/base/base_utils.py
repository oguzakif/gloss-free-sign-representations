# Adapted for sign representation training and evaluation.
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def get_data_path():
    return Path(os.environ.get("SIGN_SPOTTING_DATA_ROOT", ROOT / "data")).expanduser().resolve()

def get_checkpoint_path(base_name, name):
    root = Path(os.environ.get("SIGN_SPOTTING_CHECKPOINT_ROOT", ROOT / "checkpoints"))
    return (root / base_name / name).expanduser().resolve()

def get_lmdb_path():
    return Path(os.environ.get("SIGN_SPOTTING_LMDB_ROOT", os.environ.get("SIGN2GPT_LMDB_ROOT", get_data_path()))).expanduser().resolve()

def get_tsl_paths(vocabulary="llm"):
    root = get_data_path() / "TSL-News"
    annotations = root / ("6538_gloss" if vocabulary == "llm" else "annotations")
    vocab = root / "6538_gloss/processed_words_map.tsl_pkl" if vocabulary == "llm" else root / "processed_words.tsl3_pkl"
    return (Path(os.environ.get("SIGN_SPOTTING_ANNOTATIONS", annotations)).expanduser().resolve(),
            Path(os.environ.get("SIGN_SPOTTING_VOCAB", vocab)).expanduser().resolve())
