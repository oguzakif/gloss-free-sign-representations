# Adapted for sign representation training and evaluation.
"""Paper training entry point. Run from the repository root."""
import importlib
from absl import app
from ml_collections import config_flags
import torch
import ignite.distributed as idist
from train_utils.release_helpers import validate_config

def main(_):
    cfg = CONFIG.value
    validate_config(cfg, training=True)
    import os
    if os.environ.get("SIGN_SPOTTING_WANDB") == "1":
        cfg.logger_name = ["text", "wandb"]
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU and the environment in environment.yml.")
    count = torch.cuda.device_count()
    backend = "nccl" if count > 1 else "gloo"
    from pathlib import Path
    destination = Path(cfg.save_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "run_config.json").write_text(cfg.to_json_best_effort(), encoding="utf-8")
    runner = importlib.import_module(cfg.main_runner)
    with idist.Parallel(backend, nproc_per_node=count if count > 1 else None) as parallel:
        parallel.run(runner.Trainer, cfg)

if __name__ == "__main__":
    CONFIG = config_flags.DEFINE_config_file("config", default=None, lock_config=False)
    app.run(main)
