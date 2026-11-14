"""Evaluate a Stage-1 or translation checkpoint on dev or test."""
import argparse
import importlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--output", default="outputs/checkpoint_metrics.json")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")
    from train_utils.release_helpers import load_config, validate_config, validate_model_inputs
    cfg = load_config(args.config)
    cfg.bs = args.batch_size
    cfg.num_workers = args.num_workers
    validate_config(cfg)
    validate_model_inputs(cfg)
    cfg.run = True
    cfg.evaluation_split = args.split
    cfg.lr_scheduler = "none"
    cfg.load_from_ckpt = "explicit"
    cfg.model_checkpoint_dir = str(checkpoint)
    cfg.model_only = True
    cfg.resume = False
    cfg.save_ckpt = False
    cfg.logger_name = ["text"]
    if cfg.model_name.endswith("test_stage2_model"):
        cfg.model_params.stage1_ckpt = None  # Full Stage-2 weights already include the encoder.
    runner = importlib.import_module(cfg.main_runner)
    trainer = runner.Trainer(0, cfg)
    metrics = trainer.test_tester.state.metrics
    clean = {k: v.item() if hasattr(v, "item") else v for k, v in metrics.items()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"checkpoint": str(checkpoint), "split": args.split, "metrics": clean}, indent=2), encoding="utf-8")
    print(output)

if __name__ == "__main__":
    main()
