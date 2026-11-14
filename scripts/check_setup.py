"""Check paper inputs without starting a model or downloading weights."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--training", action="store_true", help="Also check corpus files, LMDB roots, and Stage-1 initialization.")
    args = parser.parse_args()
    from train_utils.release_helpers import load_config, validate_config
    print(json.dumps(validate_config(load_config(args.config), training=args.training), indent=2))

if __name__ == "__main__":
    main()
