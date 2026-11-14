"""Select one of the paper's spotting evaluations; remaining flags go to its evaluator."""
import argparse
import importlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    selector = argparse.ArgumentParser(description=__doc__, add_help=False)
    selector.add_argument("--method", choices=["ncc", "spatial", "model_scores", "retrieval"], default="ncc")
    selected, remaining = selector.parse_known_args()
    module = importlib.import_module("evaluation.tid_overlap_local_all_vocab_ncc_eval" if selected.method == "retrieval" else "evaluation.tid_overlap_ncc_eval")
    parser = module.build_arg_parser()
    parser.description = __doc__ + " Methods: ncc, spatial, model_scores, retrieval."
    args = parser.parse_args(remaining)
    if selected.method == "spatial":
        if args.checkpoint:
            parser.error("The spatial baseline does not take --checkpoint.")
        args.no_checkpoint = True
        args.ncc_embedding_source = "spatial"
    elif not args.checkpoint and not (selected.method == "retrieval" and args.no_checkpoint):
        parser.error("Provide an explicit --checkpoint; use --method spatial for the unadapted baseline.")
    from train_utils.release_helpers import validate_evaluation_vocabulary, validate_model_inputs
    cfg = validate_evaluation_vocabulary(args.config, args.vocab_pkl)
    validate_model_inputs(cfg)
    if selected.method != "retrieval":
        args.mode = "both"
        args.prediction_method = "model_scores" if selected.method == "model_scores" else "ncc"
        root = Path(module.__file__).resolve().parents[1]
        annotation_dir = module.resolve_existing_path(args.annotation_dir, root)
        manifest = module.resolve_existing_path(args.word_manifest, root)
        if annotation_dir is None:
            parser.error(f"Annotation directory not found: {args.annotation_dir}")
        if manifest is None:
            parser.error(f"Word manifest not found: {args.word_manifest}")
        args.label_dir = str(Path(args.output_dir) / "labels")
        module.export_labels(annotation_dir, Path(args.label_dir), manifest,
                             tid_raw_roots=module.resolve_tid_raw_roots(args.tid_raw_roots, root),
                             script_root=root)
        summary = module.evaluate_ncc(args)
    else:
        summary = module.evaluate_local_all_vocab_ncc(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
