"""Training and evaluation correctness checks using synthetic inputs."""
import csv
import importlib
import io
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

class ProtocolTests(unittest.TestCase):
    def test_evaluation_module_commands_work_from_repository_root(self):
        for module in ("evaluation.evaluate_spotting", "evaluation.evaluate_checkpoint",
                       "evaluation.tid_overlap_ncc_eval", "evaluation.tid_overlap_local_all_vocab_ncc_eval"):
            result = subprocess.run([sys.executable, "-m", module, "--help"], cwd=ROOT,
                                    capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_interval_iou_and_degenerate_intervals(self):
        from evaluation.tid_overlap_ncc_eval import interval_iou_seconds as iou
        self.assertAlmostEqual(iou(0, 2, 1, 3), 1 / 3)
        self.assertEqual(iou(0, 1, 0, 1), 1)
        self.assertEqual(iou(0, 1, 2, 3), 0)
        self.assertEqual(iou(2, 1, 0, 1), 0)

    def test_via_conversion_preserves_multiple_segments(self):
        from evaluation.tid_overlap_ncc_eval import convert_annotation_file
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "akşam_export.csv"
            with p.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                for index, interval in enumerate(([1.0, 2.0], [3.0, 4.0])):
                    writer.writerow([f"0_{index}", json.dumps(["https://example.org/144-05_cr_0.2.mp4"]), "", json.dumps(interval), "", "{}"])
            rows, skipped, summary = convert_annotation_file(p, {})
            self.assertEqual(len(rows), 2)
            self.assertEqual(skipped, [])
            self.assertEqual(summary["multi_segment_example_count"], 1)
            self.assertEqual(rows[0]["source_video_id"], "144-05_cr")

    def test_spotting_wrapper_exports_annotations_and_forwards_protocol(self):
        import evaluation.tid_overlap_ncc_eval as evaluator
        from evaluation.evaluate_spotting import main
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            annotations = root / "annotations"
            annotations.mkdir()
            with (annotations / "aksam_export.csv").open("w", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerow(["0_0", json.dumps(["https://example.org/144-05_cr_0.2.mp4"]), "", "[1,2]", "", "{}"])
            manifest = root / "overlap_words.csv"
            manifest.write_text("word_name,word_dir\naksam,aksam\n", encoding="utf-8")
            argv = ["evaluate_spotting.py", "--method", "spatial", "--annotation_dir", str(annotations), "--word_manifest", str(manifest), "--tid_raw_roots", str(root / "raw"), "--output_dir", str(root / "out")]
            with patch.object(sys, "argv", argv), patch("train_utils.release_helpers.validate_evaluation_vocabulary"), patch("train_utils.release_helpers.validate_model_inputs"), patch.object(evaluator, "evaluate_ncc", return_value={"ok": True}) as evaluate:
                main()
            args = evaluate.call_args.args[0]
            self.assertTrue(args.no_checkpoint)
            self.assertEqual(args.ncc_embedding_source, "spatial")
            self.assertTrue((root / "out/labels").is_dir())

    def test_local_retrieval_protocol_remains_distinct(self):
        from evaluation.tid_overlap_local_all_vocab_ncc_eval import build_arg_parser
        args = build_arg_parser().parse_args([])
        self.assertEqual(args.search_scope, "local_gt")
        self.assertEqual(args.gt_buffer_seconds, 0.25)
        self.assertEqual(args.ncc_peak_position, "start")

    def test_checkpoint_selection_empty_and_minimum_loss(self):
        from train_utils.checkpoint_helpers import get_best_checkpoint_details
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(get_best_checkpoint_details(temp)[0], "")
            for name in ("best_val_loss_checkpoint_2_0.40.pt", "best_val_loss_checkpoint_5_0.20.pt"):
                (Path(temp) / name).touch()
            chosen = get_best_checkpoint_details(temp, filename_prefix="best_val_loss", score_mode="min")[0]
            self.assertTrue(chosen.endswith("_5_0.20.pt"))

    def test_spatial_method_rejects_checkpoint_before_loading_model(self):
        result = subprocess.run([sys.executable, str(ROOT / "evaluation/evaluate_spotting.py"), "--method", "spatial", "--checkpoint", "unused.pt"], capture_output=True, text=True, encoding="utf-8")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("spatial baseline does not take", result.stderr)

    def test_ncc_method_requires_explicit_checkpoint(self):
        result = subprocess.run([sys.executable, str(ROOT / "evaluation/evaluate_spotting.py"), "--method", "ncc"], capture_output=True, text=True, encoding="utf-8")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("explicit --checkpoint", result.stderr)


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name)
        self.env = patch.dict(os.environ, {"SIGN_SPOTTING_DATA_ROOT": str(self.data), "SIGN_SPOTTING_STAGE1_CKPT": ""})
        self.env.start()
        self.addCleanup(self.env.stop)
        for name in ("SIGN_SPOTTING_VOCAB", "SIGN_SPOTTING_ANNOTATIONS", "SIGN_SPOTTING_LMDB_ROOT"):
            os.environ.pop(name, None)

    def vocabulary(self, relative, mapping=None):
        p = self.data / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"dict_lem_to_id": mapping or {"akşam": 0, "gel": 1}, "dict_sentence": {"akşam gel": ["akşam", "gel"]}, "dict_lem_counter": {"akşam": 1, "gel": 1}}
        p.write_bytes(pickle.dumps(payload))
        return p

    def test_morphology_config_does_not_require_llm_vocabulary(self):
        self.vocabulary("TSL-News/processed_words.tsl3_pkl")
        from configs.tslnews.tslnews_stage1_configs.TSL_example_s1_dyn_4802_config import get_config
        cfg = get_config()
        self.assertEqual(cfg.model_params.post_params.num_classes, 2)
        self.assertTrue(cfg.train_ds_params.pseudo_gloss_dir.endswith("processed_words.tsl3_pkl"))

    def test_batch_and_worker_overrides_reach_all_loaders(self):
        self.vocabulary("TSL-News/6538_gloss/processed_words_map.tsl_pkl")
        from configs.tslnews.tslnews_stage1_configs.TSL_example_s1_dyn_config import get_config
        from train_utils.release_helpers import validate_config
        cfg = get_config()
        cfg.bs, cfg.num_workers = 1, 0
        validate_config(cfg)
        for params in (cfg.train_ds_params, cfg.valid_ds_params, cfg.test_ds_params):
            self.assertEqual((params.bs, params.num_workers), (1, 0))

    def test_checkpoint_evaluation_opens_only_selected_split(self):
        self.vocabulary("TSL-News/6538_gloss/processed_words_map.tsl_pkl")
        from configs.tslnews.tslnews_stage1_configs.TSL_example_s1_dyn_config import get_config
        from trainer.base.base_trainer import BaseTrainer
        cfg = get_config()
        cfg.run, cfg.evaluation_split = True, "dev"
        trainer = BaseTrainer.__new__(BaseTrainer)
        loader, metadata = object(), {"length": 2}
        with patch("trainer.base.base_trainer.get_aug", return_value=object()), patch("trainer.base.base_trainer.get_dataset", return_value=(loader, metadata)) as get_dataset:
            result = trainer.get_dataloaders(cfg)
        get_dataset.assert_called_once()
        self.assertEqual(get_dataset.call_args.args, (cfg.valid_ds_name, cfg.valid_ds_params))
        self.assertEqual(result, (loader, loader, loader, metadata, metadata, metadata))

    def test_model_input_check_reports_missing_fasttext(self):
        self.vocabulary("TSL-News/6538_gloss/processed_words_map.tsl_pkl")
        from configs.tslnews.tslnews_stage1_configs.TSL_example_s1_dyn_config import get_config
        from train_utils.release_helpers import validate_model_inputs
        with patch.dict(os.environ, {"SIGN_SPOTTING_FASTTEXT": str(self.data / "missing.bin")}):
            with self.assertRaisesRegex(FileNotFoundError, "Missing fastText"):
                validate_model_inputs(get_config())

    def test_missing_vocabulary_never_silently_uses_3000_classes(self):
        from configs.tslnews.tslnews_stage1_configs.TSL_example_s1_dyn_config import get_config
        with self.assertRaisesRegex(FileNotFoundError, "pseudo-gloss vocabulary"):
            get_config()

    def test_cpu_config_has_positive_batch_size(self):
        self.vocabulary("TSL-News/6538_gloss/processed_words_map.tsl_pkl")
        from configs.tslnews.tslnews_stage1_configs.TSL_example_s1_dyn_config import get_config
        with patch("torch.cuda.device_count", return_value=0):
            cfg = get_config()
        self.assertEqual(cfg.bs, 4)
        self.assertEqual(cfg.num_workers, 0)

    def test_noncontiguous_vocabulary_is_rejected(self):
        p = self.vocabulary("invalid.pkl", {"akşam": 0, "gel": 2})
        from train_utils.release_helpers import read_vocabulary
        with self.assertRaisesRegex(ValueError, "contiguous"):
            read_vocabulary(p)

    def test_equal_size_different_vocabulary_order_is_rejected(self):
        original = self.vocabulary("TSL-News/6538_gloss/processed_words_map.tsl_pkl")
        wrong = self.vocabulary("wrong.pkl", {"akşam": 1, "gel": 0})
        from train_utils.release_helpers import validate_evaluation_vocabulary
        with self.assertRaisesRegex(ValueError, "different IDs"):
            validate_evaluation_vocabulary("configs/tslnews/tslnews_stage1_configs/TSL_example_s1_dyn_config.py", wrong)

    def test_config_loading_accepts_absolute_paths(self):
        self.vocabulary("TSL-News/6538_gloss/processed_words_map.tsl_pkl")
        from train_utils.release_helpers import load_config
        path = ROOT / "configs/tslnews/tslnews_stage1_configs/TSL_example_s1_dyn_config.py"
        self.assertEqual(load_config(path).seed, 1)

    def test_translation_has_no_hardcoded_stage1_checkpoint(self):
        self.vocabulary("TSL-News/6538_gloss/processed_words_map.tsl_pkl")
        from configs.tslnews.tslnews_stage2_configs.TSL_example_s2_dyn_dinov2_6538_config import get_config
        cfg = get_config()
        self.assertFalse(cfg.model_params.stage1_ckpt)
        self.assertEqual(cfg.lr, 1e-4)
        self.assertEqual(cfg.lr_scheduler_params.decay_epochs, 75)

    def test_unset_stage1_checkpoint_is_not_a_translation_baseline(self):
        self.vocabulary("TSL-News/6538_gloss/processed_words_map.tsl_pkl")
        os.environ.pop("SIGN_SPOTTING_STAGE1_CKPT", None)
        from configs.tslnews.tslnews_stage2_configs.TSL_example_s2_dyn_dinov2_6538_config import get_config
        cfg = get_config()
        self.assertFalse(cfg.translation_from_scratch)
        for params in (cfg.train_ds_params, cfg.valid_ds_params, cfg.test_ds_params):
            Path(params.csv_dir).parent.mkdir(parents=True, exist_ok=True)
            Path(params.csv_dir).touch()
            Path(params.ds_params.lmdb_video_dir).mkdir(parents=True, exist_ok=True)
        from train_utils.release_helpers import validate_config
        with self.assertRaisesRegex(FileNotFoundError, "exact Stage-1 checkpoint"):
            validate_config(cfg, training=True)

    def test_translation_baseline_requires_explicit_none(self):
        self.vocabulary("TSL-News/6538_gloss/processed_words_map.tsl_pkl")
        from configs.tslnews.tslnews_stage2_configs.TSL_example_s2_dyn_dinov2_6538_config import get_config
        with patch.dict(os.environ, {"SIGN_SPOTTING_STAGE1_CKPT": "none"}):
            cfg = get_config()
        self.assertTrue(cfg.translation_from_scratch)
        self.assertIsNone(cfg.model_params.stage1_ckpt)


class NCCTests(unittest.TestCase):
    def test_exact_template_is_best_despite_feature_scale(self):
        import torch
        from evaluation.tid_analyzer_ncc import sliding_ncc
        template = torch.eye(3)
        sentence = torch.cat([-torch.ones(2, 3), template * 7, -torch.ones(2, 3)])
        scores = sliding_ncc(sentence, template)
        self.assertEqual(int(scores.argmax()), 2)
        self.assertAlmostEqual(float(scores[2]), 1.0)

    def test_ncc_stride_short_search_and_empty_template(self):
        import torch
        from evaluation.tid_analyzer_ncc import sliding_ncc
        self.assertEqual(len(sliding_ncc(torch.ones(9, 3), torch.ones(3, 3), stride=2)), 4)
        self.assertEqual(len(sliding_ncc(torch.ones(2, 3), torch.ones(3, 3))), 1)
        self.assertEqual(len(sliding_ncc(torch.ones(2, 3), torch.empty(0, 3))), 0)


class LMDBTests(unittest.TestCase):
    def test_repeated_reads_close_environments_and_decode_frames(self):
        import lmdb
        import numpy as np
        from PIL import Image
        from dataloaders.data_utils.lmdb_utils import LMDBUtility
        with tempfile.TemporaryDirectory() as temp:
            buffer = io.BytesIO()
            Image.new("RGB", (3, 2), (100, 20, 30)).save(buffer, format="PNG")
            with lmdb.open(temp, map_size=1024 * 1024) as env:
                with env.begin(write=True) as txn:
                    txn.put(b"details", pickle.dumps({"num_frames": 1}))
                    txn.put(b"0", buffer.getvalue())
            util = LMDBUtility(temp)
            for _ in range(10):
                frame = util.get_frames([np.int64(0)])[0]
                self.assertEqual(frame.shape, (2, 3, 3))
                self.assertEqual(frame[0, 0].tolist(), [100, 20, 30])
            with self.assertRaisesRegex(KeyError, "Missing frame 1"):
                util.get_frames([1])


class PreparationTests(unittest.TestCase):
    def test_mapping_pipeline_with_synthetic_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / "input", root / "output"
            source.mkdir()
            for split in ("train", "dev", "test"):
                (source / f"TSL-News.{split}.corpus.csv").write_text("name|translation|orth\nexample|akşam|na\n", encoding="utf-8")
            mapping = root / "mapping.json"
            mapping.write_text(json.dumps({"akşam": "akşam"}, ensure_ascii=False), encoding="utf-8")
            vocabulary = root / "vocabulary.txt"
            vocabulary.write_text("akşam\ngel\n", encoding="utf-8")
            scripts = ROOT / "scripts/tslnews/llm_pseudo_gloss_extraction"
            subprocess.run([sys.executable, str(scripts / "build_mapped_tslnews_corpora.py"), "--mapping", str(mapping), "--input-dir", str(source), "--output-dir", str(output)], check=True, capture_output=True)
            pkl = output / "vocab.pkl"
            subprocess.run([sys.executable, str(scripts / "build_pseudo_pkl_from_csv.py"), "--vocab", str(vocabulary), "--input-dir", str(output), "--out", str(pkl), "--strict"], check=True, capture_output=True)
            data = pickle.loads(pkl.read_bytes())
            self.assertEqual(data["dict_lem_to_id"], {"akşam": 0, "gel": 1})
            self.assertIn("akşam", data["dict_lem_to_id"])
            self.assertTrue(data["dict_sentence"]["akşam"])


if __name__ == "__main__":
    unittest.main()
