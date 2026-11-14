# Adapted for sign representation training and evaluation.
from ml_collections import config_dict
from pathlib import Path
import os
import numpy as np
import torch
import importlib
from train_utils.checkpoint_helpers import (
    get_best_checkpoint_details,
)
from configs.base.base_utils import *


def get_config(
    stage1_module=None,
    project_suffix=None,
    stage1_ckpt=None,
    tsl_ann_root=None,
    tsl_pseudo_gloss=None,
):
    cfg = config_dict.ConfigDict()
    cfg.name = Path(os.path.realpath(__file__)).stem
    base_name = Path(os.path.realpath(__file__)).parent.name
    code_path = str(Path(os.path.realpath(__file__)).resolve().parents[3])
    cfg.project_name = "final_tslnews_translation_only_local"
    if project_suffix:
        cfg.project_name = f"{cfg.project_name}_{project_suffix}"

    ckpt_path = get_checkpoint_path(base_name, cfg.project_name)
    lmdb_path = get_lmdb_path()
    cfg.save_dir = f"{ckpt_path}"

    # Training mode (do not set cfg.run for eval-only here)
    # To evaluate only later, you can set these in a copy of this config:
    #cfg.run = True;
    #cfg.load_from_ckpt = "best"

    cfg.main_runner = "trainer.complete_translation_trainer"
    cfg.aug_name = "augmentation.video.base_video_aug"

    cfg.aug_params = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "strength": 0.2,
        "random_shift": 4,
        "stride": 2,
        "max_seq_len": 256,
    }

    base_bs = 8 # For HPC experiments, use 4 or 2 depending on GPU memory in local.
    cfg.bs = int(
        base_bs
        * max(torch.cuda.device_count(), 1)
    )
    cfg.accum = 1 # Adjust it to batch size to be 8 at least.
    cfg.num_workers = min(min(cfg.bs, int(10 * torch.cuda.device_count())), 10)

    cfg.gate_grad_multiplier = 1.0

    cfg.lr = 3e-4
    cfg.lr_scheduler = "warmupwithcosine"
    cfg.lr_scheduler_params = config_dict.ConfigDict(
        {
            "lr_scale_factor": 0.01,
            "num_cycles": 1,
            "start_value_mult": 0.7,
            "end_value_mult": 0.7,
            "warmup_epochs": 5,
        }
    )

    cfg.optimizer_name = "adamw"
    cfg.optimizer_params = config_dict.ConfigDict(
        {
            "lr": cfg.lr,
            "weight_decay": 0.001,
        }
    )

    cfg.criterion_name = "losses.base_loss"

    cfg.criterion_params = config_dict.ConfigDict(
        {
            "dict_of_loss_params": {
                "ce": {
                    "cls_name": "losses.loss_functions.ce_loss",
                    "loss_params": {
                        "src_name": "logits",
                        "tgt_name": "gt_ids",
                        "mask_name": "gt_text_mask",
                        "label_smoothing": 0.1,
                    },
                    "weight": 1.0,
                },
            }
        }
    )

    cfg.max_epochs = 100
    cfg.model_checkpoint_dir = ""

    # Reuse Phoenix-style CSV dataloader for TSL-News
    cfg.train_ds_name = "dataloaders.phoenix_video_dataset"
    cfg.valid_ds_name = "dataloaders.phoenix_video_dataset"
    cfg.test_ds_name = "dataloaders.phoenix_video_dataset"

    default_ann, default_vocab = get_tsl_paths("llm")
    tsl_ann_root = tsl_ann_root or str(default_ann)
    tsl_pseudo_gloss = tsl_pseudo_gloss or str(default_vocab)

    train_ds_params = {
        "csv_dir": f"{tsl_ann_root}/TSL-News.train.corpus.csv",
        "pseudo_gloss_dir": tsl_pseudo_gloss,
        "sep": "|",
        "name": "translation",
        "ds_params": {
            "lmdb_video_dir": f"{lmdb_path}/TSL-News/lmdb_videos/train",
            "isValid": False,
        },
        "shuffle": True,
        "num_workers": cfg.num_workers,
        "bs": cfg.bs,
        "drop_last": True,
    }
    cfg.train_ds_params = config_dict.ConfigDict(train_ds_params)

    valid_ds_params = {
        "csv_dir": f"{tsl_ann_root}/TSL-News.dev.corpus.csv",
        "pseudo_gloss_dir": tsl_pseudo_gloss,
        "sep": "|",
        "name": "translation",
        "ds_params": {
            "lmdb_video_dir": f"{lmdb_path}/TSL-News/lmdb_videos/dev",
            "isValid": True,
        },
        "shuffle": False,
        "num_workers": cfg.num_workers,
        "bs": cfg.bs,
        "drop_last": False,
    }
    cfg.valid_ds_params = config_dict.ConfigDict(valid_ds_params)

    test_ds_params = {
        "csv_dir": f"{tsl_ann_root}/TSL-News.test.corpus.csv",
        "pseudo_gloss_dir": tsl_pseudo_gloss,
        "sep": "|",
        "name": "translation",
        "ds_params": {
            "lmdb_video_dir": f"{lmdb_path}/TSL-News/lmdb_videos/test",
            "isValid": True,
        },
        "shuffle": False,
        "num_workers": cfg.num_workers,
        "bs": cfg.bs,
        "drop_last": False,
    }
    cfg.test_ds_params = config_dict.ConfigDict(test_ds_params)

    cfg.lm_name = "facebook/xglm-1.7B"
    cfg.additional_tokens = {}
    cfg.pretext = ""
    # No replacement pickle for TSL-News by default (optional)
    cfg.replacement_pickle = None

    # Stage-1 (TSL) dependency
    cfg.stage1_name = stage1_module or "configs.tslnews.tslnews_stage1_configs.TSL_example_s1_dyn_config"
    mod = importlib.import_module(cfg.stage1_name, package=None)
    stage1_config = mod.get_config()

    stage1_name = stage1_config["model_name"]
    stage1_params = stage1_config["model_params"].to_dict()
    stage1_ckpt = os.environ.get("SIGN_SPOTTING_STAGE1_CKPT", stage1_ckpt)
    cfg.translation_from_scratch = isinstance(stage1_ckpt, str) and stage1_ckpt.lower() == "none"
    if cfg.translation_from_scratch:
        stage1_ckpt = None

    cfg.model_name = "models.trial_models.test_stage2_model"
    cfg.model_params = {
        "stage1_name": stage1_name,
        "stage1_params": stage1_params,
        "stage1_ckpt": stage1_ckpt,
        "post_name": "models.post_models.linear_pos_head",
        "post_params": {"pos_type": "sine", "pre_pos": True},
        "llm_name": cfg.lm_name,
        "lang_backbone_name": "models.huggingface.modeling_xglm",
        "adaptor_params": {
            "adapt_layers": list(np.arange(0, 24, 1)),
            "lora_layers": list(np.arange(0, 24, 1)),
            "w_lora_ff": False,
            "lora_rank": 4,
            "lora_drop": 0.1,
            "gate_type": "clamp",
            "lora_a": 4.0,
            "adapt_tokens": False,
        },
        "freeze": False,
    }

    cfg.gen_params = {"max_length": 64, "temperature": 1.0, "num_beams": 4}

    cfg.seed = 1
    cfg.grad_clip_norm = 1.0
    cfg.grad_clip_value = 1.0
    cfg.mixup = False
    cfg.logger_name = ["text"]
    cfg.resume = False
    cfg.train_length = config_dict.FieldReference(None, field_type=int)
    cfg.val_length = config_dict.FieldReference(None, field_type=int)
    cfg.log_every = 100
    cfg.save_ckpt = True
    cfg.score_factor = 1
    cfg.score_name = "valid/obleu"
    cfg.checkpoint_monitors = config_dict.ConfigDict(
        {
            "best_result": config_dict.ConfigDict(
                {
                    "metric": "valid/obleu",
                    "metric_key": "bleu4",
                    "mode": "max",
                }
            ),
            "best_val_loss": config_dict.ConfigDict(
                {
                    "metric": "valid/avg_loss",
                    "metric_key": None,
                    "mode": "min",
                }
            ),
        }
    )
    cfg.bfloat16_only = False

    cfg.apply_metric_splitter = False
    cfg.append_string = ""
    cfg.watch_grad = True
    return cfg
