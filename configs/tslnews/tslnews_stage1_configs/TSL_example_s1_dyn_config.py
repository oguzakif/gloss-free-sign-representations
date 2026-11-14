# Adapted for sign representation training and evaluation.
from datetime import datetime

from ml_collections import config_dict
from pathlib import Path
import os
import torch
import pickle
from configs.base.base_utils import *


def _num_classes_from_pkl(pkl_path: str) -> int:
    from train_utils.release_helpers import read_vocabulary
    return len(read_vocabulary(pkl_path)["dict_lem_to_id"])


def get_config(vocabulary="llm"):
    cfg = config_dict.ConfigDict()
    cfg.name = Path(os.path.realpath(__file__)).stem
    base_name = Path(os.path.realpath(__file__)).parent.name
    code_path = str(Path(os.path.realpath(__file__)).resolve().parents[3])
    lmdb_path = get_lmdb_path()

    cfg.main_runner = "trainer.psuedo_gloss_trainer"
    cfg.project_name = os.environ.get("SIGN_SPOTTING_RUN_NAME", f"tslnews_pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    cfg.aug_name = "augmentation.video.base_video_aug"

    ckpt_path = get_checkpoint_path(base_name, cfg.project_name)
    cfg.save_dir = f"{ckpt_path}"

    cfg.aug_params = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "strength": 0.2,
        "random_shift": 4,
        "stride": 2,
        "max_seq_len": 256,
    }

    base_bs = 4
    cfg.bs = int(
        base_bs
        * max(torch.cuda.device_count(), 1)
    )
    cfg.accum = 2
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
                "bce": {
                    "cls_name": "losses.loss_functions.bce_loss",
                    "loss_params": {
                        "src_name": (
                            "dict_post_output",
                            "logits",
                        ),
                        "tgt_name": "pseudo_gloss_ids",
                    },
                    "weight": 10.0,
                },
            }
        }
    )

    cfg.max_epochs = 100
    cfg.model_checkpoint_dir = ""

    # Reuse Phoenix-style dataloader (expects CSV with name|translation|orth)
    cfg.train_ds_name = "dataloaders.phoenix_video_dataset"
    cfg.valid_ds_name = "dataloaders.phoenix_video_dataset"
    cfg.test_ds_name = "dataloaders.phoenix_video_dataset"

    tsl_ann_root, tsl_pseudo_gloss = map(str, get_tsl_paths(vocabulary))

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

    from configs.standards.standard_meta_model_zero_config import get_sign_encoder

    model_name, sign_model_params, dim_model = get_sign_encoder()
    cfg.model_name = model_name

    dyn_num_classes = _num_classes_from_pkl(tsl_pseudo_gloss)

    post_params = {
        "in_dim": dim_model,
        "hidden_dim": 300,
        "num_classes": dyn_num_classes,
        "dropout": 0.2,
        "class_temperature": 0.1,
        "time_temperature": 0.1,
        "dynamic_time_temperatures": True,
        "dynamic_class_temperatures": True,
        "emb_lang": "tr",
        "emb_pkl_dir": tsl_pseudo_gloss,
        "trainable_emb": False,
    }

    cfg.model_params = {
        "sign_model_name": "models.model_sign_encoder.basic_sign_encoder",
        "sign_model_params": sign_model_params,
        "post_name": "models.metaformer.post.zero_fasttext_prototype_head",
        "post_params": post_params,
    }

    cfg.seed = 1
    cfg.grad_clip_norm = 1.0
    cfg.grad_clip_value = 1.0
    cfg.logger_name = ["text"]
    cfg.resume = False
    cfg.train_length = config_dict.FieldReference(None, field_type=int)
    cfg.val_length = config_dict.FieldReference(None, field_type=int)
    cfg.log_every = 100
    cfg.save_ckpt = True
    cfg.score_factor = 1
    cfg.score_name = "valid/class_f1_score"
    cfg.checkpoint_monitors = config_dict.ConfigDict(
        {
            "best_result": config_dict.ConfigDict(
                {
                    "metric": "valid/class_f1_score",
                    "metric_key": None,
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
    cfg.mixup = False

    return cfg
