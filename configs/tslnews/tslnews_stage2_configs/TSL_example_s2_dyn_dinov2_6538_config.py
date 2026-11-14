# Adapted for sign representation training and evaluation.
from pathlib import Path
from configs.base.base_utils import get_checkpoint_path
from configs.tslnews.tslnews_stage2_configs.TSL_example_s2_dyn_config import get_config as get_base_config

def get_config():
    cfg = get_base_config(project_suffix="dinov2_6538")
    cfg.name = Path(__file__).stem + "_lr1e4_decay75"
    cfg.project_name = "tslnews_translation_dinov2_6538_lr1e4_decay75"
    cfg.save_dir = str(get_checkpoint_path(Path(__file__).parent.name, cfg.project_name))
    cfg.lr = 1e-4
    cfg.optimizer_params.lr = cfg.lr
    cfg.lr_scheduler_params.decay_epochs = 75
    cfg.resume = False
    return cfg
