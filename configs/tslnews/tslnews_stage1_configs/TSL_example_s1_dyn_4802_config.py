# Adapted for sign representation training and evaluation.
from pathlib import Path
from configs.tslnews.tslnews_stage1_configs.TSL_example_s1_dyn_config import get_config as get_base_config

def get_config():
    cfg = get_base_config(vocabulary="morphology")
    cfg.name = Path(__file__).stem
    return cfg
