"""Optional logging defaults. Export credentials in your shell when enabling W&B."""
CONFIG = {
    "WANDB_IGNORE_GLOBS": "*.patch",
    "WANDB_DISABLE_CODE": "true",
    "TOKENIZERS_PARALLELISM": "false",
}
