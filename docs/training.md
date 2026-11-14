# Training and checkpoint evaluation

Run from the repository root after preparing the inputs described in [data.md](data.md).

## Stage 1

```bash
python main.py --config configs/tslnews/tslnews_stage1_configs/TSL_example_s1_dyn_config.py
python main.py --config configs/tslnews/tslnews_stage1_configs/TSL_example_s1_dyn_4802_config.py
```

Use separate invocations for the two experiments. Each config uses the corresponding ordered vocabulary. Default training settings: 100 epochs, learning rate 3e-4, weight decay 0.001, five warmup epochs, and accumulation of two microbatches. Stage-1 configured batch size is 4 times visible GPU count; Ignite distributes this global batch across ranks. Select visible GPUs with `CUDA_VISIBLE_DEVICES`. Multi-GPU launch uses Ignite from `main.py`; do not wrap that command in another launcher.

CLI overrides are supported by ml-collections, e.g. `--config.bs=2 --config.num_workers=0`; these apply to the data loaders. Changing batch size, accumulation, or the scheduler defines a different training run. Training requires CUDA; CPU is suitable for lightweight correctness checks.

Checkpoints are saved under `checkpoints/tslnews_stage1_configs/<project_name>/` by default. The trainer saves best class-F1, best validation loss, and latest checkpoints. Set `SIGN_SPOTTING_CHECKPOINT_ROOT` to relocate them. Retain the config and exact vocabulary alongside every checkpoint. To resume Stage 1, set `SIGN_SPOTTING_RUN_NAME` to the original run name (or override `--config.save_dir` with its exact directory) and add `--config.resume=True`. Each training run saves its configuration as `run_config.json` in the checkpoint directory.


## Short startup check

Once your private inputs and pretrained weights are configured, a two-batch startup check can exercise training and validation:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/tslnews/tslnews_stage1_configs/TSL_example_s1_dyn_config.py \
  --config.bs=1 --config.num_workers=0 --config.accum=1 \
  --config.train_length=2 --config.val_length=2 --config.max_epochs=1 \
  --config.lr_scheduler=none --config.save_dir=checkpoints/startup_check
```

This uses a separate output directory and disables the full-training learning-rate schedule. Its metrics are not benchmark results. Inside Docker, use `--config.save_dir=/runs/checkpoints/startup_check` for persistent output.

## Downstream translation check

Use the fixed DINOv2/LLM-vocabulary translation configuration:

```bash
export SIGN_SPOTTING_STAGE1_CKPT=/path/to/exact_stage1_checkpoint.pt
python scripts/check_setup.py --config configs/tslnews/tslnews_stage2_configs/TSL_example_s2_dyn_dinov2_6538_config.py --training
python main.py --config configs/tslnews/tslnews_stage2_configs/TSL_example_s2_dyn_dinov2_6538_config.py
```

The recipe retains XGLM-1.7B adapters, learning rate 1e-4, and cosine decay to epoch 75. For the no-pretraining comparison, explicitly set `SIGN_SPOTTING_STAGE1_CKPT=none`, keeping all other settings fixed and using a separate save directory. Stage-1 vocabulary and Stage-2 vocabulary must have the same class IDs. XGLM pretrained files are obtained through Hugging Face on first use.

Stage 2 saves its primary best checkpoint by validation teacher-forced BLEU-4, as configured. Generated validation runs every ten epochs. Teacher-forced (`obleu`/`orouge`) and generated (`ableu`/`arouge`) metrics are different; do not report one as the other.

## Evaluate a saved checkpoint

```bash
python -m evaluation.evaluate_checkpoint \
  --config configs/tslnews/tslnews_stage2_configs/TSL_example_s2_dyn_dinov2_6538_config.py \
  --checkpoint /path/to/stage2_checkpoint.pt --split test --batch-size 1 --num-workers 0 \
  --output outputs/translation_test.json
```

The same command accepts either Stage-1 configuration to evaluate pseudo-gloss metrics. Use `--split dev` for validation. Checkpoint evaluation opens only the requested split, uses batch size 1 and zero workers by default, and does not create training checkpoint directories. Loading a full Stage-2 checkpoint does not require the original Stage-1 checkpoint; it already contains encoder weights. Exact model loading is strict. The vocabulary PKL, Turkish fastText vectors, and base pretrained model inputs are still required to construct the model.

For new experiments, use validation to select checkpoints and decoding settings, then evaluate the held-out test split once. The preparation utilities build vocabularies from all three corpus splits; use the same vocabulary protocol when comparing results.
