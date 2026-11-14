# Docker

The image provides the Linux CUDA environment for training and evaluation on an x86-64 NVIDIA GPU host. It uses PyTorch 2.3.0, CUDA 11.8, and the dependencies in `requirements.txt`, including the explicit CUDA 11.8 xFormers wheel from the [PyTorch package index](https://download.pytorch.org/whl/cu118/xformers/). No datasets, generated vocabularies, checkpoints, or pretrained weights are included in the image.

## Host setup

- Windows: install Docker Desktop, enable Linux containers with its WSL 2 backend, and configure a compatible NVIDIA driver. See [Docker's GPU setup instructions](https://docs.docker.com/desktop/features/gpu/).
- Linux: install Docker, a compatible NVIDIA driver, and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- macOS: use an NVIDIA GPU host remotely for CUDA training. This image does not provide CUDA acceleration through Apple GPUs.

## Build

Run from the repository root:

```bash
docker build -t gloss-free-sign-spotting .
```

Optional pseudo-gloss preparation and W&B dependencies can be enabled at build time:

```bash
docker build --build-arg INSTALL_PREPROCESSING=true --build-arg INSTALL_LOGGING=true -t gloss-free-sign-spotting .
```

## Start on Windows (PowerShell)

Replace the example host paths with existing folders. The data folder must contain the prepared `TSL-News/` and `TID/` layouts described in [data.md](data.md). The weights folder must contain `cc.tr.300.bin`; it can also contain local DINOv2 and experiment checkpoints. Create the runs folder before starting the container.

```powershell
docker run --rm -it --gpus all --shm-size=8g `
  --mount "type=bind,source=D:\datasets\sign-spotting,target=/data,readonly" `
  --mount "type=bind,source=D:\models\sign-spotting,target=/weights,readonly" `
  --mount "type=bind,source=D:\runs\sign-spotting,target=/runs" `
  --mount "type=volume,source=sign-spotting-cache,target=/cache" `
  gloss-free-sign-spotting
```

## Start on Linux

Replace the example host paths with existing folders:

```bash
docker run --rm -it --gpus all --shm-size=8g \
  --mount type=bind,source=/path/to/data,target=/data,readonly \
  --mount type=bind,source=/path/to/weights,target=/weights,readonly \
  --mount type=bind,source=/path/to/runs,target=/runs \
  --mount type=volume,source=sign-spotting-cache,target=/cache \
  gloss-free-sign-spotting
```

For data preparation, omit `readonly` on a working data-folder mount: preparation scripts write CSVs and LMDBs. Training and evaluation can use the read-only mounts shown above.

The container opens a shell in the repository directory. Data and weights are mounted read-only. New checkpoints go to `/runs/checkpoints`; use `/runs/outputs` for evaluation outputs. These files persist in the host runs folder. Downloaded base-model files are cached in the named Docker volume.

## Commands inside the container

Check the environment and run the synthetic tests:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m unittest discover -s tests -v
```

The input checker verifies corpus/LMDB paths, vocabulary alignment, Turkish fastText vectors, and any explicitly configured local DINOv2 weights. Check prepared inputs and train:

```bash
python scripts/check_setup.py --config configs/tslnews/tslnews_stage1_configs/TSL_example_s1_dyn_config.py --training
python main.py --config configs/tslnews/tslnews_stage1_configs/TSL_example_s1_dyn_config.py
```

To use a local DINOv2 base weight file, set `SIGN_SPOTTING_DINOV2_WEIGHTS=/weights/dinov2_vits14_pretrain.pth`. Otherwise the model loader downloads the base weights into the cache. For translation initialization, set `SIGN_SPOTTING_STAGE1_CKPT` to the matching checkpoint under `/weights` or `/runs/checkpoints`.

Run evaluation with the commands in [evaluation.md](evaluation.md), using container paths such as `/data/TSL-News/...` and `/weights/checkpoint.pt`. Set `--output_dir /runs/outputs/<run-name>` to preserve its outputs. Checkpoint evaluation supports `--output /runs/outputs/checkpoint_metrics.json`.

For optional W&B logging, supply your own credentials at runtime and enable `SIGN_SPOTTING_WANDB=1`. Credentials must not be added to the Dockerfile or build arguments.

Synthetic tests can also run without mounting data or passing a GPU:

```bash
docker run --rm gloss-free-sign-spotting python -m unittest discover -s tests -v
```

The datasets remain unreleased. A Docker image supplies the software environment; full paper experiments still require the external inputs described in [data.md](data.md).
