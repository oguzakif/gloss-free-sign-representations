FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    SIGN_SPOTTING_DATA_ROOT=/data \
    SIGN_SPOTTING_CHECKPOINT_ROOT=/runs/checkpoints \
    SIGN_SPOTTING_FASTTEXT=/weights/cc.tr.300.bin \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    XDG_CACHE_HOME=/cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates ffmpeg libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/gloss-free-sign-spotting

# Install dependencies before source files to reuse the dependency build layer.
COPY requirements.txt requirements-preprocessing.txt requirements-logging.txt ./
RUN python -m pip install "numpy==1.24.4" "pybind11==2.13.6" setuptools wheel \
    && python -m pip install --no-build-isolation "fasttext==0.9.2" \
    && python -m pip install -r requirements.txt

ARG INSTALL_PREPROCESSING=false
ARG INSTALL_LOGGING=false
RUN if [ "$INSTALL_PREPROCESSING" = "true" ]; then \
        python -m pip install -r requirements-preprocessing.txt; \
    fi \
    && if [ "$INSTALL_LOGGING" = "true" ]; then \
        python -m pip install -r requirements-logging.txt; \
    fi

COPY . .
RUN mkdir -p /data /weights /runs/checkpoints /runs/outputs /cache

CMD ["/bin/bash"]
