# syntax=docker/dockerfile:1.7
# =============================================================================
# DeepSeek-V4-Flash on SM89 (L40S) — vLLM container for Kubernetes
#
# Based on: https://github.com/yhfgyyf/vllm-deepseek-v4-sm89
# Release:  v0.23.1rc1.dev904-g8e321cc4f-cu130-sm89
# Stack:    Python 3.12 + CUDA 13.0 + torch 2.11.0+cu130 + FlashInfer 0.6.14 SM89
#
# NOTE on SM89 FlashInfer JIT: the sparse-MLA kernel is JIT-compiled from source
# at runtime by FlashInfer. Its JIT path requires `nvcc` (CUDA toolkit). We
# therefore base the runtime stage on the CUDA *devel* image so nvcc is present.
# A runtime-only base will fail with "nvcc not found" during kernel warmup.
# =============================================================================

# ---- Stage 1: fetch the two official prebuilt release wheels -----------------
FROM docker.io/curlimages/curl:8.10.1 AS wheels
USER root
ARG RELEASE_TAG=v0.23.1rc1.dev904-g8e321cc4f-cu130-sm89

# Asset names embed the release version; update together with RELEASE_TAG.
ENV VLLM_WHEEL_URL="https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/releases/download/${RELEASE_TAG}/vllm-0.23.1rc1.dev904%2Bg8e321cc4f.cu130-cp312-cp312-linux_x86_64.whl"
ENV FLASHINFER_WHEEL_URL="https://github.com/yhfgyyf/vllm-deepseek-v4-sm89/releases/download/${RELEASE_TAG}/flashinfer_python-0.6.14%2Bsm89.1-py3-none-any.whl"

RUN mkdir -p /wheels \
 && curl -fL --retry 3 -o /wheels/vllm.whl        "$VLLM_WHEEL_URL" \
 && curl -fL --retry 3 -o /wheels/flashinfer.whl  "$FLASHINFER_WHEEL_URL"

# Optional integrity check (SHA-256 from the release notes)
# RUN echo "d1a4e4ee3f64882f129a8d12e47bcd70c46992b83a16c2e6c722d85dbe87b41c  /wheels/vllm.whl"        | sha256sum -c - \
#  && echo "667e4c1c1a288681493e192a0d02976e791078a8792c47bad4d7f9186109554e  /wheels/flashinfer.whl" | sha256sum -c -

# ---- Stage 2: runtime image ------------------------------------------------
# CUDA devel = nvcc present for FlashInfer runtime JIT (matches validated env).
FROM nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Python 3.12 + tools. build-essential/gcc are needed by FlashInfer's runtime
# JIT (host-side c++ compilation of the generated .cu/.so), ffmpeg for vLLM I/O.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev \
        build-essential gcc g++ ninja-build \
        curl git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast, reproducible Python installer used by the repo)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV UV_LINK_MODE=copy \
    PATH="/root/.local/bin:${PATH}"

# Isolated venv (keeps system python clean and makes PATH management trivial)
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:${PATH}"

# Copy the prebuilt wheels from stage 1
COPY --from=wheels /wheels /wheels

# Install torch (cu130) + the two SM89 wheels + flashinfer-cubin in one resolver
# invocation. `--torch-backend=cu130` pins torch 2.11.0+cu130 & friends to the
# PyTorch cu130 index and keeps CuTe DSL at the vLLM-validated 4.5.2 build.
RUN uv pip install --python "$VIRTUAL_ENV/bin/python" \
        /wheels/flashinfer.whl \
        /wheels/vllm.whl \
        "flashinfer-cubin==0.6.13" \
        --torch-backend=cu130

# Build-time sanity check (no GPU required; catches ABI/resolution errors early)
RUN python -c "import torch, flashinfer; from importlib.metadata import version; \
print('torch', torch.__version__, '| vllm', version('vllm'), '| flashinfer ok')"

# ---- Runtime configuration required by the SM89 fork ------------------------
# FlashInfer's 0.6.14 SM89 fork JIT-compiles the sparse-MLA kernel; the release
# requires disabling the version check that would reject the patched package.
ENV FLASHINFER_DISABLE_VERSION_CHECK=1 \
    VLLM_USE_V1=1 \
    CUDA_HOME=/usr/local/cuda \
    # Model weights default mount point (override per-deployment via PVC)
    MODEL_PATH=/models/DeepSeek-V4-Flash \
    HOST=0.0.0.0 \
    PORT=8000 \
    # Library path extras from the release notes (PyNvVideoCodec for FFmpeg, venv libs)
    LD_LIBRARY_PATH="/opt/venv/lib/python3.12/site-packages/PyNvVideoCodec:/opt/venv/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

WORKDIR /workspace
EXPOSE 8000

# Entrypoint wrapper maps environment variables -> `vllm serve` flags
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
