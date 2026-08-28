FROM ghcr.io/astral-sh/uv:0.12.3-python3.12-trixie-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        curl \
        libc6-dev \
        libnuma1 \
        tmux \
        unzip && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/app/.venv/bin:$PATH" \
    UV_LINK_MODE=hardlink \
    CC=gcc \
    HF_HOME=/workspace/.cache/huggingface \
    WANDB_DIR=/workspace/wandb

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --no-cache --verbose --frozen --no-install-project \
        --group dev --group train --group eval --group vllm

COPY src/ ./src/
COPY configs/ ./configs/
COPY tests/ ./tests/

RUN uv sync --no-cache --frozen \
        --group dev --group train --group eval --group vllm

# vLLM enables FlashInfer sampling by default, but the regular FlashInfer
# wheels do not include its precompiled sampler. Without this kernel FlashInfer
# attempts a JIT build at runtime, while the slim image intentionally has no
# CUDA toolkit/nvcc. Extract only the matching CUDA 13.0 sampler from the
# official JIT-cache wheel instead of adding several GB of unused kernels.
RUN curl -fL --retry 5 \
        -o /tmp/flashinfer-jit-cache.whl \
        "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.13/flashinfer_jit_cache-0.6.13%2Bcu130-cp39-abi3-manylinux_2_28_x86_64.whl" && \
    echo "99d59b3ea32997fdb5f6c4898019611c751f96a0a4f5cc8be32efd058c1d2c46  /tmp/flashinfer-jit-cache.whl" | sha256sum -c - && \
    mkdir -p /app/.venv/lib/python3.12/site-packages/flashinfer/data/aot/sampling && \
    unzip -j /tmp/flashinfer-jit-cache.whl \
        "*/flashinfer_jit_cache/jit_cache/sampling/sampling.so" \
        -d /app/.venv/lib/python3.12/site-packages/flashinfer/data/aot/sampling && \
    rm /tmp/flashinfer-jit-cache.whl && \
    test -f /app/.venv/lib/python3.12/site-packages/flashinfer/data/aot/sampling/sampling.so

CMD ["sleep", "infinity"]
