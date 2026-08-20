FROM ghcr.io/astral-sh/uv:0.12.3-python3.12-trixie-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        libc6-dev \
        libnuma1 \
        tmux && \
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

# vLLM 0.26 enables the FlashInfer sampler by default. Its PyPI wheel omits
# the precompiled kernels, so verify that the separately locked CUDA 13 cache
# contains the sampler and will not fall back to runtime nvcc compilation.
RUN python -c "from flashinfer.jit.sampling import gen_sampling_module; assert gen_sampling_module().is_aot"

CMD ["sleep", "infinity"]
