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
    UV_COMPILE_BYTECODE=1 \
    CC=gcc \
    HF_HOME=/workspace/.cache/huggingface \
    WANDB_DIR=/workspace/wandb

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --verbose --frozen --no-install-project \
        --group dev --group train --group eval --group vllm

COPY src/ ./src/
COPY configs/ ./configs/
COPY tests/ ./tests/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --group dev --group train --group eval --group vllm

CMD ["sleep", "infinity"]
