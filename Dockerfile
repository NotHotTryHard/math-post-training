FROM ghcr.io/astral-sh/uv:0.12.3-python3.12-trixie-slim

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --group dev --group train --group eval

COPY src/ ./src/
COPY configs/ ./configs/
COPY tests/ ./tests/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --group dev --group train --group eval

CMD ["sleep", "infinity"]
