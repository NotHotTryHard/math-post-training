"""Text-generation backends with a shared, engine-neutral contract."""

from math_rl.generation.base import (
    GenerationBackend,
    GenerationConfig,
    GenerationOutput,
)

__all__ = ["GenerationBackend", "GenerationConfig", "GenerationOutput"]

