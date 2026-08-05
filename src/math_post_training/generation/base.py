"""Shared contract and value objects for text-generation backends."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Sampling options that have the same meaning across supported engines."""

    max_new_tokens: int
    num_return_sequences: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int | None = None
    stop: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GenerationOutput:
    """All completions produced for one input prompt."""

    prompt: str
    completions: tuple[str, ...]


class GenerationBackend(Protocol):
    """Engine-neutral interface implemented by generation adapters."""

    def generate(
        self,
        prompts: Sequence[str],
        config: GenerationConfig,
    ) -> list[GenerationOutput]:
        """Generate completions while preserving the order of input prompts."""
        ...
