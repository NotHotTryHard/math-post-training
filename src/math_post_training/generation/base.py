"""Options shared by text-generation backends.

Every backend exposes ``generate(prompts, config)`` and returns one list of
completions for each prompt.
"""

from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Generation options supported by every project backend."""

    max_new_tokens: int = 256
    num_return_sequences: int = 1
    do_sample: bool = True
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int | None = None
    seed: int | None = None
    stop_strings: list[str] | None = None
