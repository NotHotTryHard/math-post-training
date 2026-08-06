"""Shared loading of causal language models and their tokenizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)


class CausalLanguageModel(Protocol):
    """Operations used from an auto-loaded causal language model."""

    def eval(self) -> CausalLanguageModel: ...

    def generate(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class ModelLoadConfig:
    """Arguments shared by inference, SFT, GRPO, and evaluation model loading."""

    name_or_path: str
    dtype: str | torch.dtype = "auto"
    trust_remote_code: bool = False


def load_model_and_tokenizer(
    config: ModelLoadConfig,
    *,
    device: str | torch.device | None = None,
) -> tuple[CausalLanguageModel, PreTrainedTokenizerBase]:
    """Load a Hugging Face model ID or a local ``save_pretrained`` checkpoint."""

    tokenizer = AutoTokenizer.from_pretrained(
        config.name_or_path,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.name_or_path,
        dtype=config.dtype,
        trust_remote_code=config.trust_remote_code,
    )
    if device is not None:
        torch.nn.Module.to(model, device=device)

    return cast(CausalLanguageModel, model), tokenizer
