"""Text generation through Hugging Face Transformers, with a small CLI."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import torch
from transformers import StoppingCriteriaList, StopStringCriteria, set_seed

from math_post_training.config import ConfigError, load_yaml_config
from math_post_training.generation.base import GenerationConfig, GenerationOutput
from math_post_training.model import ModelLoadConfig, load_model_and_tokenizer

DEFAULT_CONFIG_PATH = Path("configs/current.yaml")


class TransformersBackend:
    """Generate completions using ``PreTrainedModel.generate``."""

    def __init__(
        self,
        model_config: ModelLoadConfig,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        self.device = _resolve_device(device)
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_config,
            device=self.device,
        )
        self.tokenizer.padding_side = "left"
        self.model.eval()

    def render_chat_prompt(self, user_prompt: str, *, system_prompt: str | None = None) -> str:
        """Render user input with the tokenizer's own chat template."""

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return cast(str, rendered)

    def generate(
        self,
        prompts: Sequence[str],
        config: GenerationConfig,
    ) -> list[GenerationOutput]:
        """Generate one or more completions for every pre-rendered prompt."""

        prompt_list = list(prompts)
        if not prompt_list:
            return []
        _validate_generation_config(config)

        if config.seed is not None:
            set_seed(config.seed)

        inputs = self.tokenizer(
            prompt_list,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        prompt_width = inputs["input_ids"].shape[1]

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": config.max_new_tokens,
            "num_return_sequences": config.num_return_sequences,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": config.temperature > 0,
        }
        if config.temperature > 0:
            generation_kwargs.update(temperature=config.temperature, top_p=config.top_p)
        if config.stop:
            generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [StopStringCriteria(self.tokenizer, list(config.stop))]
            )

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        completions = self.tokenizer.batch_decode(
            output_ids[:, prompt_width:],
            skip_special_tokens=True,
        )
        completions = [_trim_at_stop(text, config.stop).strip() for text in completions]

        outputs: list[GenerationOutput] = []
        for prompt_index, prompt in enumerate(prompt_list):
            start = prompt_index * config.num_return_sequences
            end = start + config.num_return_sequences
            outputs.append(
                GenerationOutput(prompt=prompt, completions=tuple(completions[start:end]))
            )
        return outputs


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None and str(device) != "auto":
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _validate_generation_config(config: GenerationConfig) -> None:
    if config.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if config.num_return_sequences < 1:
        raise ValueError("num_return_sequences must be positive")
    if config.temperature < 0:
        raise ValueError("temperature must be non-negative; use 0 for greedy decoding")
    if not 0 < config.top_p <= 1:
        raise ValueError("top_p must be in the interval (0, 1]")
    if config.temperature == 0 and config.num_return_sequences != 1:
        raise ValueError("greedy decoding supports exactly one return sequence")


def _trim_at_stop(text: str, stop_strings: Sequence[str]) -> str:
    positions = [text.find(stop) for stop in stop_strings if stop and stop in text]
    return text[: min(positions)] if positions else text


def _load_runtime_config(
    config_path: Path,
    *,
    model_override: str | None,
) -> tuple[ModelLoadConfig, GenerationConfig]:
    raw_config = load_yaml_config(config_path)
    try:
        model_values = dict(raw_config["model"])
        generation_values = dict(raw_config["generation"])
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigError("Config must contain 'model' and 'generation' mappings") from error

    if model_override is not None:
        model_values["name_or_path"] = model_override
    backend_name = generation_values.pop("backend", "transformers")
    if backend_name != "transformers":
        raise ConfigError(f"Expected generation.backend=transformers, got {backend_name!r}")
    generation_values["stop"] = tuple(generation_values.get("stop", ()))

    try:
        return ModelLoadConfig(**model_values), GenerationConfig(**generation_values)
    except TypeError as error:
        raise ConfigError(f"Unsupported model or generation option: {error}") from error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate text with a Transformers causal LM")
    parser.add_argument("prompt", help="User prompt, or a raw model prompt with --raw")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"experiment config (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument("--model", help="override model.name_or_path from the config")
    parser.add_argument("--device", default="auto", help="auto, mps, cuda, or cpu")
    parser.add_argument("--system", help="optional system message")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="skip the tokenizer chat template and pass the prompt unchanged",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for a single prompt generation smoke test."""

    args = _build_parser().parse_args(argv)
    try:
        model_config, generation_config = _load_runtime_config(
            args.config,
            model_override=args.model,
        )
        backend = TransformersBackend(model_config, device=args.device)
        prompt = (
            args.prompt
            if args.raw
            else backend.render_chat_prompt(args.prompt, system_prompt=args.system)
        )
        output = backend.generate([prompt], generation_config)[0]
    except (ConfigError, OSError, ValueError) as error:
        _build_parser().error(str(error))

    for index, completion in enumerate(output.completions, start=1):
        if len(output.completions) > 1:
            print(f"[{index}]")
        print(completion)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
