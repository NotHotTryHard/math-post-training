"""Small command-line entrypoints for local experiments."""

import argparse
from pathlib import Path

import torch

from math_post_training.config import load_yaml_config
from math_post_training.generation.base import GenerationConfig
from math_post_training.generation.transformers import TransformersBackend
from math_post_training.model import load_model_and_tokenizer
from math_post_training.prompts import render_chat_prompt

DEFAULT_CONFIG_PATH = Path("configs/current.yaml")


def _device(requested):
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _parser():
    parser = argparse.ArgumentParser(description="Generate text with a Transformers causal LM")
    parser.add_argument("prompt")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model", help="override model.name_or_path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--system")
    parser.add_argument("--raw", action="store_true", help="do not apply a chat template")
    return parser


def generate_main(argv=None):
    args = _parser().parse_args(argv)
    config = load_yaml_config(args.config)

    model_config = dict(config["model"])
    if args.model is not None:
        model_config["name_or_path"] = args.model

    generation_config = dict(config["generation"])
    backend_name = generation_config.pop("backend")
    if backend_name != "transformers":
        raise ValueError(
            f"model-generate only supports the transformers backend, got {backend_name!r}"
        )

    device = _device(args.device)
    model, tokenizer = load_model_and_tokenizer(**model_config, device=device)
    backend = TransformersBackend(model, tokenizer, device)

    prompt = args.prompt
    if not args.raw:
        prompt = render_chat_prompt(tokenizer, args.prompt, system_prompt=args.system)

    completions = backend.generate([prompt], GenerationConfig(**generation_config))[0]
    for index, completion in enumerate(completions, start=1):
        if len(completions) > 1:
            print(f"[{index}]")
        print(completion)
    return 0


if __name__ == "__main__":
    raise SystemExit(generate_main())
