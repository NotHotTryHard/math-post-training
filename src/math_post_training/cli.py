"""Small command-line entrypoints for local experiments."""

import argparse
from pathlib import Path

import torch

from math_post_training.config import load_yaml_config
from math_post_training.generation.base import GenerationConfig
from math_post_training.generation.transformers import TransformersBackend
from math_post_training.generation.vllm import VLLMBackend
from math_post_training.model import load_model_and_tokenizer
from math_post_training.prompts.inference import build_inference_prompt

DEFAULT_CONFIG_PATH = Path("configs/current.yaml")


def _device(requested):
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _generate_parser():
    parser = argparse.ArgumentParser(description="Generate text with a causal language model")
    parser.add_argument("prompt")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model", help="override model.name_or_path")
    parser.add_argument(
        "--device",
        default="auto",
        help="Transformers device; vLLM selects its accelerator from the environment",
    )
    parser.add_argument("--system")
    parser.add_argument("--raw", action="store_true", help="do not apply a chat template")
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="print the rendered prompt before generation",
    )
    return parser


def generate_main(argv=None):
    args = _generate_parser().parse_args(argv)
    config = load_yaml_config(args.config)

    model_config = dict(config["model"])
    if args.model is not None:
        model_config["name_or_path"] = args.model

    generation_config = dict(config["inference"])
    backend_name = generation_config.pop("backend")
    backend, tokenizer = _load_backend(
        backend_name,
        model_config,
        device=args.device,
        vllm_config=config.get("vllm", {}),
    )

    prompt = build_inference_prompt(
        tokenizer,
        args.prompt,
        system_prompt=args.system,
        raw=args.raw,
    )

    if args.show_prompt:
        print("=== prompt sent to model ===")
        print(prompt)
        print("=== completion ===")

    completions = backend.generate([prompt], GenerationConfig(**generation_config))[0]
    for index, completion in enumerate(completions, start=1):
        if len(completions) > 1:
            print(f"[{index}]")
        print(completion)
    return 0


def _eval_parser():
    parser = argparse.ArgumentParser(description="Evaluate a model on math benchmarks")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model", help="override model.name_or_path with a checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, help="evaluate this many examples per benchmark")
    parser.add_argument("--batch-size", type=int, help="override eval.batch_size")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="override eval.generation.max_new_tokens",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        dest="benchmarks",
        help="run only this benchmark; repeat to select several",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser


def eval_main(argv=None):
    args = _eval_parser().parse_args(argv)
    config = load_yaml_config(args.config)

    if args.batch_size is not None:
        config["eval"]["batch_size"] = args.batch_size
    if args.max_new_tokens is not None:
        config["eval"]["generation"]["max_new_tokens"] = args.max_new_tokens

    model_config = dict(config["model"])
    if args.model is not None:
        model_config["name_or_path"] = args.model

    backend_name = config["eval"]["backend"]
    backend, tokenizer = _load_backend(
        backend_name,
        model_config,
        device=args.device,
        vllm_config=config.get("vllm", {}),
    )

    from math_post_training.eval import eval_model

    run_dir, _ = eval_model(
        backend,
        tokenizer,
        config,
        model_name=model_config["name_or_path"],
        limit=args.limit,
        benchmark_names=args.benchmarks,
        output_dir=args.output_dir,
    )
    print(f"Results: {run_dir}")
    return 0


def _sft_parser():
    parser = argparse.ArgumentParser(description="Supervised fine-tune a causal language model")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    return parser


def sft_main(argv=None):
    args = _sft_parser().parse_args(argv)
    config = load_yaml_config(args.config)

    from math_post_training.sft import train_sft

    output_dir = train_sft(
        config,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(f"Model: {output_dir}")
    return 0


def _grpo_parser():
    parser = argparse.ArgumentParser(description="Train a model with verifiable GRPO rewards")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    return parser


def grpo_main(argv=None):
    args = _grpo_parser().parse_args(argv)
    config = load_yaml_config(args.config)

    from math_post_training.grpo import train_grpo

    output_dir = train_grpo(
        config,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(f"Model: {output_dir}")
    return 0


def _load_backend(backend_name, model_config, *, device, vllm_config):
    if backend_name == "transformers":
        resolved_device = _device(device)
        model, tokenizer = load_model_and_tokenizer(
            **model_config,
            device=resolved_device,
        )
        return TransformersBackend(model, tokenizer, resolved_device), tokenizer

    if backend_name == "vllm":
        backend = VLLMBackend(**model_config, **vllm_config)
        return backend, backend.tokenizer

    raise ValueError(f"Unknown generation backend: {backend_name!r}")


if __name__ == "__main__":
    raise SystemExit(generate_main())
