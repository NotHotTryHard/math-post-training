#!/usr/bin/env python3
"""Generate resumable sampled rollouts for one normalized math dataset."""

import argparse
import gzip
import json
import time
from collections import Counter
from itertools import islice
from pathlib import Path

from math_post_training.data.loaders import load_math_source
from math_post_training.generation.base import GenerationConfig
from math_post_training.generation.vllm import VLLMBackend
from math_post_training.prompts.eval import build_eval_prompt, get_eval_settings
from math_post_training.verifiers.extraction import extract_final_answer, follows_answer_format
from math_post_training.verifiers.math import check_answer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--subset")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--protocol", default="math_post_training")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--difficulty-min", type=float)
    parser.add_argument("--difficulty-max", type=float)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.rollouts < 1 or args.batch_size < 1:
        raise ValueError("rollouts and batch-size must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "rollouts.jsonl.gz"
    completed = _completed_rows(predictions_path)

    source = {
        "adapter": args.adapter,
        "path": args.dataset,
        "subset": args.subset,
        "revision": args.revision,
        "split": args.split,
        "streaming": True,
    }
    filters = {}
    if args.difficulty_min is not None:
        filters["difficulty_min"] = args.difficulty_min
    if args.difficulty_max is not None:
        filters["difficulty_max"] = args.difficulty_max
    if filters:
        source["filters"] = filters

    examples = load_math_source(source)
    stop_strings = get_eval_settings(args.protocol, args.benchmark)["stop_strings"]
    generation = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.rollouts,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        stop_strings=stop_strings,
    )
    backend = VLLMBackend(
        args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
    )
    tokenizer = backend.tokenizer

    selected = islice(examples, completed, args.limit)
    counters = _restore_counters(predictions_path)
    started_at = time.perf_counter()
    mode = "at" if completed else "wt"
    with gzip.open(predictions_path, mode, encoding="utf-8") as output:
        while batch := list(islice(selected, args.batch_size)):
            prompts = [
                build_eval_prompt(
                    tokenizer,
                    example["problem"],
                    benchmark=args.benchmark,
                    protocol=args.protocol,
                )
                for example in batch
            ]
            generated = backend.generate(prompts, generation)
            for example, completions in zip(batch, generated, strict=True):
                rollout_records = [
                    _score_completion(tokenizer, completion, example["answer"], generation)
                    for completion in completions
                ]
                record = {
                    "index": counters["prompts"],
                    "source": example["source"],
                    "problem": example["problem"],
                    "reference_answer": example["answer"],
                    "difficulty": example.get("difficulty"),
                    "topic": example.get("topic"),
                    "rollouts": rollout_records,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                _update_counters(counters, rollout_records)
            output.flush()
            _write_summary(args, counters, started_at, completed)
            print(
                f"prompts={counters['prompts']} rollouts={counters['rollouts']} "
                f"accuracy={counters['correct'] / counters['rollouts']:.4f}",
                flush=True,
            )

    _write_summary(args, counters, started_at, completed)


def _score_completion(tokenizer, completion, reference, generation):
    completion_tokens = len(tokenizer.encode(completion, add_special_tokens=False))
    truncated = completion_tokens >= generation.max_new_tokens - 1
    prediction, extraction_method = extract_final_answer(completion)
    if truncated and extraction_method == "last_number":
        parsed, correct = False, False
    else:
        parsed, correct = check_answer(reference, prediction)
    return {
        "completion": completion,
        "extracted_answer": prediction,
        "extraction_method": extraction_method,
        "parsed": parsed,
        "correct": correct,
        "format_ok": follows_answer_format(completion, answer_format="boxed"),
        "completion_tokens": completion_tokens,
        "truncated": truncated,
    }


def _empty_counters():
    return {
        "prompts": 0,
        "rollouts": 0,
        "correct": 0,
        "parsed": 0,
        "formatted": 0,
        "truncated": 0,
        "completion_tokens": 0,
        "pass_histogram": Counter(),
    }


def _update_counters(counters, rollouts):
    correct = sum(item["correct"] for item in rollouts)
    counters["prompts"] += 1
    counters["rollouts"] += len(rollouts)
    counters["correct"] += correct
    counters["parsed"] += sum(item["parsed"] for item in rollouts)
    counters["formatted"] += sum(item["format_ok"] for item in rollouts)
    counters["truncated"] += sum(item["truncated"] for item in rollouts)
    counters["completion_tokens"] += sum(item["completion_tokens"] for item in rollouts)
    counters["pass_histogram"][correct] += 1


def _completed_rows(path):
    if not path.exists():
        return 0
    with gzip.open(path, "rt", encoding="utf-8") as file:
        return sum(1 for _ in file)


def _restore_counters(path):
    counters = _empty_counters()
    if not path.exists():
        return counters
    with gzip.open(path, "rt", encoding="utf-8") as file:
        for line in file:
            _update_counters(counters, json.loads(line)["rollouts"])
    return counters


def _write_summary(args, counters, started_at, previous_prompts):
    elapsed = time.perf_counter() - started_at
    prompts = counters["prompts"]
    rollouts = counters["rollouts"]
    histogram = {
        str(key): counters["pass_histogram"].get(key, 0) for key in range(args.rollouts + 1)
    }
    mixed = sum(
        value for key, value in counters["pass_histogram"].items() if 0 < key < args.rollouts
    )
    summary = {
        "model": args.model,
        "dataset": args.dataset,
        "revision": args.revision,
        "split": args.split,
        "protocol": args.protocol,
        "generation": {
            "rollouts": args.rollouts,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
        },
        "prompts": prompts,
        "rollouts": rollouts,
        "accuracy": counters["correct"] / rollouts if rollouts else None,
        "parse_rate": counters["parsed"] / rollouts if rollouts else None,
        "format_rate": counters["formatted"] / rollouts if rollouts else None,
        "truncated": counters["truncated"],
        "truncation_rate": counters["truncated"] / rollouts if rollouts else None,
        "mean_completion_tokens": counters["completion_tokens"] / rollouts if rollouts else None,
        "pass_histogram": histogram,
        "mixed_reward_groups": mixed,
        "mixed_reward_group_rate": mixed / prompts if prompts else None,
        "elapsed_seconds_this_process": elapsed,
        "new_prompts_this_process": prompts - previous_prompts,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
