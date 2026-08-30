#!/usr/bin/env python3
"""Run two-model DeepMath rollouts on one fixed per-level sample.

The manifest is created once from a deterministic streaming shuffle and then
reused verbatim by every model, so paired comparisons never depend on dataset
iteration order.
"""

import argparse
import gzip
import hashlib
import json
import time
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path

from math_post_training.data.loaders import load_math_source
from math_post_training.evaluation.scoring import score_completion
from math_post_training.generation.base import GenerationConfig
from math_post_training.generation.vllm import VLLMBackend
from math_post_training.prompts.eval import build_eval_prompt, get_eval_settings

DEFAULT_LEVELS = (5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="zwhe99/DeepMath-103K")
    parser.add_argument(
        "--revision",
        default="5cf055d1fe3d7a2eb19719ac020211469736ae44",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--levels", type=float, nargs="+", default=DEFAULT_LEVELS)
    parser.add_argument("--samples-per-level", type=int, default=500)
    parser.add_argument("--per-level-limit", type=int)
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer-size", type=int, default=10_000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.samples_per_level < 1 or args.rollouts < 1 or args.batch_size < 1:
        raise ValueError("sample, rollout, and batch sizes must be positive")
    examples = _load_or_create_manifest(args)
    if args.per_level_limit is not None:
        examples = _limit_per_level(examples, args.per_level_limit)
    _run_rollouts(args, examples)


def _load_or_create_manifest(args):
    if args.manifest.exists():
        with gzip.open(args.manifest, "rt", encoding="utf-8") as file:
            examples = [json.loads(line) for line in file]
        _validate_manifest(args, examples)
        return examples

    source = {
        "adapter": "deepmath",
        "path": args.dataset,
        "revision": args.revision,
        "split": args.split,
        "streaming": True,
        "shuffle_seed": args.seed,
        "shuffle_buffer_size": args.shuffle_buffer_size,
        "filters": {
            "difficulty_min": min(args.levels),
            "difficulty_max": max(args.levels),
        },
    }
    wanted = set(args.levels)
    counts = Counter()
    examples = []
    seen_ids = set()
    for source_index, example in enumerate(load_math_source(source)):
        level = float(example["difficulty"])
        if level not in wanted or counts[level] >= args.samples_per_level:
            continue
        record = dict(example)
        record["difficulty"] = level
        sample_id = hashlib.sha256(
            (record["problem"] + "\0" + record["answer"]).encode()
        ).hexdigest()
        if sample_id in seen_ids:
            continue
        record["sample_id"] = sample_id
        record["shuffled_source_index"] = source_index
        examples.append(record)
        seen_ids.add(sample_id)
        counts[level] += 1
        if all(counts[level] == args.samples_per_level for level in wanted):
            break

    missing = {
        level: args.samples_per_level - counts[level]
        for level in sorted(wanted)
        if counts[level] != args.samples_per_level
    }
    if missing:
        raise RuntimeError(f"not enough examples for requested levels: {missing}")
    examples.sort(key=lambda item: (item["difficulty"], item["sample_id"]))
    _validate_manifest(args, examples)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.manifest, "wt", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")
    return examples


def _validate_manifest(args, examples):
    expected = Counter({float(level): args.samples_per_level for level in args.levels})
    actual = Counter(float(item["difficulty"]) for item in examples)
    if actual != expected:
        raise ValueError(f"manifest level counts differ: expected={expected}, actual={actual}")
    ids = [item["sample_id"] for item in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest contains duplicate sample IDs")


def _limit_per_level(examples, limit):
    if limit < 1:
        raise ValueError("per-level-limit must be positive")
    counts = Counter()
    selected = []
    for example in examples:
        level = example["difficulty"]
        if counts[level] < limit:
            selected.append(example)
            counts[level] += 1
    return selected


def _run_rollouts(args, examples):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "rollouts.jsonl.gz"
    existing = _read_existing(output_path)
    completed_ids = {item["sample_id"] for item in existing}
    selected = [item for item in examples if item["sample_id"] not in completed_ids]

    settings = get_eval_settings(args.protocol, "deepmath")
    generation = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.rollouts,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        stop_strings=settings["stop_strings"],
    )
    backend = VLLMBackend(
        args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
    )
    tokenizer = backend.tokenizer
    started_at = time.perf_counter()
    mode = "at" if existing else "wt"
    with gzip.open(output_path, mode, encoding="utf-8") as output:
        iterator = iter(selected)
        while batch := list(islice(iterator, args.batch_size)):
            prompts = [
                build_eval_prompt(
                    tokenizer,
                    example["problem"],
                    benchmark="deepmath",
                    protocol=args.protocol,
                )
                for example in batch
            ]
            generations = backend.generate(prompts, generation)
            for example, completions in zip(batch, generations, strict=True):
                record = {
                    "sample_id": example["sample_id"],
                    "problem": example["problem"],
                    "reference_answer": example["answer"],
                    "difficulty": example["difficulty"],
                    "topic": example.get("topic"),
                    "rollouts": [
                        score_completion(
                            tokenizer,
                            completion,
                            reference=example["answer"],
                            settings=settings,
                            max_new_tokens=args.max_new_tokens,
                        )
                        for completion in completions
                    ],
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                existing.append(record)
            output.flush()
            _write_summary(args, existing, started_at)
            print(f"completed={len(existing)}/{len(examples)}", flush=True)
    _write_summary(args, existing, started_at)


def _read_existing(path):
    if not path.exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def _aggregate(records):
    prompts = len(records)
    rollouts = [rollout for record in records for rollout in record["rollouts"]]
    histogram = Counter(sum(item["correct"] for item in record["rollouts"]) for record in records)
    mixed = sum(value for key, value in histogram.items() if 0 < key < 3)
    format_values = [item["format_ok"] for item in rollouts if item["format_ok"] is not None]
    return {
        "prompts": prompts,
        "rollouts": len(rollouts),
        "accuracy": sum(item["correct"] for item in rollouts) / len(rollouts) if rollouts else None,
        "parse_rate": sum(item["parsed"] for item in rollouts) / len(rollouts)
        if rollouts
        else None,
        "format_rate": sum(format_values) / len(format_values) if format_values else None,
        "truncated": sum(item["truncated"] for item in rollouts),
        "truncation_rate": sum(item["truncated"] for item in rollouts) / len(rollouts)
        if rollouts
        else None,
        "mean_completion_tokens": sum(item["completion_tokens"] for item in rollouts)
        / len(rollouts)
        if rollouts
        else None,
        "pass_histogram": {str(key): histogram.get(key, 0) for key in range(4)},
        "mixed_reward_groups": mixed,
        "mixed_reward_group_rate": mixed / prompts if prompts else None,
    }


def _write_summary(args, records, started_at):
    by_level = defaultdict(list)
    for record in records:
        by_level[float(record["difficulty"])].append(record)
    summary = {
        "model": args.model,
        "protocol": args.protocol,
        "dataset": args.dataset,
        "revision": args.revision,
        "manifest": str(args.manifest),
        "generation": {
            "rollouts": args.rollouts,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
        },
        **_aggregate(records),
        "by_difficulty": {f"{level:g}": _aggregate(by_level[level]) for level in sorted(by_level)},
        "elapsed_seconds_this_process": time.perf_counter() - started_at,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
