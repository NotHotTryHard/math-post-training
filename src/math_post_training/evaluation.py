"""Evaluate a model on the project's mathematical benchmarks."""

import gzip
import json
import time
from dataclasses import replace
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path

from math_post_training.data.loaders import load_math_source
from math_post_training.evaluation_prompts import (
    protocol_metadata,
    render_evaluation_prompt,
)
from math_post_training.generation.base import GenerationConfig
from math_post_training.verifiers.extraction import extract_final_answer, follows_answer_format
from math_post_training.verifiers.math import check_answer


def evaluate_model(
    backend,
    tokenizer,
    config,
    *,
    model_name,
    limit=None,
    benchmark_names=None,
    output_dir=None,
):
    """Run every configured benchmark and write a compact summary."""

    evaluation = config["evaluation"]
    protocol = evaluation["protocol"]
    generation = GenerationConfig(**evaluation["generation"])
    if generation.num_return_sequences != 1:
        raise ValueError("evaluation requires generation.num_return_sequences = 1")

    run_dir = _make_run_dir(
        output_dir or evaluation["output_dir"],
        config["experiment"]["name"],
    )
    predictions_path = run_dir / "predictions.jsonl.gz"
    predictions_file = None
    if evaluation.get("save_predictions", True):
        predictions_file = gzip.open(predictions_path, "wt", encoding="utf-8")

    summary = {
        "experiment": config["experiment"]["name"],
        "model": model_name,
        "created_at": datetime.now(UTC).isoformat(),
        "protocol": protocol,
        "generation": vars(generation),
        "benchmarks": {},
    }

    try:
        selected = _select_benchmarks(evaluation["benchmarks"], benchmark_names)
        for benchmark in selected:
            result = _evaluate_benchmark(
                backend,
                tokenizer,
                benchmark,
                generation,
                protocol=protocol,
                batch_size=evaluation["batch_size"],
                limit=limit,
                sample_seed=evaluation["sample_seed"],
                shuffle_buffer_size=evaluation["shuffle_buffer_size"],
                predictions_file=predictions_file,
            )
            summary["benchmarks"][benchmark["name"]] = result
    finally:
        if predictions_file is not None:
            predictions_file.close()

    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir, summary


def _evaluate_benchmark(
    backend,
    tokenizer,
    benchmark,
    generation,
    *,
    protocol,
    batch_size,
    limit,
    sample_seed,
    shuffle_buffer_size,
    predictions_file,
):
    name = benchmark["name"]
    started_at = time.perf_counter()
    metadata = protocol_metadata(protocol, name)
    benchmark_generation = replace(generation, stop_strings=metadata["stop_strings"])
    metrics = _empty_metrics()
    examples = _benchmark_examples(
        benchmark,
        limit,
        sample_seed=sample_seed,
        shuffle_buffer_size=shuffle_buffer_size,
    )

    for batch in _batched(examples, batch_size):
        prompts = [
            render_evaluation_prompt(
                tokenizer,
                example["problem"],
                benchmark=name,
                protocol=protocol,
            )
            for example in batch
        ]
        completions = backend.generate(prompts, benchmark_generation)

        for example, generated in zip(batch, completions, strict=True):
            completion = generated[0]
            prediction = extract_final_answer(
                completion,
                answer_format=metadata["answer_format"],
            )
            parsed, correct = check_answer(
                example["answer"],
                prediction,
                multiple=benchmark.get("multiple_answers", False),
            )
            format_ok = follows_answer_format(
                completion,
                answer_format=metadata["answer_format"],
            )
            completion_tokens = len(tokenizer.encode(completion, add_special_tokens=False))

            record = {
                "benchmark": name,
                "index": metrics["total"],
                "source": example["source"],
                "problem": example["problem"],
                "reference_answer": example["answer"],
                "completion": completion,
                "extracted_answer": prediction,
                "parsed": parsed,
                "correct": correct,
                "format_ok": format_ok,
                "completion_tokens": completion_tokens,
                "truncated": completion_tokens >= generation.max_new_tokens - 1,
            }
            _update_metrics(metrics, record)

            if predictions_file is not None:
                predictions_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"\r{name}: {metrics['total']} examples", end="", flush=True)

    print()
    if metrics["total"] == 0:
        raise ValueError(f"Benchmark {name!r} produced no examples")

    result = _finish_metrics(metrics)
    elapsed_seconds = time.perf_counter() - started_at
    selected_limit = limit if limit is not None else benchmark.get("limit")
    result.update(
        {
            "protocol": protocol,
            "num_shots": metadata["num_shots"],
            "dataset": benchmark["path"],
            "revision": benchmark["revision"],
            "split": benchmark["split"],
            "is_full_split": selected_limit is None,
            "sample_seed": sample_seed if selected_limit is not None else None,
            "stop_strings": metadata["stop_strings"],
            "elapsed_seconds": elapsed_seconds,
            "completion_tokens_per_second": metrics["completion_tokens"] / elapsed_seconds,
        }
    )
    format_text = "n/a" if result["format_rate"] is None else f"{result['format_rate']:.3f}"
    print(
        f"{name}: accuracy={result['accuracy']:.3f}, "
        f"parse_rate={result['parse_rate']:.3f}, format_rate={format_text}"
    )
    return result


def _benchmark_examples(benchmark, cli_limit, *, sample_seed, shuffle_buffer_size):
    selected_limit = cli_limit if cli_limit is not None else benchmark.get("limit")
    source_template = {
        "adapter": benchmark["adapter"],
        "path": benchmark["path"],
        "revision": benchmark["revision"],
        "split": benchmark["split"],
        "streaming": benchmark.get("streaming", True),
    }
    if selected_limit is not None:
        source_template["shuffle_seed"] = sample_seed
        source_template["shuffle_buffer_size"] = shuffle_buffer_size
    subsets = benchmark.get("subsets", [benchmark.get("subset")])

    def subset_examples(subset):
        source = dict(source_template)
        source["subset"] = subset
        return load_math_source(source)

    def all_examples():
        for subset in subsets:
            yield from subset_examples(subset)

    if selected_limit is None:
        return all_examples()
    if selected_limit < 1:
        raise ValueError("evaluation limit must be positive")

    if len(subsets) == 1:
        return islice(subset_examples(subsets[0]), selected_limit)
    return islice(_round_robin(subset_examples(subset) for subset in subsets), selected_limit)


def _round_robin(iterables):
    """Yield one item per subset in turn until every subset is exhausted."""

    active = [iter(iterable) for iterable in iterables]
    while active:
        remaining = []
        for iterator in active:
            try:
                yield next(iterator)
                remaining.append(iterator)
            except StopIteration:
                pass
        active = remaining


def _batched(examples, batch_size):
    if batch_size < 1:
        raise ValueError("evaluation.batch_size must be positive")

    iterator = iter(examples)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def _empty_metrics():
    return {
        "total": 0,
        "correct": 0,
        "parsed": 0,
        "formatted": 0,
        "format_total": 0,
        "truncated": 0,
        "completion_tokens": 0,
        "by_source": {},
    }


def _update_metrics(metrics, record):
    metrics["total"] += 1
    metrics["correct"] += int(record["correct"])
    metrics["parsed"] += int(record["parsed"])
    if record["format_ok"] is not None:
        metrics["format_total"] += 1
        metrics["formatted"] += int(record["format_ok"])
    metrics["truncated"] += int(record["truncated"])
    metrics["completion_tokens"] += record["completion_tokens"]

    source = metrics["by_source"].setdefault(record["source"], {"total": 0, "correct": 0})
    source["total"] += 1
    source["correct"] += int(record["correct"])


def _finish_metrics(metrics):
    total = metrics["total"]
    result = {
        "total": total,
        "correct": metrics["correct"],
        "accuracy": metrics["correct"] / total,
        "parse_rate": metrics["parsed"] / total,
        "format_rate": (
            metrics["formatted"] / metrics["format_total"] if metrics["format_total"] else None
        ),
        "truncated": metrics["truncated"],
        "mean_completion_tokens": metrics["completion_tokens"] / total,
    }

    if len(metrics["by_source"]) > 1:
        result["by_source"] = {
            source: {
                "total": values["total"],
                "correct": values["correct"],
                "accuracy": values["correct"] / values["total"],
            }
            for source, values in metrics["by_source"].items()
        }
    return result


def _select_benchmarks(benchmarks, names):
    if not names:
        return benchmarks

    requested = set(names)
    selected = [benchmark for benchmark in benchmarks if benchmark["name"] in requested]
    missing = requested - {benchmark["name"] for benchmark in selected}
    if missing:
        raise ValueError(f"Unknown benchmarks: {', '.join(sorted(missing))}")
    return selected


def _make_run_dir(output_dir, experiment_name):
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = Path(output_dir) / experiment_name / timestamp
    run_dir.mkdir(parents=True)
    return run_dir
