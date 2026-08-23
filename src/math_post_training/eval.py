"""Evaluate a model on the project's mathematical benchmarks."""

import gzip
import json
import os
import time
from dataclasses import replace
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path

from tqdm import tqdm

from math_post_training.data.loaders import load_math_source
from math_post_training.generation.base import GenerationConfig
from math_post_training.model import require_qwen_base_eos
from math_post_training.prompts.eval import (
    build_eval_prompt,
    get_eval_settings,
)
from math_post_training.verifiers.choice import check_choice_answer
from math_post_training.verifiers.extraction import extract_final_answer, follows_answer_format
from math_post_training.verifiers.math import check_answer

PREDICTION_COLUMNS = [
    "benchmark",
    "index",
    "source",
    "problem",
    "reference_answer",
    "completion",
    "extracted_answer",
    "extraction_method",
    "parsed",
    "correct",
    "format_ok",
    "completion_tokens",
    "truncated",
]


def eval_model(
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

    eval_config = config["eval"]
    protocol = eval_config["protocol"]
    if protocol == "math_post_training":
        require_qwen_base_eos(tokenizer)
    generation = GenerationConfig(**eval_config["generation"])
    if generation.num_return_sequences != 1:
        raise ValueError("eval requires generation.num_return_sequences = 1")

    run_dir = _make_run_dir(
        output_dir or eval_config["output_dir"],
        config["experiment"]["name"],
    )
    predictions_path = run_dir / "predictions.jsonl.gz"
    predictions_file = None
    if eval_config.get("save_predictions", True):
        predictions_file = gzip.open(predictions_path, "wt", encoding="utf-8")

    wandb_run, prediction_tables = _start_wandb_run(
        config,
        model_name=model_name,
        protocol=protocol,
        run_dir=run_dir,
        limit=limit,
        benchmark_names=benchmark_names,
    )

    summary = {
        "experiment": config["experiment"]["name"],
        "model": model_name,
        "created_at": datetime.now(UTC).isoformat(),
        "protocol": protocol,
        "generation": vars(generation),
        "benchmarks": {},
    }

    exit_code = 1
    try:
        try:
            selected = _select_benchmarks(eval_config["benchmarks"], benchmark_names)
            for benchmark in selected:
                result = _eval_benchmark(
                    backend,
                    tokenizer,
                    benchmark,
                    generation,
                    protocol=protocol,
                    batch_size=eval_config["batch_size"],
                    limit=limit,
                    sample_seed=eval_config["sample_seed"],
                    shuffle_buffer_size=eval_config["shuffle_buffer_size"],
                    predictions_file=predictions_file,
                    predictions_table=prediction_tables.get(benchmark["name"]),
                    wandb_run=wandb_run,
                )
                summary["benchmarks"][benchmark["name"]] = result
        finally:
            if predictions_file is not None:
                predictions_file.close()

        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _log_wandb_results(wandb_run, prediction_tables, summary)
        exit_code = 0
        return run_dir, summary
    finally:
        if wandb_run is not None:
            wandb_run.finish(exit_code=exit_code)


def _eval_benchmark(
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
    predictions_table,
    wandb_run,
):
    name = benchmark["name"]
    started_at = time.perf_counter()
    settings = get_eval_settings(protocol, name)
    benchmark_generation = replace(generation, stop_strings=settings["stop_strings"])
    metrics = _empty_metrics()
    examples = _benchmark_examples(
        benchmark,
        limit,
        sample_seed=sample_seed,
        shuffle_buffer_size=shuffle_buffer_size,
    )

    selected_limit = limit if limit is not None else benchmark.get("limit")
    with tqdm(total=selected_limit, desc=name, unit="example", dynamic_ncols=True) as progress:
        for batch in _batched(examples, batch_size):
            prompts = [
                build_eval_prompt(
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
                completion_tokens = len(tokenizer.encode(completion, add_special_tokens=False))
                truncated = completion_tokens >= generation.max_new_tokens - 1
                prediction, extraction_method = extract_final_answer(
                    completion,
                    answer_kind=settings["answer_kind"],
                )
                if truncated and extraction_method in {"last_number", "last_choice"}:
                    parsed, correct = False, False
                elif settings["answer_kind"] == "choice":
                    parsed, correct = check_choice_answer(example["answer"], prediction)
                else:
                    parsed, correct = check_answer(example["answer"], prediction)
                format_ok = follows_answer_format(
                    completion,
                    answer_format=settings["required_answer_format"],
                )

                record = {
                    "benchmark": name,
                    "index": metrics["total"],
                    "source": example["source"],
                    "problem": example["problem"],
                    "reference_answer": example["answer"],
                    "completion": completion,
                    "extracted_answer": prediction,
                    "extraction_method": extraction_method,
                    "parsed": parsed,
                    "correct": correct,
                    "format_ok": format_ok,
                    "completion_tokens": completion_tokens,
                    "truncated": truncated,
                }
                _update_metrics(metrics, record)

                if predictions_file is not None:
                    predictions_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                if predictions_table is not None:
                    predictions_table.add_data(*(record[column] for column in PREDICTION_COLUMNS))

            progress.update(len(batch))
            if wandb_run is not None:
                wandb_run.log({f"eval/{name}/processed": metrics["total"]})

    if metrics["total"] == 0:
        raise ValueError(f"Benchmark {name!r} produced no examples")

    result = _finish_metrics(metrics)
    elapsed_seconds = time.perf_counter() - started_at
    result.update(
        {
            "protocol": protocol,
            "num_shots": settings["num_shots"],
            "dataset": benchmark["path"],
            "revision": benchmark["revision"],
            "split": benchmark["split"],
            "is_full_split": selected_limit is None,
            "sample_seed": sample_seed if selected_limit is not None else None,
            "stop_strings": settings["stop_strings"],
            "elapsed_seconds": elapsed_seconds,
            "completion_tokens_per_second": metrics["completion_tokens"] / elapsed_seconds,
        }
    )
    format_text = "n/a" if result["format_rate"] is None else f"{result['format_rate']:.3f}"
    tqdm.write(
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
        "streaming": benchmark.get("streaming", False),
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
        raise ValueError("eval limit must be positive")

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
        raise ValueError("eval.batch_size must be positive")

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
        "extraction_methods": {},
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
    method = record["extraction_method"]
    metrics["extraction_methods"][method] = metrics["extraction_methods"].get(method, 0) + 1

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
        "extraction_methods": metrics["extraction_methods"],
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


def _start_wandb_run(
    config,
    *,
    model_name,
    protocol,
    run_dir,
    limit,
    benchmark_names,
):
    settings = config["eval"].get("wandb", {})
    if not settings.get("enabled", False):
        return None, {}

    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("W&B logging requires `uv sync --group eval`") from error

    tracked_config = {
        "experiment": config["experiment"],
        "model": {**config["model"], "name_or_path": model_name},
        "eval": config["eval"],
        "runtime_overrides": {
            "limit": limit,
            "benchmarks": benchmark_names,
        },
    }
    experiment_name = config["experiment"]["name"]
    backend = config["eval"].get("backend", "transformers")
    run = wandb.init(
        project=os.getenv("WANDB_PROJECT") or "math-post-training",
        entity=os.getenv("WANDB_ENTITY") or None,
        name=experiment_name,
        job_type="eval",
        tags=["eval", backend, protocol, experiment_name],
        config=tracked_config,
        dir=str(run_dir),
    )
    tables = {}
    if settings.get("log_predictions", True):
        selected = _select_benchmarks(config["eval"]["benchmarks"], benchmark_names)
        tables = {
            benchmark["name"]: wandb.Table(columns=PREDICTION_COLUMNS) for benchmark in selected
        }
    return run, tables


def _log_wandb_results(run, prediction_tables, summary):
    if run is None:
        return

    scalar_metrics = (
        "total",
        "correct",
        "accuracy",
        "parse_rate",
        "format_rate",
        "truncated",
        "mean_completion_tokens",
        "elapsed_seconds",
        "completion_tokens_per_second",
    )
    for benchmark, result in summary["benchmarks"].items():
        for metric in scalar_metrics:
            value = result.get(metric)
            if value is not None:
                run.summary[f"eval/{benchmark}/{metric}"] = value

        for source, source_result in result.get("by_source", {}).items():
            run.summary[f"eval/{benchmark}/by_source/{source}/accuracy"] = source_result["accuracy"]

    for benchmark, table in prediction_tables.items():
        run.log({f"eval/{benchmark}/predictions": table})


def _make_run_dir(output_dir, experiment_name):
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = Path(output_dir) / experiment_name / timestamp
    run_dir.mkdir(parents=True)
    return run_dir
