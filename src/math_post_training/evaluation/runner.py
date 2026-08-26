"""Orchestrate benchmark evaluation and write its durable artifacts."""

import gzip
import json
import time
from dataclasses import replace
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path

from tqdm import tqdm

from math_post_training.data.loaders import load_math_source
from math_post_training.evaluation.metrics import EvaluationMetrics
from math_post_training.evaluation.reporting import log_wandb_results, start_wandb_run
from math_post_training.evaluation.scoring import score_completion, select_majority_vote
from math_post_training.generation.base import GenerationConfig
from math_post_training.model import require_qwen_base_eos
from math_post_training.prompts.eval import (
    build_eval_prompt,
    get_eval_settings,
)

PREDICTION_COLUMNS = [
    "benchmark",
    "index",
    "source",
    "difficulty",
    "topic",
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
    aggregation = eval_config.get("aggregation", "single")
    if generation.num_return_sequences != 1 and aggregation != "majority_vote":
        raise ValueError(
            "eval requires generation.num_return_sequences = 1 unless "
            "eval.aggregation is majority_vote"
        )
    if aggregation == "majority_vote" and generation.num_return_sequences < 2:
        raise ValueError("majority_vote requires generation.num_return_sequences >= 2")
    if aggregation not in {"single", "majority_vote"}:
        raise ValueError(f"Unknown eval aggregation: {aggregation}")

    run_dir = _make_run_dir(
        output_dir or eval_config["output_dir"],
        config["experiment"]["name"],
    )
    predictions_path = run_dir / "predictions.jsonl.gz"
    predictions_file = None
    if eval_config.get("save_predictions", True):
        predictions_file = gzip.open(predictions_path, "wt", encoding="utf-8")

    selected_benchmarks = _select_benchmarks(eval_config["benchmarks"], benchmark_names)
    wandb_run, prediction_tables = start_wandb_run(
        config,
        model_name=model_name,
        protocol=protocol,
        run_dir=run_dir,
        limit=limit,
        selected_benchmarks=selected_benchmarks,
        prediction_columns=PREDICTION_COLUMNS,
    )

    summary = {
        "experiment": config["experiment"]["name"],
        "model": model_name,
        "created_at": datetime.now(UTC).isoformat(),
        "protocol": protocol,
        "aggregation": aggregation,
        "generation": vars(generation),
        "benchmarks": {},
    }

    exit_code = 1
    try:
        try:
            for benchmark in selected_benchmarks:
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
                    aggregation=aggregation,
                )
                summary["benchmarks"][benchmark["name"]] = result
        finally:
            if predictions_file is not None:
                predictions_file.close()

        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        log_wandb_results(wandb_run, prediction_tables, summary)
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
    aggregation,
):
    name = benchmark["name"]
    started_at = time.perf_counter()
    settings = get_eval_settings(protocol, name)
    benchmark_generation = replace(generation, stop_strings=settings["stop_strings"])
    metrics = EvaluationMetrics()
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
                rollout_records = [
                    score_completion(
                        tokenizer,
                        completion,
                        reference=example["answer"],
                        settings=settings,
                        max_new_tokens=generation.max_new_tokens,
                    )
                    for completion in generated
                ]
                if aggregation == "majority_vote":
                    selected_record, vote = select_majority_vote(
                        rollout_records,
                        answer_kind=settings["answer_kind"],
                    )
                else:
                    selected_record = rollout_records[0]
                    vote = None

                record = {
                    "benchmark": name,
                    "index": metrics.total,
                    "source": example["source"],
                    "difficulty": example.get("difficulty"),
                    "topic": example.get("topic"),
                    "problem": example["problem"],
                    "reference_answer": example["answer"],
                    **selected_record,
                }
                if vote is not None:
                    record.update(vote)
                    record["rollouts"] = rollout_records
                metrics.add(record)
                metrics.add_rollouts(rollout_records, vote)

                if predictions_file is not None:
                    predictions_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                if predictions_table is not None:
                    predictions_table.add_data(*(record[column] for column in PREDICTION_COLUMNS))

            progress.update(len(batch))
            if wandb_run is not None:
                wandb_run.log({f"eval/{name}/processed": metrics.total})

    if metrics.total == 0:
        raise ValueError(f"Benchmark {name!r} produced no examples")

    result = metrics.finish()
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
            "completion_tokens_per_second": (
                metrics.rollout_completion_tokens or metrics.completion_tokens
            )
            / elapsed_seconds,
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
    if "filters" in benchmark:
        source_template["filters"] = benchmark["filters"]
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
