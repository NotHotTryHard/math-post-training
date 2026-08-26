"""Optional W&B reporting for evaluation runs."""

import os


def start_wandb_run(
    config,
    *,
    model_name,
    protocol,
    run_dir,
    limit,
    selected_benchmarks,
    prediction_columns,
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
            "benchmarks": [benchmark["name"] for benchmark in selected_benchmarks],
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
        tables = {
            benchmark["name"]: wandb.Table(columns=prediction_columns)
            for benchmark in selected_benchmarks
        }
    return run, tables


def log_wandb_results(run, prediction_tables, summary):
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

        for difficulty, difficulty_result in result.get("by_difficulty", {}).items():
            prefix = f"eval/{benchmark}/by_difficulty/{difficulty}"
            run.summary[f"{prefix}/accuracy"] = difficulty_result["accuracy"]
            run.summary[f"{prefix}/parse_rate"] = difficulty_result["parse_rate"]
            run.summary[f"{prefix}/truncation_rate"] = difficulty_result["truncation_rate"]

    for benchmark, table in prediction_tables.items():
        run.log({f"eval/{benchmark}/predictions": table})
