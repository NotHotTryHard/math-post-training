"""Run GRPO in resumable segments separated by full external evaluations."""

import argparse
import json
import os
import subprocess
import uuid
from pathlib import Path

from math_post_training.checkpoint_eval import find_summary
from math_post_training.config import load_yaml_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train GRPO to configured checkpoints and evaluate between segments"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    return parser.parse_args(argv)


def checkpoint_step(path):
    return int(path.name.removeprefix("checkpoint-"))


def complete_checkpoints(root):
    return sorted(
        (
            path
            for path in root.glob("checkpoint-*")
            if (path / "trainer_state.json").is_file()
            and (path / "adapter_model.safetensors").is_file()
        ),
        key=checkpoint_step,
    )


def evaluation_score(summary, benchmark_names):
    benchmarks = summary["benchmarks"]
    return sum(benchmarks[name]["accuracy"] for name in benchmark_names) / len(
        benchmark_names
    )


def _run(command, *, env=None):
    subprocess.run([str(value) for value in command], check=True, env=env)


def _training_env(output_dir):
    run_id_path = output_dir / ".wandb-run-id"
    if run_id_path.exists():
        run_id = run_id_path.read_text(encoding="utf-8").strip()
    else:
        run_id = uuid.uuid4().hex[:8]
        output_dir.mkdir(parents=True, exist_ok=True)
        run_id_path.write_text(run_id, encoding="utf-8")
    env = os.environ.copy()
    env["WANDB_RUN_ID"] = run_id
    env["WANDB_RESUME"] = "allow"
    return env


def _write_loop_summary(path, records, *, stopped_early):
    payload = {"stopped_early": stopped_early, "evaluations": records}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_loop(config_path, *, initial_checkpoint=None):
    config = load_yaml_config(config_path)
    loop_config = config["checkpoint_eval"]
    training_config = config["grpo"]
    output_dir = Path(training_config["output_dir"])
    output_root = Path(loop_config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    max_steps = training_config["max_steps"]
    steps = sorted(set(loop_config["steps"]))
    if not steps or steps[-1] > max_steps:
        raise ValueError("checkpoint_eval.steps must be non-empty and not exceed grpo.max_steps")
    if steps[-1] != max_steps:
        steps.append(max_steps)

    early = loop_config.get("early_stopping", {})
    benchmark_names = early.get(
        "benchmarks", ["gsm8k", "gsm1k", "math", "mmlu_stem"]
    )
    patience = early.get("patience")
    min_delta = early.get("min_delta", 0.0)
    min_steps = early.get("min_steps", 0)
    train_command = loop_config.get("train_command", "model-grpo")
    eval_command = loop_config.get("eval_command", "model-eval")
    batch_size = loop_config.get("batch_size", 768)
    summary_path = output_root / "loop-summary.json"
    resume_checkpoint = Path(initial_checkpoint) if initial_checkpoint else None
    train_env = _training_env(output_dir)
    records = []
    best_score = float("-inf")
    stale_evals = 0

    for target_step in steps:
        eval_dir = output_root / f"checkpoint-{target_step}"
        result_path = find_summary(eval_dir)
        available = complete_checkpoints(output_dir)
        candidates = [path for path in available if checkpoint_step(path) == target_step]
        if candidates:
            resume_checkpoint = candidates[0]
        elif resume_checkpoint is None:
            prior = [path for path in available if checkpoint_step(path) < target_step]
            if prior:
                resume_checkpoint = prior[-1]
        if result_path is None:
            if not candidates:
                command = [
                    train_command,
                    "--config",
                    config_path,
                    "--stop-after-step",
                    target_step,
                ]
                if resume_checkpoint is not None:
                    command.extend(["--resume-from-checkpoint", resume_checkpoint])
                _run(command, env=train_env)
                candidates = [
                    path
                    for path in complete_checkpoints(output_dir)
                    if checkpoint_step(path) == target_step
                ]
                if not candidates:
                    raise RuntimeError(f"Training did not create checkpoint-{target_step}")
            resume_checkpoint = candidates[0]
            _run(
                [
                    eval_command,
                    "--config",
                    loop_config["config"],
                    "--model",
                    output_dir,
                    "--output-dir",
                    eval_dir,
                    "--batch-size",
                    batch_size,
                ]
            )
            result_path = find_summary(eval_dir)
            if result_path is None:
                raise RuntimeError(f"Evaluation did not create a summary for step {target_step}")

        result = json.loads(result_path.read_text(encoding="utf-8"))
        score = evaluation_score(result, benchmark_names)
        record = {
            "step": target_step,
            "score": score,
            "summary": str(result_path),
            "benchmarks": result["benchmarks"],
        }
        records.append(record)
        if score > best_score + min_delta:
            best_score = score
            stale_evals = 0
        else:
            stale_evals += 1
        should_stop = (
            patience is not None
            and target_step >= min_steps
            and stale_evals >= patience
        )
        _write_loop_summary(summary_path, records, stopped_early=should_stop)
        if should_stop:
            return summary_path

    return summary_path


def main(argv=None):
    args = parse_args(argv)
    summary = run_loop(args.config, initial_checkpoint=args.resume_from_checkpoint)
    print(f"Loop results: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
