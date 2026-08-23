#!/usr/bin/env python3
"""Merge and evaluate a sequence of PEFT checkpoints one at a time."""

import argparse
import gc
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--eval-command", default="model-eval")
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--every", type=int, default=1)
    return parser.parse_args()


def checkpoint_step(path):
    return int(path.name.removeprefix("checkpoint-"))


def find_summary(output_dir):
    summaries = list(output_dir.rglob("summary.json"))
    return max(summaries, key=lambda path: path.stat().st_mtime) if summaries else None


def merge_checkpoint(checkpoint, output_dir):
    peft_config = PeftConfig.from_pretrained(checkpoint)
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(base_model, checkpoint)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    tokenizer.save_pretrained(output_dir)
    del tokenizer, merged_model, peft_model, base_model
    gc.collect()


def write_index(output_root, checkpoints):
    results = []
    for checkpoint in checkpoints:
        output_dir = output_root / checkpoint.name
        summary_path = find_summary(output_dir)
        if summary_path is None:
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        results.append(
            {
                "step": checkpoint_step(checkpoint),
                "checkpoint": str(checkpoint),
                "summary": str(summary_path),
                "benchmarks": summary["benchmarks"],
            }
        )
    (output_root / "sweep-summary.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    checkpoints = sorted(
        args.checkpoint_root.glob("checkpoint-*"),
        key=checkpoint_step,
    )[:: args.every]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {args.checkpoint_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    for checkpoint in checkpoints:
        output_dir = args.output_root / checkpoint.name
        if find_summary(output_dir) is not None:
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(
            prefix=f"merged-{checkpoint.name}-",
            dir=args.work_dir,
        ) as merged_dir:
            merge_checkpoint(checkpoint, merged_dir)
            subprocess.run(
                [
                    args.eval_command,
                    "--config",
                    str(args.config),
                    "--model",
                    merged_dir,
                    "--output-dir",
                    str(output_dir),
                    "--batch-size",
                    str(args.batch_size),
                ],
                check=True,
            )
        write_index(args.output_root, checkpoints)


if __name__ == "__main__":
    main()
