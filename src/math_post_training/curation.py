"""Build a stable filtered dataset manifest from sampled rollout records."""

import argparse
import gzip
import hashlib
import json
import re
import unicodedata
from pathlib import Path

from datasets import Dataset


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drop-pass-count", type=int, nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--hub-repo")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    dropped_counts = set(args.drop_pass_count)
    rows = []
    histogram = {}

    with gzip.open(args.rollouts, "rt", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            rollout_records = record["rollouts"]
            pass_count = sum(item["correct"] for item in rollout_records)
            histogram[str(pass_count)] = histogram.get(str(pass_count), 0) + 1
            if pass_count in dropped_counts:
                continue

            problem = record["problem"]
            rollout_count = len(rollout_records)
            rows.append(
                {
                    "example_id": _example_id(problem),
                    "question": problem,
                    "answer": f"#### {record['reference_answer']}",
                    "reference_answer": record["reference_answer"],
                    "source": record["source"],
                    "difficulty": record.get("difficulty"),
                    "topic": record.get("topic"),
                    "rollout_model": args.model,
                    "rollout_count": rollout_count,
                    "pass_count": pass_count,
                    "pass_rate": pass_count / rollout_count,
                    "parse_rate": sum(item["parsed"] for item in rollout_records) / rollout_count,
                    "truncation_rate": sum(item["truncated"] for item in rollout_records)
                    / rollout_count,
                    "mean_completion_tokens": sum(
                        item["completion_tokens"] for item in rollout_records
                    )
                    / rollout_count,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_list(rows)
    dataset.to_parquet(args.output)
    manifest = {
        "rollouts": str(args.rollouts),
        "rollout_model": args.model,
        "drop_pass_count": sorted(dropped_counts),
        "source_histogram": histogram,
        "kept": len(rows),
        "dropped": sum(histogram.get(str(count), 0) for count in dropped_counts),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.hub_repo:
        dataset.push_to_hub(args.hub_repo, split="train", private=args.private)


def _example_id(problem):
    normalized = unicodedata.normalize("NFKC", problem)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.sha256(normalized.encode()).hexdigest()


if __name__ == "__main__":
    main()
