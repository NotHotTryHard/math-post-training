"""Build a policy-relative DeepMath RL pool from saved rollout records."""

import argparse
import gzip
import json
from pathlib import Path

from datasets import Dataset

from math_post_training.verifiers.math import check_answer


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hub-repo")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args(argv)


def _scores(record):
    reference = record["reference_answer"].strip()
    plain_choices = bool(reference) and all(
        part.strip().upper() in {"A", "B", "C", "D", "E"} for part in reference.split(",")
    )
    scores = []
    for rollout in record["rollouts"]:
        if rollout["truncated"] and rollout["extraction_method"] in {"last_number", "last_choice"}:
            scores.append((False, False))
        elif plain_choices:
            scores.append(check_answer(reference, rollout["extracted_answer"] or ""))
        else:
            scores.append((rollout["parsed"], rollout["correct"]))
    return scores


def main(argv=None):
    args = parse_args(argv)
    rows = []
    histogram = {str(count): 0 for count in range(4)}
    seen_ids = set()

    with gzip.open(args.rollouts, "rt", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            sample_id = record["sample_id"]
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample ID: {sample_id}")
            seen_ids.add(sample_id)

            scores = _scores(record)
            pass_count = sum(correct for _, correct in scores)
            histogram[str(pass_count)] += 1
            if pass_count not in {1, 2}:
                continue

            rollout_count = len(scores)
            rows.append(
                {
                    "example_id": sample_id,
                    "question": record["problem"],
                    "final_answer": record["reference_answer"],
                    "difficulty": float(record["difficulty"]),
                    "topic": record.get("topic") or "",
                    "r1_solution_1": "",
                    "rollout_count": rollout_count,
                    "pass_count": pass_count,
                    "pass_rate": pass_count / rollout_count,
                    "parse_rate": sum(parsed for parsed, _ in scores) / rollout_count,
                    "truncation_rate": sum(item["truncated"] for item in record["rollouts"])
                    / rollout_count,
                    "mean_completion_tokens": sum(
                        item["completion_tokens"] for item in record["rollouts"]
                    )
                    / rollout_count,
                }
            )

    dataset = Dataset.from_list(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(args.output)
    summary = {
        "source_rollouts": str(args.rollouts),
        "selection": "corrected pass_count in {1, 2} of 3",
        "source_prompts": sum(histogram.values()),
        "source_histogram": histogram,
        "kept": len(rows),
        "unique_ids": len(seen_ids),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.hub_repo:
        dataset.push_to_hub(args.hub_repo, split="train", private=args.private)


if __name__ == "__main__":
    main()
