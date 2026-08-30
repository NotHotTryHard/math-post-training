"""Rescore and compare paired DeepMath level-audit rollouts."""

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from math_post_training.verifiers.math import check_answer


def _read(path):
    with gzip.open(path, "rt", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def _score(record):
    scores = []
    reference = record["reference_answer"].strip()
    plain_choices = bool(reference) and all(
        part.strip().upper() in {"A", "B", "C", "D", "E"} for part in reference.split(",")
    )
    for rollout in record["rollouts"]:
        if rollout["truncated"] and rollout["extraction_method"] in {"last_number", "last_choice"}:
            scores.append((False, False))
        elif plain_choices:
            scores.append(check_answer(reference, rollout["extracted_answer"] or ""))
        else:
            scores.append((rollout["parsed"], rollout["correct"]))
    return scores


def _aggregate(records):
    scored = [(record, _score(record)) for record in records]
    flat = [
        (record, rollout, score)
        for record, scores in scored
        for rollout, score in zip(record["rollouts"], scores, strict=True)
    ]
    histogram = Counter(sum(correct for _, correct in scores) for _, scores in scored)
    return {
        "prompts": len(records),
        "rollouts": len(flat),
        "accuracy": sum(score[1] for _, _, score in flat) / len(flat),
        "parse_rate": sum(score[0] for _, _, score in flat) / len(flat),
        "format_rate": sum(rollout["format_ok"] for _, rollout, _ in flat) / len(flat),
        "truncation_rate": sum(rollout["truncated"] for _, rollout, _ in flat) / len(flat),
        "mean_completion_tokens": sum(rollout["completion_tokens"] for _, rollout, _ in flat)
        / len(flat),
        "pass_histogram": {str(k): histogram[k] for k in range(4)},
        "mixed_rate": (histogram[1] + histogram[2]) / len(records),
        "any_pass_rate": (len(records) - histogram[0]) / len(records),
        "all_pass_rate": histogram[3] / len(records),
    }


def _topic_family(topic):
    parts = [part.strip() for part in (topic or "Unknown").split("->")]
    return parts[1] if len(parts) > 1 else parts[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--math", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sft = {record["sample_id"]: record for record in _read(args.sft)}
    math = {record["sample_id"]: record for record in _read(args.math)}
    if set(sft) != set(math):
        common = len(set(sft) & set(math))
        raise ValueError(
            f"Sample ID mismatch: sft={len(sft)} math={len(math)} common={common}"
        )

    levels = sorted({float(record["difficulty"]) for record in sft.values()})
    by_level = {}
    matrix = Counter()
    paired_by_level = defaultdict(Counter)
    topic_rows = []

    for sample_id in sorted(sft):
        sft_record = sft[sample_id]
        math_record = math[sample_id]
        sft_passes = sum(correct for _, correct in _score(sft_record))
        math_passes = sum(correct for _, correct in _score(math_record))
        matrix[(sft_passes, math_passes)] += 1
        level = float(sft_record["difficulty"])
        paired_by_level[level][(sft_passes, math_passes)] += 1

    for level in levels:
        sft_records = [record for record in sft.values() if float(record["difficulty"]) == level]
        math_records = [
            math[sample_id]
            for sample_id, record in sft.items()
            if float(record["difficulty"]) == level
        ]
        pair = paired_by_level[level]
        by_level[f"{level:g}"] = {
            "sft": _aggregate(sft_records),
            "math": _aggregate(math_records),
            "paired": {
                "both_zero": sum(count for (a, b), count in pair.items() if a == 0 and b == 0),
                "teacher_frontier": sum(
                    count for (a, b), count in pair.items() if a == 0 and b > 0
                ),
                "sft_only": sum(count for (a, b), count in pair.items() if a > 0 and b == 0),
                "both_positive": sum(count for (a, b), count in pair.items() if a > 0 and b > 0),
            },
        }

    families = sorted({_topic_family(record.get("topic")) for record in sft.values()})
    for family in families:
        sft_records = [
            record for record in sft.values() if _topic_family(record.get("topic")) == family
        ]
        math_records = [math[record["sample_id"]] for record in sft_records]
        topic_rows.append(
            {
                "topic": family,
                "count": len(sft_records),
                "sft": _aggregate(sft_records),
                "math": _aggregate(math_records),
            }
        )

    summary = {
        "sample_ids_match": True,
        "prompts": len(sft),
        "sft": _aggregate(list(sft.values())),
        "math": _aggregate(list(math.values())),
        "by_difficulty": by_level,
        "pass_matrix": {f"sft_{a}_math_{b}": matrix[(a, b)] for a in range(4) for b in range(4)},
        "paired": {
            "both_zero": sum(count for (a, b), count in matrix.items() if a == 0 and b == 0),
            "teacher_frontier": sum(count for (a, b), count in matrix.items() if a == 0 and b > 0),
            "sft_only": sum(count for (a, b), count in matrix.items() if a > 0 and b == 0),
            "both_positive": sum(count for (a, b), count in matrix.items() if a > 0 and b > 0),
        },
        "by_topic_family": topic_rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paired-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    with (args.output_dir / "by-difficulty.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "difficulty",
                "sft_accuracy",
                "math_accuracy",
                "sft_0of3",
                "sft_mixed",
                "sft_3of3",
                "teacher_frontier",
                "both_zero",
            ]
        )
        for level, row in by_level.items():
            writer.writerow(
                [
                    level,
                    row["sft"]["accuracy"],
                    row["math"]["accuracy"],
                    row["sft"]["pass_histogram"]["0"],
                    row["sft"]["pass_histogram"]["1"] + row["sft"]["pass_histogram"]["2"],
                    row["sft"]["pass_histogram"]["3"],
                    row["paired"]["teacher_frontier"],
                    row["paired"]["both_zero"],
                ]
            )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
