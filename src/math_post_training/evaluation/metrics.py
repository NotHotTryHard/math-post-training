"""Typed accumulation of selected-completion and rollout metrics."""

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class DifficultyMetrics:
    total: int = 0
    correct: int = 0
    parsed: int = 0
    truncated: int = 0
    completion_tokens: int = 0


@dataclass
class SourceMetrics:
    total: int = 0
    correct: int = 0


@dataclass
class EvaluationMetrics:
    total: int = 0
    correct: int = 0
    parsed: int = 0
    formatted: int = 0
    format_total: int = 0
    truncated: int = 0
    completion_tokens: int = 0
    by_source: dict[str, SourceMetrics] = field(default_factory=dict)
    by_difficulty: dict[str, DifficultyMetrics] = field(default_factory=dict)
    extraction_methods: Counter = field(default_factory=Counter)
    rollout_total: int = 0
    rollout_correct: int = 0
    rollout_parsed: int = 0
    rollout_formatted: int = 0
    rollout_format_total: int = 0
    rollout_truncated: int = 0
    rollout_completion_tokens: int = 0
    groups_with_any_correct: int = 0
    vote_ties: int = 0
    vote_groups: int = 0

    def add(self, record):
        self.total += 1
        self.correct += int(record["correct"])
        self.parsed += int(record["parsed"])
        if record["format_ok"] is not None:
            self.format_total += 1
            self.formatted += int(record["format_ok"])
        self.truncated += int(record["truncated"])
        self.completion_tokens += record["completion_tokens"]
        self.extraction_methods[record["extraction_method"]] += 1

        source = self.by_source.setdefault(record["source"], SourceMetrics())
        source.total += 1
        source.correct += int(record["correct"])

        difficulty = record.get("difficulty")
        if difficulty is not None:
            bucket = self.by_difficulty.setdefault(f"{difficulty:g}", DifficultyMetrics())
            bucket.total += 1
            bucket.correct += int(record["correct"])
            bucket.parsed += int(record["parsed"])
            bucket.truncated += int(record["truncated"])
            bucket.completion_tokens += record["completion_tokens"]

    def add_rollouts(self, records, vote):
        self.rollout_total += len(records)
        self.rollout_correct += sum(record["correct"] for record in records)
        self.rollout_parsed += sum(record["parsed"] for record in records)
        self.rollout_truncated += sum(record["truncated"] for record in records)
        self.rollout_completion_tokens += sum(record["completion_tokens"] for record in records)
        formatted = [record for record in records if record["format_ok"] is not None]
        self.rollout_format_total += len(formatted)
        self.rollout_formatted += sum(record["format_ok"] for record in formatted)
        self.groups_with_any_correct += int(any(record["correct"] for record in records))
        if vote is not None:
            self.vote_groups += 1
            self.vote_ties += int(vote["vote_tied"])

    def finish(self):
        if self.total == 0:
            raise ValueError("Cannot finish empty evaluation metrics")

        result = {
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.correct / self.total,
            "parse_rate": self.parsed / self.total,
            "format_rate": self.formatted / self.format_total if self.format_total else None,
            "truncated": self.truncated,
            "mean_completion_tokens": self.completion_tokens / self.total,
            "extraction_methods": dict(self.extraction_methods),
        }
        if self.rollout_total > self.total:
            result.update(
                {
                    "rollouts_per_example": self.rollout_total / self.total,
                    "individual_rollout_accuracy": self.rollout_correct / self.rollout_total,
                    "individual_rollout_parse_rate": self.rollout_parsed / self.rollout_total,
                    "individual_rollout_format_rate": (
                        self.rollout_formatted / self.rollout_format_total
                        if self.rollout_format_total
                        else None
                    ),
                    "individual_rollout_truncated": self.rollout_truncated,
                    "mean_rollout_completion_tokens": (
                        self.rollout_completion_tokens / self.rollout_total
                    ),
                    "pass_at_n": self.groups_with_any_correct / self.total,
                    "vote_ties": self.vote_ties,
                    "vote_tie_rate": self.vote_ties / self.vote_groups,
                }
            )

        if len(self.by_source) > 1:
            result["by_source"] = {
                source: {
                    "total": values.total,
                    "correct": values.correct,
                    "accuracy": values.correct / values.total,
                }
                for source, values in self.by_source.items()
            }
        if self.by_difficulty:
            result["by_difficulty"] = {
                difficulty: {
                    "total": values.total,
                    "correct": values.correct,
                    "accuracy": values.correct / values.total,
                    "parse_rate": values.parsed / values.total,
                    "truncated": values.truncated,
                    "truncation_rate": values.truncated / values.total,
                    "mean_completion_tokens": values.completion_tokens / values.total,
                }
                for difficulty, values in sorted(
                    self.by_difficulty.items(), key=lambda item: float(item[0])
                )
            }
        return result
