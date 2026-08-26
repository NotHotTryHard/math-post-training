"""Backward-compatible facade for the evaluation package."""

from math_post_training.evaluation.metrics import EvaluationMetrics
from math_post_training.evaluation.runner import (
    PREDICTION_COLUMNS,
    eval_model,
)
from math_post_training.evaluation.runner import (
    _benchmark_examples as _benchmark_examples,
)
from math_post_training.evaluation.scoring import select_majority_vote


def _empty_metrics():
    return EvaluationMetrics()


def _update_metrics(metrics, record):
    metrics.add(record)


def _update_rollout_metrics(metrics, records, vote):
    metrics.add_rollouts(records, vote)


def _finish_metrics(metrics):
    return metrics.finish()


_select_majority_vote = select_majority_vote

__all__ = ["PREDICTION_COLUMNS", "eval_model"]
