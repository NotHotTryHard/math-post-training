from pathlib import Path

import pytest

from math_post_training.grpo_eval_loop import (
    complete_checkpoints,
    evaluation_score,
)


def test_complete_checkpoints_only_returns_resumable_checkpoints(tmp_path):
    complete = tmp_path / "checkpoint-500"
    complete.mkdir()
    (complete / "trainer_state.json").write_text("{}", encoding="utf-8")
    (complete / "adapter_model.safetensors").write_bytes(b"adapter")
    (tmp_path / "checkpoint-250").mkdir()

    assert complete_checkpoints(tmp_path) == [Path(complete)]


def test_evaluation_score_averages_requested_benchmarks():
    summary = {
        "benchmarks": {
            "gsm8k": {"accuracy": 0.8},
            "math": {"accuracy": 0.4},
        }
    }

    assert evaluation_score(summary, ["gsm8k", "math"]) == pytest.approx(0.6)
