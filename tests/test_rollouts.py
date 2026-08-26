import gzip
import json

import pytest

from math_post_training.rollouts import _empty_counters, _restore_counters, _update_counters


def _rollout(*, correct, format_ok):
    return {
        "correct": correct,
        "parsed": True,
        "format_ok": format_ok,
        "truncated": False,
        "completion_tokens": 10,
    }


def test_rollout_counters_support_protocols_without_required_format():
    counters = _empty_counters()

    _update_counters(
        counters,
        [_rollout(correct=True, format_ok=None), _rollout(correct=False, format_ok=None)],
    )

    assert counters["prompts"] == 1
    assert counters["correct"] == 1
    assert counters["format_total"] == 0
    assert counters["pass_histogram"] == {1: 1}


def test_restore_counters_reads_a_resumable_rollout_once(tmp_path):
    path = tmp_path / "rollouts.jsonl.gz"
    record = {
        "rollouts": [
            _rollout(correct=True, format_ok=True),
            _rollout(correct=False, format_ok=False),
        ]
    }
    with gzip.open(path, "wt", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

    counters = _restore_counters(path)

    assert counters["prompts"] == 1
    assert counters["rollouts"] == 2
    assert counters["formatted"] == 1
    assert counters["format_total"] == 2


def test_restore_counters_rejects_a_corrupt_resume_artifact(tmp_path):
    path = tmp_path / "rollouts.jsonl.gz"
    path.write_bytes(b"not gzip")

    with pytest.raises(RuntimeError, match="Cannot resume incomplete rollout artifact"):
        _restore_counters(path)
