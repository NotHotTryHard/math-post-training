import gzip
import json

import pytest

pytest.importorskip("math_verify")

from math_post_training import evaluation  # noqa: E402


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return messages[-1]["content"]

    def encode(self, text, add_special_tokens=False):
        return text.split()


class FakeBackend:
    def generate(self, prompts, config):
        return [["The answer is four. #### 4"] for _ in prompts]


def test_evaluation_writes_summary_and_compressed_predictions(monkeypatch, tmp_path):
    monkeypatch.setattr(
        evaluation,
        "load_math_source",
        lambda source: iter(
            [
                {
                    "problem": "What is 2 + 2?",
                    "solution": None,
                    "answer": "4",
                    "source": "fixture",
                }
            ]
        ),
    )
    config = {
        "experiment": {"name": "test-eval"},
        "evaluation": {
            "protocol": "qwen2_5_math_instruct",
            "output_dir": str(tmp_path),
            "sample_seed": 42,
            "shuffle_buffer_size": 100,
            "batch_size": 1,
            "save_predictions": True,
            "generation": {
                "max_new_tokens": 32,
                "num_return_sequences": 1,
                "do_sample": False,
            },
            "benchmarks": [
                {
                    "name": "gsm8k",
                    "adapter": "fixture",
                    "path": "fixture",
                    "revision": "fixture",
                    "split": "test",
                }
            ],
        },
    }

    run_dir, summary = evaluation.evaluate_model(
        FakeBackend(), FakeTokenizer(), config, model_name="fake-model"
    )

    assert summary["benchmarks"]["gsm8k"]["accuracy"] == 1.0
    assert json.loads((run_dir / "summary.json").read_text())["model"] == "fake-model"
    with gzip.open(run_dir / "predictions.jsonl.gz", "rt") as file:
        prediction = json.loads(file.readline())
    assert prediction["correct"] is True


def test_limited_multi_subset_benchmark_is_round_robin(monkeypatch):
    monkeypatch.setattr(
        evaluation,
        "load_math_source",
        lambda source: iter(
            [
                {
                    "problem": str(index),
                    "answer": str(index),
                    "source": source["subset"],
                }
                for index in range(3)
            ]
        ),
    )
    benchmark = {
        "adapter": "fixture",
        "path": "fixture",
        "revision": "fixture",
        "split": "test",
        "subsets": ["a", "b", "c"],
    }

    examples = list(
        evaluation._benchmark_examples(
            benchmark,
            cli_limit=5,
            sample_seed=42,
            shuffle_buffer_size=100,
        )
    )

    assert [example["source"] for example in examples] == ["a", "b", "c", "a", "b"]


def test_limited_benchmark_passes_shuffle_settings(monkeypatch):
    seen = []

    def load(source):
        seen.append(source)
        return iter([])

    monkeypatch.setattr(evaluation, "load_math_source", load)
    benchmark = {
        "adapter": "fixture",
        "path": "fixture",
        "revision": "fixture",
        "split": "test",
    }

    list(
        evaluation._benchmark_examples(
            benchmark,
            cli_limit=5,
            sample_seed=123,
            shuffle_buffer_size=456,
        )
    )

    assert seen[0]["shuffle_seed"] == 123
    assert seen[0]["shuffle_buffer_size"] == 456
