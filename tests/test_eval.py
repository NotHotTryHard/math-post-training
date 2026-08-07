import gzip
import json

import pytest

pytest.importorskip("math_verify")

from math_post_training import eval  # noqa: E402


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return messages[-1]["content"]

    def encode(self, text, add_special_tokens=False):
        return text.split()


class FakeBackend:
    def generate(self, prompts, config):
        return [["The answer is four. #### 4"] for _ in prompts]


class FakeChoiceBackend:
    def generate(self, prompts, config):
        return [["After checking the options, the answer is C."] for _ in prompts]


def test_eval_writes_summary_and_compressed_predictions(monkeypatch, tmp_path):
    monkeypatch.setattr(
        eval,
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
        "eval": {
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

    run_dir, summary = eval.eval_model(
        FakeBackend(), FakeTokenizer(), config, model_name="fake-model"
    )

    assert summary["benchmarks"]["gsm8k"]["accuracy"] == 1.0
    assert json.loads((run_dir / "summary.json").read_text())["model"] == "fake-model"
    with gzip.open(run_dir / "predictions.jsonl.gz", "rt") as file:
        prediction = json.loads(file.readline())
    assert prediction["correct"] is True
    assert prediction["extraction_method"] == "delimiter"


def test_limited_multi_subset_benchmark_is_round_robin(monkeypatch):
    monkeypatch.setattr(
        eval,
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
        eval._benchmark_examples(
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

    monkeypatch.setattr(eval, "load_math_source", load)
    benchmark = {
        "adapter": "fixture",
        "path": "fixture",
        "revision": "fixture",
        "split": "test",
    }

    list(
        eval._benchmark_examples(
            benchmark,
            cli_limit=5,
            sample_seed=123,
            shuffle_buffer_size=456,
        )
    )

    assert seen[0]["shuffle_seed"] == 123
    assert seen[0]["shuffle_buffer_size"] == 456


def test_mmlu_few_shots_are_loaded_from_each_subject_dev_split(monkeypatch):
    seen = []

    def load(source):
        seen.append(source)
        return iter(
            {
                "problem": f"demo {index}",
                "solution": None,
                "answer": "A",
                "source": f"cais/mmlu:{source['subset']}",
            }
            for index in range(5)
        )

    monkeypatch.setattr(eval, "load_math_source", load)
    benchmark = {
        "name": "mmlu_stem",
        "adapter": "mmlu",
        "path": "cais/mmlu",
        "revision": "fixture",
        "split": "test",
        "few_shot_split": "dev",
        "num_few_shots": 5,
        "subsets": ["abstract_algebra", "anatomy"],
    }

    demonstrations = eval._load_few_shot_demonstrations(benchmark)

    assert set(demonstrations) == {"abstract_algebra", "anatomy"}
    assert all(len(examples) == 5 for examples in demonstrations.values())
    assert {source["split"] for source in seen} == {"dev"}


def test_mmlu_uses_dev_few_shots_and_exact_choice_grading(monkeypatch, tmp_path):
    def load(source):
        count = 5 if source["split"] == "dev" else 1
        return iter(
            {
                "problem": f"question {index}\nA. one\nB. two\nC. three\nD. four",
                "solution": None,
                "answer": "C",
                "source": "cais/mmlu:abstract_algebra",
            }
            for index in range(count)
        )

    monkeypatch.setattr(eval, "load_math_source", load)
    config = {
        "experiment": {"name": "mmlu-eval"},
        "eval": {
            "protocol": "qwen2_5_math_base",
            "output_dir": str(tmp_path),
            "sample_seed": 42,
            "shuffle_buffer_size": 100,
            "batch_size": 1,
            "save_predictions": False,
            "generation": {
                "max_new_tokens": 32,
                "num_return_sequences": 1,
                "do_sample": False,
            },
            "benchmarks": [
                {
                    "name": "mmlu_stem",
                    "adapter": "mmlu",
                    "path": "cais/mmlu",
                    "revision": "fixture",
                    "split": "test",
                    "few_shot_split": "dev",
                    "num_few_shots": 5,
                    "subsets": ["abstract_algebra"],
                }
            ],
        },
    }

    _, summary = eval.eval_model(
        FakeChoiceBackend(),
        FakeTokenizer(),
        config,
        model_name="fake-model",
    )

    result = summary["benchmarks"]["mmlu_stem"]
    assert result["accuracy"] == 1.0
    assert result["num_shots"] == 5
    assert result["extraction_methods"] == {"answer_marker": 1}
