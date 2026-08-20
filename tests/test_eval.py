import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from math_post_training import eval
from math_post_training.config import load_yaml_config

BASELINE_CONFIGS = sorted((Path(__file__).parents[1] / "configs" / "eval").glob("*.yaml"))
MMLU_STEM_SUBSETS = {
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "electrical_engineering",
    "elementary_mathematics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_statistics",
    "machine_learning",
}


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
    class FakeTable:
        def __init__(self, columns):
            self.columns = columns
            self.rows = []

        def add_data(self, *row):
            self.rows.append(row)

    class FakeRun:
        def __init__(self):
            self.summary = {}
            self.logged = []
            self.exit_code = None

        def log(self, data):
            self.logged.append(data)

        def finish(self, exit_code):
            self.exit_code = exit_code

    wandb_run = FakeRun()
    wandb_init_kwargs = {}

    def init_wandb(**kwargs):
        wandb_init_kwargs.update(kwargs)
        return wandb_run

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=init_wandb, Table=FakeTable),
    )
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
        "model": {"name_or_path": "fake-model"},
        "eval": {
            "protocol": "qwen2_5_math_instruct",
            "output_dir": str(tmp_path),
            "sample_seed": 42,
            "shuffle_buffer_size": 100,
            "batch_size": 1,
            "save_predictions": True,
            "wandb": {"enabled": True, "log_predictions": True},
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
    assert wandb_run.summary["eval/gsm8k/accuracy"] == 1.0
    assert wandb_run.logged[-1]["eval/gsm8k/predictions"].rows
    assert wandb_run.exit_code == 0
    assert wandb_init_kwargs["tags"] == [
        "eval",
        "transformers",
        "qwen2_5_math_instruct",
        "test-eval",
    ]


def test_truncated_completion_does_not_trust_last_number(monkeypatch, tmp_path):
    class TruncatedBackend:
        def generate(self, prompts, config):
            return [["unfinished reasoning gives 4"] for _ in prompts]

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
        "experiment": {"name": "truncated-eval"},
        "eval": {
            "protocol": "qwen2_5_math_base_zero_shot",
            "output_dir": str(tmp_path),
            "sample_seed": 42,
            "shuffle_buffer_size": 100,
            "batch_size": 1,
            "save_predictions": True,
            "wandb": {"enabled": False},
            "generation": {
                "max_new_tokens": 4,
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
        TruncatedBackend(), FakeTokenizer(), config, model_name="fake-model"
    )

    with gzip.open(run_dir / "predictions.jsonl.gz", "rt") as file:
        prediction = json.loads(file.readline())
    assert prediction["truncated"] is True
    assert prediction["extraction_method"] == "last_number"
    assert prediction["extracted_answer"] == "4"
    assert prediction["parsed"] is False
    assert prediction["correct"] is False
    assert summary["benchmarks"]["gsm8k"]["parse_rate"] == 0.0


def test_metric_aggregation_separates_accuracy_from_parse_rate():
    metrics = eval._empty_metrics()
    for parsed, correct in [(True, True), (True, False), (False, False)]:
        eval._update_metrics(
            metrics,
            {
                "parsed": parsed,
                "correct": correct,
                "format_ok": None,
                "truncated": False,
                "completion_tokens": 1,
                "extraction_method": "fixture",
                "source": "fixture",
            },
        )

    result = eval._finish_metrics(metrics)

    assert result["total"] == 3
    assert result["correct"] == 1
    assert result["accuracy"] == pytest.approx(1 / 3)
    assert result["parse_rate"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("config_path", BASELINE_CONFIGS, ids=lambda path: path.stem)
def test_baseline_config_runs_every_benchmark(monkeypatch, tmp_path, config_path):
    def load(source):
        subset = source.get("subset")
        is_mmlu = source["adapter"] == "mmlu"
        return iter(
            {
                "problem": (
                    "Which option is correct?\nA. one\nB. two\nC. three\nD. four"
                    if is_mmlu
                    else "What is 2 + 2?"
                ),
                "solution": None,
                "answer": "C" if is_mmlu else "4",
                "source": f"{source['path']}:{subset}" if subset else source["path"],
            }
            for _ in range(1)
        )

    class ConfigBackend:
        def generate(self, prompts, config):
            return [
                ["The answer is C." if "Which option is correct?" in prompt else "The answer is 4."]
                for prompt in prompts
            ]

    monkeypatch.setattr(eval, "load_math_source", load)
    config = load_yaml_config(config_path)
    config["eval"]["wandb"]["enabled"] = False

    _, summary = eval.eval_model(
        ConfigBackend(),
        FakeTokenizer(),
        config,
        model_name=config["model"]["name_or_path"],
        limit=1,
        output_dir=tmp_path,
    )

    assert set(summary["benchmarks"]) == {"gsm8k", "gsm1k", "math", "mmlu_stem"}
    assert all(result["total"] == 1 for result in summary["benchmarks"].values())


@pytest.mark.parametrize("config_path", BASELINE_CONFIGS, ids=lambda path: path.stem)
def test_baseline_configs_use_the_full_mmlu_stem_test_split(config_path):
    config = load_yaml_config(config_path)
    benchmark = next(
        benchmark for benchmark in config["eval"]["benchmarks"] if benchmark["name"] == "mmlu_stem"
    )

    assert benchmark["split"] == "test"
    assert "limit" not in benchmark
    assert set(benchmark["subsets"]) == MMLU_STEM_SUBSETS


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
    assert seen[0]["streaming"] is False


def test_mmlu_uses_fixed_paper_shots_and_exact_choice_grading(monkeypatch, tmp_path):
    seen_splits = []

    def load(source):
        seen_splits.append(source["split"])
        return iter(
            {
                "problem": f"question {index}\nA. one\nB. two\nC. three\nD. four",
                "solution": None,
                "answer": "C",
                "source": "cais/mmlu:abstract_algebra",
            }
            for index in range(1)
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
    assert result["num_shots"] == 4
    assert result["extraction_methods"] == {"answer_marker": 1}
    assert seen_splits == ["test"]
