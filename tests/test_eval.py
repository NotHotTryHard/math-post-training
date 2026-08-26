import gzip
import json
from pathlib import Path

import pytest

from math_post_training.config import load_yaml_config
from math_post_training.evaluation import eval_model, runner
from math_post_training.evaluation.metrics import EvaluationMetrics
from math_post_training.evaluation.runner import _benchmark_examples
from math_post_training.evaluation.scoring import select_majority_vote

EVAL_CONFIGS = sorted(
    path
    for path in (Path(__file__).parents[1] / "configs" / "eval").glob("*.yaml")
    if "deepmath" not in path.stem
)
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
    eos_token = "<|endoftext|>"
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return messages[-1]["content"]

    def encode(self, text, add_special_tokens=False):
        return text.split()


class FakeBackend:
    def generate(self, prompts, config):
        return [["The answer is four. #### 4"] for _ in prompts]


def _eval_config(tmp_path, *, protocol="qwen2_5_math_instruct", max_new_tokens=32):
    return {
        "experiment": {"name": "test-eval"},
        "eval": {
            "protocol": protocol,
            "output_dir": str(tmp_path),
            "sample_seed": 42,
            "shuffle_buffer_size": 100,
            "batch_size": 1,
            "save_predictions": True,
            "wandb": {"enabled": False},
            "generation": {
                "max_new_tokens": max_new_tokens,
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


def _load_single_example(_source):
    return iter(
        [
            {
                "problem": "What is 2 + 2?",
                "solution": None,
                "answer": "4",
                "source": "fixture",
            }
        ]
    )


def test_majority_vote_groups_equivalent_answers_and_resolves_ties():
    math_records = [
        {"extracted_answer": answer, "parsed": True, "correct": correct}
        for answer, correct in [(r"\frac{1}{2}", True), ("0.5", True), ("2", False)]
    ]
    selected, vote = select_majority_vote(math_records, answer_kind="math")
    assert selected["correct"] is True
    assert vote["vote_count"] == 2
    assert vote["vote_tied"] is False

    choice_records = [
        {"extracted_answer": answer, "parsed": True, "correct": answer == "A"}
        for answer in ["A", "B", "A", "B"]
    ]
    selected, vote = select_majority_vote(choice_records, answer_kind="choice")
    assert selected["extracted_answer"] == "A"
    assert vote["vote_tied"] is True


def test_eval_writes_stable_summary_and_prediction_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "load_math_source", _load_single_example)

    run_dir, summary = eval_model(
        FakeBackend(),
        FakeTokenizer(),
        _eval_config(tmp_path),
        model_name="fake-model",
    )

    assert summary["benchmarks"]["gsm8k"]["accuracy"] == 1.0
    assert json.loads((run_dir / "summary.json").read_text())["model"] == "fake-model"
    with gzip.open(run_dir / "predictions.jsonl.gz", "rt") as file:
        prediction = json.loads(file.readline())
    assert prediction["correct"] is True
    assert prediction["extraction_method"] == "delimiter"


def test_truncated_completion_does_not_trust_a_fallback_answer(monkeypatch, tmp_path):
    class TruncatedBackend:
        def generate(self, prompts, config):
            return [["unfinished reasoning gives 4"] for _ in prompts]

    monkeypatch.setattr(runner, "load_math_source", _load_single_example)
    config = _eval_config(
        tmp_path,
        protocol="qwen2_5_math_base_zero_shot",
        max_new_tokens=4,
    )

    run_dir, summary = eval_model(
        TruncatedBackend(),
        FakeTokenizer(),
        config,
        model_name="fake-model",
    )

    with gzip.open(run_dir / "predictions.jsonl.gz", "rt") as file:
        prediction = json.loads(file.readline())
    assert prediction["truncated"] is True
    assert prediction["extraction_method"] == "last_number"
    assert prediction["parsed"] is False
    assert prediction["correct"] is False
    assert summary["benchmarks"]["gsm8k"]["parse_rate"] == 0.0


def test_metrics_keep_accuracy_parse_and_difficulty_statistics_separate():
    metrics = EvaluationMetrics()
    for parsed, correct, truncated, tokens in [
        (True, True, False, 100),
        (True, False, True, 200),
        (False, False, False, 300),
    ]:
        metrics.add(
            {
                "parsed": parsed,
                "correct": correct,
                "format_ok": True,
                "truncated": truncated,
                "completion_tokens": tokens,
                "extraction_method": "fixture",
                "source": "fixture",
                "difficulty": 5.0,
            }
        )

    result = metrics.finish()
    assert result["accuracy"] == pytest.approx(1 / 3)
    assert result["parse_rate"] == pytest.approx(2 / 3)
    assert result["by_difficulty"]["5"] == {
        "total": 3,
        "correct": 1,
        "accuracy": pytest.approx(1 / 3),
        "parse_rate": pytest.approx(2 / 3),
        "truncated": 1,
        "truncation_rate": pytest.approx(1 / 3),
        "mean_completion_tokens": 200,
    }


def test_eval_configs_define_the_same_full_benchmark_suite():
    for path in EVAL_CONFIGS:
        config = load_yaml_config(path)
        benchmarks = {benchmark["name"]: benchmark for benchmark in config["eval"]["benchmarks"]}
        assert set(benchmarks) == {"gsm8k", "gsm1k", "math", "mmlu_stem"}, path
        mmlu = benchmarks["mmlu_stem"]
        assert mmlu["split"] == "test", path
        assert "limit" not in mmlu, path
        assert set(mmlu["subsets"]) == MMLU_STEM_SUBSETS, path


def test_limited_multisubset_eval_round_robins_and_forwards_shuffle(monkeypatch):
    seen_sources = []

    def load(source):
        seen_sources.append(source)
        return iter(
            {"problem": str(index), "answer": str(index), "source": source["subset"]}
            for index in range(3)
        )

    monkeypatch.setattr(runner, "load_math_source", load)
    benchmark = {
        "adapter": "fixture",
        "path": "fixture",
        "revision": "fixture",
        "split": "test",
        "subsets": ["a", "b", "c"],
    }

    examples = list(
        _benchmark_examples(
            benchmark,
            cli_limit=5,
            sample_seed=123,
            shuffle_buffer_size=456,
        )
    )

    assert [example["source"] for example in examples] == ["a", "b", "c", "a", "b"]
    assert all(source["shuffle_seed"] == 123 for source in seen_sources)
    assert all(source["shuffle_buffer_size"] == 456 for source in seen_sources)
