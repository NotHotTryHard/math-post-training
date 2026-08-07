import pytest
from datasets import IterableDataset

from math_post_training.data import loaders
from math_post_training.data.preprocessing import to_grpo_example, to_sft_example
from math_post_training.data.sources import (
    gsm1k,
    gsm8k,
    hendrycks_math,
    open_math_instruct_2,
)


def test_gsm8k_normalization_keeps_solution_and_extracts_answer():
    example = gsm8k.normalize(
        {
            "question": "What is 2 + 2?",
            "answer": "Two plus two equals four.\n#### 4",
        }
    )

    assert example.problem == "What is 2 + 2?"
    assert example.solution == "Two plus two equals four.\n#### 4"
    assert example.answer == "4"
    assert example.source == "openai/gsm8k:main"


def test_open_math_instruct_normalization_keeps_provenance():
    example = open_math_instruct_2.normalize(
        {
            "problem": "What is 3 + 5?",
            "generated_solution": "Adding gives 8.",
            "expected_answer": "8",
            "problem_source": "gsm8k",
        }
    )

    assert example.answer == "8"
    assert example.source == "nvidia/OpenMathInstruct-2:gsm8k"


def test_gsm1k_normalization():
    example = gsm1k.normalize({"question": "What is 6 / 2?", "answer": "3"})

    assert example.problem == "What is 6 / 2?"
    assert example.solution is None
    assert example.answer == "3"


def test_math_normalization_extracts_nested_boxed_answer():
    example = hendrycks_math.normalize(
        {
            "problem": "Simplify one half.",
            "solution": r"Therefore the answer is $\boxed{\frac{1}{2}}$.",
            "type": "Prealgebra",
            "level": "Level 1",
        }
    )

    assert example.answer == r"\frac{1}{2}"
    assert example.source == "EleutherAI/hendrycks_math:prealgebra"


def test_sft_and_grpo_use_different_parts_of_the_same_example():
    example = {
        "problem": "What is 2 + 2?",
        "solution": "Two plus two equals four.",
        "answer": "4",
        "source": "fixture",
    }

    assert to_sft_example(example) == {
        "problem": "What is 2 + 2?",
        "completion": "Two plus two equals four.",
        "source": "fixture",
    }
    assert to_grpo_example(example) == {
        "problem": "What is 2 + 2?",
        "answer": "4",
        "source": "fixture",
    }


def test_grpo_rejects_an_example_without_reference_answer():
    with pytest.raises(ValueError, match="no reference answer"):
        to_grpo_example(
            {
                "problem": "An unverifiable problem",
                "solution": "A demonstration",
                "answer": None,
                "source": "fixture",
            }
        )


def test_dataset_sources_can_be_mixed(monkeypatch):
    source_datasets = {
        "a": IterableDataset.from_generator(lambda: iter([{"value": "a1"}, {"value": "a2"}])),
        "b": IterableDataset.from_generator(lambda: iter([{"value": "b1"}, {"value": "b2"}])),
    }
    monkeypatch.setattr(loaders, "load_math_source", lambda source: source_datasets[source["name"]])

    dataset = loaders.load_math_dataset(
        {
            "seed": 42,
            "stopping_strategy": "first_exhausted",
            "sources": [
                {"name": "a", "probability": 0.5},
                {"name": "b", "probability": 0.5},
            ],
        }
    )

    values = [row["value"] for row in dataset]
    assert values
    assert all(value[0] in {"a", "b"} for value in values)


def test_one_source_uses_the_same_sources_shape(monkeypatch):
    source_dataset = IterableDataset.from_generator(lambda: iter([{"value": "only"}]))
    monkeypatch.setattr(loaders, "load_math_source", lambda source: source_dataset)

    dataset = loaders.load_math_dataset({"sources": [{"name": "only"}]})

    assert list(dataset) == [{"value": "only"}]
