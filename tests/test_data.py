import pytest

from math_post_training.data.preprocessing import to_grpo_example, to_sft_example
from math_post_training.data.sources import gsm8k, open_math_instruct_2


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
