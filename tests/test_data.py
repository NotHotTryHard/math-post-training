import pytest
from datasets import ClassLabel, Dataset, Features, IterableDataset, Value

from math_post_training.data import loaders
from math_post_training.data.preprocessing import to_grpo_example, to_sft_example
from math_post_training.data.sources import (
    deepmath,
    gsm1k,
    gsm8k,
    hendrycks_math,
    mmlu,
    open_math_instruct_2,
)
from math_post_training.model import QWEN_BASE_EOS_TOKEN
from math_post_training.prompts.training import build_math_prompt


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


def test_deepmath_normalization_preserves_training_and_filter_metadata():
    example = deepmath.normalize(
        {
            "question": "Is two even?",
            "final_answer": "Yes",
            "difficulty": 3.5,
            "topic": "Mathematics -> Number Theory",
            "r1_solution_1": "Two is divisible by two, so the answer is yes.",
        }
    )

    assert example.problem == "Is two even?"
    assert example.solution == "Two is divisible by two, so the answer is yes."
    assert example.answer == "Yes"
    assert example.source == "zwhe99/DeepMath-103K"
    assert example.difficulty == 3.5
    assert example.topic == "Mathematics -> Number Theory"


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


def test_mmlu_normalization_renders_choices_and_answer_letter():
    example = mmlu.normalize(
        {
            "question": "What is 2 + 2?",
            "choices": ["1", "2", "4", "8"],
            "answer": 2,
            "subject": "elementary_mathematics",
        }
    )

    assert example.problem == "What is 2 + 2?\nA. 1\nB. 2\nC. 4\nD. 8"
    assert example.answer == "C"
    assert example.source == "cais/mmlu:elementary_mathematics"


def test_loader_does_not_reencode_normalized_mmlu_answer_as_class_label(monkeypatch):
    source = Dataset.from_dict(
        {
            "question": ["What is 2 + 2?"],
            "choices": [["1", "2", "4", "8"]],
            "answer": [2],
            "subject": ["elementary_mathematics"],
        }
    ).cast_column("answer", ClassLabel(names=list("ABCD")))
    monkeypatch.setattr(loaders, "load_dataset", lambda *args, **kwargs: source)

    dataset = loaders.load_math_source(
        {
            "adapter": "mmlu",
            "path": "cais/mmlu",
            "revision": "fixture",
            "split": "test",
        }
    )

    assert dataset.features["answer"].dtype == "string"
    assert dataset[0]["answer"] == "C"


def test_loader_filters_normalized_deepmath_before_limit(monkeypatch):
    source = Dataset.from_list(
        [
            {
                "question": f"problem-{difficulty}",
                "final_answer": "1",
                "difficulty": difficulty,
                "topic": "algebra",
                "r1_solution_1": "solution",
            }
            for difficulty in (3.0, 5.0, 7.0)
        ]
    )
    monkeypatch.setattr(loaders, "load_dataset", lambda *args, **kwargs: source)

    dataset = loaders.load_math_source(
        {
            "adapter": "deepmath",
            "path": "zwhe99/DeepMath-103K",
            "revision": "fixture",
            "split": "train",
            "filters": {"difficulty_min": 4, "difficulty_max": 6},
            "limit": 1,
        }
    )

    assert len(dataset) == 1
    assert dataset[0]["problem"] == "problem-5.0"
    assert dataset[0]["difficulty"] == 5.0


def test_sft_and_grpo_share_the_exact_plain_text_prompt():
    example = {
        "problem": "What is 2 + 2?",
        "solution": "Two plus two equals four.",
        "answer": "4",
        "source": "fixture",
    }

    expected_prompt = build_math_prompt("What is 2 + 2?")
    sft_example = to_sft_example(example)
    assert sft_example == {
        "prompt": expected_prompt,
        "completion": f"Two plus two equals four.{QWEN_BASE_EOS_TOKEN}",
    }
    assert to_grpo_example(example) == {
        "prompt": expected_prompt,
        "answer": "4",
        "source": "fixture",
    }


def test_grpo_keeps_optional_dataset_metadata():
    example = {
        "problem": "What is 2 + 2?",
        "solution": None,
        "answer": "4",
        "source": "fixture",
        "difficulty": 3.0,
        "topic": "arithmetic",
    }

    assert to_grpo_example(example) == {
        "prompt": build_math_prompt("What is 2 + 2?"),
        "answer": "4",
        "source": "fixture",
        "difficulty": 3.0,
        "topic": "arithmetic",
    }


def test_sft_normalizes_repeated_native_eos_to_one_token():
    example = {
        "problem": "What is 2 + 2?",
        "solution": f"Four.{QWEN_BASE_EOS_TOKEN}{QWEN_BASE_EOS_TOKEN}",
        "answer": "4",
        "source": "fixture",
    }

    assert to_sft_example(example)["completion"] == f"Four.{QWEN_BASE_EOS_TOKEN}"


@pytest.mark.parametrize("token", ["<|im_start|>", "<|im_end|>"])
def test_training_rejects_chatml_control_tokens(token):
    example = {
        "problem": "What is 2 + 2?",
        "solution": f"Four.{token}",
        "answer": "4",
        "source": "fixture",
    }

    with pytest.raises(ValueError, match="forbidden ChatML token"):
        to_sft_example(example)


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
    source_datasets = {"a": object(), "b": object()}
    calls = []

    monkeypatch.setattr(loaders, "load_math_source", lambda source: source_datasets[source["name"]])
    monkeypatch.setattr(
        loaders,
        "interleave_datasets",
        lambda datasets, **kwargs: calls.append((datasets, kwargs)) or "mixed",
    )

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

    assert dataset == "mixed"
    assert calls == [
        (
            [source_datasets["a"], source_datasets["b"]],
            {
                "probabilities": [0.5, 0.5],
                "seed": 42,
                "stopping_strategy": "first_exhausted",
            },
        )
    ]


def test_validation_holdout_is_deterministic_and_removed_from_training():
    dataset = Dataset.from_dict({"problem": [f"problem-{index}" for index in range(10)]})
    config = {"size": 3, "seed": 42}

    train_a, validation_a = loaders.split_train_validation(dataset, config)
    train_b, validation_b = loaders.split_train_validation(dataset, config)

    assert len(train_a) == 7
    assert len(validation_a) == 3
    assert validation_a["problem"] == validation_b["problem"]
    assert set(train_a["problem"]).isdisjoint(validation_a["problem"])


def test_streaming_holdout_can_replay_the_same_sequence_across_epochs():
    features = Features({"problem": Value("string")})
    dataset = IterableDataset.from_generator(
        lambda: ({"problem": f"problem-{index}"} for index in range(12)),
        features=features,
    ).take(10)

    train, validation = loaders.split_train_validation(
        dataset,
        {
            "size": 2,
            "seed": 42,
            "shuffle_buffer_size": 4,
        },
    )

    train_epochs = []
    validation_epochs = []
    for epoch in range(3):
        train.set_epoch(epoch)
        validation.set_epoch(epoch)
        train_epochs.append([row["problem"] for row in train])
        validation_epochs.append([row["problem"] for row in validation])

    assert len(train_epochs[0]) == 8
    assert train_epochs[0] == train_epochs[1] == train_epochs[2]
    assert validation_epochs[0] == validation_epochs[1] == validation_epochs[2]
    assert set(train_epochs[0]).isdisjoint(validation_epochs[0])
