import pytest

from math_post_training.prompts.evaluation import (
    build_evaluation_prompt,
    get_evaluation_settings,
)


class RecordingTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        self.messages = messages
        return "rendered prompt"


def test_base_protocol_uses_published_shot_counts():
    assert get_evaluation_settings("qwen2_5_math_base", "gsm8k")["num_shots"] == 8
    assert get_evaluation_settings("qwen2_5_math_base", "math")["num_shots"] == 4
    assert get_evaluation_settings("qwen2_5_math_base", "gsm8k")["stop_strings"] == [
        "Question:"
    ]


def test_base_gsm_prompt_is_raw_and_ends_with_target_problem():
    prompt = build_evaluation_prompt(
        RecordingTokenizer(),
        "What is 1 + 1?",
        benchmark="gsm8k",
        protocol="qwen2_5_math_base",
    )

    assert prompt.count("Question:") == 9
    assert prompt.endswith("Question: What is 1 + 1?\nLet's think step by step")


def test_base_zero_shot_prompt_asks_for_a_direct_answer():
    prompt = build_evaluation_prompt(
        RecordingTokenizer(),
        "What is 1 + 1?",
        benchmark="gsm8k",
        protocol="qwen2_5_math_base_zero_shot",
    )

    assert prompt == "Question: What is 1 + 1?\nAnswer:"
    assert (
        get_evaluation_settings(
            "qwen2_5_math_base_zero_shot",
            "gsm8k",
        )["num_shots"]
        == 0
    )


def test_base_zero_shot_cot_prompt_adds_reasoning_cue():
    prompt = build_evaluation_prompt(
        RecordingTokenizer(),
        "What is 1 + 1?",
        benchmark="gsm8k",
        protocol="qwen2_5_math_base_zero_shot_cot",
    )

    assert prompt == "Question: What is 1 + 1?\nLet's think step by step"


def test_instruct_protocol_uses_chat_template_and_boxed_system_prompt():
    tokenizer = RecordingTokenizer()
    prompt = build_evaluation_prompt(
        tokenizer,
        "What is 1 + 1?",
        benchmark="gsm8k",
        protocol="qwen2_5_math_instruct",
    )

    assert prompt == "rendered prompt"
    assert tokenizer.messages[0]["role"] == "system"
    assert r"\boxed{}" in tokenizer.messages[0]["content"]
    assert tokenizer.messages[1] == {"role": "user", "content": "What is 1 + 1?"}


def test_mmlu_prompt_adds_subject_dev_examples():
    demonstrations = [
        {
            "problem": "Demo\nA. one\nB. two\nC. three\nD. four",
            "answer": "B",
        }
    ]
    prompt = build_evaluation_prompt(
        RecordingTokenizer(),
        "Target\nA. one\nB. two\nC. three\nD. four",
        benchmark="mmlu_stem",
        protocol="qwen2_5_math_base",
        demonstrations=demonstrations,
    )

    assert prompt.startswith("Question: Demo")
    assert "Answer: B" in prompt
    assert prompt.endswith("D. four\nAnswer:")
    settings = get_evaluation_settings(
        "qwen2_5_math_base",
        "mmlu_stem",
        demonstrations,
    )
    assert settings["num_shots"] == 1
    assert settings["answer_kind"] == "choice"


def test_mmlu_requires_dev_examples():
    with pytest.raises(ValueError, match="requires dev demonstrations"):
        build_evaluation_prompt(
            RecordingTokenizer(),
            "question",
            benchmark="mmlu_stem",
            protocol="qwen2_5_math_base",
        )
