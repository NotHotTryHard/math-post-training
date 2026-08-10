import hashlib

import pytest

from math_post_training.prompts.eval import (
    MMLU_BASE_PREFIX,
    MMLU_INSTRUCT_PREFIX,
    build_eval_prompt,
    get_eval_settings,
)


class RecordingTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        self.messages = messages
        return "rendered prompt"


def test_base_protocol_uses_published_shot_counts():
    assert get_eval_settings("qwen2_5_math_base", "gsm8k")["num_shots"] == 8
    assert get_eval_settings("qwen2_5_math_base", "math")["num_shots"] == 4
    assert get_eval_settings("qwen2_5_math_base", "gsm8k")["stop_strings"] == [
        "Question:",
        "[Question]",
        "\nQ:",
    ]


def test_base_gsm_prompt_is_raw_and_ends_with_target_problem():
    prompt = build_eval_prompt(
        RecordingTokenizer(),
        "What is 1 + 1?",
        benchmark="gsm8k",
        protocol="qwen2_5_math_base",
    )

    assert prompt.count("Question:") == 9
    assert prompt.endswith("Question: What is 1 + 1?\nLet's think step by step")


def test_base_math_prompt_uses_four_shots_and_ends_with_target_problem():
    prompt = build_eval_prompt(
        RecordingTokenizer(),
        "What is 1 + 1?",
        benchmark="math",
        protocol="qwen2_5_math_base",
    )

    assert prompt.count("Problem:") == 5
    assert prompt.endswith("Problem: What is 1 + 1?\nSolution:")


def test_base_zero_shot_prompt_asks_for_a_direct_answer():
    prompt = build_eval_prompt(
        RecordingTokenizer(),
        "What is 1 + 1?",
        benchmark="gsm8k",
        protocol="qwen2_5_math_base_zero_shot",
    )

    assert prompt == "Question: What is 1 + 1?\nAnswer:"
    assert (
        get_eval_settings(
            "qwen2_5_math_base_zero_shot",
            "gsm8k",
        )["num_shots"]
        == 0
    )


def test_base_zero_shot_cot_prompt_adds_reasoning_cue():
    prompt = build_eval_prompt(
        RecordingTokenizer(),
        "What is 1 + 1?",
        benchmark="gsm8k",
        protocol="qwen2_5_math_base_zero_shot_cot",
    )

    assert prompt == "Question: What is 1 + 1?\nLet's think step by step"


def test_instruct_protocol_uses_chat_template_and_boxed_system_prompt():
    tokenizer = RecordingTokenizer()
    prompt = build_eval_prompt(
        tokenizer,
        "What is 1 + 1?",
        benchmark="gsm8k",
        protocol="qwen2_5_math_instruct",
    )

    assert prompt == "rendered prompt"
    assert tokenizer.messages[0]["role"] == "system"
    assert r"\boxed{}" in tokenizer.messages[0]["content"]
    assert tokenizer.messages[1] == {"role": "user", "content": "What is 1 + 1?"}


def test_base_mmlu_prompt_uses_published_four_shots():
    prompt = build_eval_prompt(
        RecordingTokenizer(),
        "Target\nA. one\nB. two\nC. three\nD. four",
        benchmark="mmlu_stem",
        protocol="qwen2_5_math_base",
    )

    assert prompt.count("Problem:\n") == 5
    assert "Final Answer: The final answer is (B)." in prompt
    assert prompt.endswith(
        "Target\nWhat of the following is the right choice? Explain your answer.\n"
        "(A) one\n(B) two\n(C) three\n(D) four\nSolution:"
    )
    settings = get_eval_settings("qwen2_5_math_base", "mmlu_stem")
    assert settings["num_shots"] == 4
    assert settings["answer_kind"] == "choice"
    assert settings["stop_strings"] == ["Problem:", "[Problem]", "\nQ:"]


def test_published_mmlu_prefixes_are_frozen():
    assert hashlib.sha256(MMLU_BASE_PREFIX.encode()).hexdigest() == (
        "023df72bcbdf5b7a3e7670d8d412555c7b4a3300ec46e3551c75cadb4f5ad9db"
    )
    assert hashlib.sha256(MMLU_INSTRUCT_PREFIX.encode()).hexdigest() == (
        "f8b198e7aeb41d272608cddc675f2a05f9228164185c83bbb02a5b4a07553cd4"
    )


def test_instruct_mmlu_prompt_uses_official_five_shots_in_one_user_turn():
    tokenizer = RecordingTokenizer()
    build_eval_prompt(
        tokenizer,
        "Target\nA. one\nB. two\nC. three\nD. four",
        benchmark="mmlu_stem",
        protocol="qwen2_5_math_instruct",
    )

    user_prompt = tokenizer.messages[1]["content"]
    assert user_prompt.count("Answer Choices:") == 6
    assert "secretory protein" in user_prompt
    assert user_prompt.endswith("Target\nAnswer Choices: (A) one (B) two (C) three (D) four")
    assert get_eval_settings("qwen2_5_math_instruct", "mmlu_stem")["num_shots"] == 5


def test_mmlu_requires_rendered_choices():
    with pytest.raises(ValueError, match="must end with A-D choices"):
        build_eval_prompt(
            RecordingTokenizer(),
            "question",
            benchmark="mmlu_stem",
            protocol="qwen2_5_math_base",
        )
