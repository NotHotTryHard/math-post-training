from math_post_training.prompts.evaluation import (
    build_evaluation_prompt,
    get_evaluation_settings,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return messages


def test_base_protocol_uses_published_shot_counts():
    assert get_evaluation_settings("qwen2_5_math_base", "gsm8k")["num_shots"] == 8
    assert get_evaluation_settings("qwen2_5_math_base", "math")["num_shots"] == 4
    assert get_evaluation_settings("qwen2_5_math_base", "gsm8k")["stop_strings"] == ["Question:"]


def test_base_gsm_prompt_is_raw_and_ends_with_target_problem():
    prompt = build_evaluation_prompt(
        FakeTokenizer(),
        "What is 1 + 1?",
        benchmark="gsm8k",
        protocol="qwen2_5_math_base",
    )

    assert prompt.count("Question:") == 9
    assert prompt.endswith("Question: What is 1 + 1?\nLet's think step by step")


def test_instruct_protocol_uses_chat_template_and_boxed_system_prompt():
    messages = build_evaluation_prompt(
        FakeTokenizer(),
        "What is 1 + 1?",
        benchmark="gsm8k",
        protocol="qwen2_5_math_instruct",
    )

    assert messages[0]["role"] == "system"
    assert r"\boxed{}" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "What is 1 + 1?"}
