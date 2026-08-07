from math_post_training.evaluation_prompts import (
    protocol_metadata,
    render_evaluation_prompt,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return messages


def test_base_protocol_uses_published_shot_counts():
    assert protocol_metadata("qwen2_5_math_base", "gsm8k")["num_shots"] == 8
    assert protocol_metadata("qwen2_5_math_base", "math")["num_shots"] == 4
    assert protocol_metadata("qwen2_5_math_base", "gaokao_math_cloze")["num_shots"] == 5
    assert protocol_metadata("qwen2_5_math_base", "gsm8k")["stop_strings"] == ["Question:"]


def test_base_gsm_prompt_is_raw_and_ends_with_target_problem():
    prompt = render_evaluation_prompt(
        FakeTokenizer(),
        "What is 1 + 1?",
        benchmark="gsm8k",
        protocol="qwen2_5_math_base",
    )

    assert prompt.count("Question:") == 9
    assert prompt.endswith("Question: What is 1 + 1?\nLet's think step by step")


def test_instruct_protocol_uses_chat_template_and_boxed_system_prompt():
    messages = render_evaluation_prompt(
        FakeTokenizer(),
        "What is 1 + 1?",
        benchmark="gsm8k",
        protocol="qwen2_5_math_instruct",
    )

    assert messages[0]["role"] == "system"
    assert r"\boxed{}" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": "What is 1 + 1?"}
