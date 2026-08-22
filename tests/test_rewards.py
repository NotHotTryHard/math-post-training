import pytest

from math_post_training.rewards import accuracy_reward, boxed_format_reward


def _completion(text):
    return [{"role": "assistant", "content": text}]


def test_accuracy_reward_verifies_math_answers():
    completions = [
        _completion(r"The result is \\boxed{\\frac{1}{2}}."),
        _completion(r"The result is \\boxed{2}."),
    ]

    assert accuracy_reward(completions, [r"\\frac{1}{2}", "3"]) == [1.0, 0.0]


def test_accuracy_reward_verifies_multiple_choice_answers():
    completions = [
        _completion("After checking, the final answer is C."),
        _completion("After checking, the final answer is A."),
    ]

    assert accuracy_reward(completions, ["C", "D"]) == [1.0, 0.0]


def test_boxed_format_reward_accepts_only_complete_boxes():
    completions = [
        _completion(r"Therefore, \\boxed{42}."),
        _completion(r"Therefore, \\boxed{42"),
        _completion("The answer is 42."),
    ]

    assert boxed_format_reward(completions) == [1.0, 0.0, 0.0]


def test_rewards_accept_plain_completion_text():
    assert accuracy_reward([r"\\boxed{7}"], ["7"]) == [1.0]
    assert boxed_format_reward([r"\\boxed{7}"]) == [1.0]


def test_accuracy_reward_rejects_misaligned_batches():
    with pytest.raises(ValueError):
        accuracy_reward([_completion(r"\\boxed{1}")], ["1", "2"])


def test_reward_rejects_a_malformed_conversational_completion():
    with pytest.raises(TypeError, match="non-empty list of messages"):
        boxed_format_reward([[]])
