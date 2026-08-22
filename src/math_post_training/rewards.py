"""Rule-based rewards for reinforcement learning on verifiable math tasks."""

from math_post_training.verifiers.choice import check_choice_answer
from math_post_training.verifiers.extraction import (
    extract_final_answer,
    follows_answer_format,
)
from math_post_training.verifiers.math import check_answer


def accuracy_reward(completions, answer, **_kwargs):
    """Return one when a completion's final answer matches its reference."""

    rewards = []
    for completion, reference in zip(completions, answer, strict=True):
        text = _completion_text(completion)
        if _is_choice_answer(reference):
            prediction, _ = extract_final_answer(text, answer_kind="choice")
            _, correct = check_choice_answer(reference, prediction)
        else:
            prediction, _ = extract_final_answer(text)
            _, correct = check_answer(reference, prediction)
        rewards.append(float(correct))
    return rewards


def boxed_format_reward(completions, **_kwargs):
    """Return one when a completion contains a complete final ``\\boxed{}``."""

    return [
        float(follows_answer_format(_completion_text(completion), answer_format="boxed"))
        for completion in completions
    ]


def _completion_text(completion):
    """Read either a plain or conversational completion produced by TRL."""

    if isinstance(completion, str):
        return completion
    if not completion or not isinstance(completion[-1], dict):
        raise TypeError("Completion must be text or a non-empty list of messages")

    content = completion[-1].get("content")
    if not isinstance(content, str):
        raise TypeError("The final completion message must contain text")
    return content


def _is_choice_answer(reference):
    return reference.strip().upper() in {"A", "B", "C", "D"}
