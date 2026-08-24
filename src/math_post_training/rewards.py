"""Rule-based rewards for reinforcement learning on verifiable math tasks."""

from math_post_training.verifiers.extraction import (
    extract_final_answer,
    extract_last_boxed,
    follows_answer_format,
)
from math_post_training.verifiers.math import check_answer


def accuracy_reward(completions, answer, **_kwargs):
    """Return one when a completion's final answer matches its reference."""

    rewards = []
    for completion, reference in zip(completions, answer, strict=True):
        text = _completion_text(completion)
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


def strict_boxed_reward(completions, answer, **_kwargs):
    """Reward only a verified boxed answer and penalize a missing box.

    A correct ``\\boxed{}`` earns 1, an incorrect box earns 0, and a missing or
    incomplete box earns -0.5.  Unlike :func:`accuracy_reward`, this deliberately
    has no delimiter, answer-marker, or last-number fallback.
    """

    rewards = []
    for completion, reference in zip(completions, answer, strict=True):
        prediction = extract_last_boxed(_completion_text(completion))
        if prediction is None:
            rewards.append(-0.5)
            continue
        _, correct = check_answer(reference, prediction)
        rewards.append(float(correct))
    return rewards


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
