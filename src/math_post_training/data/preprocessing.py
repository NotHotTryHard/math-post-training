"""Create stage-specific rows from canonical mathematical examples."""

from math_post_training.prompts.training import MATH_SYSTEM_PROMPT


def to_sft_example(example):
    """Build a conversational prompt-completion row for TRL SFT."""

    if not example["solution"]:
        raise ValueError("SFT example has no solution")

    return {
        "prompt": [
            {
                "role": "system",
                "content": MATH_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": example["problem"],
            }
        ],
        "completion": [
            {
                "role": "assistant",
                "content": example["solution"],
            }
        ],
    }


def to_grpo_example(example):
    """Build a prompt row while keeping metadata used by GRPO rewards."""

    if not example["answer"]:
        raise ValueError("GRPO example has no reference answer")

    return {
        "prompt": [
            {
                "role": "system",
                "content": MATH_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": example["problem"],
            },
        ],
        "answer": example["answer"],
        "source": example["source"],
    }
