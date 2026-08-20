"""Create stage-specific rows from canonical mathematical examples."""


def to_sft_example(example):
    """Build a conversational prompt-completion row for TRL SFT."""

    if not example["solution"]:
        raise ValueError("SFT example has no solution")

    return {
        "prompt": [
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
    """Keep the reference answer used by the reward function."""

    if not example["answer"]:
        raise ValueError("GRPO example has no reference answer")

    return {
        "problem": example["problem"],
        "answer": example["answer"],
        "source": example["source"],
    }
