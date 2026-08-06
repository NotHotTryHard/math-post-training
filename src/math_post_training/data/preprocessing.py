"""Create stage-specific rows from canonical mathematical examples."""


def to_sft_example(example):
    """Keep the demonstrated solution used as the SFT target."""

    if not example["solution"]:
        raise ValueError("SFT example has no solution")

    return {
        "problem": example["problem"],
        "completion": example["solution"],
        "source": example["source"],
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
