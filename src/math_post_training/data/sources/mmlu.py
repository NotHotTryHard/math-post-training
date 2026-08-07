"""MMLU multiple-choice source adapter."""

from math_post_training.data.schema import MathExample

DATASET_ID = "cais/mmlu"
LETTERS = "ABCD"


def normalize(row):
    """Render the choices and keep the correct option letter."""

    choices = row["choices"]
    if len(choices) != len(LETTERS):
        raise ValueError(f"MMLU example must have four choices, got {len(choices)}")

    answer_index = int(row["answer"])
    if answer_index not in range(len(LETTERS)):
        raise ValueError(f"Invalid MMLU answer index: {answer_index}")

    subject = row["subject"].strip()
    choices_text = "\n".join(
        f"{letter}. {choice.strip()}" for letter, choice in zip(LETTERS, choices, strict=True)
    )
    return MathExample(
        problem=f"{row['question'].strip()}\n{choices_text}",
        solution=None,
        answer=LETTERS[answer_index],
        source=f"{DATASET_ID}:{subject}",
    )
