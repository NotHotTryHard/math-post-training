"""GSM8K source adapter."""

from math_post_training.data.schema import MathExample

DATASET_ID = "openai/gsm8k"
ANSWER_DELIMITER = "####"


def normalize(row):
    """Convert a GSM8K row to ``MathExample``."""

    solution = row["answer"].strip()
    _, delimiter, answer = solution.rpartition(ANSWER_DELIMITER)
    if not delimiter or not answer.strip():
        raise ValueError("GSM8K solution has no answer after ####")

    return MathExample(
        problem=row["question"].strip(),
        solution=solution,
        answer=answer.strip(),
        source=f"{DATASET_ID}:main",
    )
