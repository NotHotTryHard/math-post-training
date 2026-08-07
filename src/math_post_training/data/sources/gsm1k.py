"""GSM1k source adapter."""

from math_post_training.data.schema import MathExample

DATASET_ID = "ScaleAI/gsm1k"


def normalize(row):
    """Convert a GSM1k row to ``MathExample``."""

    return MathExample(
        problem=row["question"].strip(),
        solution=None,
        answer=row["answer"].strip(),
        source=DATASET_ID,
    )
