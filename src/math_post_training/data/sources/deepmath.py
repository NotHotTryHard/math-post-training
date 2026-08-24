"""DeepMath-103K source adapter."""

from math_post_training.data.schema import MathExample

DATASET_ID = "zwhe99/DeepMath-103K"


def normalize(row):
    """Convert one original DeepMath-103K row to ``MathExample``."""

    solution = row["r1_solution_1"].strip()
    answer = row["final_answer"].strip()
    topic = row["topic"].strip()

    return MathExample(
        problem=row["question"].strip(),
        solution=solution or None,
        answer=answer or None,
        source=DATASET_ID,
        difficulty=float(row["difficulty"]),
        topic=topic or None,
    )
