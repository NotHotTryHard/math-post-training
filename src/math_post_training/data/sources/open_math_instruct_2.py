"""OpenMathInstruct-2 source adapter."""

from math_post_training.data.schema import MathExample

DATASET_ID = "nvidia/OpenMathInstruct-2"


def normalize(row):
    """Convert an OpenMathInstruct-2 row to ``MathExample``."""

    solution = row["generated_solution"].strip()
    answer = row["expected_answer"].strip()
    problem_source = row["problem_source"].strip()

    return MathExample(
        problem=row["problem"].strip(),
        solution=solution or None,
        answer=answer or None,
        source=f"{DATASET_ID}:{problem_source}",
    )
