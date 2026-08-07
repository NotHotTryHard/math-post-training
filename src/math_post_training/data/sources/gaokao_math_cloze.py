"""GaoKao Math Cloze source adapter."""

from math_post_training.data.schema import MathExample

DATASET_ID = "RUCAIBox/agieval:gaokao-mathcloze"


def normalize(row):
    """Convert one AGIEval GaoKao Math Cloze row to ``MathExample``."""

    parts = [row.get("passage"), row["question"]]
    problem = "\n\n".join(part.strip() for part in parts if part and part.strip())

    return MathExample(
        problem=problem,
        solution=row.get("explanation") or None,
        answer=row["label"].strip(),
        source=DATASET_ID,
    )
