"""MATH benchmark source adapter."""

from math_post_training.data.schema import MathExample

DATASET_ID = "EleutherAI/hendrycks_math"


def normalize(row):
    """Convert one MATH row and extract its boxed reference answer."""

    solution = row["solution"].strip()
    category = row["type"].strip().lower().replace(" ", "_")

    return MathExample(
        problem=row["problem"].strip(),
        solution=solution,
        answer=_last_boxed_answer(solution),
        source=f"{DATASET_ID}:{category}",
    )


def _last_boxed_answer(text):
    r"""Return the contents of the last ``\boxed{...}`` or ``\fbox{...}``."""

    boxed_at = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if boxed_at < 0:
        raise ValueError("MATH solution has no boxed answer")

    opening_brace = text.find("{", boxed_at)
    if opening_brace < 0:
        raise ValueError("MATH boxed answer has no opening brace")

    depth = 0
    for index in range(opening_brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                answer = text[opening_brace + 1 : index].strip()
                if not answer:
                    raise ValueError("MATH boxed answer is empty")
                return answer

    raise ValueError("MATH boxed answer has unbalanced braces")
