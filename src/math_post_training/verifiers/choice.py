"""Verify multiple-choice answers."""


def check_choice_answer(reference, prediction):
    """Compare one extracted A-D choice with the reference label."""

    normalized = prediction.strip().upper().strip("().")
    parsed = normalized in {"A", "B", "C", "D"}
    return parsed, parsed and normalized == reference.strip().upper()
