"""Parse and compare numeric or symbolic mathematical answers."""

from math_verify import parse, verify


def check_answer(reference, prediction):
    """Parse two mathematical answers and check their equivalence."""

    parsed_reference = _parse_answer(reference)
    parsed_prediction = _parse_answer(prediction)

    if not parsed_reference or not parsed_prediction:
        return bool(parsed_prediction), False

    return True, verify(parsed_reference, parsed_prediction)


def _parse_answer(answer):
    """Give extracted LaTeX the delimiters that ``math_verify`` expects."""

    answer = answer.replace(r"\!", "").strip()
    if not answer:
        return []

    already_delimited = answer.startswith("$") and answer.endswith("$")
    structured_expression = answer.startswith(("(", "[", "{")) and answer.endswith((")", "]", "}"))
    if not already_delimited and ("\\" in answer or structured_expression):
        answer = f"${answer}$"

    return parse(answer)
