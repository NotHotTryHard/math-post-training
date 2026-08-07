"""Parse and compare numeric or symbolic mathematical answers."""

from math_verify import parse, verify


def check_answer(reference, prediction):
    """Parse two mathematical answers and check their equivalence."""

    parsed_reference = parse(reference)
    parsed_prediction = parse(prediction)

    if not parsed_reference or not parsed_prediction:
        return bool(parsed_prediction), False

    return True, verify(parsed_reference, parsed_prediction)
