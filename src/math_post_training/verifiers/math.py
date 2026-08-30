"""Parse and compare numeric or symbolic mathematical answers."""

from math_verify import parse, verify


def check_answer(reference, prediction):
    """Parse two mathematical answers and check their equivalence."""

    reference_boolean = _boolean_answer(reference)
    if reference_boolean is not None:
        prediction_boolean = _boolean_answer(prediction)
        return prediction_boolean is not None, prediction_boolean == reference_boolean

    # DeepMath contains multiple-choice examples whose references are stored as
    # plain letters (for example, ``A``) rather than ``\text{A}``.
    reference_choices = _text_choices(reference, allow_plain=True)
    if reference_choices is not None:
        prediction_choices = _text_choices(prediction, allow_plain=True)
        return prediction_choices is not None, prediction_choices == reference_choices

    parsed_reference = _parse_answer(reference)
    parsed_prediction = _parse_answer(prediction)

    if not parsed_reference or not parsed_prediction:
        return bool(parsed_prediction), False

    return True, verify(parsed_reference, parsed_prediction)


def _boolean_answer(answer):
    """Normalize DeepMath's equivalent Boolean final-answer spellings."""

    answer = answer.strip().strip("$").strip().replace("\\\\", "\\")
    for wrapper in (r"\text{", r"\mathrm{"):
        if answer.startswith(wrapper) and answer.endswith("}"):
            answer = answer[len(wrapper) : -1].strip()
            break
    normalized = answer.casefold()
    if normalized in {"yes", "true"}:
        return True
    if normalized in {"no", "false"}:
        return False
    return None


def _text_choices(answer, *, allow_plain=False):
    """Parse MATH answers such as ``\text{C,E}`` as comma-separated choices."""

    answer = answer.strip()
    if answer.startswith(r"\text{") and answer.endswith("}"):
        answer = answer[len(r"\text{") : -1]
    elif not allow_plain:
        return None

    choices = tuple(part.strip().upper() for part in answer.split(","))
    if not choices or any(choice not in {"A", "B", "C", "D", "E"} for choice in choices):
        return None
    return choices


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
