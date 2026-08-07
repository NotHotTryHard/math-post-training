"""Parse and compare numeric or symbolic mathematical answers."""

from math_verify import parse, verify


def check_answer(reference, prediction, *, multiple=False):
    """Return ``(parsed, correct)`` for one answer or semicolon-separated blanks."""

    references = _parts(reference, multiple)
    predictions = _parts(prediction, multiple)

    parsed_references = [parse(part) for part in references]
    parsed_predictions = [parse(part) for part in predictions]
    parsed = bool(parsed_predictions) and all(parsed_predictions)

    if len(parsed_references) != len(parsed_predictions):
        return parsed, False

    correct = all(
        verify(gold, answer) if gold and answer else _same_text(reference, prediction)
        for reference, prediction, gold, answer in zip(
            references,
            predictions,
            parsed_references,
            parsed_predictions,
            strict=True,
        )
    )
    return parsed, correct


def _parts(answer, multiple):
    if not multiple:
        return [answer.strip()]
    return [part.strip() for part in answer.split(";") if part.strip()]


def _same_text(reference, prediction):
    return "".join(reference.split()).strip("$") == "".join(prediction.split()).strip("$")
