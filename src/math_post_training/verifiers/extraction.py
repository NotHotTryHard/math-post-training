"""Extract the final-answer part of a model completion."""

import re

ANSWER_MARKER = re.compile(r"(?:the\s+answer|final\s+answer)\s+is\s*:?", re.IGNORECASE)
NUMBER = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?%?")
CHOICE = re.compile(r"(?<![A-Za-z])[A-D](?![A-Za-z])", re.IGNORECASE)


def extract_final_answer(completion, *, answer_kind="math", delimiter="####"):
    """Extract one answer and report which rule found it.

    Explicit formats win. The final-number/final-choice rule is intentionally a
    last-resort heuristic, and its use is recorded in evaluation outputs.
    """

    boxed = extract_last_boxed(completion)
    if boxed is not None:
        return boxed, "boxed"

    _, found, answer = completion.rpartition(delimiter)
    if found and answer.strip():
        return _first_answer_line(answer), "delimiter"

    markers = list(ANSWER_MARKER.finditer(completion))
    if markers:
        answer = _first_answer_line(completion[markers[-1].end() :])
        if answer:
            return answer, "answer_marker"

    if answer_kind == "choice":
        choices = CHOICE.findall(completion)
        if choices:
            return choices[-1].upper(), "last_choice"
    elif answer_kind == "math":
        numbers = NUMBER.findall(completion)
        if numbers:
            return numbers[-1].replace(",", ""), "last_number"
    else:
        raise ValueError(f"Unknown answer kind: {answer_kind!r}")

    return "", "not_found"


def follows_answer_format(completion, delimiter="####", answer_format="delimiter"):
    """Check a requested final-answer format; return None when none is required."""

    if answer_format is None:
        return None
    if answer_format == "boxed":
        return extract_last_boxed(completion) is not None
    if answer_format != "delimiter":
        raise ValueError(f"Unknown answer format: {answer_format!r}")

    _, found, answer = completion.rpartition(delimiter)
    return bool(found and answer.strip())


def extract_last_boxed(text):
    """Return the contents of the final ``\\boxed{...}``, including nested braces."""

    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None

    content_start = start + len(marker)
    depth = 1
    for index in range(content_start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:index].strip()
    return None


def _first_answer_line(text):
    """Return the first non-empty line after an explicit answer marker."""

    return next((line.strip() for line in text.splitlines() if line.strip()), "")
