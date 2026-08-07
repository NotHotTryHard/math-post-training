"""Extract the final-answer part of a model completion."""


def extract_final_answer(completion, delimiter="####", answer_format=None):
    """Extract an explicitly formatted answer, or return the whole completion."""

    if answer_format == "boxed":
        boxed = extract_last_boxed(completion)
        if boxed is not None:
            return boxed

    _, found, answer = completion.rpartition(delimiter)
    if found:
        return answer.strip()
    return completion.strip()


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
