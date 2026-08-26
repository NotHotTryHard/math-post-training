"""Score completions and aggregate equivalent mathematical answers."""

from typing import TypedDict

from math_post_training.verifiers.choice import check_choice_answer
from math_post_training.verifiers.extraction import extract_final_answer, follows_answer_format
from math_post_training.verifiers.math import check_answer


class CompletionScore(TypedDict):
    completion: str
    extracted_answer: str | None
    extraction_method: str
    parsed: bool
    correct: bool
    format_ok: bool | None
    completion_tokens: int
    truncated: bool


class VoteResult(TypedDict):
    vote_count: int
    valid_vote_count: int
    vote_tied: bool


def score_completion(tokenizer, completion, *, reference, settings, max_new_tokens):
    """Parse and verify one generated completion under an evaluation protocol."""

    completion_tokens = len(tokenizer.encode(completion, add_special_tokens=False))
    truncated = completion_tokens >= max_new_tokens - 1
    prediction, extraction_method = extract_final_answer(
        completion,
        answer_kind=settings["answer_kind"],
    )
    if truncated and extraction_method in {"last_number", "last_choice"}:
        parsed, correct = False, False
    elif settings["answer_kind"] == "choice":
        parsed, correct = check_choice_answer(reference, prediction)
    else:
        parsed, correct = check_answer(reference, prediction)
    return CompletionScore(
        completion=completion,
        extracted_answer=prediction,
        extraction_method=extraction_method,
        parsed=parsed,
        correct=correct,
        format_ok=follows_answer_format(
            completion,
            answer_format=settings["required_answer_format"],
        ),
        completion_tokens=completion_tokens,
        truncated=truncated,
    )


def select_majority_vote(records, *, answer_kind):
    """Select the first completion in the largest answer-equivalence cluster."""

    clusters = []
    valid_records = [record for record in records if record["parsed"]]
    for record in valid_records:
        for cluster in clusters:
            if _answers_equivalent(
                cluster[0]["extracted_answer"],
                record["extracted_answer"],
                answer_kind=answer_kind,
            ):
                cluster.append(record)
                break
        else:
            clusters.append([record])

    if not clusters:
        return records[0], VoteResult(
            vote_count=0,
            valid_vote_count=0,
            vote_tied=False,
        )

    vote_count = max(len(cluster) for cluster in clusters)
    winners = [cluster for cluster in clusters if len(cluster) == vote_count]
    return winners[0][0], VoteResult(
        vote_count=vote_count,
        valid_vote_count=len(valid_records),
        vote_tied=len(winners) > 1,
    )


def _answers_equivalent(left, right, *, answer_kind):
    if answer_kind == "choice":
        parsed, equivalent = check_choice_answer(left, right)
    else:
        parsed, equivalent = check_answer(left, right)
    return parsed and equivalent
