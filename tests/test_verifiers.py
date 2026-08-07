import pytest

pytest.importorskip("math_verify")

from math_post_training.verifiers.choice import check_choice_answer  # noqa: E402
from math_post_training.verifiers.extraction import (  # noqa: E402
    extract_final_answer,
    extract_last_boxed,
    follows_answer_format,
)
from math_post_training.verifiers.math import check_answer  # noqa: E402


def test_final_answer_extraction_uses_last_delimiter():
    completion = "A draft #### 3\nCorrection: #### $\\frac{1}{2}$"

    assert extract_final_answer(completion) == ("$\\frac{1}{2}$", "delimiter")
    assert follows_answer_format(completion)


def test_boxed_answer_extraction_handles_nested_braces():
    completion = r"Therefore, the answer is \\boxed{\\frac{1}{2}}."

    assert extract_last_boxed(completion) == r"\\frac{1}{2}"
    assert extract_final_answer(completion) == (r"\\frac{1}{2}", "boxed")
    assert follows_answer_format(completion, answer_format="boxed")


def test_math_verifier_accepts_equivalent_forms():
    parsed, correct = check_answer(r"\frac{1}{2}", "$0.5$")

    assert parsed
    assert correct


@pytest.mark.parametrize(
    ("reference", "prediction"),
    [
        (r"\sqrt{13}", r"\sqrt{13}"),
        (r"-\frac{24}{25}", r"$-0.96$"),
        (r"987,\!436", "987436"),
        ("(-13,-16,-18)", r"$(-13,-16,-18)$"),
    ],
)
def test_math_verifier_parses_extracted_latex(reference, prediction):
    assert check_answer(reference, prediction) == (True, True)


def test_unparseable_text_is_not_silently_accepted():
    assert check_answer("not math", "not math") == (False, False)


def test_answer_marker_wins_over_numbers_in_reasoning():
    completion = "We started with 100 and divided by 4. The answer is 25."

    assert extract_final_answer(completion) == ("25.", "answer_marker")


def test_last_number_is_an_explicit_fallback():
    completion = "We started with 100 and eventually obtained 25"

    assert extract_final_answer(completion) == ("25", "last_number")


def test_choice_fallback_uses_the_last_standalone_choice():
    completion = "A seems plausible, but after checking I choose C"

    assert extract_final_answer(completion, answer_kind="choice") == ("C", "last_choice")


def test_choice_verifier_uses_exact_option_letter():
    assert check_choice_answer("C", "(c)") == (True, True)
    assert check_choice_answer("C", "D") == (True, False)
    assert check_choice_answer("C", "maybe") == (False, False)
