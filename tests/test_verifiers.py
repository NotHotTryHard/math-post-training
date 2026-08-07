import pytest

pytest.importorskip("math_verify")

from math_post_training.verifiers.extraction import (  # noqa: E402
    extract_final_answer,
    extract_last_boxed,
    follows_answer_format,
)
from math_post_training.verifiers.math import check_answer  # noqa: E402


def test_final_answer_extraction_uses_last_delimiter():
    completion = "A draft #### 3\nCorrection: #### $\\frac{1}{2}$"

    assert extract_final_answer(completion) == "$\\frac{1}{2}$"
    assert follows_answer_format(completion)


def test_boxed_answer_extraction_handles_nested_braces():
    completion = r"Therefore, the answer is \\boxed{\\frac{1}{2}}."

    assert extract_last_boxed(completion) == r"\\frac{1}{2}"
    assert extract_final_answer(completion, answer_format="boxed") == r"\\frac{1}{2}"
    assert follows_answer_format(completion, answer_format="boxed")


def test_math_verifier_accepts_equivalent_forms():
    parsed, correct = check_answer(r"\frac{1}{2}", "$0.5$")

    assert parsed
    assert correct


def test_multiple_answer_verifier_checks_every_blank():
    assert check_answer("$5$;$10$", "$5$; $10$", multiple=True) == (True, True)
    assert check_answer("$5$;$10$", "$7$; $10$", multiple=True) == (True, False)


def test_textual_gaokao_answer_has_exact_match_fallback():
    answer = "三组对面分别平行;对角线互相平分"

    assert check_answer(answer, answer, multiple=True) == (False, True)
