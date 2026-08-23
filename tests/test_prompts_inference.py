from math_post_training.prompts.inference import build_inference_prompt
from math_post_training.prompts.training import build_math_prompt


def test_inference_prompt_uses_the_shared_math_template():
    assert build_inference_prompt("What is 2 + 2?") == build_math_prompt(
        "What is 2 + 2?"
    )


def test_raw_inference_prompt_is_returned_unchanged():
    assert build_inference_prompt("raw text", raw=True) == "raw text"
