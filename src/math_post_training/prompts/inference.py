"""Build the prompt used by the interactive ``model-generate`` command."""

from math_post_training.prompts.training import build_math_prompt


def build_inference_prompt(
    user_prompt,
    *,
    raw=False,
):
    """Return the exact plain-text string that the generation backend tokenizes."""

    if raw:
        return user_prompt
    return build_math_prompt(user_prompt)
