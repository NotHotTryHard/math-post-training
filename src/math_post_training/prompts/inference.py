"""Build the prompt used by the interactive ``model-generate`` command."""


def build_inference_prompt(
    tokenizer,
    user_prompt,
    *,
    system_prompt=None,
    raw=False,
):
    """Return the exact string that the generation backend will tokenize."""

    if raw:
        return user_prompt

    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str):
        raise TypeError("apply_chat_template(tokenize=False) did not return a string")
    return rendered
