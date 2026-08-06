"""Prompt construction and model chat-template helpers."""


def render_chat_prompt(
    tokenizer,
    user_prompt,
    *,
    system_prompt=None,
):
    """Turn chat messages into the exact text expected by the model."""

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
