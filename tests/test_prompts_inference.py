from math_post_training.prompts.inference import build_inference_prompt


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        self.messages = messages
        return "rendered prompt"


def test_inference_prompt_builds_messages_and_applies_chat_template():
    tokenizer = FakeTokenizer()

    prompt = build_inference_prompt(
        tokenizer,
        "Hello",
        system_prompt="Be concise",
    )

    assert prompt == "rendered prompt"
    assert tokenizer.messages == [
        {"role": "system", "content": "Be concise"},
        {"role": "user", "content": "Hello"},
    ]


def test_raw_inference_prompt_is_returned_unchanged():
    assert build_inference_prompt(None, "raw text", raw=True) == "raw text"
