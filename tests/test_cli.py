from math_post_training import cli


class RecordingTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        assert messages == [{"role": "user", "content": "hello"}]
        return "rendered prompt"


class FakeBackend:
    def __init__(self, model, tokenizer, device):
        pass

    def generate(self, prompts, config):
        assert prompts == ["rendered prompt"]
        return [["completion"]]


class FakeVLLMBackend:
    def __init__(self, **kwargs):
        assert kwargs == {
            "name_or_path": "fake-model",
            "tensor_parallel_size": 2,
        }
        self.tokenizer = RecordingTokenizer()

    def generate(self, prompts, config):
        assert prompts == ["rendered prompt"]
        return [["completion"]]


def test_generate_can_show_exact_rendered_prompt(monkeypatch, capsys):
    config = {
        "model": {"name_or_path": "fake-model"},
        "inference": {"backend": "transformers"},
    }
    monkeypatch.setattr(cli, "load_yaml_config", lambda path: config)
    monkeypatch.setattr(
        cli,
        "load_model_and_tokenizer",
        lambda **kwargs: (object(), RecordingTokenizer()),
    )
    monkeypatch.setattr(cli, "TransformersBackend", FakeBackend)

    assert cli.generate_main(["hello", "--show-prompt"]) == 0
    assert capsys.readouterr().out == (
        "=== prompt sent to model ===\nrendered prompt\n=== completion ===\ncompletion\n"
    )


def test_generate_supports_vllm_backend(monkeypatch, capsys):
    config = {
        "model": {"name_or_path": "fake-model"},
        "vllm": {"tensor_parallel_size": 2},
        "inference": {"backend": "vllm"},
    }
    monkeypatch.setattr(cli, "load_yaml_config", lambda path: config)
    monkeypatch.setattr(cli, "VLLMBackend", FakeVLLMBackend)

    assert cli.generate_main(["hello"]) == 0
    assert capsys.readouterr().out == "completion\n"
